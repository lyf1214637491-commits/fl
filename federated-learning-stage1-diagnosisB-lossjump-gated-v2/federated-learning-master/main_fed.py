#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy
import csv
import os
import time
from collections import deque
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from utils.attacks import DriftConfig, AttackConfig, build_label_mapping
from utils.diagnosis import DiagnosisState, true_cause_for_round
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img, test_asr, test_asr_nt, test_img_with_label_map


def _severity_tv(
    cur_counts: torch.Tensor,
    cur_prob_sums: torch.Tensor,
    ref_counts: torch.Tensor,
    ref_prob_sums: torch.Tensor,
) -> float:
    """Compute conditional-output total variation severity in [0,1].

    Inputs are aggregated across selected clients:
      - counts[y] = number of (clean) monitor samples with label y
      - prob_sums[y,k] = sum of predicted probabilities for class k over those samples

    The conditional mean vector is p_hat(·|y) = prob_sums[y]/counts[y].
    Severity is the label-frequency-weighted mean TV distance to the reference.
    """

    # Defensive: move to CPU and float.
    cc = cur_counts.detach().cpu().long()
    cp = cur_prob_sums.detach().cpu().float()
    rc = ref_counts.detach().cpu().long()
    rp = ref_prob_sums.detach().cpu().float()

    num_classes = int(cp.shape[0])
    eps = 1e-12

    # Build conditional matrices (rows: true label y; cols: predicted class k).
    cur_mat = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    ref_mat = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    for y in range(num_classes):
        if int(cc[y].item()) > 0:
            cur_mat[y] = cp[y] / max(float(cc[y].item()), 1.0)
        if int(rc[y].item()) > 0:
            ref_mat[y] = rp[y] / max(float(rc[y].item()), 1.0)
        else:
            # If reference has no samples for a class, fall back to uniform.
            ref_mat[y] = torch.full((num_classes,), 1.0 / float(num_classes))

    # TV per label.
    tv_y = 0.5 * torch.abs(cur_mat - ref_mat).sum(dim=1)  # [num_classes]

    total = float(cc.sum().item())
    if total <= 0:
        return 0.0
    w = cc.float() / (total + eps)
    sev = float((w * tv_y).sum().item())
    # Clamp defensively.
    return float(max(0.0, min(1.0, sev)))


def _severity_to_p(sev: str) -> float:
    """Probability of label remapping per sample.

    For *abrupt* drift we want the concept to switch at once for the affected
    clients. Therefore we remap all labels once drift starts.

    Drift "severity" is controlled by how *many* classes participate in the
    mapping (see build_label_mapping), not by partially remapping samples.
    """
    sev = (sev or '').lower()
    if sev in {'mild', 'moderate', 'severe'}:
        return 1.0
    raise ValueError(f'Unknown drift severity: {sev}')


def _auto_trigger_value(dataset: str) -> float:
    """Return a reasonable trigger pixel value *after normalization*.

    We set the patch to correspond to raw pixel value 1.0 under the dataset's normalization.
    """
    if dataset == 'mnist':
        mean, std = 0.1307, 0.3081
        return float((1.0 - mean) / std)
    if dataset == 'cifar':
        mean, std = 0.5, 0.5
        return float((1.0 - mean) / std)
    return 5.0


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu)
                               if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # seeds (best-effort determinism)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        # MNIST is single-channel.
        # The upstream code requires passing --num_channels 1; we enforce it here
        # to avoid silent misconfiguration.
        args.num_channels = 1
        trans_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        # CIFAR-10 is 3-channel RGB.
        args.num_channels = 3
        trans_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    # trigger value (auto if < 0)
    if getattr(args, 'trigger_value', -1.0) < 0:
        args.trigger_value = _auto_trigger_value(args.dataset)

    # determine drift/attacker client sets
    rng = np.random.RandomState(args.seed)
    drift_users = set()
    attacker_users = set()

    if args.scenario in ['drift', 'both']:
        n_drift = max(1, int(round(args.drift_frac * args.num_users)))
        drift_users = set(rng.choice(range(args.num_users), n_drift, replace=False).tolist())

    if args.scenario in ['backdoor', 'both']:
        n_att = max(1, int(round(args.attack_frac * args.num_users)))
        attacker_users = set(rng.choice(range(args.num_users), n_att, replace=False).tolist())

    # configs
    drift_cfg = None
    if args.scenario in ['drift', 'both']:
        p_remap = _severity_to_p(args.drift_sev)
        mapping_np = build_label_mapping(args.num_classes, args.drift_sev, seed=args.seed + 17)
        drift_cfg = DriftConfig(
            drift_round=int(args.drift_round),
            p_remap=float(p_remap),
            mapping=torch.from_numpy(mapping_np).long()
        )

    attack_cfg = None
    if args.scenario in ['backdoor', 'both']:
        attack_cfg = AttackConfig(
            attack_start_round=int(args.attack_start_round),
            poison_rate=float(args.poison_rate),
            target_label=int(args.target_label),
            trigger_size=int(args.trigger_size),
            trigger_value=float(args.trigger_value),
        )

    # build model
    img_size = dataset_train[0][0].shape
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()

    # init CSV log
    os.makedirs(args.save_dir, exist_ok=True)
    if args.log_csv:
        csv_path = args.log_csv
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(args.save_dir, f"stage1_{args.dataset}_{args.model}_{args.scenario}_{ts}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'round', 'avg_train_loss', 'test_acc', 'test_loss', 'asr', 'asr_nt',
            'drift_test_acc', 'drift_test_loss',
            'sev_score',
            'm', 'selected_users', 'n_drift_sel', 'n_att_sel',
            'scenario', 'drift_round', 'drift_frac', 'drift_sev',
            'attack_start_round', 'attack_frac', 'poison_rate', 'target_label',
            'trigger_size', 'trigger_value', 'attack_scale',
            'alarm', 'pred_cause', 'true_cause'
        ])

    print(f"[log] csv_path={csv_path}")
    if drift_users:
        print(f"[config] drift_users={len(drift_users)}/{args.num_users}, drift_round={args.drift_round}, sev={args.drift_sev}")
    if attacker_users:
        print(f"[config] attacker_users={len(attacker_users)}/{args.num_users}, attack_start_round={args.attack_start_round}, poison_rate={args.poison_rate}")

    # training
    loss_train = []

    # stage-1 diagnosis (Path B)
    diag_state = DiagnosisState.from_args(args)

    # Path B severity (loss-jump) configuration
    sev_loss_window = int(getattr(args, 'sev_loss_window', 5))
    sev_loss_warmup = int(getattr(args, 'sev_loss_warmup', 20))
    sev_test_loss_gate = float(getattr(args, 'sev_test_loss_gate', 0.20))
    loss_hist = deque(maxlen=max(sev_loss_window + 1, 2))

    # Keep num_classes for the optional per-client monitoring stats returned by LocalUpdate.
    num_classes = int(getattr(args, 'num_classes', 10))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [net_glob.state_dict() for _ in range(args.num_users)]

    for r in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        n_drift_sel = int(sum((int(u) in drift_users) for u in idxs_users))
        n_att_sel = int(sum((int(u) in attacker_users) for u in idxs_users))
        w_global = copy.deepcopy(net_glob.state_dict())

        # Optional monitoring stats across selected clients (reserved for future
        # severity definitions). Not used by the current loss-jump proxy.
        mon_counts_round = torch.zeros(num_classes, dtype=torch.long)
        mon_prob_sums_round = torch.zeros((num_classes, num_classes), dtype=torch.float32)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, mon_counts, mon_prob_sums = local.train(
                net=copy.deepcopy(net_glob).to(args.device),
                client_id=int(idx),
                round_idx=int(r),
                drift_cfg=drift_cfg,
                attack_cfg=attack_cfg,
                is_drift_client=(idx in drift_users),
                is_attacker=(idx in attacker_users),
            )

            # Accumulate Path B monitoring stats.
            try:
                mon_counts_round += mon_counts.detach().cpu().long()
                mon_prob_sums_round += mon_prob_sums.detach().cpu().float()
            except Exception:
                pass

            # Optional: amplify attacker update (model-replacement style) to make backdoor effect observable in stage-1.
            if attack_cfg is not None and (idx in attacker_users) and r >= int(attack_cfg.attack_start_round):
                s = float(getattr(args, 'attack_scale', 1.0))
                if s != 1.0:
                    for k in w.keys():
                        w[k] = w_global[k] + s * (w[k] - w_global[k])


            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(float(loss))

        # update global weights
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        loss_avg = float(sum(loss_locals) / max(len(loss_locals), 1))
        loss_train.append(loss_avg)
        # Update the loss history for the loss-jump severity proxy.
        loss_hist.append(float(loss_avg))

        # evaluation
        do_eval = (r % max(int(args.eval_every), 1) == 0) or (r == args.epochs - 1)
        sev_score = None
        if do_eval:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            asr = test_asr(net_glob, dataset_test, args)
            asr_nt = test_asr_nt(net_glob, dataset_test, args)
            # If drift is active, also evaluate under the *post-drift* concept
            # (i.e., the ground-truth labels are remapped).
            if drift_cfg is not None and r >= int(drift_cfg.drift_round):
                acc_drift, loss_drift = test_img_with_label_map(net_glob, dataset_test, args, drift_cfg.mapping)
            else:
                acc_drift, loss_drift = None, None
            net_glob.train()

            # ---- Path B severity (loss-jump proxy) ----
            # Compute a robust drift proxy that is relatively insensitive to
            # gradual training dynamics: the *positive* jump in average train loss.
            if (r < int(sev_loss_warmup)) or (len(loss_hist) < int(sev_loss_window) + 1):
                sev_score = 0.0
            else:
                prev = list(loss_hist)[-int(sev_loss_window) - 1:-1]
                base = float(np.mean(prev)) if prev else float(loss_avg)
                sev_score = max(0.0, float(loss_avg) - base)

            # Gates to reduce false drift evidence during normal training or
            # attack-induced transients.
            # 1) If clean test loss is low, ignore the loss jump (it is likely
            #    caused by sampling variance rather than a concept shift).
            if (loss_test is not None) and (float(loss_test) < float(sev_test_loss_gate)):
                sev_score = 0.0
            # 2) If ASR-NT is already high, this round is dominated by backdoor
            #    effects; do not treat it as drift evidence.
            if (asr_nt is not None) and (float(asr_nt) >= float(getattr(args, 'diag_drift_asr_gate', 20.0))):
                sev_score = 0.0

            # diagnosis (Path B)
            alarm, pred_cause = diag_state.update(
                round_idx=int(r),
                severity=None if sev_score is None else float(sev_score),
                asr_nt=float(asr_nt),
            )
            true_cause = true_cause_for_round(
                scenario=str(args.scenario),
                round_idx=int(r),
                drift_round=int(getattr(args, 'drift_round', 0)),
                attack_start_round=int(getattr(args, 'attack_start_round', 0)),
            )
        else:
            acc_test, loss_test, asr, asr_nt = None, None, None, None
            acc_drift, loss_drift = None, None
            alarm, pred_cause, true_cause = None, None, None

        # log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                r, loss_avg,
                '' if acc_test is None else acc_test,
                '' if loss_test is None else loss_test,
                '' if asr is None else asr,
                '' if asr_nt is None else asr_nt,
                '' if acc_drift is None else acc_drift,
                '' if loss_drift is None else loss_drift,
                '' if sev_score is None else sev_score,
                m, ' '.join([str(int(x)) for x in idxs_users]), n_drift_sel, n_att_sel,
                args.scenario, getattr(args, 'drift_round', ''), getattr(args, 'drift_frac', ''), getattr(args, 'drift_sev', ''),
                getattr(args, 'attack_start_round', ''), getattr(args, 'attack_frac', ''), getattr(args, 'poison_rate', ''), getattr(args, 'target_label', ''),
                getattr(args, 'trigger_size', ''), getattr(args, 'trigger_value', ''), float(getattr(args, 'attack_scale', 1.0)),
                '' if alarm is None else int(alarm),
                '' if pred_cause is None else str(pred_cause),
                '' if true_cause is None else str(true_cause),
            ])

        # print progress
        if acc_test is None:
            print(f"Round {r:3d}, avg_train_loss {loss_avg:.4f}")
        else:
            msg = f"Round {r:3d}, avg_train_loss {loss_avg:.4f}, test_acc {acc_test:.2f}, test_loss {loss_test:.4f}, ASR {asr:.2f}"
            if acc_drift is not None:
                msg += f", drift_test_acc {acc_drift:.2f}, drift_test_loss {loss_drift:.4f}"
            if pred_cause is not None:
                msg += f", alarm {int(alarm)}, pred {pred_cause}, true {true_cause}"
            print(msg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.xlabel('round')
    plot_path = os.path.join(args.save_dir, f"loss_{args.dataset}_{args.model}_{args.scenario}.png")
    plt.savefig(plot_path)
    print(f"[log] loss_plot={plot_path}")
