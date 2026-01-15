#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

from utils.attacks import DriftConfig, AttackConfig, remap_labels, poison_batch


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(
        self,
        net,
        client_id=None,
        round_idx=0,
        drift_cfg: DriftConfig = None,
        attack_cfg: AttackConfig = None,
        is_drift_client: bool = False,
        is_attacker: bool = False,
    ):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        # ---- Path B severity monitoring ----
        # We compute conditional output statistics on a small number of early
        # batches (clean-only) to form p_hat(·|y).
        num_classes = int(getattr(self.args, 'num_classes', 10))
        monitor_batches = int(getattr(self.args, 'sev_monitor_batches', 1))
        mon_counts = torch.zeros(num_classes, dtype=torch.long)
        mon_prob_sums = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # Stage-1 experiments: abrupt drift (label remap) and backdoor poisoning
                if drift_cfg is not None and is_drift_client and round_idx >= int(drift_cfg.drift_round):
                    labels = remap_labels(labels, drift_cfg.mapping, float(drift_cfg.p_remap))

                if attack_cfg is not None and is_attacker and round_idx >= int(attack_cfg.attack_start_round):
                    images, labels, poison_mask = poison_batch(images, labels, attack_cfg)
                else:
                    poison_mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)

                net.zero_grad()
                log_probs = net(images)

                # Collect monitoring stats (clean-only, early batches of first epoch).
                if iter == 0 and batch_idx < monitor_batches:
                    with torch.no_grad():
                        probs = torch.softmax(log_probs.detach(), dim=1)
                        clean_mask = ~poison_mask
                        if clean_mask.any():
                            y = labels.detach()[clean_mask].long().cpu()
                            p = probs.detach()[clean_mask].float().cpu()
                            for c in range(num_classes):
                                m = (y == c)
                                if m.any():
                                    mon_prob_sums[c] += p[m].sum(dim=0)
                                    mon_counts[c] += int(m.sum().item())

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), mon_counts, mon_prob_sums

