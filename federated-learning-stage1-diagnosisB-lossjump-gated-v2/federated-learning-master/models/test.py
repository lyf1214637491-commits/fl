#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.attacks import apply_trigger


def test_img(net_g, datatest, args):
    """Standard clean evaluation."""
    net_g.eval()
    test_loss = 0.0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(args.device)
            target = target.to(args.device)

            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    if getattr(args, 'verbose', False):
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss


def test_asr(net_g, datatest, args):
    """Backdoor attack success rate (ASR) on triggered test inputs.

    ASR is defined as: P( f(x ⊕ trigger) == y_target ).
    """
    net_g.eval()
    target_label = int(getattr(args, 'target_label', 0))
    trigger_size = int(getattr(args, 'trigger_size', 3))
    trigger_value = float(getattr(args, 'trigger_value', 5.0))

    correct_target = 0
    total = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(args.device)
            data_t = apply_trigger(data, trigger_size=trigger_size, trigger_value=trigger_value)
            log_probs = net_g(data_t)
            y_pred = log_probs.data.max(1, keepdim=False)[1]
            correct_target += (y_pred == target_label).long().sum().item()
            total += y_pred.numel()

    asr = 100.0 * correct_target / max(total, 1)
    return asr



def test_asr_nt(net_g, datatest, args):
    """ASR on triggered inputs, excluding samples whose *true* label is already the target label.

    This avoids the ~10% baseline inflation on balanced 10-class datasets.
    Returns percentage in [0,100].
    """
    net_g.eval()
    target_label = int(getattr(args, 'target_label', 0))
    trigger_size = int(getattr(args, 'trigger_size', 3))
    trigger_value = float(getattr(args, 'trigger_value', 5.0))

    correct_target = 0
    total = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)
    with torch.no_grad():
        for data, y_true in data_loader:
            data = data.to(args.device)
            y_true = y_true.to(args.device)
            # only evaluate non-target true labels
            mask = (y_true != target_label)
            if mask.any():
                data_t = apply_trigger(data[mask], trigger_size, trigger_value)
                log_probs = net_g(data_t)
                y_pred = log_probs.data.max(1, keepdim=False)[1]
                correct_target += (y_pred == target_label).long().sum().item()
                total += y_pred.numel()

    return 100.0 * correct_target / max(total, 1)

def test_img_with_label_map(net_g, datatest, args, label_map: torch.Tensor):
    """Evaluate under a *new concept* where labels are remapped.

    label_map: tensor M on CPU with shape [num_classes], where new_y = M[old_y].
    """
    net_g.eval()
    test_loss = 0.0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False)
    map_dev = label_map.to(args.device)

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(args.device)
            target = target.to(args.device)
            target_new = map_dev[target]

            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target_new, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target_new.data.view_as(y_pred)).long().sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return accuracy, test_loss
