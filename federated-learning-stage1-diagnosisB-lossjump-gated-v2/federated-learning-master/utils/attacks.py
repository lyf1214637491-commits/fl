#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Utilities for drift and backdoor experiments (stage-1)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class DriftConfig:
    drift_round: int
    p_remap: float
    mapping: torch.Tensor  # shape [num_classes], dtype long, on CPU


@dataclass(frozen=True)
class AttackConfig:
    attack_start_round: int
    poison_rate: float
    target_label: int
    trigger_size: int
    trigger_value: float


def build_label_mapping(num_classes: int, severity: str, seed: int) -> np.ndarray:
    """Return an array M where new_label = M[old_label]."""
    severity = severity.lower()
    if num_classes <= 1:
        raise ValueError('num_classes must be > 1')

    if severity not in {'mild', 'moderate', 'severe'}:
        raise ValueError(f'Unknown drift severity: {severity}')

    mapping = np.arange(num_classes, dtype=np.int64)

    if severity == 'mild':
        swaps = [(0, 1)]
        for a, b in swaps:
            if a < num_classes and b < num_classes:
                mapping[a], mapping[b] = mapping[b], mapping[a]
    elif severity == 'moderate':
        swaps = [(0, 1), (2, 3)]
        for a, b in swaps:
            if a < num_classes and b < num_classes:
                mapping[a], mapping[b] = mapping[b], mapping[a]
    else:  # severe
        rng = np.random.RandomState(seed)
        mapping = rng.permutation(num_classes).astype(np.int64)

    return mapping


def remap_labels(labels: torch.Tensor, mapping: torch.Tensor, p_remap: float) -> torch.Tensor:
    """Randomly remap a subset of labels with probability p_remap per sample."""
    if p_remap <= 0:
        return labels
    # mapping is stored on CPU; always move it to the label tensor's device
    mapping_dev = mapping.to(device=labels.device)

    if p_remap >= 1:
        # IMPORTANT: ensure the returned tensor stays on the same device as labels
        return mapping_dev[labels]

    mask = torch.rand(labels.shape, device=labels.device) < float(p_remap)
    out = labels.clone()
    out[mask] = mapping_dev[out[mask]]
    return out


def apply_trigger(images: torch.Tensor, trigger_size: int, trigger_value: float) -> torch.Tensor:
    """Apply a fixed square patch trigger at the bottom-right corner."""
    if trigger_size <= 0:
        return images
    if images.dim() != 4:
        raise ValueError('images must be a 4D tensor [N,C,H,W]')
    out = images.clone()
    _, _, h, w = out.shape
    ts = min(trigger_size, h, w)
    out[:, :, h - ts:h, w - ts:w] = float(trigger_value)
    return out


def poison_batch(images: torch.Tensor,
                 labels: torch.Tensor,
                 cfg: AttackConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poison a random subset of the batch. Returns (images_p, labels_p, mask)."""
    if cfg.poison_rate <= 0:
        mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
        return images, labels, mask

    mask = torch.rand(labels.shape, device=labels.device) < float(cfg.poison_rate)
    if mask.any():
        images_p = images.clone()
        labels_p = labels.clone()
        images_p[mask] = apply_trigger(images_p[mask], cfg.trigger_size, cfg.trigger_value)
        labels_p[mask] = int(cfg.target_label)
        return images_p, labels_p, mask
    return images, labels, mask
