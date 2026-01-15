#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stage-1 root-cause diagnosis (Path B).

Path B removes the "oracle" drift proxy (drift_test_acc) and replaces it with a
more realistic *severity* signal computed from observed training telemetry.

In stage-1 we focus on **abrupt** drift. A robust proxy in this codebase is the
server-observable *jump* in the averaged client training loss:

  severity_t = loss_avg_t - mean(loss_avg_{t-W : t-1})

Intuition:
  - Abrupt concept/label-mapping drift makes a non-trivial fraction of clients
    suddenly incur higher loss, creating a clear positive jump.
  - Pure backdoor poisoning (even with model-replacement amplification) tends to
    raise ASR/ASR-NT but does not create a consistent loss jump of the same form.

Diagnosis distinguishes {normal, drift, backdoor, both} using:
  - Drift proxy: severity >= sev_thr (with an optional drift-warmup)
  - Backdoor proxy: ASR-NT time-consistency
  - Temporal fusion: evidence "hold" and BOTH "latch"

Note: the severity definition above is deliberately simple for stage-1. It is
easy to replace by more sophisticated drift statistics in later stages.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


LABEL_NORMAL = "normal"
LABEL_DRIFT = "drift"
LABEL_BACKDOOR = "backdoor"
LABEL_BOTH = "both"


@dataclass
class DiagnosisConfig:
    """Configuration for rule-based diagnosis."""

    warmup_rounds: int = 10

    # Separate warmup for drift proxy (avoid early-round training transients).
    drift_warmup_rounds: int = 20

    # Drift proxy threshold (severity is a loss jump; scale depends on model/data).
    sev_thr: float = 0.20

    # Backdoor proxy thresholds (on ASR-NT).
    asr_thr: float = 15.0
    asr_h: int = 2
    asr_std_thr: float = 50.0

    # Temporal memory / latching.
    # For stage-1, drift/backdoor are persistent once detected; keep evidence long.
    hold_rounds: int = 200
    both_latch_rounds: int = 40

    # Gate drift proxy when ASR-NT indicates strong backdoor activity (avoid
    # misclassifying attack-induced transients as drift).
    drift_asr_gate: float = 20.0


class DiagnosisState:
    """Stateful diagnosis with sliding-window statistics."""

    def __init__(self, cfg: DiagnosisConfig):
        self.cfg = cfg
        self._asr_hist: Deque[float] = deque(maxlen=max(int(cfg.asr_h), 1))
        self._last_drift_round = None
        self._last_backdoor_round = None
        self._last_both_round = None

    @staticmethod
    def from_args(args) -> "DiagnosisState":
        cfg = DiagnosisConfig(
            warmup_rounds=int(getattr(args, "diag_warmup", 10)),
            drift_warmup_rounds=int(getattr(args, "diag_drift_warmup", 20)),
            sev_thr=float(getattr(args, "diag_sev_thr", 0.20)),
            asr_thr=float(getattr(args, "diag_asr_thr", 15.0)),
            asr_h=int(getattr(args, "diag_asr_h", 2)),
            asr_std_thr=float(getattr(args, "diag_asr_std_thr", 50.0)),
            hold_rounds=int(getattr(args, "diag_hold_rounds", 200)),
            both_latch_rounds=int(getattr(args, "diag_both_latch_rounds", 40)),
            drift_asr_gate=float(getattr(args, "diag_drift_asr_gate", 20.0)),
        )
        return DiagnosisState(cfg)

    def update(
        self,
        round_idx: int,
        severity: Optional[float],
        asr_nt: Optional[float],
    ) -> Tuple[int, str]:
        """Update state and return (alarm, pred_cause)."""

        # ASR-NT is required for backdoor evidence. Severity is optional; when
        # missing we treat it as 0.0 (i.e., no drift evidence for this round).
        if asr_nt is None:
            return 0, LABEL_NORMAL
        sev_v = 0.0 if severity is None else float(severity)

        r = int(round_idx)
        if r < int(self.cfg.warmup_rounds):
            self._asr_hist.append(float(asr_nt))
            return 0, LABEL_NORMAL

        # --- Drift proxy (severity) ---
        drift_flag = False
        if r >= int(self.cfg.drift_warmup_rounds):
            drift_flag = float(sev_v) >= float(self.cfg.sev_thr)

        # ASR gate: if the system is clearly experiencing strong backdoor
        # activity, do not activate drift on this round. This reduces
        # false BOTH during attack pulses.
        if float(asr_nt) >= float(self.cfg.drift_asr_gate):
            drift_flag = False

        # --- Backdoor proxy (ASR-NT consistency) ---
        self._asr_hist.append(float(asr_nt))
        backdoor_flag = False
        if len(self._asr_hist) == self._asr_hist.maxlen:
            mean = float(np.mean(self._asr_hist))
            std = float(np.std(self._asr_hist))
            if (mean >= float(self.cfg.asr_thr)) and (std <= float(self.cfg.asr_std_thr)):
                backdoor_flag = True

        # --- Temporal fusion ---
        if drift_flag:
            self._last_drift_round = r
        if backdoor_flag:
            self._last_backdoor_round = r

        hold = int(self.cfg.hold_rounds)
        latch = int(self.cfg.both_latch_rounds)

        drift_active = (self._last_drift_round is not None) and ((r - int(self._last_drift_round)) <= hold)
        backdoor_active = (self._last_backdoor_round is not None) and ((r - int(self._last_backdoor_round)) <= hold)

        both_active = False
        if drift_active and backdoor_active:
            both_active = True
            self._last_both_round = r

        if (self._last_both_round is not None) and ((r - int(self._last_both_round)) <= latch):
            both_active = True

        if both_active:
            pred = LABEL_BOTH
        elif drift_active:
            pred = LABEL_DRIFT
        elif backdoor_active:
            pred = LABEL_BACKDOOR
        else:
            pred = LABEL_NORMAL

        alarm = int(pred != LABEL_NORMAL)
        return alarm, pred


def true_cause_for_round(
    scenario: str,
    round_idx: int,
    drift_round: int,
    attack_start_round: int,
) -> str:
    """Return the ground-truth cause label for a given round."""

    scenario = (scenario or "").lower()
    r = int(round_idx)
    dr = int(drift_round)
    ar = int(attack_start_round)

    if scenario == "baseline":
        return LABEL_NORMAL
    if scenario == "drift":
        return LABEL_DRIFT if r >= dr else LABEL_NORMAL
    if scenario == "backdoor":
        return LABEL_BACKDOOR if r >= ar else LABEL_NORMAL
    if scenario == "both":
        drift_on = r >= dr
        attack_on = r >= ar
        if drift_on and attack_on:
            return LABEL_BOTH
        if drift_on:
            return LABEL_DRIFT
        if attack_on:
            return LABEL_BACKDOOR
        return LABEL_NORMAL
    return LABEL_NORMAL
