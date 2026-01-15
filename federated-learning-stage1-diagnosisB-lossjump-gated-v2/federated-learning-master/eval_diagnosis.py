#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate stage-1 diagnosis results from CSV logs.

Usage examples:

  python eval_diagnosis.py --csv ./save/stage1_mnist_cnn_drift_*.csv
  python eval_diagnosis.py --csv ./save/...drift.csv ./save/...backdoor.csv ./save/...both.csv

The script prints:
  - Confusion matrix (rows: true, cols: pred)
  - Per-class precision/recall/F1 and macro-F1
  - Simple detection delay metrics
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np


def _maybe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _recompute_diagnosis(rows: List[Dict[str, str]], args) -> List[Dict[str, str]]:
    """Optionally recompute alarm/pred_cause from logged metrics.

    This avoids rerunning training when tuning thresholds.
    """

    try:
        from utils.diagnosis import DiagnosisState  # type: ignore
    except Exception:
        # If import fails, return rows unchanged.
        return rows

    st = DiagnosisState.from_args(args)

    from collections import deque
    loss_hist = deque(maxlen=max(int(getattr(args, "sev_loss_window", 5)) + 1, 2))
    sev_loss_window = int(getattr(args, "sev_loss_window", 5))
    sev_loss_warmup = int(getattr(args, "sev_loss_warmup", 20))
    force_sev = bool(getattr(args, "force_sev_from_loss", False))

    out: List[Dict[str, str]] = []
    for r in rows:
        rr = dict(r)
        ridx = int(_maybe_float(rr.get("round", "0")))
        sev = _maybe_float(rr.get("sev_score", ""))
        asr_nt = _maybe_float(rr.get("asr_nt", ""))
        test_loss = _maybe_float(rr.get("test_loss", ""))
        loss_avg = _maybe_float(rr.get("avg_train_loss", ""))
        # Update loss history for optional sev recomputation
        if not np.isnan(loss_avg):
            loss_hist.append(float(loss_avg))

        sev_v = None if np.isnan(sev) else float(sev)
        if force_sev or np.isnan(sev):
            # Recompute severity using the stage-1 loss-jump proxy
            if (ridx < int(sev_loss_warmup)) or (len(loss_hist) < int(sev_loss_window) + 1):
                sev_v = 0.0
            else:
                prev = list(loss_hist)[-int(sev_loss_window) - 1:-1]
                base = float(np.mean(prev)) if prev else float(loss_avg)
                sev_v = max(0.0, float(loss_avg) - base)

        asr_nt_v = None if np.isnan(asr_nt) else float(asr_nt)

        # Apply the same severity gates used online (Path B):
        # - Ignore loss-jump severity when clean test loss is low.
        # - Ignore drift evidence on rounds dominated by backdoor activity.
        sev_test_loss_gate = float(getattr(args, "sev_test_loss_gate", 0.20))
        drift_asr_gate = float(getattr(args, "diag_drift_asr_gate", 20.0))
        if sev_v is not None:
            if (not np.isnan(test_loss)) and (float(test_loss) < sev_test_loss_gate):
                sev_v = 0.0
            if (asr_nt_v is not None) and (float(asr_nt_v) >= drift_asr_gate):
                sev_v = 0.0
        alarm, pred = st.update(
            round_idx=ridx,
            severity=sev_v,
            asr_nt=asr_nt_v,
        )
        rr["alarm"] = str(int(alarm))
        rr["pred_cause"] = str(pred)
        out.append(rr)
    return out


LABELS = ["normal", "drift", "backdoor", "both"]
IDX = {l: i for i, l in enumerate(LABELS)}


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    import csv

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _confusion(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, int]:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    n = 0
    for r in rows:
        y_true = (r.get("true_cause", "") or "").strip().lower()
        y_pred = (r.get("pred_cause", "") or "").strip().lower()
        if y_true not in IDX or y_pred not in IDX:
            continue
        cm[IDX[y_true], IDX[y_pred]] += 1
        n += 1
    return cm, n


def _prf(cm: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], float, float, float, Dict[str, int]]:
    per: Dict[str, Dict[str, float]] = {}
    f1s = []
    supports: Dict[str, int] = {}
    for i, lab in enumerate(LABELS):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
        supports[lab] = int(cm[i, :].sum())

    macro_f1_all = float(np.mean(f1s)) if f1s else 0.0
    present_f1s = [per[lab]["f1"] for lab in LABELS if supports.get(lab, 0) > 0]
    macro_f1_present = float(np.mean(present_f1s)) if present_f1s else 0.0

    total = float(cm.sum())
    acc = float(cm.trace() / total) if total > 0 else 0.0
    return per, macro_f1_all, macro_f1_present, acc, supports


def _filter_rows(
    rows: List[Dict[str, str]],
    only_alarm: bool,
    window: int,
    scenario_window: str,
) -> List[Dict[str, str]]:
    """Filter rows for alternative evaluation views.

    - only_alarm: keep rows with alarm==1
    - window>0: keep a window of rounds after an event start
      * drift: start=drift_round
      * backdoor: start=attack_start_round
      * both: start depends on scenario_window
          - 'both_active': start=max(drift_round, attack_start_round)
          - 'anomaly_start': start=min(drift_round, attack_start_round)
    """

    out = rows
    if only_alarm:
        out = [r for r in out if (r.get("alarm", "") or "").strip() == "1"]

    if window and window > 0 and out:
        scenario = ((out[0].get("scenario", "") if out else "") or "").strip().lower()
        dr, ar = _event_starts(out)
        if scenario == "drift":
            start = dr
        elif scenario == "backdoor":
            start = ar
        elif scenario == "both":
            if scenario_window == "both_active":
                start = max(dr, ar)
            else:
                start = min(dr, ar)
        else:
            start = 0
        end = start + int(window)

        def _round(rr):
            try:
                return int(float(rr.get("round", "0")))
            except Exception:
                return -1

        out = [r for r in out if (_round(r) >= start and _round(r) < end)]

    return out


def _event_starts(rows: List[Dict[str, str]]) -> Tuple[int, int]:
    """Return (drift_round, attack_start_round) from CSV metadata."""
    # Take the first non-empty occurrence.
    dr, ar = 0, 0
    for r in rows:
        if dr == 0:
            v = (r.get("drift_round", "") or "").strip()
            if v != "":
                try:
                    dr = int(float(v))
                except Exception:
                    pass
        if ar == 0:
            v = (r.get("attack_start_round", "") or "").strip()
            if v != "":
                try:
                    ar = int(float(v))
                except Exception:
                    pass
        if dr != 0 and ar != 0:
            break
    return dr, ar


def _delay(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute simple delay metrics based on first correct prediction."""
    scenario = ((rows[0].get("scenario", "") if rows else "") or "").strip().lower()
    dr, ar = _event_starts(rows)

    def first_round_where(cond) -> int:
        for rr in rows:
            try:
                t = int(float(rr.get("round", "0")))
            except Exception:
                continue
            if cond(rr):
                return t
        return -1

    # When does the *anomaly* begin?
    if scenario == "drift":
        start = dr
        target_label = "drift"
    elif scenario == "backdoor":
        start = ar
        target_label = "backdoor"
    elif scenario == "both":
        start = min(dr, ar)
        target_label = "both"  # used for the 'both-active' delay below
    else:
        return {"delay_first_correct": float("nan"), "delay_both": float("nan")}

    # First correct *non-normal* label after anomaly start.
    t_first = first_round_where(lambda rr: int(float(rr.get("round", "0"))) >= start and (rr.get("pred_cause", "").strip().lower() == rr.get("true_cause", "").strip().lower()) and (rr.get("true_cause", "").strip().lower() != "normal"))
    delay_first = float(t_first - start) if t_first >= 0 else float("nan")

    # For both: delay until we correctly output BOTH after both are active.
    delay_both = float("nan")
    if scenario == "both":
        both_active = max(dr, ar)
        t_both = first_round_where(lambda rr: int(float(rr.get("round", "0"))) >= both_active and rr.get("pred_cause", "").strip().lower() == "both" and rr.get("true_cause", "").strip().lower() == "both")
        delay_both = float(t_both - both_active) if t_both >= 0 else float("nan")

    return {"delay_first_correct": delay_first, "delay_both": delay_both}


def _print_cm(cm: np.ndarray) -> None:
    header = "".ljust(10) + "".join([f"{l:>10}" for l in LABELS])
    print(header)
    for i, lab in enumerate(LABELS):
        row = "".ljust(0)
        row += f"{lab:<10}" + "".join([f"{cm[i, j]:>10d}" for j in range(len(LABELS))])
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="CSV path(s) or glob patterns")
    ap.add_argument("--recompute_pred", action="store_true", help="Recompute alarm/pred_cause from logged metrics (no retraining)")

    # Diagnosis thresholds (used when --recompute_pred is enabled)
    ap.add_argument("--diag_warmup", type=int, default=10)
    ap.add_argument("--diag_drift_warmup", type=int, default=20)
    ap.add_argument("--diag_sev_thr", type=float, default=0.20)
    ap.add_argument("--diag_drift_asr_gate", type=float, default=20.0)
    ap.add_argument("--sev_test_loss_gate", type=float, default=0.20)
    ap.add_argument("--sev_loss_window", type=int, default=5)
    ap.add_argument("--sev_loss_warmup", type=int, default=20)
    ap.add_argument("--force_sev_from_loss", action="store_true", help="Ignore logged sev_score and recompute from avg_train_loss")
    ap.add_argument("--diag_asr_thr", type=float, default=15.0)
    ap.add_argument("--diag_asr_h", type=int, default=2)
    ap.add_argument("--diag_asr_std_thr", type=float, default=50.0)
    ap.add_argument("--diag_hold_rounds", type=int, default=200)
    ap.add_argument("--diag_both_latch_rounds", type=int, default=40)
    ap.add_argument("--only_alarm", action="store_true", help="Evaluate only rounds where alarm==1")
    ap.add_argument("--window", type=int, default=0, help="If >0, evaluate only a fixed window (in rounds) after event start")
    ap.add_argument(
        "--scenario_window",
        type=str,
        default="both_active",
        choices=["both_active", "anomaly_start"],
        help="For scenario='both', define the window start used by --window",
    )
    ap.add_argument("--combined", action="store_true", help="Also print a combined confusion matrix across all provided CSVs")
    args = ap.parse_args()

    paths: List[str] = []
    for p in args.csv:
        if any(ch in p for ch in "*?["):
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)

    if not paths:
        raise SystemExit("No CSV files matched.")

    combined_rows: List[Dict[str, str]] = []

    for csv_path in paths:
        print("=" * 80)
        print(f"File: {csv_path}")
        rows = _load_rows(csv_path)
        if args.recompute_pred:
            rows = _recompute_diagnosis(rows, args)
        rows_f = _filter_rows(rows, only_alarm=bool(args.only_alarm), window=int(args.window), scenario_window=str(args.scenario_window))
        combined_rows.extend(rows_f)

        cm, n = _confusion(rows_f)
        print(f"Usable rows: {n}")
        if args.only_alarm:
            print("View: alarm-only")
        if args.window and args.window > 0:
            print(f"View: window={int(args.window)} rounds, scenario_window={args.scenario_window}")
        print("Confusion matrix (rows=true, cols=pred):")
        _print_cm(cm)
        per, macro_f1_all, macro_f1_present, acc, supports = _prf(cm)
        print("Per-class metrics:")
        for lab in LABELS:
            m = per[lab]
            sup = supports.get(lab, 0)
            print(f"  {lab:<8}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  support={sup}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro-F1 (all classes): {macro_f1_all:.3f}")
        print(f"Macro-F1 (present classes): {macro_f1_present:.3f}")
        d = _delay(rows)
        if not np.isnan(d["delay_first_correct"]):
            print(f"Delay to first correct non-normal label: {d['delay_first_correct']:.1f} rounds")
        if not np.isnan(d["delay_both"]):
            print(f"Delay to BOTH after both active: {d['delay_both']:.1f} rounds")

    if args.combined:
        print("=" * 80)
        print("Combined view across all input CSVs")
        cm, n = _confusion(combined_rows)
        print(f"Usable rows: {n}")
        if args.only_alarm:
            print("View: alarm-only")
        if args.window and args.window > 0:
            print(f"View: window={int(args.window)} rounds, scenario_window={args.scenario_window}")
        print("Confusion matrix (rows=true, cols=pred):")
        _print_cm(cm)
        per, macro_f1_all, macro_f1_present, acc, supports = _prf(cm)
        print("Per-class metrics:")
        for lab in LABELS:
            m = per[lab]
            sup = supports.get(lab, 0)
            print(f"  {lab:<8}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  support={sup}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Macro-F1 (all classes): {macro_f1_all:.3f}")
        print(f"Macro-F1 (present classes): {macro_f1_present:.3f}")


if __name__ == "__main__":
    main()
