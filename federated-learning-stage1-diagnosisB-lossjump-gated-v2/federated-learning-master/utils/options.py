#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # stage-1 experiment arguments (drift / backdoor)
    parser.add_argument('--scenario', type=str, default='baseline',
                        choices=['baseline', 'drift', 'backdoor', 'both'],
                        help="experiment scenario")
    parser.add_argument('--drift_round', type=int, default=60, help="round index when drift starts (inclusive)")
    parser.add_argument('--drift_frac', type=float, default=0.5, help="fraction of drift clients among all clients")
    parser.add_argument('--drift_sev', type=str, default='moderate', choices=['mild', 'moderate', 'severe'],
                        help="abrupt drift severity level")

    parser.add_argument('--attack_start_round', type=int, default=0, help="round index when backdoor attack starts (inclusive)")
    parser.add_argument('--attack_frac', type=float, default=0.2, help="fraction of attacker clients among all clients")
    parser.add_argument('--poison_rate', type=float, default=0.3, help="poisoning rate within attacker local batches")
    parser.add_argument('--attack_scale', type=float, default=1.0, help="scale attacker model delta before aggregation (>=1 strengthens attack)")
    parser.add_argument('--target_label', type=int, default=0, help="backdoor target label")
    parser.add_argument('--trigger_size', type=int, default=3, help="trigger patch size (square)")
    parser.add_argument('--trigger_value', type=float, default=-1.0, help="trigger pixel value (after normalization)")

    parser.add_argument('--eval_every', type=int, default=1, help="evaluate every N rounds")
    parser.add_argument('--save_dir', type=str, default='./save', help="directory to save logs/plots")
    parser.add_argument('--log_csv', type=str, default='', help="optional csv path; if empty, auto-generate")

    # stage-1 diagnosis (Path B: severity + asr_nt + temporal fusion)
    parser.add_argument('--diag_warmup', type=int, default=10, help="warmup rounds before enabling diagnosis")
    parser.add_argument('--diag_drift_warmup', type=int, default=20,
                        help="warmup rounds before enabling drift proxy (avoid early training transients)")
    parser.add_argument('--diag_sev_thr', type=float, default=0.20,
                        help="drift proxy: severity threshold (loss-jump scale)")
    parser.add_argument('--diag_drift_asr_gate', type=float, default=20.0,
                        help='gate drift proxy when ASR-NT is >= this value (attack-dominant rounds)')

    # Backdoor proxy: ASR-NT time-consistency (lower latency than loss-based alarm).
    parser.add_argument('--diag_asr_thr', type=float, default=15.0, help="backdoor proxy: ASR-NT mean threshold")
    parser.add_argument('--diag_asr_h', type=int, default=2, help="backdoor proxy: sliding window length")
    parser.add_argument('--diag_asr_std_thr', type=float, default=50.0, help="backdoor proxy: window std threshold")

    # Temporal fusion / latching.
    parser.add_argument('--diag_hold_rounds', type=int, default=200, help="keep drift/backdoor evidence active for N rounds")
    parser.add_argument('--diag_both_latch_rounds', type=int, default=40, help="keep BOTH prediction for N rounds")

    # Severity monitoring controls (Path B)
    # Current stage-1 implementation uses a loss-jump proxy.
    parser.add_argument('--sev_loss_window', type=int, default=5, help="window size W for loss-jump severity")
    parser.add_argument('--sev_loss_warmup', type=int, default=20, help="do not compute loss-jump severity before this round")
    parser.add_argument('--sev_test_loss_gate', type=float, default=0.20,
                        help='gate severity: if clean test_loss < this, ignore loss-jump severity')
    
    # Deprecated (kept for compatibility): conditional-output monitoring.
    parser.add_argument('--sev_ref_rounds', type=int, default=10, help="[deprecated] rounds used to build severity reference")
    parser.add_argument('--sev_monitor_batches', type=int, default=1, help="[deprecated] monitoring batches per client")

    args = parser.parse_args()
    return args
