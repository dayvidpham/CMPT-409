#!/usr/bin/env python3
"""
Simple script to print a specific data array from results.npz file.

Usage:
    python utils/print_array.py <results_file> <optimizer> <metric> [--lr LR] [--rho RHO] [--seed SEED]

Examples:
    python utils/print_array.py results.npz GD loss_train --lr 0.1 --seed 0
    python utils/print_array.py results.npz SAM error_test --lr 0.1 --rho 1.0 --seed 0
    python utils/print_array.py results.npz LossNGD gradnorm --lr 1.0
"""

import argparse
import numpy as np
from read_results import ResultsReader


def main():
    parser = argparse.ArgumentParser(
        description='Print a specific data array from results.npz file'
    )
    parser.add_argument('filepath', help='Path to results.npz file')
    parser.add_argument('optimizer', help='Optimizer name (e.g., GD, SAM, LossNGD)')
    parser.add_argument('metric', help='Metric name (e.g., loss_train, error_test, gradnorm)')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--rho', type=float, help='Rho parameter (for SAM optimizers)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--full', action='store_true', help='Print full array (default: first/last 10)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')

    args = parser.parse_args()

    # Load results
    reader = ResultsReader(args.filepath)

    # Build params dict
    params = {'lr': args.lr}
    if args.rho is not None:
        params['rho'] = args.rho

    # Get data
    try:
        data = reader.get_data(
            optimizer=args.optimizer,
            params=params,
            seed=args.seed,
            metric=args.metric
        )
    except KeyError as e:
        print(f"Error: {e}")
        print(f"\nAvailable optimizers: {reader.optimizers}")
        print(f"Available metrics: {reader.metrics}")
        return 1

    # Print info
    param_str = ', '.join([f'{k}={v}' for k, v in sorted(params.items())])
    print('='*70)
    print(f'{args.optimizer}({param_str}) seed={args.seed} - {args.metric}')
    print('='*70)
    print(f'Shape: {data.shape}')

    if args.stats or len(data) > 20:
        print(f'\nStatistics:')
        print(f'  Min:    {data.min():.6e}')
        print(f'  Max:    {data.max():.6e}')
        print(f'  Mean:   {data.mean():.6e}')
        print(f'  Std:    {data.std():.6e}')
        print(f'  First:  {data[0]:.6e}')
        print(f'  Last:   {data[-1]:.6e}')

    if args.stats:
        return 0

    print(f'\nArray values:')
    if args.full or len(data) <= 20:
        np.set_printoptions(threshold=np.inf, linewidth=100)
        print(data)
    else:
        print(f'First 10: {data[:10]}')
        print(f'...')
        print(f'Last 10:  {data[-10:]}')
        print(f'\nUse --full to see all {len(data)} values or --stats for statistics only')

    return 0


if __name__ == '__main__':
    exit(main())
