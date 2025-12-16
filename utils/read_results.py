#!/usr/bin/env python3
"""
Utility script to read and analyze results.npz files from experiments.

Usage:
    python read_results.py <path_to_results.npz>

Example:
    python read_results.py experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ResultsReader:
    """Read and parse results from .npz experiment files."""

    def __init__(self, filepath: str):
        """Load results from npz file."""
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        self.data = np.load(filepath)
        self.keys = list(self.data.keys())
        self._parse_structure()

    def _parse_structure(self):
        """Parse the structure of the results file."""
        self.optimizers = set()
        self.seeds = set()
        self.metrics = set()
        self.hyperparams = {}

        # Pattern to match keys like: GD(lr=0.1)_seed0_loss_train
        # or SAM(lr=0.1,rho=1.0)_seed0_loss_train
        pattern = r'^(.+?)_seed(\d+)_(.+)$'

        for key in self.keys:
            match = re.match(pattern, key)
            if match:
                optimizer_with_params = match.group(1)
                seed = int(match.group(2))
                metric = match.group(3)

                self.seeds.add(seed)
                self.metrics.add(metric)

                # Parse optimizer name and hyperparameters
                if '(' in optimizer_with_params:
                    opt_name = optimizer_with_params[:optimizer_with_params.index('(')]
                    self.optimizers.add(opt_name)

                    if opt_name not in self.hyperparams:
                        self.hyperparams[opt_name] = []

                    # Extract hyperparameters
                    params_str = optimizer_with_params[optimizer_with_params.index('(')+1:-1]
                    params = {}
                    for param in params_str.split(','):
                        key, val = param.split('=')
                        params[key] = float(val)

                    if params not in self.hyperparams[opt_name]:
                        self.hyperparams[opt_name].append(params)

        self.optimizers = sorted(self.optimizers)
        self.seeds = sorted(self.seeds)
        self.metrics = sorted(self.metrics)

        for opt in self.hyperparams:
            self.hyperparams[opt] = sorted(
                self.hyperparams[opt],
                key=lambda x: tuple(x.values())
            )

    def get_data(self, optimizer: str, params: Dict[str, float],
                 seed: int, metric: str) -> np.ndarray:
        """
        Get data for specific optimizer, parameters, seed, and metric.

        Args:
            optimizer: Optimizer name (e.g., 'GD', 'SAM', 'LossNGD')
            params: Dictionary of hyperparameters (e.g., {'lr': 0.1, 'rho': 1.0})
            seed: Random seed number
            metric: Metric name (e.g., 'loss_train', 'error_test')

        Returns:
            NumPy array containing the metric values
        """
        # Construct the key
        param_str = ','.join([f'{k}={v}' for k, v in sorted(params.items())])
        key = f'{optimizer}({param_str})_seed{seed}_{metric}'

        if key not in self.data:
            raise KeyError(f"Key not found: {key}")

        return self.data[key]

    def get_all_runs(self, optimizer: str, metric: str) -> Dict[Tuple, np.ndarray]:
        """
        Get all runs for a specific optimizer and metric.

        Args:
            optimizer: Optimizer name
            metric: Metric name

        Returns:
            Dictionary mapping (params_tuple, seed) to metric values
        """
        results = {}

        if optimizer not in self.hyperparams:
            return results

        for params in self.hyperparams[optimizer]:
            for seed in self.seeds:
                try:
                    data = self.get_data(optimizer, params, seed, metric)
                    key = (tuple(sorted(params.items())), seed)
                    results[key] = data
                except KeyError:
                    continue

        return results

    def summary(self) -> str:
        """Generate a summary of the results file."""
        lines = [
            f"Results file: {self.filepath}",
            f"Total arrays: {len(self.keys)}",
            "",
            f"Optimizers: {', '.join(self.optimizers)}",
            f"Seeds: {', '.join(map(str, self.seeds))}",
            f"Metrics: {', '.join(self.metrics)}",
            "",
            "Hyperparameters per optimizer:"
        ]

        for opt in self.optimizers:
            if opt in self.hyperparams:
                lines.append(f"\n  {opt}:")
                for params in self.hyperparams[opt]:
                    param_str = ', '.join([f'{k}={v}' for k, v in sorted(params.items())])
                    lines.append(f"    {param_str}")

        return '\n'.join(lines)

    def list_keys(self, filter_str: Optional[str] = None) -> List[str]:
        """
        List all keys, optionally filtered by a substring.

        Args:
            filter_str: Optional substring to filter keys

        Returns:
            List of matching keys
        """
        if filter_str:
            return [k for k in self.keys if filter_str in k]
        return self.keys

    def get_final_values(self, optimizer: str, metric: str) -> Dict[Tuple, float]:
        """
        Get final values for a specific optimizer and metric across all runs.

        Args:
            optimizer: Optimizer name
            metric: Metric name

        Returns:
            Dictionary mapping (params_tuple, seed) to final metric value
        """
        results = {}

        for params in self.hyperparams.get(optimizer, []):
            for seed in self.seeds:
                try:
                    data = self.get_data(optimizer, params, seed, metric)
                    key = (tuple(sorted(params.items())), seed)
                    results[key] = data[-1] if len(data) > 0 else np.nan
                except KeyError:
                    continue

        return results

    def to_dataframe(self, keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Args:
            keys: Optional list of keys to include. If None, includes all keys.

        Returns:
            DataFrame with index as step number and columns as data keys.
            Arrays of different lengths are padded with NaN.
        """
        if keys is None:
            keys = self.keys

        if not keys:
            return pd.DataFrame()

        # Find the maximum length across all arrays
        max_length = max(len(self.data[key]) for key in keys)

        # Create a dictionary to store the data
        data_dict = {}
        for key in keys:
            array = self.data[key]
            # Pad with NaN if necessary
            if len(array) < max_length:
                padded = np.full(max_length, np.nan)
                padded[:len(array)] = array
                data_dict[key] = padded
            else:
                data_dict[key] = array

        return pd.DataFrame(data_dict)

    def to_dataframe_by_optimizer(self, optimizer: str, metric: Optional[str] = None) -> pd.DataFrame:
        """
        Convert results for a specific optimizer to a pandas DataFrame.

        Args:
            optimizer: Optimizer name (e.g., 'GD', 'SAM', 'LossNGD')
            metric: Optional metric name to filter by (e.g., 'loss_train')

        Returns:
            DataFrame with columns for each param/seed combination for the optimizer.
        """
        # Filter keys that start with the optimizer name
        optimizer_keys = [k for k in self.keys if k.startswith(f'{optimizer}(')]

        # Further filter by metric if specified
        if metric:
            optimizer_keys = [k for k in optimizer_keys if k.endswith(f'_{metric}')]

        return self.to_dataframe(keys=optimizer_keys)

    def to_dataframe_by_metric(self, metric: str) -> pd.DataFrame:
        """
        Convert results for a specific metric to a pandas DataFrame.

        Args:
            metric: Metric name (e.g., 'loss_train', 'error_test')

        Returns:
            DataFrame with index as step number and columns for each
            optimizer/params/seed combination for the specified metric.
        """
        # Filter keys that match the metric
        metric_keys = [k for k in self.keys if k.endswith(f'_{metric}')]
        return self.to_dataframe(keys=metric_keys)


def main():
    parser = argparse.ArgumentParser(
        description='Read and analyze experiment results from .npz files'
    )
    parser.add_argument(
        'filepath',
        type=str,
        help='Path to results.npz file'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Filter by optimizer name'
    )
    parser.add_argument(
        '--metric',
        type=str,
        help='Filter by metric name'
    )
    parser.add_argument(
        '--list-keys',
        action='store_true',
        help='List all keys in the file'
    )
    parser.add_argument(
        '--final-values',
        action='store_true',
        help='Show final values for specified optimizer and metric'
    )
    parser.add_argument(
        '--dataframe',
        action='store_true',
        help='Convert to pandas DataFrame and display (filters by optimizer, defaults to GD)'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=10,
        help='Number of rows to display when using --dataframe (default: 10)'
    )

    args = parser.parse_args()

    # Load results
    reader = ResultsReader(args.filepath)

    # Print summary by default
    if not any([args.list_keys, args.final_values, args.dataframe]):
        print(reader.summary())

    # List keys
    if args.list_keys:
        filter_str = None
        if args.optimizer:
            filter_str = args.optimizer
        if args.metric:
            filter_str = args.metric if not filter_str else f"{filter_str}_{args.metric}"

        keys = reader.list_keys(filter_str)
        print(f"\nFound {len(keys)} matching keys:")
        for key in keys:
            shape = reader.data[key].shape
            print(f"  {key}: {shape}")

    # Show final values
    if args.final_values:
        if not args.optimizer or not args.metric:
            print("\nError: --final-values requires both --optimizer and --metric")
            return

        final_vals = reader.get_final_values(args.optimizer, args.metric)
        print(f"\nFinal {args.metric} values for {args.optimizer}:")
        for (params, seed), value in sorted(final_vals.items()):
            param_str = ', '.join([f'{k}={v}' for k, v in params])
            print(f"  {param_str}, seed={seed}: {value:.6e}")

    # Display DataFrame
    if args.dataframe:
        # Default to 'GD' optimizer if not specified
        optimizer = args.optimizer if args.optimizer else 'GD'

        # Get DataFrame for the specified optimizer
        df = reader.to_dataframe_by_optimizer(optimizer, metric=args.metric)

        if df.empty:
            print(f"\nNo data found for optimizer '{optimizer}'")
            if args.metric:
                print(f"with metric '{args.metric}'")
        else:
            # Print info
            if args.metric:
                print(f"\nDataFrame for optimizer '{optimizer}', metric '{args.metric}':")
            else:
                print(f"\nDataFrame for optimizer '{optimizer}' (all metrics):")

            print(f"Shape: {df.shape} rows × {df.shape[1]} columns\n")

            # Display the dataframe
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_rows', args.head)
            print(df.head(args.head))


if __name__ == '__main__':
    main()
