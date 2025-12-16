#!/usr/bin/env python3
"""
Example script demonstrating how to use ResultsReader programmatically.

This shows how to:
1. Load a results file
2. Access specific data
3. Iterate over all runs
4. Extract and analyze metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from read_results import ResultsReader


def example_basic_usage():
    """Basic usage example."""
    # Load results
    reader = ResultsReader('experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz')

    # Print summary
    print(reader.summary())
    print("\n" + "="*80 + "\n")

    # Access specific data
    loss_train = reader.get_data(
        optimizer='GD',
        params={'lr': 0.1},
        seed=0,
        metric='loss_train'
    )
    print(f"GD with lr=0.1, seed=0 training loss:")
    print(f"  Initial: {loss_train[0]:.6e}")
    print(f"  Final: {loss_train[-1]:.6e}")
    print(f"  Number of evaluations: {len(loss_train)}")
    print()


def example_compare_optimizers():
    """Compare final test errors across different optimizers."""
    reader = ResultsReader('experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz')

    print("Comparing final test errors:\n")

    for optimizer in ['GD', 'LossNGD', 'VecNGD']:
        final_errors = reader.get_final_values(optimizer, 'error_test')

        if not final_errors:
            continue

        print(f"{optimizer}:")
        # Get best performing hyperparameters
        best_params, best_error = min(final_errors.items(), key=lambda x: x[1])
        param_dict = dict(best_params[0])
        seed = best_params[1]

        print(f"  Best: {best_error:.6e} with {param_dict}, seed={seed}")
        print()


def example_plot_learning_curves():
    """Plot learning curves for different learning rates."""
    reader = ResultsReader('experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz')

    optimizer = 'GD'
    metric = 'loss_train'

    plt.figure(figsize=(10, 6))

    # Plot curves for different learning rates
    for params in reader.hyperparams[optimizer]:
        data = reader.get_data(optimizer, params, seed=0, metric=metric)
        steps = reader.get_data(optimizer, params, seed=0, metric='steps')

        lr = params['lr']
        plt.plot(steps, data, label=f"lr={lr}", alpha=0.7)

    plt.xlabel('Steps')
    plt.ylabel('Training Loss')
    plt.title(f'{optimizer} Learning Curves')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    print("Learning curves saved to learning_curves.png")


def example_iterate_all_runs():
    """Iterate over all runs for a specific optimizer and metric."""
    reader = ResultsReader('experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz')

    optimizer = 'SAM'
    metric = 'error_test'

    all_runs = reader.get_all_runs(optimizer, metric)

    print(f"Found {len(all_runs)} runs for {optimizer} with metric {metric}\n")

    # Show first 5 runs
    for i, ((params, seed), data) in enumerate(all_runs.items()):
        if i >= 5:
            print(f"... and {len(all_runs) - 5} more runs")
            break

        param_dict = dict(params)
        print(f"Run {i+1}:")
        print(f"  Params: {param_dict}")
        print(f"  Seed: {seed}")
        print(f"  Final value: {data[-1]:.6e}")
        print(f"  Array shape: {data.shape}")
        print()


if __name__ == '__main__':
    print("Example 1: Basic Usage")
    print("="*80)
    example_basic_usage()

    print("\nExample 2: Compare Optimizers")
    print("="*80)
    example_compare_optimizers()

    print("\nExample 3: Iterate All Runs")
    print("="*80)
    example_iterate_all_runs()

    print("\nExample 4: Plot Learning Curves")
    print("="*80)
    example_plot_learning_curves()
