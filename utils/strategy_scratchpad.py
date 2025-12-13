#!/usr/bin/env python
"""
Strategy Scratchpad - Minimal example for experimenting with custom plotting strategies.

This script provides a quick way to test new plotting strategies and visualizations
using a small dataset (n=200, D=2). Modify the custom_experiment() function to
experiment with your own strategies.

Usage:
    python utils/strategy_scratchpad.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from engine import (
    LinearModel,
    make_soudry_dataset,
    split_train_test,
    Metric,
    Optimizer,
    MetricsCollector,
    exponential_loss,
    get_error_rate,
    get_angle,
    get_direction_distance,
    get_empirical_max_margin,
    run_training,
)
from engine.optimizers import step_gd, step_sam_stable
from engine.optimizers.base import make_optimizer
from engine.plotting import plot_all
from engine.strategies import (
    PlotStrategy,
    AxisScale,
    SafeLog,
    Scale,
    Clamp,
    LogLogStrategy,
    PercentageStrategy,
)


def generate_data_and_train():
    """Generate small dataset and run short training."""
    print("=" * 70)
    print("GENERATING DATA AND RUNNING TRAINING")
    print("=" * 70)

    device = "cpu"

    # Small dataset for quick iteration (n=200, D=2)
    print("\nGenerating dataset (n=200, d=2)...")
    X, y, v_pop = make_soudry_dataset(n=200, d=2, device=device, margin=0.1, sigma=0.3)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=40, random_state=42)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Train samples: {datasets[0].X.shape[0]}")
    print(f"  Test samples: {datasets[1].X.shape[0]}")

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], device=device)

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
            },
            w_star=w_star,
        )

    # Optimizers (GD and SAM for comparison)
    print("\nSetting up optimizers (GD, SAM)...")
    optimizers = {
        Optimizer.GD: make_optimizer(step_gd),
        Optimizer.SAM: make_optimizer(step_sam_stable),
    }

    # Run training (short run for quick iteration)
    print("\nRunning training (1000 iterations, 2 learning rates)...")
    learning_rates = [0.1, 1.0]
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        total_iters=1000,
        debug=True,
    )
    print("Training complete!\n")

    return results, learning_rates, optimizers


def custom_experiment(results, learning_rates, optimizers):
    """
    YOUR CUSTOM EXPERIMENT GOES HERE!

    Modify this function to experiment with your own plotting strategies.
    The results, learning_rates, and optimizers are provided for you.

    Example strategies to try:
    - Different axis scales (Linear, Log, Symlog, Logit)
    - Custom transforms (SafeLog, Scale, Clamp, or create your own)
    - Different y_lim ranges
    - Custom y_label_suffix
    """
    print("=" * 70)
    print("CUSTOM EXPERIMENT - MODIFY THIS FUNCTION!")
    print("=" * 70)

    # Example: Create a custom strategy for Error metric
    # This uses log-log scale with percentage, different from the default
    my_custom_strategy = PlotStrategy(
        transforms=[SafeLog(), Scale(100.0)],
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Log,  # Changed from Linear to Log
        y_label_suffix=" (%) [my custom scale]",
    )

    # Apply your custom strategy
    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(optimizers.keys()),
        experiment_name="scratchpad/my_custom_experiment",
        strategy_overrides={Metric.Error: my_custom_strategy},
    )

    print("\n✓ Custom experiment complete!")
    print("  Plots saved to: experiments/scratchpad/my_custom_experiment/")
    print("\nTry modifying the strategy above and run again!")


def example_experiments(results, learning_rates, optimizers):
    """
    Example experiments to demonstrate different strategies.
    Use these as templates for your own experiments.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE EXPERIMENTS")
    print("=" * 70)

    # Example 1: Clamped loss values
    print("\nExample 1: Clamped loss (min=1e-10, max=1e2)...")
    loss_clamped = LogLogStrategy().pipe(Clamp(min_val=1e-10, max_val=1e2))
    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(optimizers.keys()),
        experiment_name="scratchpad/example_1_clamped_loss",
        strategy_overrides={Metric.Loss: loss_clamped},
    )
    print("  ✓ Saved to: experiments/scratchpad/example_1_clamped_loss/")

    # Example 2: Symmetric log scale (handles negative values)
    print("\nExample 2: Symmetric log scale...")
    symlog_strategy = PlotStrategy(
        transforms=[],  # No SafeLog needed with symlog
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Symlog,
        y_label_suffix=" [symlog]",
    )
    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(optimizers.keys()),
        experiment_name="scratchpad/example_2_symlog",
        strategy_overrides={Metric.Loss: symlog_strategy},
    )
    print("  ✓ Saved to: experiments/scratchpad/example_2_symlog/")

    # Example 3: Linear scale for loss (unusual but possible)
    print("\nExample 3: Linear scale for loss...")
    linear_loss = PlotStrategy(
        transforms=[SafeLog()],
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Linear,  # Linear instead of log
        y_label_suffix=" [linear scale]",
    )
    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(optimizers.keys()),
        experiment_name="scratchpad/example_3_linear_loss",
        strategy_overrides={Metric.Loss: linear_loss},
    )
    print("  ✓ Saved to: experiments/scratchpad/example_3_linear_loss/")

    print("\n✓ All example experiments complete!")


def main():
    """Main scratchpad workflow."""
    print("\n" + "🎨 " * 35)
    print("STRATEGY SCRATCHPAD")
    print("🎨 " * 35 + "\n")

    # Generate data and run training
    results, learning_rates, optimizers = generate_data_and_train()

    # Run your custom experiment (MODIFY THIS!)
    custom_experiment(results, learning_rates, optimizers)

    # Optionally run example experiments
    print("\n" + "?" * 70)
    print("Run example experiments? (shows different strategy patterns)")
    response = input("Enter 'y' to run examples, or any other key to skip: ").strip().lower()
    print("?" * 70)

    if response == 'y':
        example_experiments(results, learning_rates, optimizers)

    # Done
    print("\n" + "=" * 70)
    print("✅ SCRATCHPAD SESSION COMPLETE!")
    print("=" * 70)
    print("\nQuick Reference - Creating Custom Strategies:")
    print("""
    # Basic pattern:
    my_strategy = PlotStrategy(
        transforms=[SafeLog(), Scale(100.0), Clamp(min_val=0, max_val=100)],
        x_scale=AxisScale.Log,      # Linear, Log, Symlog, Logit
        y_scale=AxisScale.Linear,   # Linear, Log, Symlog, Logit
        y_label_suffix=" (my unit)",
        y_lim=(0, 100),             # Optional
    )

    # Or use fluent pipe interface:
    my_strategy = LogLogStrategy().pipe(Clamp(min_val=1e-8))

    # Apply with strategy_overrides:
    plot_all(..., strategy_overrides={Metric.Loss: my_strategy})
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
