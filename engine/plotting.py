import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Optional
from datetime import datetime
from .types import Optimizer, MetricKey, Metric
from .history import TrainingHistory
from .strategies import PlotStrategy, PlotContext


def plot_all(
    results: Dict[float, Dict[Optimizer, Union[TrainingHistory, List[TrainingHistory]]]],
    learning_rates: List[float],
    optimizers: List[Optimizer],
    experiment_name: str,
    save_combined: bool = True,
    save_separate: bool = True,
    save_aggregated: bool = True,
    post_training: bool = False,
    strategy_overrides: Optional[Dict[Metric, PlotStrategy]] = None
):
    """
    Unified plotting for all model types.
    Automatically detects which metrics are available from TrainingHistory.

    Args:
        strategy_overrides: Optional dict mapping Metric -> PlotStrategy to override defaults
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        base_dir = Path("experiments") / experiment_name / timestamp
    else:
        base_dir = Path("experiments") / "test_plots" / timestamp
        
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save results as NPZ if not already saved
    if not post_training:
        save_results_npz(results, learning_rates, optimizers, base_dir / "results.npz")

    # Detect metric keys
    first_hist = results[learning_rates[0]][optimizers[0]]
    if isinstance(first_hist, list):
        if len(first_hist) == 0:
            print("Warning: No history found to detect metrics.")
            return
        first_hist = first_hist[0]
    metric_keys = first_hist.metric_keys

    # Group metrics
    metric_types = {}
    for key in metric_keys:
        metric_name = str(key.metric.name.lower())
        if metric_name not in metric_types:
            metric_types[metric_name] = []
        metric_types[metric_name].append(key)

    # Resolve strategy overrides
    overrides = strategy_overrides or {}

    # Generate Plots
    for metric_name, keys in metric_types.items():
        # Get the metric enum from the first key
        metric = keys[0].metric

        # Resolve: Override > Metric Default
        strategy = overrides.get(metric, metric.strategy)

        if save_combined:
            plot_combined(results, learning_rates, optimizers, keys, strategy,
                          base_dir / "combined" / f"{metric_name}.png")

        if save_separate:
            for opt in optimizers:
                plot_separate(results, learning_rates, opt, keys, strategy,
                              base_dir / "separate" / opt.name / f"{metric_name}.png")

        if save_aggregated:
            plot_aggregated(results, learning_rates, optimizers, keys, strategy,
                            base_dir / "aggregated" / f"{metric_name}_comparison.png")

def plot_aggregated(results, learning_rates, optimizers, metric_keys, strategy, filepath):
    """Paper-style plotting with overlaid runs and mean."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ncols = len(optimizers)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    lr_colors = {lr: cmap(i % 10) for i, lr in enumerate(learning_rates)}

    for i, opt in enumerate(optimizers):
        ax = axes[i]

        # Configure axis using strategy (only once per axis)
        key = metric_keys[0]  # Primary metric
        strategy.configure_axis(ax, base_label=key.metric.name)

        for lr in learning_rates:
            entry = results[lr][opt]
            histories = entry if isinstance(entry, list) else [entry]

            if not histories:
                continue

            all_values = []
            steps = None

            for h in histories:
                h_cpu = h.copy_cpu()
                # Get raw values
                vals = np.array(h_cpu.get(key))
                curr_steps = np.array(h_cpu.get_steps())

                all_values.append(vals)
                if steps is None or len(curr_steps) > len(steps):
                    steps = curr_steps

            color = lr_colors[lr]

            # 1. Plot individual runs (faint)
            if len(histories) > 1:
                for vals in all_values:
                    run_steps = steps[:len(vals)]
                    ctx = PlotContext(
                        ax=ax, x=run_steps, y=vals,
                        label="", color=color, alpha=0.15, linestyle="-"
                    )
                    strategy.plot(ctx)

            # 2. Plot Mean (solid)
            if all_values:
                min_len = min(len(v) for v in all_values)
                truncated_values = np.stack([v[:min_len] for v in all_values])
                mean_values = np.mean(truncated_values, axis=0)
                mean_steps = steps[:min_len]

                ctx = PlotContext(
                    ax=ax, x=mean_steps, y=mean_values,
                    label=f"lr={lr}", color=color, alpha=1.0, linestyle="-"
                )
                strategy.plot(ctx)

        ax.set_title(opt.name)
        ax.set_xlabel("Steps")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_combined(results, learning_rates, optimizers, metric_keys, strategy, filepath):
    """Simple combined view."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ncols = len(learning_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, lr in enumerate(learning_rates):
        ax = axes[i]

        # Configure axis using strategy (once per axis)
        strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

        for opt in optimizers:
            entry = results[lr][opt]
            history = entry[0] if isinstance(entry, list) and entry else entry

            if isinstance(entry, list) and not entry:
                continue

            history = history.copy_cpu()
            steps = np.array(history.get_steps())

            for key in metric_keys:
                # Get raw values
                values = np.array(history.get(key))
                label = opt.name + (f"_{key.split.name}" if key.split else "")

                ctx = PlotContext(
                    ax=ax, x=steps, y=values,
                    label=label, alpha=0.7
                )
                strategy.plot(ctx)

        ax.set_xlabel("Steps")
        ax.set_title(f"LR = {lr}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_separate(results, learning_rates, optimizer, metric_keys, strategy, filepath):
    """Single optimizer view."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Configure axis using strategy
    strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

    for lr in learning_rates:
        entry = results[lr][optimizer]
        history = entry[0] if isinstance(entry, list) and entry else entry

        if isinstance(entry, list) and not entry:
            continue

        history = history.copy_cpu()
        steps = np.array(history.get_steps())

        for key in metric_keys:
            # Get raw values
            values = np.array(history.get(key))
            label = f"lr={lr}" + (f"_{key.split.name}" if key.split else "")

            ctx = PlotContext(
                ax=ax, x=steps, y=values,
                label=label, alpha=0.7
            )
            strategy.plot(ctx)

    ax.set_xlabel("Steps")
    ax.set_title(f"{optimizer.name}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def save_results_npz(results, learning_rates, optimizers, filepath):
    """Save flattened results as NPZ"""
    data = {}
    for lr in learning_rates:
        for opt in optimizers:
            entry = results[lr][opt]
            histories = entry if isinstance(entry, list) else [entry]
            
            for seed, history in enumerate(histories):
                hist_cpu = history.copy_cpu()
                hist_dict = hist_cpu.to_dict()
                
                for key, values in hist_dict.items():
                    npz_key = f"lr{lr}_{opt.name}_seed{seed}_{key}"
                    data[npz_key] = values
    np.savez(filepath, **data)
