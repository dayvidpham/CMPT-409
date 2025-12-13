import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Mapping
from datetime import datetime
from .types import (
    OptimizerConfig,
    MetricKey,
    Metric,
    Hyperparam,
    DatasetSplit,
    Optimizer,
)
from .history import TrainingHistory
from .strategies import PlotStrategy, PlotContext

# Type alias for results - use Mapping for covariance
ResultsType = Mapping[OptimizerConfig, Union[TrainingHistory, List[TrainingHistory]]]


def plot_all(
    results: ResultsType,
    experiment_name: str,
    save_combined: bool = True,
    save_separate: bool = True,
    save_aggregated: bool = True,
    post_training: bool = False,
    strategy_overrides: Optional[Dict[Metric, PlotStrategy]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Unified plotting for all model types.
    Automatically detects which metrics are available from TrainingHistory.

    Args:
        results: Training results dict[OptimizerConfig] = TrainingHistory
        experiment_name: Name for experiment directory
        save_combined: Save combined plots (grouped by learning rate)
        save_separate: Save separate plots per optimizer
        save_aggregated: Save aggregated comparison plots
        post_training: If True, skip saving results.npz (already saved)
        strategy_overrides: Optional dict mapping Metric -> PlotStrategy to override defaults
        output_dir: Optional custom output directory (overrides experiment_name-based path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_dir = output_dir
    elif experiment_name:
        base_dir = Path("experiments") / experiment_name / timestamp
    else:
        base_dir = Path("experiments") / "test_plots" / timestamp

    base_dir.mkdir(parents=True, exist_ok=True)

    # Group configs by learning rate
    configs_by_lr: Dict[float, List[OptimizerConfig]] = {}
    for config in results.keys():
        lr = config.learning_rate
        if lr not in configs_by_lr:
            configs_by_lr[lr] = []
        configs_by_lr[lr].append(config)

    learning_rates = sorted(configs_by_lr.keys())
    all_configs = list(results.keys())

    # Save results as NPZ if not already saved
    if not post_training:
        save_results_npz(results, base_dir / "results.npz")

    # Detect metric keys from first history
    first_config = all_configs[0]
    first_entry = results[first_config]
    if isinstance(first_entry, list):
        if len(first_entry) == 0:
            print("Warning: No history found to detect metrics.")
            return
        first_hist = first_entry[0]
    else:
        first_hist = first_entry
    metric_keys = first_hist.metric_keys

    # Group metrics by type
    metric_types: Dict[str, List[MetricKey]] = {}
    for key in metric_keys:
        metric_name = str(key.metric.name.lower())
        if metric_name not in metric_types:
            metric_types[metric_name] = []
        metric_types[metric_name].append(key)

    # Resolve strategy overrides
    overrides = strategy_overrides or {}

    # Generate Plots
    for metric_name, keys in metric_types.items():
        metric = keys[0].metric
        strategy = overrides.get(metric, metric.strategy)

        if save_combined:
            plot_combined(
                results,
                configs_by_lr,
                learning_rates,
                keys,
                strategy,
                base_dir / "combined" / f"{metric_name}.png",
            )

        if save_separate:
            for config in all_configs:
                plot_separate(
                    results,
                    config,
                    keys,
                    strategy,
                    base_dir / "separate" / config.dir_name / f"{metric_name}.png",
                )

        if save_aggregated:
            plot_aggregated(
                results,
                configs_by_lr,
                learning_rates,
                keys,
                strategy,
                base_dir / "aggregated" / f"{metric_name}_comparison.png",
            )

    # Detect if we have hyperparameter sweeps (configs with rho)
    has_rho_sweeps = any(
        config.get(Hyperparam.Rho) is not None for config in all_configs
    )

    if has_rho_sweeps:
        # Extract unique rho and learning rate values
        rho_values = sorted(
            set(
                config.get(Hyperparam.Rho)
                for config in all_configs
                if config.get(Hyperparam.Rho) is not None
            )
        )
        learning_rates_for_grid = sorted(
            set(
                config.learning_rate
                for config in all_configs
                if config.get(Hyperparam.Rho) is not None
            )
        )

        # Generate hyperparameter grid plots for each metric
        for metric_name, keys in metric_types.items():
            metric = keys[0].metric
            strategy = overrides.get(metric, metric.strategy)

            plot_hyperparam_grid(
                results,
                learning_rates_for_grid,
                rho_values,
                keys,
                strategy,
                base_dir / "hyperparam_grid" / f"{metric_name}_grid.png",
            )

    # Detect if we have base/SAM optimizer pairs
    optimizer_pairs = _identify_optimizer_pairs(results)

    if optimizer_pairs:
        # Generate SAM comparison plots for each metric
        for metric_name, keys in metric_types.items():
            metric = keys[0].metric
            strategy = overrides.get(metric, metric.strategy)

            plot_sam_comparison(
                results,
                keys,
                strategy,
                base_dir / "sam_comparison" / f"{metric_name}_sam_comparison.png",
            )


def _get_history(
    entry: Union[TrainingHistory, List[TrainingHistory]],
) -> Optional[TrainingHistory]:
    """Extract single history from entry, handling both single and list cases."""
    if isinstance(entry, list):
        return entry[0] if entry else None
    return entry


def _get_histories(
    entry: Union[TrainingHistory, List[TrainingHistory]],
) -> List[TrainingHistory]:
    """Extract list of histories from entry."""
    if isinstance(entry, list):
        return entry
    return [entry]


def plot_aggregated(
    results: ResultsType,
    configs_by_lr: Mapping[float, List[OptimizerConfig]],
    learning_rates: List[float],
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Paper-style plotting: one subplot per optimizer type, colored by learning rate."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Filter to only Test split if available, otherwise use first available
    test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]
    if test_keys:
        selected_keys = test_keys
    else:
        # Use reference metrics (no split) or whatever is available
        selected_keys = metric_keys

    key = selected_keys[0]
    split_name = key.split.name if key.split else ""

    # Group configs by optimizer type (ignoring hyperparams)
    optimizer_types: Dict[str, List[OptimizerConfig]] = {}
    for config in results.keys():
        opt_name = config.optimizer.name
        if opt_name not in optimizer_types:
            optimizer_types[opt_name] = []
        optimizer_types[opt_name].append(config)

    opt_names = sorted(optimizer_types.keys())
    ncols = len(opt_names)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    lr_colors = {lr: cmap(i % 10) for i, lr in enumerate(learning_rates)}

    for i, opt_name in enumerate(opt_names):
        ax = axes[i]
        strategy.configure_axis(ax, base_label=key.metric.name)

        # Get all configs for this optimizer type
        opt_configs = optimizer_types[opt_name]

        for lr in learning_rates:
            # Find configs with this lr
            lr_configs = [c for c in opt_configs if c.learning_rate == lr]
            if not lr_configs:
                continue

            color = lr_colors[lr]

            for config in lr_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                all_values: List[np.ndarray] = []
                steps: Optional[np.ndarray] = None

                for h in histories:
                    h_cpu = h.copy_cpu()
                    vals = np.array(h_cpu.get(key))
                    curr_steps = np.array(h_cpu.get_steps())
                    all_values.append(vals)
                    if steps is None or len(curr_steps) > len(steps):
                        steps = curr_steps

                # Plot individual runs (faint) if multiple
                if len(histories) > 1 and steps is not None:
                    for vals in all_values:
                        run_steps = steps[: len(vals)]
                        ctx = PlotContext(
                            ax=ax,
                            x=run_steps,
                            y=vals,
                            label="",
                            plot_kwargs={
                                "color": color,
                                "alpha": 0.15,
                                "linestyle": "-",
                            },
                        )
                        strategy.plot(ctx)

                # Plot mean (or single run)
                if all_values and steps is not None:
                    min_len = min(len(v) for v in all_values)
                    truncated = np.stack([v[:min_len] for v in all_values])
                    mean_vals = np.mean(truncated, axis=0)
                    mean_steps = steps[:min_len]

                    # Label with config name for non-trivial hyperparams
                    label = config.name if len(lr_configs) > 1 else f"lr={lr}"
                    ctx = PlotContext(
                        ax=ax,
                        x=mean_steps,
                        y=mean_vals,
                        label=label,
                        plot_kwargs={"color": color, "alpha": 1.0, "linestyle": "-"},
                    )
                    strategy.plot(ctx)

        # Set subplot title with split name
        split_prefix = f"{split_name} " if split_name else ""
        ax.set_title(f"{split_prefix}{key.metric.display_name} of {opt_name}")
        ax.set_xlabel("Steps")
        ax.legend(fontsize=8)

    # Explicitly enable y-tick labels on all subplots
    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Add figure title with split and repeat count
    # Determine number of repeats from first config
    first_config = next(iter(results.keys()))
    first_entry = results[first_config]
    num_repeats = len(_get_histories(first_entry))

    repeat_text = f", Repeated x{num_repeats}" if num_repeats > 1 else ""
    split_prefix = f"{split_name} " if split_name else ""
    fig.suptitle(
        f"{split_prefix}{key.metric.display_name} Comparison for each Optimizer{repeat_text}",
        fontsize=13,
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined(
    results: ResultsType,
    configs_by_lr: Mapping[float, List[OptimizerConfig]],
    learning_rates: List[float],
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Combined view: one subplot per learning rate, all optimizers overlaid."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ncols = len(learning_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), sharey=True)
    if ncols == 1:
        axes = [axes]

    # Check if we have train/test splits
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    # Color map for different optimizers
    import colorsys

    cmap = plt.get_cmap("tab10")

    # Collect all optimizer names for consistent coloring
    all_optimizer_names = sorted(
        set(
            config.optimizer.name
            for lr_configs in configs_by_lr.values()
            for config in lr_configs
        )
    )
    opt_colors = {opt: cmap(i % 10) for i, opt in enumerate(all_optimizer_names)}

    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

        configs = configs_by_lr.get(lr, [])
        for config in configs:
            entry = results[config]
            history = _get_history(entry)

            if history is None:
                continue

            history_cpu = history.copy_cpu()
            steps = np.array(history_cpu.get_steps())

            # Get base color for this optimizer
            base_color = opt_colors[config.optimizer.name]
            r, g, b = base_color[:3]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            if show_split_styles:
                # Plot train and test with different lightness
                train_keys = [k for k in metric_keys if k.split == DatasetSplit.Train]
                test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                # Plot train split (lighter)
                for key in train_keys:
                    values = np.array(history_cpu.get(key))
                    train_v = min(v * 1.3, 1.0)
                    train_color = colorsys.hsv_to_rgb(h, s, train_v)

                    ctx = PlotContext(
                        ax=ax,
                        x=steps,
                        y=values,
                        label=f"{config.name} (train)",
                        plot_kwargs={
                            "color": train_color,
                            "alpha": 0.7,
                            "linestyle": "-",
                        },
                    )
                    strategy.plot(ctx)

                # Plot test split (darker)
                for key in test_keys:
                    values = np.array(history_cpu.get(key))
                    test_v = v * 0.7
                    test_color = colorsys.hsv_to_rgb(h, s, test_v)

                    ctx = PlotContext(
                        ax=ax,
                        x=steps,
                        y=values,
                        label=f"{config.name} (test)",
                        plot_kwargs={
                            "color": test_color,
                            "alpha": 0.7,
                            "linestyle": "-",
                        },
                    )
                    strategy.plot(ctx)
            else:
                # No split differentiation
                for key in metric_keys:
                    values = np.array(history_cpu.get(key))
                    label = config.name + (f"_{key.split.name}" if key.split else "")
                    ctx = PlotContext(
                        ax=ax,
                        x=steps,
                        y=values,
                        label=label,
                        plot_kwargs={"color": base_color, "alpha": 0.7},
                    )
                    strategy.plot(ctx)

        ax.set_xlabel("Steps")
        ax.set_title(f"LR = {lr}")

    # Explicitly enable y-tick labels on all subplots
    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Create single legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 4),
        fontsize=8,
        frameon=True,
    )

    # Add figure title
    fig.suptitle(
        f"{metric_keys[0].metric.display_name} for each Learning Rate",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for legend at top
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_separate(
    results: ResultsType,
    config: OptimizerConfig,
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Single optimizer config view."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

    entry = results[config]
    histories = _get_histories(entry)

    if not histories:
        plt.close()
        return

    for h in histories:
        h_cpu = h.copy_cpu()
        steps = np.array(h_cpu.get_steps())

        for key in metric_keys:
            values = np.array(h_cpu.get(key))
            label = key.split.name if key.split else key.metric.name
            ctx = PlotContext(
                ax=ax, x=steps, y=values, label=label, plot_kwargs={"alpha": 0.7}
            )
            strategy.plot(ctx)

    ax.set_xlabel("Steps")
    ax.legend()

    # Add figure title
    fig.suptitle(
        f"{metric_keys[0].metric.display_name}: {config.name}", fontsize=12, y=0.995
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hyperparam_grid(
    results: ResultsType,
    learning_rates: List[float],
    rho_values: List[float],
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """
    Hyperparameter grid: rows=rho, cols=learning_rate.
    Each subplot shows all optimizers at that (lr, rho) combination.
    Legend is placed outside to avoid clutter.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    nrows = len(rho_values)
    ncols = len(learning_rates)

    # Create grid with shared axes
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Color map for different optimizers
    cmap = plt.get_cmap("tab10")

    # Collect all optimizer types for consistent coloring
    all_optimizer_types = sorted(
        set(config.optimizer.name for config in results.keys())
    )
    opt_colors = {opt: cmap(i % 10) for i, opt in enumerate(all_optimizer_types)}

    for row_idx, rho in enumerate(rho_values):
        for col_idx, lr in enumerate(learning_rates):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

            # Find all configs with this lr and rho
            matching_configs = [
                config
                for config in results.keys()
                if config.learning_rate == lr
                and config.get(Hyperparam.Rho, None) == rho
            ]

            if not matching_configs:
                # Empty subplot - just show the parameter values
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Determine if we should show train/test differentiation
            has_splits = any(key.split is not None for key in metric_keys)
            is_loss_or_error = any(
                key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
            )
            show_split_styles = has_splits and is_loss_or_error

            for config in matching_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                color = opt_colors[config.optimizer.name]

                # Group metric keys by split if applicable
                if show_split_styles:
                    import colorsys

                    train_keys = [
                        k for k in metric_keys if k.split == DatasetSplit.Train
                    ]
                    test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                    # Convert base color to HSV for value adjustment
                    r, g, b = color[:3]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)

                    # Plot train split (lighter value, solid line)
                    for key in train_keys:
                        all_values: List[np.ndarray] = []
                        steps: Optional[np.ndarray] = None

                        for h_obj in histories:
                            h_cpu = h_obj.copy_cpu()
                            vals = np.array(h_cpu.get(key))
                            curr_steps = np.array(h_cpu.get_steps())
                            all_values.append(vals)
                            if steps is None or len(curr_steps) > len(steps):
                                steps = curr_steps

                        if all_values and steps is not None:
                            min_len = min(len(v) for v in all_values)
                            truncated = np.stack([v[:min_len] for v in all_values])
                            mean_vals = np.mean(truncated, axis=0)
                            mean_steps = steps[:min_len]

                            # Lighter color for train (increase value)
                            train_v = min(v * 1.3, 1.0)  # Make lighter but cap at 1.0
                            train_color = colorsys.hsv_to_rgb(h, s, train_v)

                            label = config.optimizer.name
                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=label,
                                plot_kwargs={
                                    "color": train_color,
                                    "alpha": 0.9,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

                    # Plot test split (darker value, solid line)
                    for key in test_keys:
                        all_values = []
                        steps = None

                        for h_obj in histories:
                            h_cpu = h_obj.copy_cpu()
                            vals = np.array(h_cpu.get(key))
                            curr_steps = np.array(h_cpu.get_steps())
                            all_values.append(vals)
                            if steps is None or len(curr_steps) > len(steps):
                                steps = curr_steps

                        if all_values and steps is not None:
                            min_len = min(len(v) for v in all_values)
                            truncated = np.stack([v[:min_len] for v in all_values])
                            mean_vals = np.mean(truncated, axis=0)
                            mean_steps = steps[:min_len]

                            # Darker color for test (decrease value)
                            test_v = v * 0.7  # Make darker
                            test_color = colorsys.hsv_to_rgb(h, s, test_v)

                            # Plot with solid line (no markers, no label to avoid duplicates)
                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label="",  # Don't duplicate in legend
                                plot_kwargs={
                                    "color": test_color,
                                    "alpha": 0.9,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

                else:
                    # No split differentiation (reference metrics or single split)
                    for key in metric_keys:
                        all_values = []
                        steps = None

                        for h in histories:
                            h_cpu = h.copy_cpu()
                            vals = np.array(h_cpu.get(key))
                            curr_steps = np.array(h_cpu.get_steps())
                            all_values.append(vals)
                            if steps is None or len(curr_steps) > len(steps):
                                steps = curr_steps

                        if all_values and steps is not None:
                            min_len = min(len(v) for v in all_values)
                            truncated = np.stack([v[:min_len] for v in all_values])
                            mean_vals = np.mean(truncated, axis=0)
                            mean_steps = steps[:min_len]

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=config.optimizer.name,
                                plot_kwargs={
                                    "color": color,
                                    "alpha": 1.0,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

            # Titles for top row (learning rates)
            if row_idx == 0:
                ax.set_title(f"lr={lr}", fontsize=10)

            # Y-labels for leftmost column (rho values)
            if col_idx == 0:
                ax.set_ylabel(f"rho={rho}\n{metric_keys[0].metric.name}", fontsize=9)

            # X-labels for bottom row
            if row_idx == nrows - 1:
                ax.set_xlabel("Steps", fontsize=9)

    # Explicitly enable y-tick labels on all subplots
    for row in axes:
        for ax in row:
            ax.tick_params(axis="y", which="both", labelleft=True)
            ax.tick_params(labelsize=8)

    # Determine if we showed train/test differentiation
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    # Create legends
    # 1. Optimizer legend (placed outside on right)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Optimizers",
            fontsize=9,
        )

    # 2. Train/test legend (horizontal at top) if applicable
    if show_split_styles:
        from matplotlib.lines import Line2D

        # Show lighter color for train, darker for test
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=(0.7, 0.7, 0.7),
                linestyle="-",
                linewidth=2,
                label="Train (lighter)",
            ),
            Line2D(
                [0],
                [0],
                color=(0.3, 0.3, 0.3),
                linestyle="-",
                linewidth=2,
                label="Test (darker)",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            fontsize=9,
            frameon=True,
        )

    # Add figure title
    fig.suptitle(
        f"{metric_keys[0].metric.display_name} Hyperparameter Grid (Rows=ρ, Cols=LR)",
        fontsize=14,
        y=0.998,
    )

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sam_comparison(
    results: ResultsType,
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """
    SAM comparison plot: base vs SAM variants.
    Rows = base optimizers, Cols = [Base, SAM variant]
    Lines = hyperparameter combinations
    Color = learning rate, Outline (SAM only) = rho lightness
    Train/test differentiation for Loss/Error only.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Identify optimizer pairs
    optimizer_pairs = _identify_optimizer_pairs(results)

    if not optimizer_pairs:
        # No matching base/SAM pairs, skip this plot
        return

    # Determine if this metric has train/test splits
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    nrows = len(optimizer_pairs)
    ncols = 2  # Base and SAM

    # Create figure with extra space for legend on the right
    fig = plt.figure(figsize=(8 * ncols + 2, 5 * nrows + 0.5))

    # Create gridspec for subplots
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(
        nrows,
        ncols,
        hspace=0.3,
        wspace=0.15,
        top=0.92,  # Leave space at top for legend
        bottom=0.08,
        left=0.08,
        right=0.75,  # Leave space on right for legend
    )

    # Create axes with shared y-axis per row
    axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                # First column - no sharing
                axes[i, j] = fig.add_subplot(gs[i, j])
            else:
                # Share y-axis with first column of same row
                axes[i, j] = fig.add_subplot(gs[i, j], sharey=axes[i, 0])

    base_optimizers = sorted(optimizer_pairs.keys(), key=lambda x: x.name)

    # Get all configs and unique rho/lr values
    all_configs = list(results.keys())
    all_rhos = sorted(
        set(
            c.get(Hyperparam.Rho, 0.0)
            for c in all_configs
            if c.get(Hyperparam.Rho) is not None
        )
    )
    all_lrs = sorted(set(c.learning_rate for c in all_configs))

    # Color mapping: learning rate → hue, rho → lightness
    import colorsys

    lr_cmap = plt.get_cmap("tab10")

    # Assign each learning rate a hue from qualitative palette
    lr_to_hue = {}
    for i, lr in enumerate(all_lrs):
        base_color = lr_cmap(i % 10)
        r, g, b = base_color[:3]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        lr_to_hue[lr] = h

    def get_color_for_config(lr, rho):
        """Get color based on LR (hue) and rho (lightness)."""
        if lr not in lr_to_hue:
            return lr_cmap(0)

        h = lr_to_hue[lr]
        s = 0.7  # Fixed saturation for vividness

        # Map rho to value (lightness): low rho = lighter, high rho = darker
        if rho is not None and all_rhos and len(all_rhos) > 1:
            rho_idx = all_rhos.index(rho)
            # Use range [0.4, 0.9] to avoid too dark or too light
            v = 0.9 - 0.5 * (rho_idx / max(1, len(all_rhos) - 1))
        else:
            # For non-SAM optimizers (no rho), use full brightness
            v = 0.9

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)

    for row_idx, base_opt in enumerate(base_optimizers):
        sam_opt = optimizer_pairs[base_opt]

        for col_idx, opt in enumerate([base_opt, sam_opt]):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

            # Get all configs for this optimizer
            opt_configs = [c for c in results.keys() if c.optimizer == opt]

            if not opt_configs:
                continue

            # Check if this is a SAM variant (has rho parameter)
            is_sam_variant = col_idx == 1

            for config in opt_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                lr = config.learning_rate
                rho = config.get(Hyperparam.Rho, None)

                # Get color based on LR (hue) and rho (lightness)
                color = get_color_for_config(lr, rho)
                linewidth = 2.0
                line_alpha = 0.6 if is_sam_variant else 0.9

                # Only plot test split (or reference metrics without split)
                if show_split_styles:
                    test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                    # Plot test split only (solid lines)
                    for key in test_keys:
                        all_values = []
                        steps = None

                        for h in histories:
                            h_cpu = h.copy_cpu()
                            vals = np.array(h_cpu.get(key))
                            curr_steps = np.array(h_cpu.get_steps())
                            all_values.append(vals)
                            if steps is None or len(curr_steps) > len(steps):
                                steps = curr_steps

                        if all_values and steps is not None:
                            min_len = min(len(v) for v in all_values)
                            truncated = np.stack([v[:min_len] for v in all_values])
                            mean_vals = np.mean(truncated, axis=0)
                            mean_steps = steps[:min_len]

                            # Plot with solid line
                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label="",
                                plot_kwargs={
                                    "color": color,
                                    "linewidth": linewidth,
                                    "alpha": line_alpha,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

                else:
                    # No split differentiation (reference metrics)
                    for key in metric_keys:
                        all_values = []
                        steps = None

                        for h in histories:
                            h_cpu = h.copy_cpu()
                            vals = np.array(h_cpu.get(key))
                            curr_steps = np.array(h_cpu.get_steps())
                            all_values.append(vals)
                            if steps is None or len(curr_steps) > len(steps):
                                steps = curr_steps

                        if all_values and steps is not None:
                            min_len = min(len(v) for v in all_values)
                            truncated = np.stack([v[:min_len] for v in all_values])
                            mean_vals = np.mean(truncated, axis=0)
                            mean_steps = steps[:min_len]

                            # Plot with solid line
                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label="",
                                plot_kwargs={
                                    "color": color,
                                    "linewidth": linewidth,
                                    "alpha": line_alpha,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

            # Add title to each subplot with optimizer name
            ax.set_title(f"{opt.name}", fontsize=11, pad=10)

            # X-labels for bottom row
            if row_idx == nrows - 1:
                ax.set_xlabel("Steps", fontsize=10)

    # Enable y-tick labels on leftmost subplots only (due to sharey)
    for row_idx in range(nrows):
        axes[row_idx, 0].tick_params(axis="y", which="both", labelleft=True)

    # Add legend showing learning rate colors and rho lightness
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = []

    # Add learning rate color indicators (hue)
    if all_lrs:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="Learning Rate (hue):")
        )
        for lr in all_lrs:
            # Show with middle rho value for SAM variants
            mid_rho = all_rhos[len(all_rhos) // 2] if all_rhos else None
            color = get_color_for_config(lr, mid_rho)
            lr_label = f"{lr:.0e}" if lr < 0.01 else f"{lr:.2g}"
            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, label=f"  {lr_label}")
            )

    # Add rho lightness indicators (only for SAM variants)
    if all_rhos:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="ρ (lightness, SAM only):")
        )
        # Show a few representative rho values with first LR
        sample_lr = all_lrs[0] if all_lrs else 0.01
        rho_indices = (
            [0, len(all_rhos) // 2, -1]
            if len(all_rhos) > 2
            else list(range(len(all_rhos)))
        )
        for rho_idx in rho_indices:
            if rho_idx < len(all_rhos):
                rho = all_rhos[rho_idx]
                color = get_color_for_config(sample_lr, rho)
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=3, label=f"  ρ={rho:.2g}")
                )

    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(0.77, 0.5),
            fontsize=9,
            frameon=True,
            fancybox=True,
        )

    # Add figure title
    split_note = " (Test Split)" if show_split_styles else ""
    fig.suptitle(
        f"{metric_keys[0].metric.display_name}: Base vs SAM Variants{split_note}",
        fontsize=14,
        y=0.995,
    )

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def _identify_optimizer_pairs(
    results: ResultsType,
) -> Dict[Optimizer, Optimizer]:
    """
    Identify base optimizer -> SAM variant pairs present in results.
    Returns dict mapping base optimizer to its SAM counterpart.
    """
    # Define all possible base -> SAM mappings
    sam_mappings = {
        Optimizer.GD: Optimizer.SAM,
        Optimizer.NGD: Optimizer.SAM_NGD,
        Optimizer.Adam: Optimizer.SAM_Adam,
        Optimizer.AdaGrad: Optimizer.SAM_AdaGrad,
    }

    # Get all optimizers present in results
    present_optimizers = set(config.optimizer for config in results.keys())

    # Filter to only pairs where both base and SAM variant exist
    pairs = {}
    for base, sam in sam_mappings.items():
        if base in present_optimizers and sam in present_optimizers:
            pairs[base] = sam

    return pairs


def _compute_hyperparam_score(config: OptimizerConfig) -> float:
    """
    Compute sorting score for hyperparameters using lr × rho product.
    Returns lr for configs without rho, or lr × rho for SAM variants.
    """
    lr = config.learning_rate
    rho = config.get(Hyperparam.Rho)

    if rho is None:
        # Non-SAM optimizer, use lr only
        return lr

    return lr * rho


def _assign_sequential_colors(
    configs: List[OptimizerConfig],
    colormap_name: str = "viridis",
) -> Dict[OptimizerConfig, Any]:
    """
    Assign sequential colors to configs based on hyperparameter scores.
    Returns dict mapping config to matplotlib color.
    """
    if not configs:
        return {}

    # Compute scores and sort
    scores = [_compute_hyperparam_score(c) for c in configs]

    # Get colormap
    cmap = plt.get_cmap(colormap_name)

    # Normalize scores to [0, 1] for colormap
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All same score, use single color
        return {c: cmap(0.5) for c in configs}

    color_map = {}
    for config, score in zip(configs, scores):
        normalized = (score - min_score) / (max_score - min_score)
        color_map[config] = cmap(normalized)

    return color_map


def _assign_2d_colors(
    configs: List[OptimizerConfig],
    lr_colormap: str = "Blues",
    rho_colormap: str = "Oranges",
) -> tuple[Dict[OptimizerConfig, Any], tuple[float, float], tuple[float, float]]:
    """
    Assign 2D colors based on lr and rho using two sequential colormaps.
    Blends colors from both dimensions.

    Returns:
        - Dict mapping config to color
        - (lr_min, lr_max) range
        - (rho_min, rho_max) range
    """
    if not configs:
        return {}, (0, 1), (0, 1)

    # Extract lr and rho values
    lr_values = [c.learning_rate for c in configs]
    rho_values = [c.get(Hyperparam.Rho, 0.0) for c in configs]

    # Get ranges
    lr_min, lr_max = min(lr_values), max(lr_values)
    rho_min, rho_max = min(rho_values), max(rho_values)

    # Get colormaps - use darker portions to avoid yellow/light colors
    lr_cmap = plt.get_cmap(lr_colormap)
    rho_cmap = plt.get_cmap(rho_colormap)

    color_map = {}
    for config in configs:
        lr = config.learning_rate
        rho = config.get(Hyperparam.Rho, 0.0)

        # Normalize to [0.3, 0.9] to avoid very light colors
        if lr_max == lr_min:
            lr_norm = 0.6
        else:
            lr_norm = 0.3 + 0.6 * (lr - lr_min) / (lr_max - lr_min)

        if rho_max == rho_min:
            rho_norm = 0.6
        else:
            rho_norm = 0.3 + 0.6 * (rho - rho_min) / (rho_max - rho_min)

        # Get colors from each map
        lr_color = np.array(lr_cmap(lr_norm)[:3])  # RGB only
        rho_color = np.array(rho_cmap(rho_norm)[:3])

        # Blend using multiplication (darker where both are high)
        blended = lr_color * rho_color
        # Renormalize to avoid too dark colors
        blended = blended / blended.max() if blended.max() > 0 else blended

        color_map[config] = tuple(blended)

    return color_map, (lr_min, lr_max), (rho_min, rho_max)


def _assign_color_and_linewidth(
    configs: List[OptimizerConfig],
    lr_colormap: str = "viridis",
    min_linewidth: float = 1.0,
    max_linewidth: float = 4.0,
) -> tuple[
    Dict[OptimizerConfig, tuple[Any, float]], tuple[float, float], tuple[float, float]
]:
    """
    Assign colors based on learning rate and line widths based on rho.

    Visual mapping:
    - Color (sequential): Learning rate
    - Line width: Rho (thicker = higher rho)

    Returns:
        - Dict mapping config to (color, linewidth)
        - (lr_min, lr_max) range
        - (rho_min, rho_max) range
    """
    if not configs:
        return {}, (0, 1), (0, 1)

    # Extract lr and rho values
    lr_values = [c.learning_rate for c in configs]
    rho_values = [c.get(Hyperparam.Rho, 0.0) for c in configs]

    # Get ranges
    lr_min, lr_max = min(lr_values), max(lr_values)
    rho_min, rho_max = min(rho_values), max(rho_values)

    # Get colormap - use middle-to-dark range to avoid very light colors
    # Range [0.2, 0.85] avoids both too-light and too-dark extremes
    cmap = plt.get_cmap(lr_colormap)

    style_map = {}
    for config in configs:
        lr = config.learning_rate
        rho = config.get(Hyperparam.Rho, 0.0)

        # Normalize lr to [0.2, 0.85] to avoid pale colors
        if lr_max == lr_min:
            lr_norm = 0.5
        else:
            lr_norm = 0.2 + 0.65 * (lr - lr_min) / (lr_max - lr_min)

        color = cmap(lr_norm)

        # Normalize rho to linewidth range
        if rho_max == rho_min:
            linewidth = (min_linewidth + max_linewidth) / 2
        else:
            linewidth = min_linewidth + (max_linewidth - min_linewidth) * (
                rho - rho_min
            ) / (rho_max - rho_min)

        style_map[config] = (color, linewidth)

    return style_map, (lr_min, lr_max), (rho_min, rho_max)


def save_results_npz(
    results: ResultsType,
    filepath: Path,
) -> None:
    """Save results as NPZ with config names as keys."""
    import torch

    data: Dict[str, np.ndarray] = {}
    for config, entry in results.items():
        histories = _get_histories(entry)

        for seed, history in enumerate(histories):
            hist_cpu = history.copy_cpu()
            hist_dict = hist_cpu.to_dict()

            for key, values in hist_dict.items():
                # Use config name which includes all hyperparams
                npz_key = f"{config.name}_seed{seed}_{key}"
                # Convert Tensor to numpy if needed
                if isinstance(values, torch.Tensor):
                    values = values.cpu().numpy()
                data[npz_key] = values

    np.savez(filepath, **data)  # type: ignore[call-arg]
