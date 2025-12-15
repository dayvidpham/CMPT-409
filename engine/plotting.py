import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Mapping
from datetime import datetime
from dataclasses import dataclass
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


@dataclass
class PlotTask:
    """
    Defines a specific plotting job.
    Can represent a single split (e.g., Train Loss) or a mixed split (e.g., Combined Loss).
    """

    metric: Metric
    keys: List[MetricKey]
    filename_prefix: str  # e.g., "train_loss" or "loss"
    display_title: str  # e.g., "Train Loss" or "Loss"

    @property
    def base_label(self) -> str:
        """Return raw metric name to avoid double-application of strategy suffix."""
        return self.keys[0].metric.name if self.keys else self.display_title


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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    # Group keys by Metric Enum
    metric_groups: Dict[Metric, List[MetricKey]] = {}
    for key in metric_keys:
        if key.metric not in metric_groups:
            metric_groups[key.metric] = []
        metric_groups[key.metric].append(key)

    # Resolve strategy overrides
    overrides = strategy_overrides or {}

    # Iterate Metrics and Generate Tasks
    for metric, keys in metric_groups.items():
        strategy = overrides.get(metric, metric.strategy)
        tasks: List[PlotTask] = []

        # --- Task A: Combined / Mixed Split ---
        # Contains ALL keys for this metric
        tasks.append(
            PlotTask(
                metric=metric,
                keys=keys,
                filename_prefix=metric.name.lower(),  # e.g. "loss"
                display_title=metric.display_name,  # e.g. "Loss"
            )
        )

        # --- Task B & C: Specific Splits (Loss metric only) ---
        # Only create split-specific tasks for Loss metric
        if metric == Metric.Loss:
            # Group keys by DatasetSplit Enum
            keys_by_split: Dict[DatasetSplit, List[MetricKey]] = {}
            for key in keys:
                if key.split:  # Handle keys that have a valid split
                    if key.split not in keys_by_split:
                        keys_by_split[key.split] = []
                    keys_by_split[key.split].append(key)

            # Create distinct tasks for each split found (Train, Test)
            for split, split_keys in keys_by_split.items():
                tasks.append(
                    PlotTask(
                        metric=metric,
                        keys=split_keys,
                        filename_prefix=f"{split.name.lower()}_{metric.name.lower()}",  # e.g. "train_loss"
                        display_title=f"{split.name} {metric.display_name}",  # e.g. "Train Loss"
                    )
                )

        # --- Dispatch Tasks to Plotters ---
        def get_path(task: PlotTask, folder: str, suffix: str = "") -> Path:
            return base_dir / folder / f"{task.filename_prefix}{suffix}.png"

        # Detect if we have hyperparameter sweeps (configs with rho)
        has_rho_sweeps = any(
            config.get(Hyperparam.Rho) is not None for config in all_configs
        )

        for task in tasks:
            # Helper to build consistent paths

            if save_combined:
                plot_combined(
                    results,
                    configs_by_lr,
                    learning_rates,
                    task,
                    strategy,
                    get_path(task, "combined"),
                )

            if save_separate:
                for config in all_configs:
                    plot_separate(
                        results,
                        config,
                        task,
                        strategy,
                        base_dir
                        / "separate"
                        / config.dir_name
                        / f"{task.filename_prefix}.png",
                    )

            if save_aggregated:
                plot_aggregated(
                    results,
                    configs_by_lr,
                    learning_rates,
                    task,
                    strategy,
                    get_path(task, "aggregated", "_comparison"),
                )

        # Generate hyperparameter grid plots for appropriate tasks
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

            for task in tasks:
                plot_hyperparam_grid(
                    results,
                    learning_rates_for_grid,
                    rho_values,
                    task,
                    strategy,
                    get_path(task, "hyperparam_grid", "_grid"),
                )

        # Detect if we have base/SAM optimizer pairs
        optimizer_pairs = _identify_optimizer_pairs(results)

        if optimizer_pairs:
            # Generate SAM comparison plots for each task
            for task in tasks:
                plot_sam_comparison(
                    results,
                    task,
                    strategy,
                    get_path(task, "sam_comparison", "_sam_comparison"),
                )

    # Detect if we have stability metrics and dispatch stability analysis
    # Stability metrics should have split=None
    stability_keys = [
        k for k in metric_keys if k.metric.requires_model_artifact and k.split is None
    ]

    if stability_keys:
        stability_task = PlotTask(
            metric=Metric.WeightNorm,  # Placeholder Metric type for the group
            keys=stability_keys,
            filename_prefix="stability_analysis",
            display_title="Numerical Stability Metrics",
        )

        plot_stability_analysis(
            results,
            stability_task,
            base_dir / "stability_analysis" / "stability_analysis.png",
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
    task: PlotTask,
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Paper-style plotting: one subplot per optimizer type, colored by learning rate."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Use keys from the task
    selected_keys = task.keys

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
    fig, axes = plt.subplots(
        1, ncols, figsize=(6 * ncols, 5), sharey=True, constrained_layout=True
    )
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

    # Explicitly enable y-tick labels on all subplots
    for ax in axes:
        ax.tick_params(axis="y", which="both", labelleft=True)

    # Add figure title with split and repeat count BEFORE legend
    # Determine number of repeats from first config
    first_config = next(iter(results.keys()))
    first_entry = results[first_config]
    num_repeats = len(_get_histories(first_entry))

    repeat_text = f", Repeated x{num_repeats}" if num_repeats > 1 else ""
    split_prefix = f"{split_name} " if split_name else ""
    strategy.apply_suptitle(
        fig,
        f"{split_prefix}{key.metric.display_name} Comparison for each Optimizer{repeat_text}",
    )

    # Create single legend above the plots (after suptitle for proper spacing)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(len(labels), 6),
            fontsize=8,
            frameon=True,
        )

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined(
    results: ResultsType,
    configs_by_lr: Mapping[float, List[OptimizerConfig]],
    learning_rates: List[float],
    task: PlotTask,
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Combined view: one subplot per learning rate, all optimizers overlaid."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = task.keys
    ncols = len(learning_rates)
    fig, axes = plt.subplots(
        1, ncols, figsize=(5 * ncols, 4.5), sharey=True, constrained_layout=True
    )
    if ncols == 1:
        axes = [axes]

    # Check if we have train/test splits
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    # Color map for different optimizers
    # Collect all optimizer names for consistent colormap assignment
    all_optimizer_names = sorted(
        set(
            config.optimizer.name
            for lr_configs in configs_by_lr.values()
            for config in lr_configs
        )
    )

    # Assign different colormaps to each optimizer type for variety
    colormap_names = [
        "Blues",
        "Oranges",
        "Greens",
        "Reds",
        "Purples",
        "YlOrBr",
        "PuBu",
        "RdPu",
    ]
    opt_colormaps = {
        opt: plt.get_cmap(colormap_names[i % len(colormap_names)])
        for i, opt in enumerate(all_optimizer_names)
    }

    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

        configs = configs_by_lr.get(lr, [])

        # Group configs by optimizer for color assignment
        configs_by_opt: Dict[str, List[OptimizerConfig]] = {}
        for config in configs:
            opt_name = config.optimizer.name
            if opt_name not in configs_by_opt:
                configs_by_opt[opt_name] = []
            configs_by_opt[opt_name].append(config)

        # Assign colors to each config based on hyperparameters within optimizer type
        config_colors: Dict[OptimizerConfig, Any] = {}
        for opt_name, opt_configs in configs_by_opt.items():
            cmap = opt_colormaps[opt_name]

            if len(opt_configs) == 1:
                # Single config for this optimizer, use mid-range color
                config_colors[opt_configs[0]] = cmap(0.6)
            else:
                # Multiple configs, use hyperparameter score to assign colors
                scores = [_compute_hyperparam_score(c) for c in opt_configs]
                min_score = min(scores)
                max_score = max(scores)

                for config, score in zip(opt_configs, scores):
                    if max_score == min_score:
                        normalized = 0.6
                    else:
                        # Map to range [0.3, 0.9] to avoid too light/dark colors
                        normalized = 0.3 + 0.6 * (score - min_score) / (
                            max_score - min_score
                        )
                    config_colors[config] = cmap(normalized)

        for config in configs:
            entry = results[config]
            history = _get_history(entry)

            if history is None:
                continue

            history_cpu = history.copy_cpu()
            steps = np.array(history_cpu.get_steps())

            # Get assigned color for this config
            base_color = config_colors[config]

            if show_split_styles:
                # Plot train and test with solid/dashed line differentiation
                train_keys = [k for k in metric_keys if k.split == DatasetSplit.Train]
                test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                # Plot train split (dashed, lower opacity, thinner)
                for key in train_keys:
                    values = np.array(history_cpu.get(key))

                    ctx = PlotContext(
                        ax=ax,
                        x=steps,
                        y=values,
                        label=f"{config.name} (train)",
                        plot_kwargs={
                            "color": base_color,
                            "alpha": 0.5,
                            "linestyle": "--",
                            "linewidth": 1.5,
                        },
                    )
                    strategy.plot(ctx)

                # Plot test split (solid, full opacity, thicker)
                for key in test_keys:
                    values = np.array(history_cpu.get(key))

                    ctx = PlotContext(
                        ax=ax,
                        x=steps,
                        y=values,
                        label=f"{config.name} (test)",
                        plot_kwargs={
                            "color": base_color,
                            "alpha": 1.0,
                            "linestyle": "-",
                            "linewidth": 2.0,
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

    # Add figure title BEFORE legend for proper spacing
    strategy.apply_suptitle(
        fig, f"{metric_keys[0].metric.display_name} for each Learning Rate"
    )

    # Create single legend at the top (after suptitle)
    handles, labels = axes[0].get_legend_handles_labels()

    # Add split style indicators if needed
    from matplotlib.lines import Line2D

    if show_split_styles:
        # Remove duplicates from automatic legend
        by_label = dict(zip(labels, handles))

        # Add manual legend entries for split styles
        style_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=2,
                label="Test (Solid)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Train (Dashed)",
            ),
            Line2D([0], [0], color="none", label=" "),  # Spacer
        ]

        final_handles = style_handles + list(by_label.values())
        final_labels = [h.get_label() for h in style_handles] + list(by_label.keys())
    else:
        final_handles = handles
        final_labels = labels

    fig.legend(
        final_handles,
        final_labels,
        loc="outside upper center",
        ncol=min(len(final_labels), 4),
        fontsize=8,
        frameon=True,
    )

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_separate(
    results: ResultsType,
    config: OptimizerConfig,
    task: PlotTask,
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Single optimizer config view."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = task.keys
    fig, ax = plt.subplots(figsize=(8, 6))
    strategy.configure_axis(ax, base_label=task.base_label)

    entry = results[config]
    histories = _get_histories(entry)

    if not histories:
        plt.close()
        return

    # Check if we have train/test splits
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    for h in histories:
        h_cpu = h.copy_cpu()
        steps = np.array(h_cpu.get_steps())

        for key in metric_keys:
            values = np.array(h_cpu.get(key))
            label = key.split.name if key.split else key.metric.name

            # Apply split-specific styling
            if show_split_styles and key.split == DatasetSplit.Train:
                # Train: dashed, lower opacity, thinner
                plot_kwargs = {
                    "alpha": 0.5,
                    "linestyle": "--",
                    "linewidth": 1.5,
                }
            elif show_split_styles and key.split == DatasetSplit.Test:
                # Test: solid, full opacity, thicker
                plot_kwargs = {
                    "alpha": 1.0,
                    "linestyle": "-",
                    "linewidth": 2.0,
                }
            else:
                # No split differentiation
                plot_kwargs = {"alpha": 0.7}

            ctx = PlotContext(
                ax=ax, x=steps, y=values, label=label, plot_kwargs=plot_kwargs
            )
            strategy.plot(ctx)

    ax.set_xlabel("Steps")

    # Create legend with split styles if needed
    from matplotlib.lines import Line2D

    if show_split_styles:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Add manual legend entries for split styles
        style_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=2,
                label="Test (Solid)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Train (Dashed)",
            ),
            Line2D([0], [0], color="none", label=" "),  # Spacer
        ]

        final_handles = style_handles + list(by_label.values())
        final_labels = [h.get_label() for h in style_handles] + list(by_label.keys())

        ax.legend(final_handles, final_labels)
    else:
        ax.legend()

    # Add figure title
    strategy.apply_suptitle(fig, f"{task.display_title}: {config.name}")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hyperparam_grid(
    results: ResultsType,
    learning_rates: List[float],
    rho_values: List[float],
    task: PlotTask,
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """
    Hyperparameter grid with improved color disambiguation.
    Change: Uses Line Style + Opacity for Train/Test split instead of just Darkness.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = task.keys
    has_splits = any(key.split is not None for key in metric_keys)
    is_loss_or_error = any(
        key.metric in (Metric.Loss, Metric.Error) for key in metric_keys
    )
    show_split_styles = has_splits and is_loss_or_error

    nrows = len(rho_values)
    ncols = len(learning_rates)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    strategy.apply_suptitle(
        fig, f"{task.display_title} Hyperparameter Grid (Rows=ρ, Cols=LR)"
    )

    # Use a high-contrast colormap (Set1 or Dark2 are better for lines than tab10)
    # But sticking to tab10 for consistency with your other plots is fine if we fix the value issue.
    cmap = plt.get_cmap("tab10")

    all_optimizer_types = sorted(
        set(config.optimizer.name for config in results.keys())
    )
    opt_colors = {opt: cmap(i % 10) for i, opt in enumerate(all_optimizer_types)}

    for row_idx, rho in enumerate(rho_values):
        for col_idx, lr in enumerate(learning_rates):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=task.base_label)

            matching_configs = [
                config
                for config in results.keys()
                if config.learning_rate == lr
                and config.get(Hyperparam.Rho, None) == rho
            ]

            # Also include base optimizers (non-SAM variants) with this lr but no rho
            # They appear on every rho row since they're not SAM variants
            base_optimizers_no_rho = [
                config
                for config in results.keys()
                if config.learning_rate == lr
                and config.get(Hyperparam.Rho, None) is None
                and config.optimizer
                in (
                    Optimizer.GD,
                    Optimizer.NGD,  # Backward compatibility
                    Optimizer.LossNGD,  # New
                    Optimizer.VecNGD,  # New
                    Optimizer.Adam,
                    Optimizer.AdaGrad,
                )
            ]
            matching_configs.extend(base_optimizers_no_rho)

            if not matching_configs:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            for config in matching_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                # Get the base color for this optimizer
                base_rgb = opt_colors[config.optimizer.name]

                if show_split_styles:
                    train_keys = [
                        k for k in metric_keys if k.split == DatasetSplit.Train
                    ]
                    test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                    # --- PLOT TRAIN (Background Context) ---
                    # Strategy: Dashed line, lower opacity.
                    # We do NOT change the color hue/value drastically, keeping it recognizable.
                    for key in train_keys:
                        all_values, steps = _collect_data(histories, key)
                        if all_values and steps is not None:
                            mean_vals, mean_steps = _aggregate_runs(all_values, steps)

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=f"{config.optimizer.name} (Train)",
                                plot_kwargs={
                                    "color": base_rgb,
                                    "alpha": 0.5,  # Make it visually recessive
                                    "linestyle": "--",  # Distinct texture
                                    "linewidth": 1.5,
                                },
                            )
                            strategy.plot(ctx)

                    # --- PLOT TEST (Foreground Focus) ---
                    # Strategy: Solid line, full opacity.
                    # This fixes the "muddy" issue by keeping colors vivid.
                    for key in test_keys:
                        all_values, steps = _collect_data(histories, key)
                        if all_values and steps is not None:
                            mean_vals, mean_steps = _aggregate_runs(all_values, steps)

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=f"{config.optimizer.name} (Test)",  # Label this one for the legend
                                plot_kwargs={
                                    "color": base_rgb,
                                    "alpha": 1.0,  # Full saturation/visibility
                                    "linestyle": "-",  # Solid
                                    "linewidth": 2.0,  # Slightly thicker
                                },
                            )
                            strategy.plot(ctx)

                else:
                    # No split differentiation
                    for key in metric_keys:
                        all_values, steps = _collect_data(histories, key)
                        if all_values and steps is not None:
                            mean_vals, mean_steps = _aggregate_runs(all_values, steps)
                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=config.optimizer.name,
                                plot_kwargs={
                                    "color": base_rgb,
                                    "alpha": 1.0,
                                    "linestyle": "-",
                                },
                            )
                            strategy.plot(ctx)

            if row_idx == 0:
                ax.set_title(f"lr={lr}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"rho={rho}\n{task.display_title}", fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel("Steps", fontsize=9)

    for row in axes:
        for ax in row:
            ax.tick_params(axis="y", which="both", labelleft=True)
            ax.tick_params(labelsize=8)

    # 1. Optimizer Legend (Colors)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Filter duplicates from legend
    by_label = dict(zip(labels, handles))

    if has_splits:
        # 2. Split Legend (Line Styles) - Manually created
        style_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=2,
                label="Test (Solid)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Train (Dashed)",
            ),
            Line2D([0], [0], color="none", label=" "),  # Spacer
        ]
    else:
        style_handles = []

    # Combine
    final_handles = style_handles + list(by_label.values())
    final_labels = [h.get_label() for h in style_handles] + list(by_label.keys())

    fig.legend(
        final_handles,
        final_labels,
        loc="outside upper center",
        ncol=min(len(final_labels), 6),
        fontsize=9,
        frameon=True,
    )

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


# --- Helper functions to keep the code clean ---
def _collect_data(histories, key):
    all_values = []
    steps = None
    for h_obj in histories:
        h_cpu = h_obj.copy_cpu()
        vals = np.array(h_cpu.get(key))
        curr_steps = np.array(h_cpu.get_steps())
        all_values.append(vals)
        if steps is None or len(curr_steps) > len(steps):
            steps = curr_steps
    return all_values, steps


def _aggregate_runs(all_values, steps):
    min_len = min(len(v) for v in all_values)
    truncated = np.stack([v[:min_len] for v in all_values])
    mean_vals = np.mean(truncated, axis=0)
    mean_steps = steps[:min_len]
    return mean_vals, mean_steps


# -------------------------------------------


def plot_stability_analysis(
    results: ResultsType,
    task: PlotTask,
    filepath: Path,
) -> None:
    """
    Plots stability metrics in a grid:
    Rows = Base Optimizer Types (GD, LossNGD, VecNGD, Adam, AdaGrad)
    Cols = [Base: W_Norm, Update_Norm, Ratio] | [SAM: W_Norm, Update_Norm, Ratio]

    Color scheme:
    - Hue: Learning Rate
    - Lightness: Rho (for SAM variants)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Identify optimizer pairs (base -> SAM)
    pairs = _identify_optimizer_pairs(results)
    if not pairs:
        return

    # Extract stability metrics from task in explicit order
    metric_keys = task.keys
    stability_metrics = [
        Metric.WeightNorm,
        Metric.GradNorm,
        Metric.UpdateNorm,
        Metric.GradLossRatio,
    ]

    # Preserve explicit order: filter metric_keys by stability_metrics order
    metrics = []
    for metric in stability_metrics:
        if any(k.metric == metric for k in metric_keys):
            metrics.append(metric)

    if not metrics:
        return

    # Sort base optimizers by name
    base_opts = sorted(pairs.keys(), key=lambda x: x.name)
    nrows = len(pairs)
    ncols = len(metrics) * 2  # N metrics × 2 groups (Base, SAM)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),  # Wider subplots for better readability
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    # --- Color Setup (Hue=LR, Lightness=Rho) ---
    all_configs = list(results.keys())
    all_lrs = sorted(set(c.learning_rate for c in all_configs))
    all_rhos = sorted(
        set(
            c.get(Hyperparam.Rho, 0.0)
            for c in all_configs
            if c.get(Hyperparam.Rho) is not None
        )
    )

    lr_cmap = plt.get_cmap("tab10")

    lr_to_hue = {}
    for i, lr in enumerate(all_lrs):
        r, g, b = lr_cmap(i % 10)[:3]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        lr_to_hue[lr] = h

    def get_color(lr, rho):
        """Get color based on learning rate (hue) and rho (lightness)."""
        if lr not in lr_to_hue:
            return lr_cmap(0)

        h = lr_to_hue[lr]

        # Map rho to value (lightness)
        # Higher rho -> Darker color (Lower Value)
        if rho is not None and rho != 0.0 and len(all_rhos) > 1:
            rho_idx = all_rhos.index(rho)
            # Map index to value range [0.9, 0.4]
            v = 0.9 - 0.5 * (rho_idx / max(1, len(all_rhos) - 1))
        else:
            v = 0.9  # Full brightness for base optimizers or single rho

        return colorsys.hsv_to_rgb(h, 0.7, v)

    # --- Plotting Loop ---
    for row_idx, base_opt in enumerate(base_opts):
        sam_opt = pairs[base_opt]

        # Iterate: Base and SAM interleaved for each metric
        # Layout: [Base: Metric1, SAM: Metric1, Base: Metric2, SAM: Metric2, ...]
        for m_idx, metric in enumerate(metrics):
            col_offset = m_idx * 2  # Each metric takes 2 columns (Base + SAM)

            for group_idx, opt in enumerate([base_opt, sam_opt]):
                ax = axes[row_idx, col_offset + group_idx]
                configs = [c for c in results.keys() if c.optimizer == opt]

                # Filter metric keys for this specific metric
                m_keys = [k for k in task.keys if k.metric == metric]

                # Stability metrics should not have splits
                assert all(k.split is None for k in m_keys), (
                    f"Stability metric {metric.name} should not have dataset splits"
                )

                # Configure axis with metric's strategy
                strategy = metric.strategy
                strategy.configure_axis(ax, base_label=metric.name)

                # Plot data
                for config in configs:
                    history = _get_history(results[config])
                    if history is None:
                        continue

                    histories = _get_histories(results[config])
                    color = get_color(
                        config.learning_rate, config.get(Hyperparam.Rho, 0.0)
                    )

                    for h in histories:
                        h_cpu = h.copy_cpu()
                        steps = h_cpu.get_steps().numpy()

                        for key in m_keys:
                            if key in h_cpu.metric_keys:
                                values = h_cpu.get(key).numpy()
                                ctx = PlotContext(
                                    ax=ax,
                                    x=steps,
                                    y=values,
                                    # Titles: Show optimizer and metric
                                    label=f"{opt.name}: {metric.name}",
                                    plot_kwargs={
                                        "color": color,
                                        "alpha": 0.7,
                                        "linestyle": "-",
                                    },
                                )
                                strategy.plot(ctx)

                # Add subplot title: {Optimizer}: {Metric}
                ax.set_title(f"{opt.name}: {metric.display_name}", fontsize=11, pad=10)

                # Y-axis labels
                col = col_offset + group_idx
                if col == 0:
                    # Leftmost column: show optimizer name
                    ax.set_ylabel(f"{base_opt.name}", fontsize=10)
                elif group_idx == 0:
                    # First column of each metric pair: show metric name
                    ax.set_ylabel(f"{metric.display_name}", fontsize=9)
                else:
                    # Second column of each metric pair: no y-label (redundant)
                    ax.set_ylabel("")

    # Set X labels for bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel("Steps", fontsize=10)

    # --- Legend (Outside Upper Center) ---
    legend_elements = []

    # 1. Learning Rate (Hue)
    if all_lrs:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="LR (Hue):")
        )
        for lr in all_lrs:
            mid_rho = all_rhos[len(all_rhos) // 2] if all_rhos else None
            c = get_color(lr, mid_rho)
            lr_label = f"{lr:.0e}" if lr < 0.01 else f"{lr:.2g}"
            legend_elements.append(
                Line2D([0], [0], color=c, lw=3, label=f"  {lr_label}")
            )

    # 2. Rho (Lightness)
    if all_rhos:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="  "))
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="Rho (Value):")
        )
        sample_lr = all_lrs[0] if all_lrs else 0.01

        # Show Min, Mid, Max rho
        indices = (
            [0, len(all_rhos) // 2, -1] if len(all_rhos) > 2 else range(len(all_rhos))
        )
        for i in sorted(list(set(indices))):
            rho = all_rhos[i]
            c = get_color(sample_lr, rho)
            legend_elements.append(Line2D([0], [0], color=c, lw=3, label=f"  ρ={rho}"))

    fig.suptitle(f"Stability Analysis: {task.display_title}", fontsize=14, y=1.20)

    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="outside upper center",
            ncol=min(len(legend_elements), 8),
            fontsize=9,
            frameon=True,
        )

    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_sam_comparison(
    results: ResultsType,
    task: PlotTask,
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

    metric_keys = task.keys

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

    # Create figure with constrained layout
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

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
        if rho is not None and rho != 0.0 and all_rhos and len(all_rhos) > 1:
            rho_idx = all_rhos.index(rho)
            # Use range [0.4, 0.9] to avoid too dark or too light
            v = 0.9 - 0.5 * (rho_idx / max(1, len(all_rhos) - 1))
        else:
            # For non-SAM optimizers (no rho or rho=0.0), use full brightness
            v = 0.9

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)

    for row_idx, base_opt in enumerate(base_optimizers):
        sam_opt = optimizer_pairs[base_opt]

        for col_idx, opt in enumerate([base_opt, sam_opt]):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=task.base_label)

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
                rho = config.get(Hyperparam.Rho, 0.0)

                # Get color based on LR (hue) and rho (lightness)
                color = get_color_for_config(lr, rho)
                linewidth = 2.0
                line_alpha = 0.6 if is_sam_variant else 0.9

                # Plot all keys from the task
                if show_split_styles:
                    # Plot with solid/dashed differentiation for train/test
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

                            # Apply split-specific styling
                            if key.split == DatasetSplit.Train:
                                # Train: dashed, lower opacity, thinner
                                linestyle = "--"
                                alpha = 0.5
                                lw = 1.5
                            elif key.split == DatasetSplit.Test:
                                # Test: solid, full opacity, thicker
                                linestyle = "-"
                                alpha = 1.0
                                lw = 2.0
                            else:
                                # Fallback
                                linestyle = "-"
                                alpha = line_alpha
                                lw = linewidth

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label="",
                                plot_kwargs={
                                    "color": color,
                                    "linewidth": lw,
                                    "alpha": alpha,
                                    "linestyle": linestyle,
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

    legend_elements = []

    # Add split style indicators if needed
    if show_split_styles:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=2,
                label="Test (Solid)",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Train (Dashed)",
            )
        )
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))

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

    strategy.apply_suptitle(fig, f"{task.display_title}: Base vs SAM Variants")

    # Add legend at the top (after suptitle)
    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="outside upper center",
            ncol=min(len(legend_elements), 6),
            fontsize=9,
            frameon=True,
            fancybox=True,
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
        Optimizer.LossNGD: Optimizer.SAM_LossNGD,
        Optimizer.VecNGD: Optimizer.SAM_VecNGD,
        Optimizer.NGD: Optimizer.SAM_NGD,  # Backward compatibility
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
