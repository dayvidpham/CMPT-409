import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Mapping, Tuple
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
from .colors import ColorManagerFactory

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
                    output_path = (
                        base_dir
                        / "separate"
                        / config.dir_name
                        / f"{task.filename_prefix}.png"
                    )
                    plot_separate(
                        results,
                        config,
                        task,
                        strategy,
                        output_path,
                    )

            if save_aggregated:
                output_path = get_path(task, "aggregated", "_comparison")
                plot_aggregated(
                    results,
                    configs_by_lr,
                    learning_rates,
                    task,
                    strategy,
                    output_path,
                )

        # Generate hyperparameter grid plots for appropriate tasks
        if has_rho_sweeps:
            # Extract unique rho and learning rate values (only actual hyperparameter values)
            rho_values = sorted(
                set(
                    rho_val
                    for config in all_configs
                    if (rho_val := config.get(Hyperparam.Rho)) is not None
                )
            )

            # Include learning rates from both SAM and base optimizers
            learning_rates_for_grid = sorted(
                set(
                    config.learning_rate
                    for config in all_configs
                    if config.get(Hyperparam.Rho) is not None or (
                        config.get(Hyperparam.Rho) is None
                        and config.optimizer in (
                            Optimizer.GD,
                            Optimizer.NGD,
                            Optimizer.LossNGD,
                            Optimizer.VecNGD,
                            Optimizer.Adam,
                            Optimizer.AdaGrad,
                        )
                    )
                )
            )

            for task in tasks:
                output_path = get_path(task, "hyperparam_grid", "_grid")
                plot_hyperparam_grid(
                    results,
                    learning_rates_for_grid,
                    rho_values,
                    task,
                    strategy,
                    output_path,
                )
                print(
                    f"Finished plotting `{task.display_title}` (hyperparameter grid) to `{output_path}`"
                )

        # Detect if we have base/SAM optimizer pairs
        optimizer_pairs = _identify_optimizer_pairs(results)

        if optimizer_pairs:
            # Generate SAM comparison plots for each task
            for task in tasks:
                output_path = get_path(task, "sam_comparison", "_sam_comparison")
                plot_sam_comparison(
                    results,
                    task,
                    strategy,
                    output_path,
                )
                print(
                    f"Finished plotting `{task.display_title}` (SAM comparison) to `{output_path}`"
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

        output_path = base_dir / "stability_analysis" / "stability_analysis.png"
        plot_stability_analysis(
            results,
            stability_task,
            output_path,
        )
        print(
            f"Finished plotting `{stability_task.display_title}` (stability analysis) to `{output_path}`"
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
        fig, f"{task.display_title} Hyperparameter Grid (Rows=rho, Cols=LR)"
    )

    # Use paired optimizer colors for hyperparameter grid
    # Get all optimizer types for this task
    all_optimizer_configs = list(results.keys())
    optimizer_types = sorted(
        set(config.optimizer.name for config in all_optimizer_configs)
    )

    # Create color manager for paired optimizer colors
    colors = ColorManagerFactory.create_paired_optimizer_manager(
        optimizer_types, rho_values
    )

    # Get colors for each optimizer type
    opt_name_colors = colors.legend_colors()

    # Pre-compute matching configs for each (lr, rho) pair to avoid O(n^3) iteration
    # This significantly improves performance for large config sets
    configs_by_lr_rho = {}
    base_optimizers_by_lr = {}

    for config in results.keys():
        lr = config.learning_rate
        rho = config.get(Hyperparam.Rho, None)

        # Store configs with explicit rho values
        if rho is not None:
            key = (lr, rho)
            if key not in configs_by_lr_rho:
                configs_by_lr_rho[key] = []
            configs_by_lr_rho[key].append(config)

        # Store base optimizers (no rho) separately for rho=0.0 row
        if rho is None and config.optimizer in (
            Optimizer.GD,
            Optimizer.NGD,  # Backward compatibility
            Optimizer.LossNGD,  # New
            Optimizer.VecNGD,  # New
            Optimizer.Adam,
            Optimizer.AdaGrad,
        ):
            if lr not in base_optimizers_by_lr:
                base_optimizers_by_lr[lr] = []
            base_optimizers_by_lr[lr].append(config)

    for row_idx, rho in enumerate(rho_values):
        for col_idx, lr in enumerate(learning_rates):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=task.base_label)

            # Use pre-computed configs instead of iterating through all configs
            matching_configs = configs_by_lr_rho.get((lr, rho), [])

            # Include base optimizers in EVERY row (as reference for comparison)
            matching_configs = list(matching_configs) + base_optimizers_by_lr.get(lr, [])

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

                # Get the color for this optimizer WITHOUT rho opacity
                # PairedOptimizerColorStrategy provides pastel for base, vibrant for SAM
                color = colors.color_config(config.optimizer.name, rho=None)
                base_rgb = color[:3]  # RGB part

                if show_split_styles:
                    train_keys = [
                        k for k in metric_keys if k.split == DatasetSplit.Train
                    ]
                    test_keys = [k for k in metric_keys if k.split == DatasetSplit.Test]

                    # --- PLOT TRAIN (Background Context) ---
                    # Strategy: Dashed line, lower opacity.
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
                                    "alpha": 0.4,  # Lower opacity for train
                                    "linestyle": "--",  # Distinct texture
                                    "linewidth": 1.5,
                                },
                            )
                            strategy.plot(ctx)

                    # --- PLOT TEST (Foreground Focus) ---
                    # Strategy: Solid line, default opacity.
                    for key in test_keys:
                        all_values, steps = _collect_data(histories, key)
                        if all_values and steps is not None:
                            mean_vals, mean_steps = _aggregate_runs(all_values, steps)

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label=f"{config.optimizer.name} (Test)",
                                plot_kwargs={
                                    "color": base_rgb,
                                    "alpha": 0.8,  # Default opacity for test
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
                                    "alpha": 0.8,  # Default opacity
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

    # 1. Optimizer Type Legend (Bright Colors)
    opt_handles = []
    opt_labels = []
    opt_colors = colors.legend_colors()

    for opt_name in sorted(optimizer_types):
        opt_handles.append(
            Line2D(
                [0], [0], color=opt_colors[opt_name][:3], lw=3, label=f"  {opt_name}"
            )
        )
        opt_labels.append(opt_name)

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
        # Combine with spacer
        final_handles = style_handles + opt_handles
        final_labels = [h.get_label() for h in style_handles] + opt_labels
    else:
        # Just optimizer colors
        final_handles = opt_handles
        final_labels = opt_labels

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


def _compute_rho_vibrancy_color(lr: float, rho: float, all_lrs: List[float], all_rhos: List[float]) -> Tuple[float, float, float]:
    """
    Compute color where LR determines hue and rho determines vibrancy.
    Higher rho = more vibrant (higher saturation, lower lightness).
    Lower rho = more pastel (lower saturation, higher lightness).

    Args:
        lr: Learning rate value
        rho: Rho value
        all_lrs: List of all learning rates (sorted)
        all_rhos: List of all rho values (sorted)

    Returns:
        RGB tuple normalized to [0,1]
    """
    try:
        import hsluv
    except ImportError:
        hsluv = None

    # Map LR to hue position around color wheel
    sorted_lrs = sorted(all_lrs)
    n_lrs = len(sorted_lrs)
    if lr in sorted_lrs:
        lr_rank = sorted_lrs.index(lr)
        lr_normalized = lr_rank / max(1, n_lrs - 1)
    else:
        lr_normalized = 0.5

    hue = (lr_normalized * 330.0) % 360.0  # Use 330° to avoid red-red overlap

    # Map rho to vibrancy (saturation and lightness)
    # Special case: rho=0.0 (base optimizer) should be MOST vibrant
    # SAM variants (rho > 0) vary from less vibrant to more vibrant
    if rho == 0.0:
        # Base optimizer: maximum vibrancy
        saturation = 95.0
        lightness = 45.0
    else:
        # SAM variants: scale vibrancy with rho
        sorted_rhos = sorted([r for r in all_rhos if r > 0.0])  # Only non-zero rhos
        n_rhos = len(sorted_rhos)
        if n_rhos > 0 and rho in sorted_rhos:
            rho_rank = sorted_rhos.index(rho)
            rho_normalized = rho_rank / max(1, n_rhos - 1)
        else:
            rho_normalized = 0.5

        # Higher rho = more vibrant (higher saturation, lower lightness)
        saturation = 30.0 + 65.0 * rho_normalized  # 30% to 95%
        lightness = 85.0 - 40.0 * rho_normalized  # 85% to 45%

    if hsluv is None:
        # Fallback to simple HSV conversion
        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, lightness / 100.0)
    else:
        rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

    return (rgb[0], rgb[1], rgb[2])


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

    # --- Color Setup (LR determines hue, rho determines vibrancy) ---
    all_configs = list(results.keys())
    all_lrs = sorted(set(c.learning_rate for c in all_configs))
    all_rhos = sorted(set(c.get(Hyperparam.Rho, 0.0) for c in all_configs))

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

                    # Compute color with rho-based vibrancy (higher rho = more vibrant)
                    lr = config.learning_rate
                    rho = config.get(Hyperparam.Rho, 0.0)
                    color_rgb = _compute_rho_vibrancy_color(lr, rho, all_lrs, all_rhos)

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
                                        "color": color_rgb,
                                        "alpha": 0.8,
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

    # 1. Learning Rate Colors (Hue)
    if all_lrs:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):")
        )
        for lr in sorted(all_lrs):
            # Use mid-rho for LR color examples
            mid_rho_idx = len(all_rhos) // 2 if all_rhos else 0
            sample_rho = all_rhos[mid_rho_idx] if all_rhos else 0.0
            c_rgb = _compute_rho_vibrancy_color(lr, sample_rho, all_lrs, all_rhos)
            legend_elements.append(Line2D([0], [0], color=c_rgb, lw=3, label=f"  lr={lr}"))

    # 2. Rho (Vibrancy: higher rho = more intense)
    if all_rhos and len(all_rhos) > 1:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="  "))
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="rho (Vibrancy):")
        )
        sample_lr = all_lrs[0] if all_lrs else 0.01

        # Show Min, Mid, Max rho
        indices = (
            [0, len(all_rhos) // 2, -1] if len(all_rhos) > 2 else range(len(all_rhos))
        )
        for i in sorted(list(set(indices))):
            rho = all_rhos[i]
            c_rgb = _compute_rho_vibrancy_color(sample_lr, rho, all_lrs, all_rhos)
            legend_elements.append(Line2D([0], [0], color=c_rgb, lw=3, alpha=0.8, label=f"  rho={rho}"))

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
    Color = learning rate, Opacity (SAM only) = rho opacity
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
    all_rhos = sorted(set(c.get(Hyperparam.Rho, 0.0) for c in all_configs))
    all_lrs = sorted(set(c.learning_rate for c in all_configs))

    for row_idx, base_opt in enumerate(base_optimizers):
        sam_opt = optimizer_pairs[base_opt]

        for col_idx, opt in enumerate([base_opt, sam_opt]):
            ax = axes[row_idx, col_idx]
            strategy.configure_axis(ax, base_label=task.base_label)

            # Get all configs for this optimizer
            opt_configs = [c for c in results.keys() if c.optimizer == opt]

            if not opt_configs:
                continue

            for config in opt_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                # Compute color with rho-based vibrancy (higher rho = more vibrant)
                lr = config.learning_rate
                rho = config.get(Hyperparam.Rho, 0.0)
                color_rgb = _compute_rho_vibrancy_color(lr, rho, all_lrs, all_rhos)
                linewidth = 2.0

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
                                alpha = 0.4
                                lw = 1.5
                            elif key.split == DatasetSplit.Test:
                                # Test: solid, default opacity, thicker
                                linestyle = "-"
                                alpha = 0.8
                                lw = 2.0
                            else:
                                # Fallback
                                linestyle = "-"
                                alpha = 0.8
                                lw = linewidth

                            ctx = PlotContext(
                                ax=ax,
                                x=mean_steps,
                                y=mean_vals,
                                label="",
                                plot_kwargs={
                                    "color": color_rgb,
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
                                    "color": color_rgb,
                                    "linewidth": linewidth,
                                    "alpha": 0.8,
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

    # Add legend showing learning rate colors and rho opacity
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
            Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):")
        )
        # Use mid-rho for LR color examples
        mid_rho_idx = len(all_rhos) // 2 if all_rhos else 0
        sample_rho = all_rhos[mid_rho_idx] if all_rhos else 0.0
        for lr in sorted(all_lrs):
            c_rgb = _compute_rho_vibrancy_color(lr, sample_rho, all_lrs, all_rhos)
            legend_elements.append(
                Line2D([0], [0], color=c_rgb, linewidth=3, label=f"  lr={lr}")
            )

    # Add rho vibrancy indicators (SAM variants)
    if all_rhos and len(all_rhos) > 1:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))
        legend_elements.append(
            Patch(facecolor="none", edgecolor="none", label="rho (Vibrancy, SAM only):")
        )
        # Show a few representative rho values with first LR
        sample_lr = all_lrs[0] if all_lrs else 0.01
        rho_indices = (
            [0, len(all_rhos) // 2, -1]
            if len(all_rhos) > 2
            else list(range(len(all_rhos)))
        )
        for i in rho_indices:
            rho = all_rhos[i]
            c_rgb = _compute_rho_vibrancy_color(sample_lr, rho, all_lrs, all_rhos)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=c_rgb,
                    linewidth=3,
                    alpha=0.8,
                    label=f"  rho={rho}",
                )
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
