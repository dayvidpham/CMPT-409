import torch
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from tqdm import tqdm

from .types import DatasetSplit, OptimizerConfig, Hyperparam
from .optimizers import OptimizerState
from .models.base import Model
from .metrics import MetricsCollector
from .history import TrainingHistory


def _print_optimizer_configs(
    optimizers: Dict[OptimizerConfig, Callable[[], OptimizerState]],
) -> None:
    """
    Print optimizer configurations in a condensed format.

    Groups by optimizer type and shows learning rates and other hyperparameters as arrays.

    Args:
        optimizers: Dictionary of OptimizerConfig to factory callables
    """
    # Group configs by optimizer type
    grouped: Dict[str, List[OptimizerConfig]] = {}
    for config in optimizers.keys():
        opt_name = config.optimizer.name
        if opt_name not in grouped:
            grouped[opt_name] = []
        grouped[opt_name].append(config)

    print(f"Running {len(optimizers)} optimizer configurations:")

    for optimizer_name in sorted(grouped.keys()):
        opt_configs = grouped[optimizer_name]

        # Extract unique learning rates
        lrs = sorted(set(config.learning_rate for config in opt_configs))

        # Check if there are other hyperparameters (besides learning rate)
        other_params = set()
        for config in opt_configs:
            for hp, _ in config.hyperparams:
                if hp != Hyperparam.LearningRate:
                    other_params.add(hp)

        # Build the display string
        parts = [f"lr=[{', '.join(str(lr) for lr in lrs)}]"]

        # Add other hyperparameters if present
        for param in sorted(other_params, key=lambda x: x.value):
            values = sorted(
                v
                for v in {config.get(param) for config in opt_configs}
                if v is not None
            )
            parts.append(f"{param.value}=[{', '.join(str(v) for v in values)}]")

        print(f"  - {optimizer_name}: {', '.join(parts)} ({len(opt_configs)} configs)")


def _compute_log_spaced_record_points(
    total_iters: int, num_points: int = 200
) -> List[int]:
    """Compute log-spaced iteration indices for recording metrics."""
    points = np.unique(np.logspace(0, np.log10(total_iters), num_points).astype(int))
    points = sorted(set(points) | {1})  # Always include iteration 1
    return [int(p) for p in points if p <= total_iters]


def run_training(
    datasets: Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]],
    model_factory: Callable[[], Model],
    optimizers: Dict[OptimizerConfig, Callable[[], OptimizerState]],
    metrics_collector_factory: Callable[[Model], MetricsCollector],
    train_split: DatasetSplit = DatasetSplit.Train,
    # Mutually exclusive mode parameters
    total_iters: Optional[int] = None,  # Full-batch mode
    num_epochs: Optional[int] = None,  # Mini-batch mode
    batch_size: Optional[int] = None,  # Mini-batch mode
    drop_last: bool = True,  # PyTorch DataLoader convention
    debug: bool = True,
) -> Dict[OptimizerConfig, TrainingHistory]:
    """
    Train models with specified optimizer configurations.

    Learning rate is extracted from each OptimizerConfig.

    Two training modes:
    - Full-batch: Set total_iters, leave batch_size=None
    - Mini-batch: Set num_epochs and batch_size, leave total_iters=None

    Args:
        datasets: Dict mapping splits to (X, y) tuples
        model_factory: Function returning fresh model instance
        optimizers: Dict mapping OptimizerConfig to optimizer factory.
            Each OptimizerConfig must contain Hyperparam.LearningRate.
        metrics_collector_factory: Function creating MetricsCollector from model
        train_split: Which split to use for training
        total_iters: Number of training iterations (full-batch mode)
        num_epochs: Number of epochs (mini-batch mode)
        batch_size: Batch size (mini-batch mode)
        drop_last: If True, drop incomplete final batch (PyTorch default).
                   If False, include partial final batch.
        debug: Show progress bars

    Returns:
        results[config] = TrainingHistory (flat dict keyed by OptimizerConfig)
    """
    # Print optimizer configurations in condensed format
    if debug:
        _print_optimizer_configs(optimizers)

    # Validate mode parameters
    full_batch_mode = total_iters is not None
    mini_batch_mode = batch_size is not None

    if not full_batch_mode and not mini_batch_mode:
        raise ValueError("Must specify either total_iters or (num_epochs + batch_size)")
    if full_batch_mode and mini_batch_mode:
        raise ValueError("Cannot specify both total_iters and batch_size")
    if mini_batch_mode and num_epochs is None:
        raise ValueError("num_epochs required when batch_size is specified")

    if mini_batch_mode:
        assert num_epochs is not None and batch_size is not None
        return _run_minibatch_training(
            datasets,
            model_factory,
            optimizers,
            metrics_collector_factory,
            train_split,
            num_epochs,
            batch_size,
            drop_last,
            debug,
        )
    else:
        assert total_iters is not None
        return _run_fullbatch_training(
            datasets,
            model_factory,
            optimizers,
            metrics_collector_factory,
            train_split,
            total_iters,
            debug,
        )


def _run_fullbatch_training(
    datasets: Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]],
    model_factory: Callable[[], Model],
    optimizers: Dict[OptimizerConfig, Callable[[], OptimizerState]],
    metrics_collector_factory: Callable[[Model], MetricsCollector],
    train_split: DatasetSplit,
    total_iters: int,
    debug: bool,
) -> Dict[OptimizerConfig, TrainingHistory]:
    """Full-batch training."""
    # Logarithmic recording steps (200 points, always includes t=1)
    record_steps = set(
        np.unique(np.logspace(0, np.log10(total_iters), 200).astype(int))
    )
    record_steps.add(1)
    record_steps_list = sorted(record_steps)

    results: Dict[OptimizerConfig, TrainingHistory] = {}

    for config, optimizer_factory in optimizers.items():
        lr = config.learning_rate

        if debug:
            print(f"\nRunning {config.name}")

        # Create fresh model and metrics collector
        model = model_factory()
        X_train, y_train = datasets[train_split]
        collector = metrics_collector_factory(model)
        generic_optim = optimizer_factory()

        # Link optimizer to metrics collector
        generic_optim.metrics_collector = collector

        # Pre-allocate history buffer
        metric_keys = collector.get_metric_keys(list(datasets.keys()))
        history = TrainingHistory(
            metric_keys=metric_keys,
            num_records=len(record_steps_list),
            device=model.device,
            metadata={
                "optimizer": config.name,
                "learning_rate": lr,
                "total_iters": total_iters,
                "mode": "full_batch",
            },
        )

        # Training loop
        iterator = (
            tqdm(range(1, total_iters + 1)) if debug else range(1, total_iters + 1)
        )

        for t in iterator:
            try:
                generic_optim.step(model, X_train, y_train, lr)

                # Record metrics at logging steps
                if t in record_steps:
                    metrics = collector.compute_all(model, datasets)
                    history.record(t, metrics)
            except Exception as e:
                print(f"Error at step {t}: {e}")
                raise e

        results[config] = history

    return results


def _run_minibatch_training(
    datasets: Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]],
    model_factory: Callable[[], Model],
    optimizers: Dict[OptimizerConfig, Callable[[], OptimizerState]],
    metrics_collector_factory: Callable[[Model], MetricsCollector],
    train_split: DatasetSplit,
    num_epochs: int,
    batch_size: int,
    drop_last: bool,
    debug: bool,
) -> Dict[OptimizerConfig, TrainingHistory]:
    """Mini-batch training with DataLoader-style batching."""
    X_train, y_train = datasets[train_split]
    n_samples = X_train.shape[0]
    device = X_train.device

    # Compute batches per epoch (PyTorch DataLoader style)
    if drop_last:
        batches_per_epoch = n_samples // batch_size
        samples_per_epoch = batches_per_epoch * batch_size
    else:
        batches_per_epoch = (n_samples + batch_size - 1) // batch_size  # ceil division
        samples_per_epoch = n_samples

    if batches_per_epoch == 0:
        raise ValueError(
            f"batch_size ({batch_size}) is larger than n_samples ({n_samples})"
        )

    total_iters = num_epochs * batches_per_epoch

    if debug:
        print(f"Mini-batch training: {n_samples} samples, batch_size={batch_size}")
        print(f"  {batches_per_epoch} batches/epoch, {samples_per_epoch} samples/epoch")
        print(f"  {total_iters} total iterations over {num_epochs} epochs")
        if drop_last and samples_per_epoch < n_samples:
            print(
                f"  Note: dropping {n_samples - samples_per_epoch} samples/epoch (drop_last=True)"
            )

    # Log-spaced recording points (200 points)
    record_iters = _compute_log_spaced_record_points(total_iters, num_points=200)

    results: Dict[OptimizerConfig, TrainingHistory] = {}

    for config, optimizer_factory in optimizers.items():
        lr = config.learning_rate

        if debug:
            print(f"\nRunning {config.name}")

        # Fresh model, optimizer, metrics collector
        model = model_factory()
        generic_optim = optimizer_factory()
        collector = metrics_collector_factory(model)

        # Link optimizer to metrics collector
        generic_optim.metrics_collector = collector

        # Pre-allocate history
        metric_keys = collector.get_metric_keys(list(datasets.keys()))
        history = TrainingHistory(
            metric_keys=metric_keys,
            num_records=len(record_iters),
            device=str(device),
            metadata={
                "optimizer": config.name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "drop_last": drop_last,
                "mode": "mini_batch",
            },
        )

        t_global = 0
        record_idx = 0

        # Epoch loop
        epoch_iter = (
            tqdm(range(num_epochs), desc=config.name) if debug else range(num_epochs)
        )

        for epoch in epoch_iter:
            # Shuffle indices (sampling without replacement within epoch)
            perm = torch.randperm(n_samples, device=device)

            # Mini-batch loop
            for batch_idx in range(batches_per_epoch):
                t_global += 1

                # Extract batch
                start = batch_idx * batch_size
                if drop_last:
                    end = start + batch_size
                else:
                    end = min(start + batch_size, n_samples)  # Clamp for partial batch

                batch_indices = perm[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Optimizer step
                generic_optim.step(model, X_batch, y_batch, lr)

                # Record metrics at log-spaced points
                if (
                    record_idx < len(record_iters)
                    and t_global >= record_iters[record_idx]
                ):
                    metrics = collector.compute_all(model, datasets)
                    history.record(t_global, metrics)
                    record_idx += 1

        results[config] = history

    return results
