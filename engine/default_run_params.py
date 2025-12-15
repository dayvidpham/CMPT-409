"""Default configuration parameters for running experiments.

This module centralizes common configuration used across all run_*.py scripts.
"""

from typing import Optional, Dict, Any, Callable
import torch

from engine import (
    LinearModel,
    TwoLayerModel,
    Metric,
    Optimizer,
    Hyperparam,
    MetricsCollector,
    make_soudry_dataset,
    get_error_rate,
    get_angle,
    get_direction_distance,
    expand_sweep_grid,
)
from engine.losses import LogisticLoss
from engine.metrics import get_weight_norm, compute_update_norm
from engine.optimizers import (
    step_gd,
    step_sam_stable,
    step_loss_ngd,
    step_vec_ngd,
    step_sam_loss_ngd,
    step_sam_vec_ngd,
    make_optimizer_factory,
    make_stateful_optimizer_factory,
)
from engine.optimizers.manual import (
    ManualAdam,
    ManualAdaGrad,
    ManualSAM_Adam,
    ManualSAM_AdaGrad,
    ManualGD,
    ManualLossNGD,
    ManualVecNGD,
    ManualSAM,
    ManualSAM_LossNGD,
    ManualSAM_VecNGD,
)


# ============================================================================
# General
# ============================================================================


def default_dataset_generation(
    n: int = 200,
    d: int = 5000,
    margin: float = 1.0,
    test_size: int = 40,
    device: str = "cuda",
    **kwargs,
) -> Dict[str, Any]:
    """Returns default dataset generation parameters.

    Args:
        n: Number of samples
        d: Number of dimensions
        margin: Margin for separability
        test_size: Number of test samples
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Additional parameters

    Returns:
        Dictionary with dataset parameters
    """
    params = {
        "n": n,
        "d": d,
        "margin": margin,
        "test_size": test_size,
        "device": device,
    }
    params.update(kwargs)
    return params


def default_metrics(w_star, loss_fn, **kwargs):
    """Returns default metrics factory.

    Args:
        w_star: Optimal weight vector for linear models
        loss_fn: Loss function to use
        **kwargs: Additional parameters

    Returns:
        Metrics factory function
    """

    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: loss_fn,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
                Metric.WeightNorm: get_weight_norm,
                Metric.GradNorm: get_weight_norm,  # Function not used, optimizer provides grad_norm
                Metric.UpdateNorm: compute_update_norm,  # Function not used, optimizer provides update_norm
                Metric.GradLossRatio: loss_fn,  # Function not used, computed from grad_norm/loss - DISABLED
            },
            w_star=w_star,
        )

    return metrics_factory


# ============================================================================
# Model type
# ============================================================================


def default_model_linear(input_dim: int, device: str = "cuda", **kwargs):
    """Returns default linear model factory.

    Args:
        input_dim: Input dimension
        device: Device to use
        **kwargs: Additional parameters

    Returns:
        Model factory function
    """

    def model_factory():
        return LinearModel(input_dim, device=device)

    return model_factory


def default_model_twolayer(
    input_dim: int, hidden_dim: int = 50, device: str = "cuda", **kwargs
):
    """Returns default two-layer model factory.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer width
        device: Device to use
        **kwargs: Additional parameters

    Returns:
        Model factory function
    """

    def model_factory():
        return TwoLayerModel(input_dim, hidden_dim, device=device)

    return model_factory


# ============================================================================
# Hyperparams
# ============================================================================


def default_hyperparams_lr(**kwargs):
    """Returns default learning rate values for hyperparameter sweeps.

    Args:
        **kwargs: Additional parameters

    Returns:
        List of learning rates
    """
    return [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]


def default_hyperparams_rho(**kwargs):
    """Returns default rho values for SAM hyperparameter sweeps.

    Args:
        **kwargs: Additional parameters

    Returns:
        List of rho values
    """
    return [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]


# ============================================================================
# Optimizers
# ============================================================================


def default_optimizer_family_gd(model, loss_fn, use_manual: bool = False, **kwargs):
    """Returns default GD optimizer family configurations.

    Includes: GD, SAM, LossNGD, VecNGD, SAM_LossNGD, SAM_VecNGD

    Args:
        model: Model type (LinearModel or TwoLayerModel class)
        loss_fn: Loss function to use
        use_manual: If True, use manual implementations (for TwoLayer models)
        **kwargs: Additional parameters

    Returns:
        Tuple of (optimizer_factories, sweeps)
    """
    learning_rates = default_hyperparams_lr()
    rho_values = default_hyperparams_rho()

    if use_manual:
        # Manual implementations for TwoLayer models
        optimizer_factories = {
            Optimizer.GD: make_stateful_optimizer_factory(ManualGD, loss=loss_fn),
            Optimizer.LossNGD: make_stateful_optimizer_factory(
                ManualLossNGD, loss=loss_fn
            ),
            Optimizer.VecNGD: make_stateful_optimizer_factory(
                ManualVecNGD, loss=loss_fn
            ),
            Optimizer.SAM: make_stateful_optimizer_factory(ManualSAM, loss=loss_fn),
            Optimizer.SAM_LossNGD: make_stateful_optimizer_factory(
                ManualSAM_LossNGD, loss=loss_fn
            ),
            Optimizer.SAM_VecNGD: make_stateful_optimizer_factory(
                ManualSAM_VecNGD, loss=loss_fn
            ),
        }
    else:
        # Standard implementations for Linear models
        optimizer_factories = {
            Optimizer.GD: make_optimizer_factory(step_gd, loss=loss_fn),
            Optimizer.LossNGD: make_optimizer_factory(step_loss_ngd, loss=loss_fn),
            Optimizer.VecNGD: make_optimizer_factory(step_vec_ngd, loss=loss_fn),
            Optimizer.SAM: make_optimizer_factory(step_sam_stable, loss=loss_fn),
            Optimizer.SAM_LossNGD: make_optimizer_factory(
                step_sam_loss_ngd, loss=loss_fn
            ),
            Optimizer.SAM_VecNGD: make_optimizer_factory(
                step_sam_vec_ngd, loss=loss_fn
            ),
        }

    sweeps = {
        Optimizer.GD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.LossNGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.VecNGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_LossNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_VecNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
    }

    return optimizer_factories, sweeps


def default_optimizer_family_adaptive(model, loss_fn, **kwargs):
    """Returns default adaptive optimizer family configurations.

    Includes: Adam, AdaGrad, SAM_Adam, SAM_AdaGrad

    Args:
        model: Model type (LinearModel or TwoLayerModel class)
        loss_fn: Loss function to use
        **kwargs: Additional parameters

    Returns:
        Tuple of (optimizer_factories, sweeps)
    """
    learning_rates = default_hyperparams_lr()
    rho_values = default_hyperparams_rho()

    optimizer_factories = {
        Optimizer.Adam: make_stateful_optimizer_factory(ManualAdam, loss=loss_fn),
        Optimizer.AdaGrad: make_stateful_optimizer_factory(ManualAdaGrad, loss=loss_fn),
        Optimizer.SAM_Adam: make_stateful_optimizer_factory(
            ManualSAM_Adam, loss=loss_fn
        ),
        Optimizer.SAM_AdaGrad: make_stateful_optimizer_factory(
            ManualSAM_AdaGrad, loss=loss_fn
        ),
    }

    sweeps = {
        Optimizer.Adam: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM_Adam: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
    }

    return optimizer_factories, sweeps


# ============================================================================
# Run args
# ============================================================================


def default_deterministic_run(
    total_iters: int = 10_000,
    debug: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Returns default arguments for deterministic (full-batch) training runs.

    Args:
        total_iters: Total number of training iterations
        debug: Enable debug output
        **kwargs: Additional parameters

    Returns:
        Dictionary with run_training parameters
    """
    from engine import DatasetSplit

    params = {
        "train_split": DatasetSplit.Train,
        "total_iters": total_iters,
        "debug": debug,
    }
    params.update(kwargs)
    return params


def default_stochastic_run(
    num_epochs: int = 10_000,
    batch_size: int = 32,
    drop_last: bool = True,
    debug: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Returns default arguments for stochastic (mini-batch) training runs.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for mini-batch training
        drop_last: Whether to drop the last incomplete batch
        debug: Enable debug output
        **kwargs: Additional parameters

    Returns:
        Dictionary with run_training parameters
    """
    from engine import DatasetSplit

    params = {
        "train_split": DatasetSplit.Train,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "debug": debug,
    }
    params.update(kwargs)
    return params


# ============================================================================
# Plot args
# ============================================================================


def default_plot_options(
    experiment_name: str,
    save_separate: bool = False,
    save_aggregated: bool = False,
    save_combined: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Returns default plotting options.

    Args:
        experiment_name: Name for the experiment (used in output path) - REQUIRED
        save_separate: Save separate plots for each metric
        save_aggregated: Save aggregated plots (deprecated)
        save_combined: Save combined plots (deprecated)
        **kwargs: Additional parameters

    Returns:
        Dictionary with plot_all parameters
    """
    params = {
        "experiment_name": experiment_name,
        "save_separate": save_separate,
        "save_aggregated": save_aggregated,
        "save_combined": save_combined,
    }
    params.update(kwargs)
    return params
