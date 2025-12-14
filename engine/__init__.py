"""Unified optimization experiment framework"""

# Core types
from .types import (
    DatasetSplit,
    Metric,
    Optimizer,
    MetricKey,
    OptimizerConfig,
    Hyperparam,
)

# Data management
from .data import (
    make_soudry_dataset,
    split_train_test,
)

# Metrics
from .metrics import (
    MetricsCollector,
    get_empirical_max_margin,
    get_angle,
    get_direction_distance,
    get_error_rate,
)

# Loss functions
from .losses import (
    Loss,
    ExponentialLoss,
    LogisticLoss,
)

# History
from .history import TrainingHistory

# Models
from .models import LinearModel, TwoLayerModel

# Training
from .trainer import run_training

# Sweeps
from .sweeps import expand_sweep_grid

__all__ = [
    # Types
    "DatasetSplit",
    "Metric",
    "Optimizer",
    "MetricKey",
    "OptimizerConfig",
    "Hyperparam",
    # Data
    "make_soudry_dataset",
    "split_train_test",
    # Metrics
    "MetricsCollector",
    "get_empirical_max_margin",
    "get_angle",
    "get_direction_distance",
    "get_error_rate",
    # Loss functions
    "Loss",
    "ExponentialLoss",
    "LogisticLoss",
    # History
    "TrainingHistory",
    # Models
    "LinearModel",
    "TwoLayerModel",
    # Training
    "run_training",
    # Sweeps
    "expand_sweep_grid",
]
