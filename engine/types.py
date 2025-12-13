from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Tuple
import torch

# Everything is Torch now
ArrayLike = torch.Tensor

# Import strategy components
from .strategies import PlotStrategy, LogLogStrategy, PercentageStrategy


class DatasetSplit(Enum):
    Train = auto()
    Val = auto()
    Test = auto()


class Metric(Enum):
    """
    Metric definitions composed with their plotting strategies.
    Format: Name = (unique_slug, strategy_instance)
    """

    _strategy: PlotStrategy  # Class-level annotation for type checker

    Loss = ("loss", LogLogStrategy())
    Error = ("error", PercentageStrategy())
    Angle = ("angle", LogLogStrategy())
    Distance = ("dist", LogLogStrategy())

    def __new__(cls, value: str, strategy: PlotStrategy):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._strategy = strategy
        return obj

    @property
    def requires_reference(self) -> bool:
        """Metrics like Angle/Distance need w_star."""
        return self in (Metric.Angle, Metric.Distance)

    @property
    def strategy(self) -> PlotStrategy:
        """Accessor for the default plotting strategy."""
        return self._strategy

    @property
    def display_name(self) -> str:
        """Formatted name for display in titles, using strategy's suffix."""
        return f"{self.name}{self.strategy.display_name_suffix}"


class Optimizer(Enum):
    GD = auto()
    SAM = auto()
    NGD = auto()
    SAM_NGD = auto()
    Adam = auto()
    AdaGrad = auto()
    SAM_Adam = auto()
    SAM_AdaGrad = auto()


class Hyperparam(Enum):
    """Names for optimizer hyperparameters that can be swept."""

    Rho = "rho"
    LearningRate = "lr"


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Immutable, hashable optimizer configuration for training runs.

    Used as dictionary keys for results storage and provides self-documenting
    names for plots and logs. Hyperparameters use Hyperparam enum keys.
    """

    optimizer: Optimizer
    hyperparams: Tuple[Tuple[Hyperparam, float], ...] = ()

    @property
    def name(self) -> str:
        """Human-readable name for plots/logs (e.g., 'SAM(lr=0.01,rho=0.1)')."""
        if not self.hyperparams:
            return self.optimizer.name
        params = ",".join(f"{hp.value}={v}" for hp, v in self.hyperparams)
        return f"{self.optimizer.name}({params})"

    @property
    def dir_name(self) -> str:
        """Directory-safe name (e.g., 'SAM--lr=0.01_rho=0.1')."""
        if not self.hyperparams:
            return self.optimizer.name
        params = "_".join(f"{hp.value}={v}" for hp, v in self.hyperparams)
        return f"{self.optimizer.name}--{params}"

    @property
    def learning_rate(self) -> float:
        """Get learning rate (required hyperparameter)."""
        lr = self.get(Hyperparam.LearningRate)
        if lr is None:
            raise ValueError(
                f"OptimizerConfig {self.optimizer.name} missing required LearningRate"
            )
        return lr

    def get(self, key: Hyperparam, default: Optional[float] = None) -> Optional[float]:
        """Get a hyperparameter value by Hyperparam enum."""
        for hp, v in self.hyperparams:
            if hp == key:
                return v
        return default

    def get_step_fn_kwargs(self) -> dict:
        """Get hyperparams as dict with string keys for step function kwargs (excludes lr)."""
        return {
            hp.value: v for hp, v in self.hyperparams if hp != Hyperparam.LearningRate
        }

    @classmethod
    def with_params(cls, optimizer: Optimizer, **kwargs: float) -> "OptimizerConfig":
        """Convenience constructor - converts string keys to Hyperparam enums."""
        hyperparams = tuple((Hyperparam(k), v) for k, v in kwargs.items())
        return cls(optimizer=optimizer, hyperparams=hyperparams)


@dataclass(frozen=True)
class MetricKey:
    """
    Composite key for metric storage.

    Uses explicit __eq__ and __hash__ based on string representation
    to ensure that two keys with the same metric name/split are treated
    as identical, even if the underlying Metric objects are different instances.
    """

    metric: Metric
    split: Optional[DatasetSplit] = None

    def __post_init__(self):
        """
        Validates the metric/split combination immediately upon creation.
        """
        # Validate: reference metrics must have split=None
        if self.metric.requires_reference and self.split is not None:
            raise ValueError(
                f"Metric '{self.metric.name}' is a reference metric; split must be None."
            )

        # Validate: non-reference metrics must have a split
        if not self.metric.requires_reference and self.split is None:
            raise ValueError(
                f"Metric '{self.metric.name}' requires a specific dataset split."
            )

    def __str__(self) -> str:
        """
        Returns the canonical string representation.
        e.g., 'accuracy_test' or 'bleu'
        """
        if self.split is None:
            return self.metric.name.lower()
        return f"{self.metric.name.lower()}_{self.split.name.lower()}"

    def __eq__(self, other: Any) -> bool:
        """
        Force equality to be based on the unique string representation.
        This fixes issues where two identical metrics are different objects in memory.
        """
        if not isinstance(other, MetricKey):
            return NotImplemented
        return str(self) == str(other)

    def __hash__(self) -> int:
        """
        Force hashing to match the equality definition.
        """
        return hash(str(self))
