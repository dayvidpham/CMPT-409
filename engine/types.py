from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional
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

class Optimizer(Enum):
    GD = auto()
    SAM = auto()
    NGD = auto()
    SAM_NGD = auto()
    Adam = auto()
    AdaGrad = auto()
    SAM_Adam = auto()
    SAM_AdaGrad = auto()


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
            raise ValueError(f"Metric '{self.metric.name}' is a reference metric; split must be None.")
        
        # Validate: non-reference metrics must have a split
        if not self.metric.requires_reference and self.split is None:
            raise ValueError(f"Metric '{self.metric.name}' requires a specific dataset split.")

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
