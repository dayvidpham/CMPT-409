"""
Flexible, extensible plotting strategy system using composition and the command pattern.

This module provides:
- AxisScale: Typed enum for matplotlib scale types
- Transform: Abstract base class for composable data transformations
- PlotContext: Dataclass bundling plot data
- PlotStrategy: Mutable strategy class with fluent pipe() interface
- Factory functions: Common presets (LogLogStrategy, PercentageStrategy)
"""

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Type for matplotlib colors (string names, hex codes, or RGBA tuples)
ColorType = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float], None
]


class AxisScale(Enum):
    """Axis scale types for matplotlib."""

    Linear = "linear"
    Log = "log"
    Symlog = "symlog"
    Logit = "logit"


class Transform(ABC):
    """Abstract base class for data transformations."""

    @abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        pass


class SafeLog(Transform):
    """Clamp to epsilon before log to prevent log(0)."""

    def __init__(self, eps: float = 1e-16):
        self.eps = eps

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(values, self.eps)


class Scale(Transform):
    """Multiply by constant (e.g., 100 for percentage)."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values * self.factor


class Clamp(Transform):
    """Clip values to a range."""

    def __init__(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.min_val, self.max_val)


@dataclass
class PlotContext:
    """Encapsulates all data required to render a single line."""

    ax: Any  # matplotlib.axes.Axes
    x: np.ndarray
    y: np.ndarray
    label: str = ""
    plot_kwargs: dict = None  # Additional kwargs for ax.plot()

    def __post_init__(self):
        if self.plot_kwargs is None:
            self.plot_kwargs = {}


class PlotStrategy:
    """
    Defines how a metric should be plotted.
    Uses composition of Transforms to modify data before rendering.
    """

    def __init__(
        self,
        transforms: Optional[List[Transform]] = None,
        x_scale: AxisScale = AxisScale.Log,
        y_scale: AxisScale = AxisScale.Linear,
        y_label_suffix: str = "",
        display_name_suffix: str = "",
        y_lim: Optional[tuple[float, float]] = None,
        **style_kwargs,
    ):
        self.transforms = transforms or []
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.y_label_suffix = y_label_suffix
        self.display_name_suffix = display_name_suffix or y_label_suffix
        self.y_lim = y_lim
        self.style_kwargs = style_kwargs

    def pipe(self, transform: Transform) -> "PlotStrategy":
        """Fluent interface to add a transform to the pipeline."""
        self.transforms.append(transform)
        return self

    def process_values(self, values: np.ndarray) -> np.ndarray:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            values = t(values)
        return values

    def configure_axis(self, ax: Any, base_label: str) -> None:
        """Apply axis configuration (scales, limits, labels)."""
        ax.set_xscale(self.x_scale.value)
        ax.set_yscale(self.y_scale.value)
        ax.set_ylabel(f"{base_label}{self.display_name_suffix}")

        if self.y_lim:
            ax.set_ylim(*self.y_lim)

        # Ensure y-axis tick labels are visible on all subplots (for sharey=True)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True)

        ax.grid(True, which="major", alpha=0.3)
        if self.x_scale == AxisScale.Log or self.y_scale == AxisScale.Log:
            ax.grid(True, which="minor", alpha=0.1)

    def plot(self, ctx: PlotContext) -> None:
        """Render the plot on the provided axes."""
        # Merge strategy defaults, context kwargs, and label
        style = {**self.style_kwargs, **ctx.plot_kwargs, "label": ctx.label}

        y_transformed = self.process_values(ctx.y)
        ctx.ax.plot(ctx.x, y_transformed, **style)


# Factory Functions (Presets)


def LogLogStrategy(eps: float = 1e-16, **kwargs) -> PlotStrategy:
    """Preset for Log-Log plots (Loss, Angle, Distance)."""
    base_transforms = [SafeLog(eps=eps)]
    user_transforms = kwargs.pop("transforms", [])
    return PlotStrategy(
        transforms=base_transforms + user_transforms,
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Log,
        **kwargs,
    )


def PercentageStrategy(**kwargs) -> PlotStrategy:
    """Preset for percentage plots with automatic y-axis scaling (Error Rate)."""
    base_transforms = [Scale(100.0)]
    user_transforms = kwargs.pop("transforms", [])
    return PlotStrategy(
        transforms=base_transforms + user_transforms,
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Linear,
        y_label_suffix=" (%)",
        display_name_suffix=" Rate (%)",
        **kwargs,
    )
