"""
Color management system for optimizer visualization.

Provides perceptually uniform color mapping for learning rates and rho values
with support for different visualization strategies (HSLuv, Paired).

Main classes:
    ColorStrategy: Abstract base for color generation strategies
    HUSLColorStrategy: Perceptually uniform colors using HSLuv
    PairedColorStrategy: Consistent colors for hyperparameter grids
    PairedOptimizerColorStrategy: Paired colors for optimizer families
    SequentialLRColorStrategy: Sequential colors for learning rates
    ColorManagerFactory: Factory for creating color managers

Usage:
    colors = ColorManagerFactory.create_husl_manager([0.01, 0.1], [0.0, 0.05])
    color = colors.color_config(lr, rho)  # Get RGBA color
    bright_color = colors.color_lr(lr)    # Get RGB color for legends
    legend_dict = colors.legend_colors()  # Get all legend colors
"""

from abc import ABC, abstractmethod
from typing import (
    List,
    Tuple,
    Dict,
    Callable,
    Optional,
    Any,
    Union,
    Protocol,
    runtime_checkable,
)
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

try:
    import hsluv
except ImportError:
    hsluv = None


class ColorFunction(Protocol):
    """Protocol for color function objects."""

    def color_lr(self, lr: Union[float, str]) -> Tuple[float, float, float]:
        """Get RGB color for learning rate or optimizer name."""
        ...

    def color_rho(self, rho: float) -> float:
        """Get opacity for rho value."""
        ...

    def color_config(
        self, lr: Union[float, str], rho: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """Get RGBA color for config."""
        ...

    def legend_colors(
        self,
    ) -> Dict[Union[float, str], Tuple[float, float, float, float]]:
        """Get bright colors for legends."""
        ...

    @property
    def all_lrs(self) -> List[Union[float, str]]:
        """All learning rates or optimizer names."""
        ...

    @property
    def all_rhos(self) -> List[float]:
        """All rho values."""
        ...

    def color_rho(self, rho: float) -> float:
        """Get opacity for rho value."""
        ...

    def color_config(
        self, lr: Union[float, str], rho: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """Get RGBA color for config."""
        ...

    def legend_colors(
        self,
    ) -> Dict[Union[float, str], Tuple[float, float, float, float]]:
        """Get bright colors for legends."""
        ...

    @property
    def all_lrs(self) -> List[Union[float, str]]:
        """All learning rates or optimizer names."""
        ...

    @property
    def all_rhos(self) -> List[float]:
        """All rho values."""
        ...


class ColorStrategy(ABC):
    """Abstract base class for color generation strategies.

    Different strategies implement different visual mappings:
    - HUSLColorStrategy: Perceptually uniform HSLuv colors with rho opacity
    - PairedColorStrategy: Consistent LR colors across rho facets
    - PairedOptimizerColorStrategy: Paired colors for optimizer families
    - SequentialLRColorStrategy: Sequential colors for learning rates
    """

    @abstractmethod
    def create_color_function(
        self, all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create a color function closure with methods.

        Returns:
            ColorFunction object with methods:
            - color_lr(lr): Get RGB color for learning rate
            - color_rho(rho): Get opacity for rho value
            - color_config(lr, rho): Get RGBA color for config
            - legend_colors(): Get bright colors for legends
        """
        pass


class HUSLColorStrategy(ColorStrategy):
    """HSLuv-based color strategy for optimizer-based coloring.

    Generates colors using the HSLuv color space where each optimizer gets
    its own base hue, and learning rates vary the saturation and lightness:
    - Higher LR: Brighter, more saturated colors
    - Lower LR: More muted, pastel colors

    Visual mapping:
    - Optimizer → Base hue (perceptually uniform distribution)
    - Learning rate → Saturation and lightness variation
    - Rho → Opacity (0.3 to 0.9 range)
    - Legends → Bright colors (opacity = 1.0)
    """

    def __init__(self, base_saturation: float = 70.0, base_lightness: float = 60.0):
        """Initialize HSLuv strategy.

        Args:
            base_saturation: Base saturation in HSLuv space (0-100)
            base_lightness: Base lightness in HSLuv space (0-100)

        Raises:
            ImportError: If hsluv package is not available
        """
        if hsluv is None:
            raise ImportError("hsluv package is required for HUSLColorStrategy")
        self.base_saturation = base_saturation
        self.base_lightness = base_lightness

    def _generate_optimizer_hues(self, n_optimizers: int) -> List[float]:
        """Generate n perceptually uniform hues for optimizers.

        Args:
            n_optimizers: Number of optimizers

        Returns:
            List of hue values in [0, 360] range
        """
        hues = []
        for i in range(n_optimizers):
            # Distribute hues evenly around the color wheel
            hue = (i * 360.0 / n_optimizers) % 360.0
            hues.append(hue)
        return hues

    def create_color_function(
        self, all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create closure with optimizer-based HSLuv colors.

        Args:
            all_lrs: List of all learning rate values (treated as optimizer indices)
            all_rhos: List of all rho values

        Returns:
            ColorFunction object with color mapping methods
        """
        # Treat all_lrs as optimizer indices for this strategy
        n_optimizers = len(all_lrs)
        optimizer_hues = self._generate_optimizer_hues(n_optimizers)

        # Sort learning rates to map them to brightness/saturation levels
        if all_lrs and isinstance(all_lrs[0], (int, float)):
            sorted_lrs = sorted(all_lrs)  # type: ignore
        else:
            sorted_lrs = all_lrs  # type: ignore
        n_lrs = len(sorted_lrs)

        # Pre-compute opacity mapping for rho values
        n_rhos = len(all_rhos)
        rho_to_opacity = {}
        for i, rho in enumerate(all_rhos):
            # Map rho index to opacity: 0.3 (min) to 0.9 (max)
            opacity = 0.3 + 0.6 * (i / max(1, n_rhos - 1))
            rho_to_opacity[rho] = opacity

        def color_optimizer(
            optimizer_idx: int, lr: Union[float, str]
        ) -> Tuple[float, float, float]:
            """Get color for optimizer with learning rate variation.

            Args:
                optimizer_idx: Index of the optimizer (0, 1, 2, ...)
                lr: Learning rate value

            Returns:
                RGB tuple normalized to [0,1]
            """
            # Get base hue for this optimizer
            base_hue = optimizer_hues[optimizer_idx]

            # Map learning rate to saturation and lightness
            # Higher LR = more saturated and brighter
            # Lower LR = less saturated and more pastel
            if isinstance(lr, (int, float)) and lr in sorted_lrs:
                lr_rank = sorted_lrs.index(lr)  # type: ignore
                lr_normalized = lr_rank / max(1, n_lrs - 1)

                # Saturation: 40% (lowest LR) to 90% (highest LR)
                saturation = 40.0 + 50.0 * lr_normalized

                # Lightness: 75% (lowest LR, more pastel) to 55% (highest LR, more vibrant)
                lightness = 75.0 - 20.0 * lr_normalized
            else:
                # Default values for non-numeric LRs
                saturation = 65.0
                lightness = 60.0

            if hsluv is None:
                # Fallback to simple HSV conversion if hsluv not available
                import colorsys

                rgb = colorsys.hsv_to_rgb(
                    base_hue / 360.0, saturation / 100.0, lightness / 100.0
                )
            else:
                rgb = hsluv.hsluv_to_rgb((base_hue, saturation, lightness))
            return (rgb[0], rgb[1], rgb[2])

        def color_lr(lr: Union[float, str]) -> Tuple[float, float, float]:
            """Get bright learning rate color (for legends) - uses highest LR variation.

            Args:
                lr: Learning rate value (treated as optimizer index)

            Returns:
                RGB tuple normalized to [0,1]
            """
            # For legend, use the brightest version (highest LR equivalent)
            optimizer_idx = all_lrs.index(lr)
            base_hue = optimizer_hues[optimizer_idx]
            if hsluv is None:
                # Fallback to simple HSV conversion if hsluv not available
                import colorsys

                rgb = colorsys.hsv_to_rgb(base_hue / 360.0, 0.9, 0.55)
            else:
                rgb = hsluv.hsluv_to_rgb((base_hue, 90.0, 55.0))
            return (rgb[0], rgb[1], rgb[2])

        def color_rho(rho: float) -> float:
            """Get opacity for rho value.

            Args:
                rho: Rho value

            Returns:
                Opacity value in [0.3, 0.9] range
            """
            return rho_to_opacity[rho]

        def color_config(
            lr: Union[float, str], rho: Optional[float] = None
        ) -> Tuple[float, float, float, float]:
            """Get color for specific config (optimizer index + rho).

            Args:
                lr: Learning rate value (treated as optimizer index)
                rho: Optional rho value (uses full opacity if None or 0.0)

            Returns:
                RGBA tuple with opacity based on rho
            """
            optimizer_idx = all_lrs.index(lr)
            base_rgb = color_optimizer(optimizer_idx, lr)
            if rho is None or rho == 0.0 or len(all_rhos) <= 1:
                return (*base_rgb, 0.9)  # Full opacity for base optimizers (rho=0.0)
            return (*base_rgb, rho_to_opacity[rho])

        def legend_colors() -> Dict[
            Union[float, str], Tuple[float, float, float, float]
        ]:
            """Get bright colors for all optimizers (for legends).

            Returns:
                Dict mapping optimizer index to bright RGBA colors (opacity = 1.0)
            """
            legend_dict = {}
            for i, lr in enumerate(all_lrs):
                base_hue = optimizer_hues[i]
                if hsluv is None:
                    import colorsys

                    rgb = colorsys.hsv_to_rgb(base_hue / 360.0, 0.9, 0.55)
                else:
                    rgb = hsluv.hsluv_to_rgb((base_hue, 90.0, 55.0))
                legend_dict[lr] = (rgb[0], rgb[1], rgb[2], 1.0)
            return legend_dict

        # Return proper class instance with all color methods
        class ColorFunctionImpl:
            """Color function closure with optimizer-based HSLuv color mapping."""

            def __init__(self):
                self.color_lr = color_lr
                self.color_rho = color_rho
                self.color_config = color_config
                self.legend_colors = legend_colors
                self._all_lrs = all_lrs
                self._all_rhos = all_rhos

            @property
            def all_lrs(self) -> List[Union[float, str]]:
                return self._all_lrs

            @property
            def all_rhos(self) -> List[float]:
                return self._all_rhos

        return ColorFunctionImpl()


class PairedColorStrategy(ColorStrategy):
    """Paired color strategy for hyperparameter grids with consistent LR colors.

    Uses matplotlib's Paired colormap to provide consistent colors across
    rho facets in hyperparameter grids. Each learning rate gets a consistent
    color regardless of rho value.

    Visual mapping:
    - Learning rate → Consistent color from Paired colormap
    - Rho → Ignored (always uses full opacity)
    - Legends → Bright colors (opacity = 1.0)
    """

    def __init__(self, colormap_name: str = "Paired"):
        """Initialize paired color strategy.

        Args:
            colormap_name: Name of matplotlib colormap to use
        """
        self.colormap_name = colormap_name

    def create_color_function(
        self, all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create closure with paired colors (no rho opacity variation).

        Args:
            all_lrs: List of all learning rate values
            all_rhos: List of all rho values (ignored in this strategy)

        Returns:
            ColorFunction object with paired color mapping
        """
        cmap = plt.get_cmap(self.colormap_name)

        # Pre-compute colors for each learning rate
        lr_to_rgb = {}
        for i, lr in enumerate(all_lrs):
            # Use even indices: 0, 2, 4, 6, 8, 10 for consistency
            color_idx = (i * 2) % 12
            lr_to_rgb[lr] = cmap(color_idx / 12.0)

        def color_lr(lr: Union[float, str]) -> Tuple[float, float, float]:
            """Get learning rate color.

            Args:
                lr: Learning rate value

            Returns:
                RGB tuple from paired colormap
            """
            color = lr_to_rgb[lr]
            return (color[0], color[1], color[2])

        def color_rho(rho: float) -> float:
            """Paired strategy ignores rho (always full opacity).

            Args:
                rho: Rho value (ignored)

            Returns:
                Always returns 0.9
            """
            return 0.9

        def color_config(
            lr: Union[float, str], rho: Optional[float] = None
        ) -> Tuple[float, float, float, float]:
            """Get color (ignores rho, always full opacity).

            Args:
                lr: Learning rate value
                rho: Optional rho value (ignored)

            Returns:
                RGBA tuple with full opacity
            """
            base_rgb = lr_to_rgb[lr]
            return (base_rgb[0], base_rgb[1], base_rgb[2], 0.9)

        def legend_colors() -> Dict[
            Union[float, str], Tuple[float, float, float, float]
        ]:
            """Get bright colors for legends.

            Returns:
                Dict mapping LR to bright RGBA colors (opacity = 1.0)
            """
            return {
                lr: (lr_to_rgb[lr][0], lr_to_rgb[lr][1], lr_to_rgb[lr][2], 1.0)
                for lr in all_lrs
            }

        # Return proper class instance with all color methods
        class ColorFunctionImpl:
            """Color function closure with paired color mapping."""

            def __init__(self):
                self.color_lr = color_lr
                self.color_rho = color_rho
                self.color_config = color_config
                self.legend_colors = legend_colors
                self._all_lrs = all_lrs
                self._all_rhos = all_rhos

            @property
            def all_lrs(self) -> List[Union[float, str]]:
                return self._all_lrs

            @property
            def all_rhos(self) -> List[float]:
                return self._all_rhos

        return ColorFunctionImpl()


class PairedOptimizerColorStrategy(ColorStrategy):
    """Paired optimizer color strategy for hyperparameter grids.

    Creates paired colors where base optimizers (GD, VecNGD, LossNGD) get pastel
    shades and their SAM variants (SAM_GD, SAM_VecNGD, SAM_LossNGD) get
    vibrant shades of the same base hue.

    Visual mapping:
    - Optimizer family → Base hue (GD=red, VecNGD=blue, LossNGD=green, etc.)
    - SAM variant → Vibrant shade, Base variant → Pastel shade
    - Rho → Opacity (0.3 to 0.9 range)
    - Legends → Bright colors (opacity = 1.0)
    """

    def __init__(self):
        """Initialize paired optimizer strategy."""
        pass

    def _get_optimizer_family(self, optimizer_name: str) -> str:
        """Extract base optimizer family name.

        Args:
            optimizer_name: Full optimizer name (e.g., 'SAM_GD', 'VecNGD', 'SAM')

        Returns:
            Base family name (e.g., 'GD', 'VecNGD')
        """
        # Special case: "SAM" (without suffix) is the SAM variant of GD
        if optimizer_name == "SAM":
            return "GD"
        # Remove SAM_ prefix if present
        if optimizer_name.startswith("SAM_"):
            return optimizer_name[4:]  # Remove 'SAM_' prefix
        return optimizer_name

    def _is_sam_variant(self, optimizer_name: str) -> bool:
        """Check if optimizer is a SAM variant.

        Args:
            optimizer_name: Optimizer name

        Returns:
            True if SAM variant, False otherwise
        """
        return optimizer_name == "SAM" or optimizer_name.startswith("SAM_")

    def _generate_family_hues(self, families: List[str]) -> Dict[str, float]:
        """Generate base hues for each optimizer family.

        Args:
            families: List of unique optimizer family names

        Returns:
            Dict mapping family name to hue value
        """
        n_families = len(families)
        family_to_hue = {}

        for i, family in enumerate(families):
            # Distribute hues evenly around the color wheel
            hue = (i * 360.0 / n_families) % 360.0
            family_to_hue[family] = hue

        return family_to_hue

    def create_color_function(
        self, all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create closure with paired optimizer colors.

        Args:
            all_lrs: List of optimizer names
            all_rhos: List of all rho values

        Returns:
            ColorFunction object with paired optimizer color mapping
        """
        # Extract optimizer names
        optimizer_names = [str(lr) for lr in all_lrs]

        # Group optimizers by family
        families = sorted(
            set(self._get_optimizer_family(name) for name in optimizer_names)
        )
        family_to_hue = self._generate_family_hues(families)

        # Pre-compute opacity mapping for rho values
        n_rhos = len(all_rhos)
        rho_to_opacity = {}
        for i, rho in enumerate(all_rhos):
            # Map rho index to opacity: 0.3 (min) to 0.9 (max)
            opacity = 0.3 + 0.6 * (i / max(1, n_rhos - 1))
            rho_to_opacity[rho] = opacity

        def color_optimizer(optimizer_name: str) -> Tuple[float, float, float]:
            """Get color for optimizer with pastel/vibrant pairing.

            Args:
                optimizer_name: Full optimizer name

            Returns:
                RGB tuple normalized to [0,1]
            """
            family = self._get_optimizer_family(optimizer_name)
            is_sam = self._is_sam_variant(optimizer_name)
            base_hue = family_to_hue[family]

            if is_sam:
                # Vibrant shade for SAM variants: maximum saturation, low lightness
                hue = base_hue
                saturation = 100.0
                lightness = 40.0
            else:
                # Pastel paired color for base variants: shift hue slightly for distinction
                # This creates a harmonious but clearly different color
                hue = (base_hue + 25.0) % 360.0  # Shift hue by 25 degrees
                saturation = 50.0  # Medium saturation for pastel effect
                lightness = 75.0   # Higher lightness for pastel effect

            if hsluv is None:
                # Fallback to simple HSV conversion
                import colorsys

                rgb = colorsys.hsv_to_rgb(
                    hue / 360.0, saturation / 100.0, lightness / 100.0
                )
            else:
                rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

            return (rgb[0], rgb[1], rgb[2])

        def color_lr(optimizer_name: Union[float, str]) -> Tuple[float, float, float]:
            """Get bright optimizer color for legends.

            Args:
                optimizer_name: Optimizer name

            Returns:
                RGB tuple normalized to [0,1]
            """
            name_str = str(optimizer_name)
            family = self._get_optimizer_family(name_str)
            base_hue = family_to_hue[family]

            # Always use vibrant colors for legends
            if hsluv is None:
                import colorsys

                rgb = colorsys.hsv_to_rgb(base_hue / 360.0, 0.9, 0.55)
            else:
                rgb = hsluv.hsluv_to_rgb((base_hue, 90.0, 55.0))

            return (rgb[0], rgb[1], rgb[2])

        def color_rho(rho: float) -> float:
            """Get opacity for rho value.

            Args:
                rho: Rho value

            Returns:
                Opacity value in [0.3, 0.9] range
            """
            return rho_to_opacity[rho]

        def color_config(
            optimizer_name: Union[float, str], rho: Optional[float] = None
        ) -> Tuple[float, float, float, float]:
            """Get color for specific optimizer config with rho opacity.

            Args:
                optimizer_name: Optimizer name
                rho: Optional rho value

            Returns:
                RGBA tuple with opacity based on rho
            """
            name_str = str(optimizer_name)
            base_rgb = color_optimizer(name_str)
            if rho is None or rho == 0.0 or len(all_rhos) <= 1:
                return (*base_rgb, 0.9)  # Full opacity for base optimizers
            return (*base_rgb, rho_to_opacity[rho])

        def legend_colors() -> Dict[
            Union[float, str], Tuple[float, float, float, float]
        ]:
            """Get colors for all optimizers (for legends).
            Uses the same paired color logic: base optimizers get shifted pastel colors,
            SAM variants get vibrant colors.

            Returns:
                Dict mapping optimizer name to RGBA colors
            """
            legend_dict = {}
            for optimizer_name in optimizer_names:
                family = self._get_optimizer_family(optimizer_name)
                is_sam = self._is_sam_variant(optimizer_name)
                base_hue = family_to_hue[family]

                if is_sam:
                    # Vibrant colors for SAM variants
                    hue = base_hue
                    saturation = 100.0
                    lightness = 40.0
                else:
                    # Pastel paired colors for base variants (with hue shift)
                    hue = (base_hue + 25.0) % 360.0
                    saturation = 50.0
                    lightness = 75.0

                if hsluv is None:
                    import colorsys
                    rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, lightness / 100.0)
                else:
                    rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

                legend_dict[optimizer_name] = (rgb[0], rgb[1], rgb[2], 1.0)

            return legend_dict

        # Return proper class instance
        class ColorFunctionImpl:
            """Color function closure with paired optimizer color mapping."""

            def __init__(self):
                self.color_lr = color_lr
                self.color_rho = color_rho
                self.color_config = color_config
                self.legend_colors = legend_colors
                self._all_lrs = all_lrs
                self._all_rhos = all_rhos

            @property
            def all_lrs(self) -> List[Union[float, str]]:
                return self._all_lrs

            @property
            def all_rhos(self) -> List[float]:
                return self._all_rhos

        return ColorFunctionImpl()


class SequentialLRColorStrategy(ColorStrategy):
    """Sequential learning rate color strategy for faceted plots.

    Creates colors where learning rate determines both hue (position on color wheel)
    and saturation/lightness (pastel to vibrant). Lower learning rates are more
    pastel/muted, higher learning rates are more vibrant.

    Visual mapping:
    - Learning rate → Hue (color wheel position) + Saturation/Lightness (pastel→vibrant)
    - Lower LR → Pastel/muted colors, Higher LR → Vibrant colors
    - Rho → Opacity (0.3 to 0.9 range)
    - Legends → Bright colors (opacity = 1.0)
    """

    def __init__(self):
        """Initialize sequential LR strategy."""
        pass

    def create_color_function(
        self, all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create closure with sequential LR colors.

        Args:
            all_lrs: List of all learning rate values
            all_rhos: List of all rho values

        Returns:
            ColorFunction object with sequential LR color mapping
        """
        # Sort learning rates for consistent mapping
        numeric_lrs = [float(lr) for lr in all_lrs if isinstance(lr, (int, float))]
        sorted_lrs = sorted(numeric_lrs)
        n_lrs = len(sorted_lrs)

        # Pre-compute opacity mapping for rho values
        n_rhos = len(all_rhos)
        rho_to_opacity = {}
        for i, rho in enumerate(all_rhos):
            # Map rho index to opacity: 0.3 (min) to 0.9 (max)
            opacity = 0.3 + 0.6 * (i / max(1, n_rhos - 1))
            rho_to_opacity[rho] = opacity

        def color_lr(lr: Union[float, str]) -> Tuple[float, float, float]:
            """Get color for learning rate with sequential variation.

            Args:
                lr: Learning rate value

            Returns:
                RGB tuple normalized to [0,1]
            """
            lr_float = float(lr)
            # Find rank of this learning rate
            if lr_float in sorted_lrs:
                lr_rank = sorted_lrs.index(lr_float)
                lr_normalized = lr_rank / max(1, n_lrs - 1)
            else:
                # Default to middle if not in list
                lr_normalized = 0.5

            # Map to hue position around color wheel
            hue = (lr_normalized * 330.0) % 360.0  # Use 330° to avoid red-red overlap

            # Map LR to saturation and lightness (pastel → vibrant)
            # Lower LR = more pastel (lower saturation, higher lightness)
            # Higher LR = more vibrant (higher saturation, lower lightness)
            saturation = 30.0 + 60.0 * lr_normalized  # 30% to 90%
            lightness = 80.0 - 25.0 * lr_normalized  # 80% to 55%

            if hsluv is None:
                # Fallback to simple HSV conversion
                import colorsys

                rgb = colorsys.hsv_to_rgb(
                    hue / 360.0, saturation / 100.0, lightness / 100.0
                )
            else:
                rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

            return (rgb[0], rgb[1], rgb[2])

        def color_rho(rho: float) -> float:
            """Get opacity for rho value.

            Args:
                rho: Rho value

            Returns:
                Opacity value in [0.3, 0.9] range
            """
            return rho_to_opacity[rho]

        def color_config(
            lr: Union[float, str], rho: Optional[float] = None
        ) -> Tuple[float, float, float, float]:
            """Get color for specific LR config with rho opacity.

            Args:
                lr: Learning rate value
                rho: Optional rho value

            Returns:
                RGBA tuple with opacity based on rho
            """
            base_rgb = color_lr(lr)
            if rho is None or rho == 0.0 or len(all_rhos) <= 1:
                return (*base_rgb, 0.9)  # Full opacity for non-SAM
            return (*base_rgb, rho_to_opacity[rho])

        def legend_colors() -> Dict[
            Union[float, str], Tuple[float, float, float, float]
        ]:
            """Get bright colors for all learning rates (for legends).

            Returns:
                Dict mapping LR to bright RGBA colors
            """
            legend_dict = {}
            for lr in sorted_lrs:
                lr_rank = sorted_lrs.index(lr)
                lr_normalized = lr_rank / max(1, n_lrs - 1)

                # Use vibrant colors for legends
                hue = (lr_normalized * 330.0) % 360.0

                if hsluv is None:
                    import colorsys

                    rgb = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 0.55)
                else:
                    rgb = hsluv.hsluv_to_rgb((hue, 90.0, 55.0))

                legend_dict[lr] = (rgb[0], rgb[1], rgb[2], 1.0)

            return legend_dict

        # Return proper class instance
        class ColorFunctionImpl:
            """Color function closure with sequential LR color mapping."""

            def __init__(self):
                self.color_lr = color_lr
                self.color_rho = color_rho
                self.color_config = color_config
                self.legend_colors = legend_colors
                self._all_lrs = all_lrs
                self._all_rhos = all_rhos

            @property
            def all_lrs(self) -> List[Union[float, str]]:
                return self._all_lrs

            @property
            def all_rhos(self) -> List[float]:
                return self._all_rhos

        return ColorFunctionImpl()


class ColorManagerFactory:
    """Factory for creating color managers.

    Provides convenient methods to create color managers with different
    strategies based on the visualization needs.

    Usage:
        # HSLuv colors for general plots
        colors = ColorManagerFactory.create_husl_manager([0.01, 0.1], [0.0, 0.05])

        # Paired colors for hyperparameter grids
        colors = ColorManagerFactory.create_paired_manager([0.01, 0.1], [0.0, 0.05])

        # Paired optimizer colors for hyperparam grids
        colors = ColorManagerFactory.create_paired_optimizer_manager(['GD', 'SAM_GD'], [0.0, 0.05])

        # Sequential LR colors for faceted plots
        colors = ColorManagerFactory.create_sequential_lr_manager([0.001, 0.01, 0.1], [0.0, 0.05])
    """

    @staticmethod
    def create_husl_manager(
        all_lrs: List[Union[float, str]], all_rhos: List[float]
    ) -> ColorFunction:
        """Create HSLuv color manager with perceptually uniform colors.

        Args:
            all_lrs: List of all learning rate values
            all_rhos: List of all rho values

        Returns:
            ColorFunction with HSLuv-based color mapping

        Raises:
            ImportError: If hsluv package is not available
        """
        strategy = HUSLColorStrategy()
        return strategy.create_color_function(all_lrs, all_rhos)

    @staticmethod
    def create_paired_manager(
        all_lrs: List[Union[float, str]],
        all_rhos: List[float],
        colormap_name: str = "Paired",
    ) -> ColorFunction:
        """Create paired color manager for hyperparameter grids.

        Args:
            all_lrs: List of all learning rate values
            all_rhos: List of all rho values (ignored in paired strategy)
            colormap_name: Name of matplotlib colormap to use

        Returns:
            ColorFunction with paired color mapping
        """
        strategy = PairedColorStrategy(colormap_name)
        return strategy.create_color_function(all_lrs, all_rhos)

    @staticmethod
    def create_paired_optimizer_manager(
        optimizer_names: List[str], all_rhos: List[float]
    ) -> ColorFunction:
        """Create paired optimizer color manager for hyperparameter grids.

        Args:
            optimizer_names: List of optimizer names
            all_rhos: List of all rho values

        Returns:
            ColorFunction with paired optimizer color mapping
        """
        strategy = PairedOptimizerColorStrategy()
        return strategy.create_color_function(optimizer_names, all_rhos)

    @staticmethod
    def create_sequential_lr_manager(
        all_lrs: List[float], all_rhos: List[float]
    ) -> ColorFunction:
        """Create sequential LR color manager for faceted plots.

        Args:
            all_lrs: List of all learning rate values
            all_rhos: List of all rho values

        Returns:
            ColorFunction with sequential LR color mapping
        """
        strategy = SequentialLRColorStrategy()
        return strategy.create_color_function(all_lrs, all_rhos)

    @staticmethod
    def create_color_manager(use_paired: bool = False, **kwargs) -> ColorFunction:
        """Factory method for creating color managers.

        Args:
            use_paired: If True, create paired manager; otherwise HSLuv
            **kwargs: Additional arguments passed to specific manager creation

        Returns:
            ColorFunction with appropriate color mapping strategy
        """
        if use_paired:
            return ColorManagerFactory.create_paired_manager(**kwargs)
        else:
            return ColorManagerFactory.create_husl_manager(**kwargs)
