from typing import Dict, Optional, Callable
from sklearn.svm import LinearSVC
from .types import Metric, MetricKey, DatasetSplit, ArrayLike
from .models import Model
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Reference Metric Computation (w_star)
# -----------------------------------------------------------------------------


def get_empirical_max_margin(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute max-margin classifier using LinearSVC.

    Args:
        X: Input features (torch.Tensor)
        y: Labels (torch.Tensor)

    Returns:
        w_star: Normalized max-margin weight vector (torch.Tensor)
    """
    # Convert to numpy for sklearn
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    # Compute max-margin classifier
    svm = LinearSVC(C=1e6, fit_intercept=False, max_iter=100000)
    svm.fit(X_np, y_np)
    w = svm.coef_.ravel()
    w_normalized = w / np.linalg.norm(w)

    # Convert back to torch on same device as X
    return torch.tensor(w_normalized, dtype=X.dtype, device=X.device)


# -----------------------------------------------------------------------------
# Vector Metrics (w vs w_star)
# -----------------------------------------------------------------------------


def get_angle(w: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
    """
    Compute angle between two vectors in radians.

    Args:
        w: Weight vector
        w_star: Reference weight vector

    Returns:
        Angle in radians
    """
    w_flat = w.flatten()
    w_star_flat = w_star.flatten()

    # Ensure same device
    if w_star.device != w.device:
        w_star_flat = w_star_flat.to(w.device)

    # Compute angle
    n_w = torch.norm(w_flat)
    n_star = torch.norm(w_star_flat)
    dot_val = torch.dot(n_w, n_star)
    cos_angle = torch.clamp(dot_val, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    return angle


def get_direction_distance(w: torch.Tensor, w_star: torch.Tensor) -> torch.Tensor:
    """
    L2 distance between normalized directions.

    Args:
        w: Weight vector
        w_star: Reference weight vector

    Returns:
        Distance (Python float)
    """
    eps = 1e-12

    # Ensure same device
    if w_star.device != w.device:
        w_star = w_star.to(w.device)

    # Normalize both vectors
    w_norm = w / (torch.norm(w) + eps)
    w_star_norm = w_star / (torch.norm(w_star) + eps)

    # Compute distance
    diff = w_norm - w_star_norm
    distance = torch.norm(diff)

    return distance


def get_norm(w: torch.Tensor, _unused=None) -> torch.Tensor:
    """
    L2 norm of weight vector.

    Args:
        w: Weight vector
        _unused: Unused parameter for API compatibility

    Returns:
        Norm (Python float)
    """
    return torch.norm(w)


# -----------------------------------------------------------------------------
# Model Metrics (loss, error)
# -----------------------------------------------------------------------------


def get_error_rate(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute classification error rate.

    Args:
        scores: Model predictions (N,)
        y: Labels {-1, +1} (N,)

    Returns:
        Error rate in [0, 1] (Python float)
    """
    predictions = torch.sign(scores)
    errors = (predictions != y).float()
    error_rate = torch.mean(errors)
    return error_rate


# -----------------------------------------------------------------------------
# Stability Metrics (W_norm, UpdateNorm, Weight/Loss Ratio)
# -----------------------------------------------------------------------------


def get_weight_norm(model: Model) -> torch.Tensor:
    """
    Compute L2 norm of model weights as a 0-d tensor.

    Avoids CPU synchronization by returning a tensor instead of float.
    For linear models, returns ||w||. For general models, returns
    the total norm across all parameters.

    Args:
        model: Model to compute norm of

    Returns:
        0-d tensor containing the weight norm
    """
    with torch.no_grad():
        return model.effective_weight.norm()


# Placeholder signature for update_norm tracking
# This will be populated during compute_all
def compute_update_norm(w_current: torch.Tensor, w_prev: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute the norm of the weight update.

    Args:
        w_current: Current weight vector
        w_prev: Previous weight vector (None if first call)

    Returns:
        0-d tensor containing the update norm
    """
    with torch.no_grad():
        if w_prev is None:
            return torch.zeros((), device=w_current.device, dtype=w_current.dtype, requires_grad=False)
        return (w_current - w_prev).norm()


# -----------------------------------------------------------------------------
# MetricsCollector
# -----------------------------------------------------------------------------


class MetricsCollector:
    """
    Computes multiple metrics on a model without forcing CPU synchronization.
    All computations stay on device until final `.item()` call.
    """

    def __init__(
        self, metric_fns: Dict[Metric, Callable], w_star: Optional[torch.Tensor] = None
    ):
        """
        Args:
            metric_fns: Dict mapping Metric enum to computation function
            w_star: Reference solution for angle/distance metrics (optional)
        """
        self.metric_fns = metric_fns
        self.w_star = w_star
        self.w_prev = None  # Track previous weights for UpdateNorm stability metric
        self.grad_norm: Optional[torch.Tensor] = None  # Gradient norm set by optimizer
        self.train_loss: Optional[torch.Tensor] = None  # Training loss set by optimizer

    def compute_all(
        self,
        model: Model,
        datasets: Dict[DatasetSplit, tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[MetricKey, torch.Tensor]:
        """
        Compute all metrics for all dataset splits.

        Args:
            model: Trained model
            datasets: Dict mapping DatasetSplit to (X, y) tuples

        Returns:
            Dict mapping MetricKey to metric values (Python floats)
        """
        results = {}

        # Get model weights (lazily, only if needed)
        w_eff = None

        # First pass: reference metrics and dataset metrics (Loss, Error)
        for metric, metric_fn in self.metric_fns.items():
            if metric.requires_reference:
                # Reference metrics (Angle, Distance): compare w vs w_star
                if self.w_star is None:
                    continue  # Skip if no reference available

                # Lazy fetch of effective weight
                if w_eff is None:
                    w_eff = model.effective_weight

                value = metric_fn(w_eff, self.w_star)
                key = MetricKey(metric=metric, split=None)
                results[key] = value

            elif metric.requires_split:
                # Dataset metrics (Loss, Error): compute on each split
                for split, (X, y) in datasets.items():
                    # For training loss, require optimizer to provide it
                    if metric == Metric.Loss and split == DatasetSplit.Train:
                        if self.train_loss is None:
                            raise ValueError(
                                "Training loss must be set by the optimizer. "
                                "Ensure the optimizer has a reference to this MetricsCollector and sets train_loss during step()."
                            )
                        value = self.train_loss
                        # Reset for next iteration
                        self.train_loss = None
                    else:
                        # Forward pass
                        with torch.no_grad():
                            scores = model.forward(X)

                        # Compute metric
                        value = metric_fn(scores, y)
                    key = MetricKey(metric=metric, split=split)
                    results[key] = value

        # Second pass: stability metrics (may reference results from first pass)
        for metric, metric_fn in self.metric_fns.items():
            if metric.requires_model_artifact:
                # Stability metrics: computed once per iteration, no dataset splits
                # Keep as tensors to avoid CPU synchronization
                if metric == Metric.WeightNorm:
                    value = get_weight_norm(model)
                elif metric == Metric.UpdateNorm:
                    if self.grad_norm is None:
                        raise ValueError(
                            "UpdateNorm metric requires grad_norm to be set by the optimizer. "
                            "Ensure the optimizer has a reference to this MetricsCollector and sets grad_norm during step()."
                        )
                    # Use the gradient norm provided by the optimizer (unscaled by learning rate)
                    value = self.grad_norm
                elif metric == Metric.GradLossRatio:
                    # Compute ratio: grad_norm / loss
                    # Both should have been set by the optimizer
                    if self.grad_norm is None:
                        raise ValueError(
                            "GradLossRatio requires grad_norm to be set by the optimizer. "
                            "Ensure the optimizer has a reference to this MetricsCollector and sets grad_norm during step()."
                        )
                    # Reuse loss from results (should have been computed in first pass)
                    loss_key = MetricKey(metric=Metric.Loss, split=DatasetSplit.Train)
                    if loss_key not in results:
                        raise ValueError(
                            "GradLossRatio requires training loss to be computed first. "
                            "Ensure Loss metric is in the metrics list before GradLossRatio."
                        )
                    loss_val = results[loss_key]
                    # Compute ratio: grad_norm / loss (tensor operation)
                    value = self.grad_norm / loss_val
                else:
                    continue

                key = MetricKey(metric=metric, split=None)
                results[key] = value.detach()

        # Reset grad_norm and train_loss after all metrics computed
        # (They will be set again by optimizer on next step)
        self.grad_norm = None
        self.train_loss = None

        return results


    def get_metric_keys(self, splits: list[DatasetSplit]) -> list[MetricKey]:
        """
        Return list of all metric keys that will be computed.

        Args:
            splits: List of dataset splits available

        Returns:
            List of MetricKey objects
        """
        keys = []
        for metric in self.metric_fns.keys():
            if not metric.requires_split:
                # Reference metrics: one key without split
                keys.append(MetricKey(metric=metric, split=None))
            else:
                # Dataset metrics: one key per split
                for split in splits:
                    keys.append(MetricKey(metric=metric, split=split))
        return keys
