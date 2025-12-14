from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..types import ArrayLike
from ..models import Model
from ..constants import EPS, GRAD_TOL, CLAMP_MIN, CLAMP_MAX
from ..losses import Loss, ExponentialLoss
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Optimizer Base Classes
# -----------------------------------------------------------------------------


class OptimizerState(ABC):
    """
    Base class for optimizers.
    Now operates on the Model instance, not a flat vector.
    """

    def __init__(self, metrics_collector=None):
        """
        Args:
            metrics_collector: Optional MetricsCollector to update with gradient norms
        """
        from ..metrics import MetricsCollector
        self.metrics_collector: Optional[MetricsCollector] = metrics_collector

    def collect_metrics(self, grad_norm: torch.Tensor, train_loss: torch.Tensor):
        """
        Store gradient norm and training loss in the metrics collector.

        Args:
            grad_norm: Norm of the gradient (unscaled by learning rate)
            train_loss: Training loss value
        """
        if self.metrics_collector is not None:
            self.metrics_collector.grad_norm = grad_norm
            self.metrics_collector.train_loss = train_loss

    @abstractmethod
    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        """
        Performs one optimization step.
        Updates model parameters IN-PLACE.
        """
        pass

    def reset(self):
        """Reset internal state (momentum, etc.)."""
        pass

    # def __call__(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float) -> ArrayLike:
    #     """Allow calling the object like a function"""
    #     return self.step(model, X, y, lr)


class StatelessOptimizer(OptimizerState):
    """Wrapper for stateless optimizers (GD, SAM, NGD)"""

    def __init__(self, step_fn: Callable, loss: Optional[Loss] = None, metrics_collector=None):
        super().__init__(metrics_collector)
        self.step_fn = step_fn
        self.loss_fn = loss or ExponentialLoss()  # Create once, reuse across all steps

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # Pass the reusable loss function and metrics collector to the step function
        self.step_fn(model, X, y, lr, self.loss_fn, self.metrics_collector)

    def reset(self):
        pass  # No state to reset


class StatefulOptimizer(OptimizerState):
    """
    Wraps standard PyTorch optimizers (Adam, SGD with momentum).
    Automatically handles initialization on the first step.
    """

    def __init__(self, torch_opt_class: type[torch.optim.Optimizer], loss: Optional[Loss] = None, **kwargs):
        self.opt_class = torch_opt_class
        self.kwargs = kwargs
        self.loss_fn = loss or ExponentialLoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # 1. Lazy Initialization: Create optimizer on first call
        if self.optimizer is None:
            # We assume model.parameters() returns torch tensors
            self.optimizer = self.opt_class(model.parameters(), lr=lr, **self.kwargs)  # type: ignore[call-arg]

        # 2. Update Learning Rate (if changed)
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr

        # 3. Standard PyTorch Training Step
        self.optimizer.zero_grad()

        # Forward
        scores = model.forward(X)
        loss = self.loss_fn(scores, y)

        # Backward & Step
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.optimizer = None


class SAMOptimizer(OptimizerState):
    """
    Sharpness-Aware Minimization wrapper for PyTorch optimizers.
    Performs SAM's double forward/backward pass with global norm calculation.
    Matches the numerical stability and optimization of first_order.py and manual.py.
    """

    def __init__(
        self, torch_opt_class: type[torch.optim.Optimizer], rho: float = 0.05, loss: Optional[Loss] = None, **kwargs
    ):
        self.opt_class = torch_opt_class
        self.kwargs = kwargs
        self.rho = rho
        self.loss_fn = loss or ExponentialLoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # 1. Lazy Initialization
        if self.optimizer is None:
            self.optimizer = self.opt_class(model.parameters(), lr=lr, **self.kwargs)  # type: ignore[call-arg]

        # 2. First forward/backward to compute gradient at current point
        self.optimizer.zero_grad()
        scores = model.forward(X)
        loss = self.loss_fn(scores, y)
        loss.backward()

        # Gather params with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # 3. Compute GLOBAL norm across all parameters (like first_order.py and manual.py)
            per_tensor_norms = torch._foreach_norm(grads, 2)
            global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))

            # Save original parameters
            original_params = [p.clone() for p in params_with_grad]

            # 4. SAM Perturbation using global norm and GRAD_TOL
            if global_norm > GRAD_TOL:
                scale = self.rho / global_norm
                # Perturb all parameters: p = p + (rho/||g||) * g
                torch._foreach_add_(params_with_grad, grads, alpha=scale)

        # 5. Second forward/backward at perturbed point
        self.optimizer.zero_grad()
        scores_adv = model.forward(X)
        loss_adv = self.loss_fn(scores_adv, y)
        loss_adv.backward()

        # 6. Restore original parameters
        with torch.no_grad():
            for p, p_orig in zip(params_with_grad, original_params):
                p.copy_(p_orig)

        # 7. Apply optimizer update using adversarial gradients
        self.optimizer.step()

    def reset(self):
        self.optimizer = None


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def make_optimizer(step_fn: Callable) -> OptimizerState:
    """Create optimizer from stateless step function"""
    return StatelessOptimizer(step_fn)


def make_optimizer_factory(
    step_fn: Callable,
    loss: Optional[Loss] = None,
    **fixed_kwargs,
) -> Callable[..., StatelessOptimizer]:
    """
    Create optimizer factory with optional fixed hyperparameters and loss.

    Args:
        step_fn: The optimizer step function (e.g., step_gd, step_sam_stable)
        loss: Loss function to use (defaults to ExponentialLoss)
        **fixed_kwargs: Hyperparameters to fix at factory creation time

    Returns:
        Factory callable that accepts additional kwargs and returns optimizer.

    Example:
        >>> # Factory with ExponentialLoss
        >>> gd_factory = make_optimizer_factory(step_gd)
        >>> opt = gd_factory()
        >>>
        >>> # Factory with LogisticLoss
        >>> from engine.losses import LogisticLoss
        >>> gd_logistic_factory = make_optimizer_factory(step_gd, loss=LogisticLoss())
        >>> opt = gd_logistic_factory()
    """
    from functools import partial

    def factory(**kwargs) -> StatelessOptimizer:
        merged = {**fixed_kwargs, **kwargs}
        return StatelessOptimizer(partial(step_fn, **merged), loss=loss)

    return factory
