from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..types import ArrayLike
from ..models import Model
import torch
import torch.nn as nn

# Global constants
# Machine epsilon for float64 (Double Precision)
EPS = 2.2e-16
GRAD_TOL = 1e-30
CLAMP_MIN = -300
CLAMP_MAX = 300

# -----------------------------------------------------------------------------
# Optimizer Base Classes
# -----------------------------------------------------------------------------

class OptimizerState(ABC):
    """
    Base class for optimizers.
    Now operates on the Model instance, not a flat vector.
    """
    
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

    def __init__(self, step_fn: Callable):
        self.step_fn = step_fn
        self.loss_fn = ExponentialLoss()  # Create once, reuse across all steps

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # Pass the reusable loss function to the step function
        self.step_fn(model, X, y, lr, self.loss_fn)

    def reset(self):
        pass  # No state to reset


class StatefulOptimizer(OptimizerState):
    """
    Wraps standard PyTorch optimizers (Adam, SGD with momentum).
    Automatically handles initialization on the first step.
    """
    def __init__(self, torch_opt_class: type[torch.optim.Optimizer], **kwargs):
        self.opt_class = torch_opt_class
        self.kwargs = kwargs
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # 1. Lazy Initialization: Create optimizer on first call
        if self.optimizer is None:
            # We assume model.parameters() returns torch tensors
            self.optimizer = self.opt_class(model.parameters(), lr=lr, **self.kwargs)
            self.loss_fn = ExponentialLoss()

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
    Performs SAM's double forward/backward pass.
    """
    def __init__(self, torch_opt_class: type[torch.optim.Optimizer], rho: float = 0.05, **kwargs):
        self.opt_class = torch_opt_class
        self.kwargs = kwargs
        self.rho = rho
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def step(self, model: Model, X: ArrayLike, y: ArrayLike, lr: float):
        # 1. Lazy Initialization
        if self.optimizer is None:
            self.optimizer = self.opt_class(model.parameters(), lr=lr, **self.kwargs)
            self.loss_fn = ExponentialLoss()

        # 2. First forward/backward to compute gradient at current point
        self.optimizer.zero_grad()
        scores = model.forward(X)
        loss = self.loss_fn(scores, y)
        loss.backward()

        # 3. Compute perturbation and save current parameters
        original_params = []
        with torch.no_grad():
            for p in model.parameters():
                original_params.append(p.clone())
                if p.grad is not None:
                    grad_norm = p.grad.norm() + 1e-12
                    p.add_(p.grad, alpha=self.rho / grad_norm)

        # 4. Second forward/backward at perturbed point
        self.optimizer.zero_grad()
        scores_adv = model.forward(X)
        loss_adv = self.loss_fn(scores_adv, y)
        loss_adv.backward()

        # 5. Restore original parameters and apply update
        with torch.no_grad():
            for p, p_orig in zip(model.parameters(), original_params):
                p.copy_(p_orig)

        self.optimizer.step()

    def reset(self):
        self.optimizer = None


class ExponentialLoss(nn.Module):
    """
    Computes the mean exponential loss: mean(exp(-y * y_pred))
    """
    def __init__(self, clamp_min=-50, clamp_max=100):
        super().__init__()
        # Clamping is essential because exp() grows excessively fast.
        # exp(100) is ~2e43, which is safe in float64 but huge.
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: The raw scores (logits) from the network. Shape (N, 1) or (N,)
            target: The targets, MUST be {-1, 1}. Shape (N, 1) or (N,)
        """
        # Ensure shapes match
        if input.shape != target.shape:
            target = target.view_as(input)

        # Compute margins: y * f(x)
        margins = target * input

        # Numerical stability (prevents Inf/NaN gradients)
        margins = torch.clamp(margins, min=self.clamp_min, max=self.clamp_max)

        # Loss
        return torch.mean(torch.exp(-margins))


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def make_optimizer(step_fn: Callable) -> OptimizerState:
    """Create optimizer from stateless step function"""
    return StatelessOptimizer(step_fn)
