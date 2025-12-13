"""
Loss function implementations with gradient support.

All losses support two interfaces:
1. __call__(scores, y) -> scalar loss for metrics and NGD normalization
2. grad_linear(X, y, w) -> gradient for linear models f(x) = w^T x
"""

from abc import ABC, abstractmethod
import torch
from .constants import EPS, GRAD_TOL, CLAMP_MIN, CLAMP_MAX


class Loss(ABC):
    """Abstract base for loss functions with gradient support."""

    def __init__(
        self,
        clamp_min: float = CLAMP_MIN,
        clamp_max: float = CLAMP_MAX,
    ):
        """
        Args:
            clamp_min: Lower bound for margin clamping (prevents underflow)
            clamp_max: Upper bound for margin clamping (prevents overflow)
        """
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @abstractmethod
    def __call__(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value (for metrics and NGD normalization).

        Args:
            scores: Model predictions, shape (N,) or (N, 1)
            y: Binary labels {-1, 1}, shape (N,) or (N, 1)

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def grad_linear(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient for linear model: ∂L/∂w where scores = Xw.

        Args:
            X: Feature matrix, shape (N, D)
            y: Binary labels {-1, 1}, shape (N,)
            w: Weight vector, shape (D,)

        Returns:
            Gradient ∂L/∂w, shape (D,)
        """
        pass


class ExponentialLoss(Loss):
    """
    Exponential loss: L(w) = mean(exp(-y * w^T x))

    Gradient: ∇L = -1/N * X^T (y * exp(-margins))

    Numerical stability:
    - Margins clamped to [clamp_min, clamp_max] before exp()
    - Uses matrix multiplication to avoid large intermediate tensors
    """

    def __call__(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute exponential loss with numerical stability."""
        y = y.view_as(scores)
        margins = y * scores
        # Clamp to prevent exp() overflow/underflow
        margins_clamped = torch.clamp(margins, self.clamp_min, self.clamp_max)
        return torch.mean(torch.exp(-margins_clamped))

    def grad_linear(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient optimized for GPU.
        Uses Matrix-Vector multiplication to avoid large intermediate broadcasts.

        Matches the original _linear_grad_exponential from first_order.py
        """
        # 1. Compute Scores: (N, D) @ (D,) -> (N,)
        scores = X @ w
        y = y.view_as(scores)

        # 2. Compute margins and clamp for numerical stability
        margins = y * scores
        margins_clamped = torch.clamp(margins, self.clamp_min, self.clamp_max)

        # 3. exp(-m)
        exp_neg_margins = torch.exp(-margins_clamped)

        # 4. scalar_term: y * exp(-m), shape (N, 1)
        scalar_term = (y * exp_neg_margins).view(-1, 1)

        # 5. Compute Gradient via Matrix Multiplication
        # Optimized: -1/N * (X.T @ scalar_term) instead of broadcasting
        # X.T is (D, N), scalar_term is (N, 1) -> Result (D, 1)
        N = X.shape[0]
        grad = torch.matmul(X.T, scalar_term)
        grad.div_(-N)  # In-place division

        return grad.view_as(w)

    def grad_linear_with_loss(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both gradient and loss simultaneously for efficiency.

        Returns:
            (gradient, loss) tuple
        """
        # 1. Compute Scores
        scores = X @ w
        y = y.view_as(scores)

        # 2. Compute margins and clamp
        margins = y * scores
        margins_clamped = torch.clamp(margins, self.clamp_min, self.clamp_max)

        # 3. exp(-m)
        exp_neg_margins = torch.exp(-margins_clamped)

        # 4. Gradient computation
        scalar_term = (y * exp_neg_margins).view(-1, 1)
        N = X.shape[0]
        grad = torch.matmul(X.T, scalar_term)
        grad.div_(-N)

        # 5. Loss computation
        loss = exp_neg_margins.mean()

        return grad.view_as(w), loss


class LogisticLoss(Loss):
    """
    Logistic loss: L(w) = mean(log(1 + exp(-y * w^T x)))

    Gradient: ∇L = -1/N * X^T (y * σ(-margins))
    where σ(z) = 1/(1 + exp(-z)) is the sigmoid function.

    Numerical stability:
    - Uses torch.nn.functional.softplus for stable log(1 + exp(x))
    - Uses torch.sigmoid for stable σ(-m) = 1/(1 + exp(m))
    - No explicit clamping needed (sigmoid and softplus are inherently stable)
    """

    def __call__(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute logistic loss with numerical stability."""
        y = y.view_as(scores)
        margins = y * scores
        # softplus(x) = log(1 + exp(x)) is numerically stable
        # We want log(1 + exp(-m)) = softplus(-m)
        return torch.mean(torch.nn.functional.softplus(-margins))

    def grad_linear(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient optimized for GPU.

        For logistic loss: -ℓ'(m) = σ(-m) = 1/(1 + exp(m))
        Gradient: ∇L = -1/N * X^T (y * σ(-m))
        """
        # 1. Compute Scores: (N, D) @ (D,) -> (N,)
        scores = X @ w
        y = y.view_as(scores)

        # 2. Compute margins
        margins = y * scores

        # 3. σ(-m) = 1/(1 + exp(m)) - numerically stable via torch.sigmoid
        # Note: sigmoid(x) = 1/(1 + exp(-x)), so sigmoid(-m) = 1/(1 + exp(m))
        sigmoid_neg_margins = torch.sigmoid(-margins)

        # 4. scalar_term: y * σ(-m), shape (N, 1)
        scalar_term = (y * sigmoid_neg_margins).view(-1, 1)

        # 5. Compute Gradient via Matrix Multiplication
        N = X.shape[0]
        grad = torch.matmul(X.T, scalar_term)
        grad.div_(-N)  # In-place division

        return grad.view_as(w)

    def grad_linear_with_loss(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both gradient and loss simultaneously for efficiency.

        Returns:
            (gradient, loss) tuple
        """
        # 1. Compute Scores
        scores = X @ w
        y = y.view_as(scores)

        # 2. Compute margins
        margins = y * scores

        # 3. Gradient computation
        sigmoid_neg_margins = torch.sigmoid(-margins)
        scalar_term = (y * sigmoid_neg_margins).view(-1, 1)
        N = X.shape[0]
        grad = torch.matmul(X.T, scalar_term)
        grad.div_(-N)

        # 4. Loss computation
        loss = torch.nn.functional.softplus(-margins).mean()

        return grad.view_as(w), loss
