import torch
from .base import GRAD_TOL
from ..losses import Loss


# -----------------------------------------------------------------------------
# Helper function for metrics collection
# -----------------------------------------------------------------------------


def _collect_metrics(metrics_collector, grad_norm, train_loss, update_norm=None):
    """Helper to collect metrics in stateless optimizer functions."""
    if metrics_collector is not None:
        metrics_collector.grad_norm = grad_norm
        metrics_collector.train_loss = train_loss
        if update_norm is not None:
            metrics_collector.update_norm = update_norm


# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support


def step_gd(model, X, y, lr, loss_fn: Loss, metrics_collector=None):
    """Gradient Descent step with configurable loss."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            # For GD: update_direction = gradient, so update_norm = grad_norm
            update_norm = grad_norm
            _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
            model.w -= lr * grad
    else:
        raise NotImplementedError("Use ManualGD instead")


def step_sgd(model, X, y, lr, loss_fn: Loss, metrics_collector=None):
    """Stochastic Gradient Descent step - classic SGD without momentum."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            # For SGD: update_direction = gradient, so update_norm = grad_norm
            update_norm = grad_norm
            _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
            model.w -= lr * grad
    else:
        raise NotImplementedError("Use ManualSGD instead")


def step_loss_ngd(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    grad_tol=GRAD_TOL,
):
    """
    Loss-Normalized Gradient Descent (per Nacson et al. Eq 11).

    Update: w = w - lr * (grad / loss)

    This effectively increases the step size as the loss decreases, counteracting
    vanishing gradients of exponential loss.
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # Compute gradient and loss together for efficiency
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            # For LossNGD: update_direction = gradient / loss, so update_norm = ||gradient|| / loss
            update_norm = (
                grad_norm / loss
                if grad_norm > grad_tol
                else torch.zeros_like(grad_norm)
            )
            _collect_metrics(metrics_collector, grad_norm, loss, update_norm)

            if grad_norm > grad_tol:
                model.w -= lr * (grad / loss)
            # else: gradient effectively zero, no update
    else:
        raise NotImplementedError("Use ManualLossNGD instead")


def step_vec_ngd(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    grad_tol=GRAD_TOL,
):
    """
    Vector-Normalized Gradient Descent (Standard NGD).

    Update: w = w - lr * (grad / ||grad||)

    Normalizes by gradient norm, not loss value.
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            # For VecNGD: update_direction = gradient / ||gradient||, so update_norm = ||gradient / ||gradient|| || = 1
            if grad_norm > grad_tol:
                update_vector = grad / grad_norm
                update_norm = (
                    update_vector.norm()
                )  # This should be 1.0, but compute it properly
            else:
                update_norm = torch.zeros_like(grad_norm)
            _collect_metrics(metrics_collector, grad_norm, loss, update_norm)

            if grad_norm > grad_tol:
                model.w -= lr * (grad / grad_norm)


def step_sam_stable(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    rho=0.05,
    grad_tol=GRAD_TOL,
):
    """Sharpness-Aware Minimization (SAM) with configurable loss."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. Compute Gradient
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()

            # 2. Compute Perturbation (Strictly enforcing norm = rho)
            if grad_norm > grad_tol:
                eps = (rho / grad_norm) * grad
            else:
                eps = torch.zeros_like(grad)  # Or random direction if preferred

            # 3. Adversarial Step
            w_adv = model.w + eps
            grad_adv = loss_fn.grad_linear(X, y, w_adv)

            # For SAM: update_direction = grad_adv, so update_norm = ||grad_adv||
            update_norm = grad_adv.norm()
            _collect_metrics(metrics_collector, grad_norm, loss, update_norm)

            # 4. Update
            model.w -= lr * grad_adv
    else:
        raise NotImplementedError("Use ManualSAM instead")


def step_sam_loss_ngd(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    rho=0.05,
    grad_tol=GRAD_TOL,
):
    """
    SAM + Loss-Normalized GD (per Nacson et al. Eq 11).

    Performs SAM perturbation, then applies loss-normalized gradient descent
    at adversarial point: w = w - lr * (grad_adv / loss_adv)
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation (uses gradient norm)
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad
                w_adv = model.w + eps

                # 2. Loss-normalized GD update at adversarial point
                if hasattr(loss_fn, "grad_linear_with_loss"):
                    grad_adv, loss_adv = loss_fn.grad_linear_with_loss(X, y, w_adv)
                else:
                    grad_adv = loss_fn.grad_linear(X, y, w_adv)
                    scores_adv = X @ w_adv
                    loss_adv = loss_fn(scores_adv, y)

                grad_adv_norm = grad_adv.norm()

                if grad_adv_norm > grad_tol:
                    # For SAM_LossNGD: update_direction = grad_adv / loss_adv
                    update_norm = grad_adv_norm / loss_adv
                    _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
                    # Loss-normalized update
                    model.w -= lr * (grad_adv / loss_adv)
                else:
                    # No update if adversarial gradient is too small
                    update_norm = torch.zeros_like(grad_norm)
                    _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
            else:
                # No update, treat as zero
                update_norm = torch.zeros_like(grad_norm)
                _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
    else:
        raise NotImplementedError("Use ManualSAM_NGD instead")


def step_sam_vec_ngd(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    rho=0.05,
    grad_tol=GRAD_TOL,
):
    """
    SAM + Vector-Normalized GD.

    Performs SAM perturbation, then applies vector-normalized gradient descent
    at adversarial point: w = w - lr * (grad_adv / ||grad_adv||)
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation (uses gradient norm)
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad
                w_adv = model.w + eps

                # 2. Vector-normalized GD update at adversarial point
                grad_adv = loss_fn.grad_linear(X, y, w_adv)
                grad_adv_norm = grad_adv.norm()

                if grad_adv_norm > grad_tol:
                    # For SAM_VecNGD: update_direction = grad_adv / ||grad_adv||, so update_norm = ||grad_adv / ||grad_adv|| || = 1
                    update_vector = grad_adv / grad_adv_norm
                    update_norm = (
                        update_vector.norm()
                    )  # This should be 1.0, but compute it properly
                    _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
                    # Vector-normalized update
                    model.w -= lr * (grad_adv / grad_adv_norm)
                else:
                    # No update if adversarial gradient is too small
                    update_norm = torch.zeros_like(grad_norm)
                    _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
            else:
                # No update if initial gradient is too small
                update_norm = torch.zeros_like(grad_norm)
                _collect_metrics(metrics_collector, grad_norm, loss, update_norm)
    else:
        raise NotImplementedError("Use ManualSAM_VecNGD instead")
