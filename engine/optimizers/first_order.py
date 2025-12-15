import torch
from .base import GRAD_TOL
from ..losses import Loss


# -----------------------------------------------------------------------------
# Helper function for metrics collection
# -----------------------------------------------------------------------------


def _collect_metrics(metrics_collector, grad_norm, train_loss):
    """Helper to collect metrics in stateless optimizer functions."""
    if metrics_collector is not None:
        metrics_collector.grad_norm = grad_norm
        metrics_collector.train_loss = train_loss


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
            _collect_metrics(metrics_collector, grad.norm(), loss)
            model.w -= lr * grad
    else:
        raise NotImplementedError("Use ManualGD instead")



def step_sgd(model, X, y, lr, loss_fn: Loss, metrics_collector=None):
    """Stochastic Gradient Descent step - classic SGD without momentum."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            _collect_metrics(metrics_collector, grad.norm(), loss)
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
    the vanishing gradients of exponential loss.
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # Compute gradient and loss together for efficiency
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            _collect_metrics(metrics_collector, grad_norm, loss)

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

    Normalizes by the gradient norm, not the loss value.
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            _collect_metrics(metrics_collector, grad_norm, loss)

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
            _collect_metrics(metrics_collector, grad_norm, loss)

            # 2. Compute Perturbation (Strictly enforcing norm = rho)
            if grad_norm > grad_tol:
                eps = (rho / grad_norm) * grad
            else:
                eps = torch.zeros_like(grad)  # Or random direction if preferred

            # 3. Adversarial Step
            w_adv = model.w + eps
            grad_adv = loss_fn.grad_linear(X, y, w_adv)

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
    at the adversarial point: w = w - lr * (grad_adv / loss_adv)
    """
    return _step_sam_loss_ngd_impl(
        model,
        X,
        y,
        lr,
        loss_fn,
        metrics_collector=metrics_collector,
        rho=rho,
        grad_tol=grad_tol,
        perturb_sign=1.0,
    )


def step_sam_messy_loss_ngd(
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
    Variant of SAM + Loss-Normalized GD that perturbs in the negative direction
    (w_adv = w - eps) to intentionally destabilize the adversarial point.
    """
    return _step_sam_loss_ngd_impl(
        model,
        X,
        y,
        lr,
        loss_fn,
        metrics_collector=metrics_collector,
        rho=rho,
        grad_tol=grad_tol,
        perturb_sign=-1.0,
    )


def _step_sam_loss_ngd_impl(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    rho=0.05,
    grad_tol=GRAD_TOL,
    perturb_sign: float = 1.0,
):
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation (uses gradient norm)
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            _collect_metrics(metrics_collector, grad_norm, loss)

            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad

                if perturb_sign >= 0:
                    w_adv = model.w + eps

                    # 2. Loss-normalized GD update at adversarial point
                    if hasattr(loss_fn, 'grad_linear_with_loss'):
                        grad_adv, loss_adv = loss_fn.grad_linear_with_loss(X, y, w_adv)
                    else:
                        grad_adv = loss_fn.grad_linear(X, y, w_adv)
                        scores_adv = X @ w_adv
                        loss_adv = loss_fn(scores_adv, y)

                    grad_adv_norm = grad_adv.norm()

                    if grad_adv_norm > grad_tol:
                        # Loss-normalized update
                        model.w -= lr * (grad_adv / loss_adv)
                else:
                    # Messy variant: gradient sampled at w + eps, denominator from ||grad(w - eps)||.
                    # w_pos = model.w + eps
                    w_neg = model.w - eps

                    if hasattr(loss_fn, 'grad_linear_with_loss'):
                        #grad_pos, _ = loss_fn.grad_linear_with_loss(X, y, w_pos)
                        grad_neg, loss_neg = loss_fn.grad_linear_with_loss(X, y, w_neg)
                    else:
                        # grad_pos = loss_fn.grad_linear(X, y, w_pos)
                        grad_neg = loss_fn.grad_linear(X, y, w_neg)
                        scores_neg = X @ w_neg
                        loss_neg = loss_fn(scores_neg, y)

                    # grad_neg_norm = grad_neg.norm()

                    # Previous behavior for messy SAM Loss-NGD:
                    if loss_neg > grad_tol:
                        model.w -= lr * (grad_neg / loss_neg)

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
    at the adversarial point: w = w - lr * (grad_adv / ||grad_adv||)
    """
    return _step_sam_vec_ngd_impl(
        model,
        X,
        y,
        lr,
        loss_fn,
        metrics_collector=metrics_collector,
        rho=rho,
        grad_tol=grad_tol,
        perturb_sign=1.0,
    )


def step_sam_messy_vec_ngd(
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
    Variant of SAM + Vector-Normalized GD that flips the adversarial sign
    (w_adv = w - eps) for debugging/ablation experiments.
    """
    return _step_sam_vec_ngd_impl(
        model,
        X,
        y,
        lr,
        loss_fn,
        metrics_collector=metrics_collector,
        rho=rho,
        grad_tol=grad_tol,
        perturb_sign=-1.0,
    )


def _step_sam_vec_ngd_impl(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    metrics_collector=None,
    rho=0.05,
    grad_tol=GRAD_TOL,
    perturb_sign: float = 1.0,
):
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation (uses gradient norm)
            grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            grad_norm = grad.norm()
            _collect_metrics(metrics_collector, grad_norm, loss)

            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad

                if perturb_sign >= 0:
                    w_adv = model.w + eps

                    # 2. Vector-normalized GD update at adversarial point
                    grad_adv = loss_fn.grad_linear(X, y, w_adv)
                    grad_adv_norm = grad_adv.norm()

                    if grad_adv_norm > grad_tol:
                        # Vector-normalized update
                        model.w -= lr * (grad_adv / grad_adv_norm)
                else:
                    # Messy variant: use gradient from w + eps, scale by ||grad(w - eps)||.
                    # w_pos = model.w + eps
                    w_neg = model.w - eps
                    # grad_pos = loss_fn.grad_linear(X, y, w_pos)
                    grad_neg = loss_fn.grad_linear(X, y, w_neg)
                    grad_neg_norm = grad_neg.norm()

                    # Previous behavior for messy SAM Vec-NGD:
                    # if grad_neg_norm > grad_tol:
                    #     model.w -= lr * (grad_neg / grad_neg_norm)

                    if grad_neg_norm > grad_tol:
                        model.w -= lr * (grad_neg / grad_neg_norm)
    else:
        raise NotImplementedError("Use ManualSAM_VecNGD instead")
