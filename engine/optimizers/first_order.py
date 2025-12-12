import torch

# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support

# Machine epsilon for float64 (Double Precision)
EPS = 2.2e-16
GRAD_TOL = 1e-30
CLAMP_MIN = -300
CLAMP_MAX = 300

def _linear_grad_exponential(X, y, w, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX):
    """
    Helper to compute the gradient of Exponential Loss for a linear model manually.
    """
    scores = X @ w

    # Ensure y matches scores shape (N,) or (N,1)
    y = y.view_as(scores)

    margins = y * scores
    margins_clamped = torch.clamp(margins, clamp_min, clamp_max)
    exp_neg_margins = torch.exp(-margins_clamped)

    # FIX: The scalar term must be (N, 1) to broadcast against X (N, D)
    # regardless of whether scores was (N,) or (N, 1).
    scalar_term = (y * exp_neg_margins).view(-1, 1)

    # Gradient: -mean(scalar_term * X)
    grad = -torch.mean(scalar_term * X, dim=0)

    # Ensure gradient shape matches weight shape
    return grad.view_as(w)


def step_gd(model, X, y, lr, loss_fn):
    """Gradient Descent step."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w)
            model.w -= lr * grad
    else:
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad


def step_ngd_stable(model, X, y, lr, loss_fn, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """
    Normalized Gradient Descent with numerical stability.

    Uses clamped gradient computation to prevent overflow, and a hard threshold
    check to preserve NGD's constant step-size property (avoiding EPS degradation).
    """
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                model.w -= lr * (grad / grad_norm)
            # else: gradient effectively zero, no update
    else:
        # General case: autograd path
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if grad_norm > grad_tol:
                        param -= lr * (param.grad / grad_norm)


def step_sam_stable(model, X, y, lr, loss_fn, rho=0.05, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """Sharpness-Aware Minimization (SAM) - Corrected."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. Compute Gradient
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            # 2. Compute Perturbation (Strictly enforcing norm = rho)
            if grad_norm > grad_tol:  # Use a tiny threshold significantly smaller than EPS
                eps = (rho / grad_norm) * grad
            else:
                eps = torch.zeros_like(grad) # Or random direction if preferred

            # 3. Adversarial Step
            w_adv = model.w + eps
            grad_adv = _linear_grad_exponential(X, y, w_adv, clamp_min, clamp_max)

            # 4. Update
            model.w -= lr * grad_adv
    else:
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        original_params = []
        with torch.no_grad():
            for param in model.parameters():
                original_params.append(param.clone())
                if param.grad is not None:
                    grad_norm = param.grad.norm() + EPS
                    param.add_(param.grad, alpha=rho / grad_norm)

        model.zero_grad()
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        with torch.no_grad():
            for param, param_orig in zip(model.parameters(), original_params):
                grad_adv = param.grad
                param.copy_(param_orig)
                if grad_adv is not None:
                    param -= lr * grad_adv


def step_sam_ngd_stable(model, X, y, lr, loss_fn, rho=0.05, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """
    SAM + Normalized GD with numerical stability.
    """
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation
            # Use the clamped helper to prevent overflow in exp()
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            # Check against a tiny tolerance to avoid div-by-zero
            # 1e-30 is safe for float64 and allows training to continue much longer
            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad
                w_adv = model.w + eps

                # 2. NGD update at adversarial point
                grad_adv = _linear_grad_exponential(X, y, w_adv, clamp_min, clamp_max)
                grad_adv_norm = grad_adv.norm()

                if grad_adv_norm > grad_tol:
                    # Standard NGD update
                    model.w -= lr * (grad_adv / grad_adv_norm)
    else:
        # General case: autograd path
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        original_params = []
        with torch.no_grad():
            for param in model.parameters():
                original_params.append(param.clone())
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if grad_norm > grad_tol:
                        param.add_(param.grad, alpha=rho / grad_norm)

        model.zero_grad()
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        with torch.no_grad():
            for param, param_orig in zip(model.parameters(), original_params):
                param.copy_(param_orig)
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if grad_norm > grad_tol:
                        param -= lr * (param.grad / grad_norm)


