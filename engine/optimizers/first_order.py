import torch
from .base import GRAD_TOL
from ..losses import Loss


# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support


def step_gd(model, X, y, lr, loss_fn: Loss):
    """Gradient Descent step with configurable loss."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = loss_fn.grad_linear(X, y, model.w)
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


def step_sgd(model, X, y, lr, loss_fn: Loss):
    """Stochastic Gradient Descent step - classic SGD without momentum."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = loss_fn.grad_linear(X, y, model.w)
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


def step_ngd_stable(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    grad_tol=GRAD_TOL,
):
    """
    Normalized Gradient Descent (Loss-Normalized per Nacson et al. Eq 11).

    Update: w = w - lr * (grad / loss)

    This effectively increases the step size as the loss decreases, counteracting
    the vanishing gradients of exponential loss.
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # Compute gradient and loss together for efficiency
            if hasattr(loss_fn, 'grad_linear_with_loss'):
                grad, loss = loss_fn.grad_linear_with_loss(X, y, model.w)
            else:
                # Fallback for Loss implementations without grad_linear_with_loss
                grad = loss_fn.grad_linear(X, y, model.w)
                scores = X @ model.w
                loss = loss_fn(scores, y)

            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                model.w -= lr * (grad / loss)
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
                    # Normalize by LOSS, not grad norm
                    if grad_norm > grad_tol:
                        param -= lr * (param.grad / loss)


def step_sam_stable(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    rho=0.05,
    grad_tol=GRAD_TOL,
):
    """Sharpness-Aware Minimization (SAM) with configurable loss."""
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. Compute Gradient
            grad = loss_fn.grad_linear(X, y, model.w)
            grad_norm = grad.norm()

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
        # 1. Compute Gradients at current w
        model.zero_grad(set_to_none=True)
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        # Gather params with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # --- GLOBAL NORM CALCULATION ---
            # Efficiently compute norm across all tensors
            per_tensor_norms = torch._foreach_norm(grads, 2)
            global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))

            # Save original parameters
            original_params = [p.clone() for p in params_with_grad]

            # 2. SAM Perturbation (Global)
            if global_norm > grad_tol:
                scale = rho / global_norm
                # p = p + (rho/||g||) * g
                torch._foreach_add_(params_with_grad, grads, alpha=scale)

        # 3. Compute Gradients at adversarial point
        model.zero_grad(set_to_none=True)
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        # Refresh gradients (grads_adv)
        grads_adv = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # 4. Restore & Update

            # First, restore original weights: w = w_orig
            for p, p_orig in zip(params_with_grad, original_params):
                p.copy_(p_orig)

            # Then apply standard GD update using adversarial gradients
            # w = w - lr * g_adv
            # Note: SAM uses standard GD update, NOT normalized update
            torch._foreach_add_(params_with_grad, grads_adv, alpha=-lr)


def step_sam_ngd_stable(
    model,
    X,
    y,
    lr,
    loss_fn: Loss,
    rho=0.05,
    grad_tol=GRAD_TOL,
):
    """
    SAM + Loss-Normalized GD (per Nacson et al. Eq 11).

    Performs SAM perturbation, then applies loss-normalized gradient descent
    at the adversarial point: w = w - lr * (grad_adv / loss_adv)
    """
    if hasattr(model, "w") and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation (uses gradient norm)
            grad = loss_fn.grad_linear(X, y, model.w)
            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad
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
                # No update, treat as zero
                pass
    else:
        # 1. Compute Gradients at current w
        model.zero_grad(set_to_none=True)
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        # Gather params that actually have gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # --- GLOBAL NORM (for SAM perturbation) ---
            per_tensor_norms = torch._foreach_norm(grads, 2)
            global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))

            # Store original parameters (clone is unavoidable for SAM)
            original_params = [p.clone() for p in params_with_grad]

            # 2. SAM Perturbation (Fused)
            if global_norm > grad_tol:
                scale = rho / global_norm
                # Fused add: p = p + scale * g
                torch._foreach_add_(params_with_grad, grads, alpha=scale)

        # 3. Compute Gradients at adversarial point
        model.zero_grad(set_to_none=True)
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        # Refresh grads list
        grads_adv = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # 4. Loss-Normalized GD Update & Restore

            # Restore weights first: p.copy_(p_orig)
            for p, p_orig in zip(params_with_grad, original_params):
                p.copy_(p_orig)

            # Check if adversarial gradients are non-negligible
            per_tensor_norms_adv = torch._foreach_norm(grads_adv, 2)
            global_adv_norm = torch.linalg.vector_norm(
                torch.stack(per_tensor_norms_adv)
            )

            if global_adv_norm > grad_tol:
                # Loss-normalized update: p = p - (lr / loss_adv) * g_adv
                update_scale = -lr / loss_adv
                torch._foreach_add_(params_with_grad, grads_adv, alpha=update_scale)
