import torch


def forward_two_layer(model, X):
    Z = model.W1 @ X.T
    pred = model.W2 @ Z
    return pred.mean(0)

def exponential_loss_2layer_torch(model, X, y):
    f = forward_two_layer(model, X)
    return torch.exp(-torch.clamp(y * f, min=-50, max=50)).mean()
# ---------------------------------------------------------
# 1. Adam
# ---------------------------------------------------------
def make_torch_adam_step(W1_shape, W2_shape, lr):
    """
    Returns a step_fn(model, X, y, lr) that internally uses
    a persistent Adam optimizer.
    """
    opt = None    # persistent state

    def step(model, X, y, lr_unused=None):
        nonlocal opt
        if opt is None:
            opt = torch.optim.Adam(model.parameters(), lr=lr)

        loss = exponential_loss_2layer_torch(model, X, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return model

    return step


# ---------------------------------------------------------
# 2. Adagrad
# ---------------------------------------------------------
def make_torch_adagrad_step(W1_shape, W2_shape, lr):
    opt = None

    def step(model, X, y, lr_unused=None):
        nonlocal opt
        if opt is None:
            opt = torch.optim.Adagrad(model.parameters(), lr=lr)

        loss = exponential_loss_2layer_torch(model, X, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return model

    return step


# ---------------------------------------------------------
# 3. SAM + Adam
# ---------------------------------------------------------
def make_torch_sam_adam_step_2layer(W1_shape, W2_shape, lr, rho=0.05):
    opt = None

    def step(model, X, y, lr_unused=None):
        nonlocal opt
        if opt is None:
            opt = torch.optim.Adam(model.parameters(), lr=lr)

        # ----- base grad -----
        loss = exponential_loss_2layer_torch(model, X, y)
        opt.zero_grad()
        loss.backward()

        grads = [p.grad.clone() for p in model.parameters()]
        gvec = torch.cat([g.view(-1) for g in grads])
        gnorm = gvec.norm() + 1e-12

        # save original weights
        base = [p.clone().detach() for p in model.parameters()]

        # perturb
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                n = p.numel()
                p.add_(rho * gvec[idx:idx+n].view_as(p) / gnorm)
                idx += n

        # ----- gradient at perturbed -----
        opt.zero_grad()
        loss2 = exponential_loss_2layer_torch(model, X, y)
        loss2.backward()

        # restore original weights
        with torch.no_grad():
            for p, b in zip(model.parameters(), base):
                p.copy_(b)

        # perform Adam update using perturbed grads
        opt.step()

        return model

    return step


# ---------------------------------------------------------
# 4. SAM + Adagrad
# ---------------------------------------------------------
def make_torch_sam_adagrad_step_2layer(W1_shape, W2_shape, lr, rho=0.05):
    opt = None

    def step(model, X, y, lr_unused=None):
        nonlocal opt
        if opt is None:
            opt = torch.optim.Adagrad(model.parameters(), lr=lr)

        # base gradient
        loss = exponential_loss_2layer_torch(model, X, y)
        opt.zero_grad()
        loss.backward()

        grads = [p.grad.clone() for p in model.parameters()]
        gvec = torch.cat([g.view(-1) for g in grads])
        gnorm = gvec.norm() + 1e-12

        # save state
        base = [p.clone().detach() for p in model.parameters()]

        # perturb
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                n = p.numel()
                p.add_(rho * gvec[idx:idx+n].view_as(p) / gnorm)
                idx += n

        # gradient at perturbed
        opt.zero_grad()
        loss2 = exponential_loss_2layer_torch(model, X, y)
        loss2.backward()

        # restore
        with torch.no_grad():
            for p, b in zip(model.parameters(), base):
                p.copy_(b)

        opt.step()
        return model

    return step
