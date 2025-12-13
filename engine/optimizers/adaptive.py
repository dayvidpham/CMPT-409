import torch
import math
from .base import OptimizerState, StatefulOptimizer, SAMOptimizer

# -----------------------------------------------------------------------------
# Public Optimizer Constructors
# -----------------------------------------------------------------------------

def Adam(betas=(0.9, 0.999), eps=1e-8):
    """Returns a StatefulOptimizer using Adam."""
    return StatefulOptimizer(torch.optim.Adam, betas=betas, eps=eps)

def AdaGrad(eps=1e-8):
    """Returns a StatefulOptimizer using AdaGrad."""
    return StatefulOptimizer(torch.optim.Adagrad, eps=eps)

def SAM_Adam(rho=0.05, betas=(0.9, 0.999), eps=1e-8):
    """Returns a SAMOptimizer using SAM-Adam."""
    return SAMOptimizer(torch.optim.Adam, rho=rho, betas=betas, eps=eps)

def SAM_AdaGrad(rho=0.05, eps=1e-8):
    """Returns a SAMOptimizer using SAM-AdaGrad."""
    return SAMOptimizer(torch.optim.Adagrad, rho=rho, eps=eps)


