from .first_order import (
    step_gd,
    step_sgd,
    step_loss_ngd,
    step_vec_ngd,
    step_sam_stable,
    step_sam_loss_ngd,
    step_sam_messy_loss_ngd,
    step_sam_vec_ngd,
    step_sam_messy_vec_ngd,
)
# Backward compatibility aliases
step_ngd_stable = step_loss_ngd
step_sam_ngd_stable = step_sam_loss_ngd
from .adaptive import (
    Adam,
    AdaGrad,
    SAM_Adam,
    SAM_AdaGrad,
)

from .manual import (
    ManualAdam,
    ManualAdaGrad,
    ManualSAM_Adam,
    ManualSAM_AdaGrad,
    ManualGD,
    ManualLossNGD,
    ManualVecNGD,
    ManualSAM,
    ManualSAM_LossNGD,
    ManualSAM_VecNGD,
)
# Backward compatibility
ManualNGD = ManualLossNGD
ManualSAM_NGD = ManualSAM_LossNGD

from .base import (
    make_optimizer,
    make_optimizer_factory,
    OptimizerState,
    StatelessOptimizer,
    StatefulOptimizer,
    SAMOptimizer,
)

__all__ = [
    "step_gd",
    "step_sgd",
    "step_loss_ngd",
    "step_vec_ngd",
    "step_sam_stable",
    "step_sam_loss_ngd",
    "step_sam_messy_loss_ngd",
    "step_sam_vec_ngd",
    "step_sam_messy_vec_ngd",
    "step_ngd_stable",  # Backward compatibility
    "step_sam_ngd_stable",  # Backward compatibility
    "make_optimizer",
    "make_optimizer_factory",
    "OptimizerState",
    "StatelessOptimizer",
    "StatefulOptimizer",
    "SAMOptimizer",
    "Adam",
    "AdaGrad",
    "SAM_Adam",
    "SAM_AdaGrad",
    "ManualAdam",
    "ManualAdaGrad",
    "ManualSAM_Adam",
    "ManualSAM_AdaGrad",
    "ManualGD",
    "ManualLossNGD",
    "ManualVecNGD",
    "ManualNGD",  # Backward compatibility
    "ManualSAM",
    "ManualSAM_LossNGD",
    "ManualSAM_VecNGD",
    "ManualSAM_NGD",  # Backward compatibility
]
