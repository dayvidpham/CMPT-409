# ===============================================================
# run_adam_experiments.py   (Adam / Adagrad / SAM-Adam / SAM-Adagrad)
# ===============================================================
import numpy as np
import torch
import torch.nn as nn

from linearLayer.dataset import make_soudry_dataset
from linearLayer.trainer import run_training
from linearLayer.plotting import plot_all
from linearLayer.adam_optimizers import *







# ===============================================================
# Main experiment
# ===============================================================

def main():
    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(N, D)
    print("Dataset ready:", X.shape, y.shape)

    # to torch tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # ----------------------------------------------------------
    # Width k
    # ----------------------------------------------------------
    k = 50

    # ----------------------------------------------------------
    # Learning rates
    # ----------------------------------------------------------
    learning_rates = [1e-4]
    total_iters = 100_000

    # ----------------------------------------------------------
    # Optimizer registry
    # ----------------------------------------------------------
    optimizers = {}

    optimizers = {}

    for lr in learning_rates:

        optimizers[f"Adam_lr{lr}"] = make_torch_adam_step((k, D), (10, k), lr)
        optimizers[f"Adagrad_lr{lr}"] = make_torch_adagrad_step((k, D), (10, k), lr)
        optimizers[f"SAM_Adam_lr{lr}"] = make_torch_sam_adam_step_2layer((k, D), (10, k), lr)
        optimizers[f"SAM_Adagrad_lr{lr}"] = make_torch_sam_adagrad_step_2layer((k, D), (10, k), lr)

    # ----------------------------------------------------------
    # Run training
    # ----------------------------------------------------------
    results = run_training(
        X_t, y_t,
        optimizers=optimizers,
        learning_rates=learning_rates,
        k=k,
        total_iters=total_iters,
        debug=True
    )

    # ----------------------------------------------------------
    # Plot summary
    # ----------------------------------------------------------
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="2layers_adam_family"
    )


if __name__ == "__main__":
    main()
