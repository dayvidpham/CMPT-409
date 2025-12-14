from engine import (
    run_training,
    LinearModel,
    DatasetSplit,
    Metric,
    Optimizer,
    OptimizerConfig,
    Hyperparam,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_empirical_max_margin,
    ExponentialLoss,
    LogisticLoss,
    get_error_rate,
    get_angle,
    get_direction_distance,
    expand_sweep_grid,
)
from engine.metrics import get_weight_norm, compute_update_norm
from engine.optimizers import Adam, AdaGrad, SAM_Adam, SAM_AdaGrad
from engine.plotting import plot_all
import numpy as np
import random
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

# For repeatable experiments
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset (Torch only now)
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)

    print("Angle(v, w*):", get_angle(v_pop, w_star))

    # Split data
    # datasets = split_train_test(X, y, test_size=0.2, random_state=SEED)
    datasets = split_train_test(X, y, test_size=40)

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], device=device)

    # === Loss function (configurable) ===
    # Choose which loss function to use
    # loss_fn = ExponentialLoss()  # Uncomment for default exponential loss
    loss_fn = LogisticLoss()  # Using LogisticLoss as example

    # Metrics factory (includes Angle/Distance for linear model)
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: loss_fn,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
                Metric.WeightNorm: get_weight_norm,
                Metric.UpdateNorm: compute_update_norm,
                Metric.GradLossRatio: loss_fn,  # Function not used, computed from grad_norm/loss
            },
            w_star=w_star
        )

    # Optimizer factories (using adaptive.py for LinearModel)
    optimizer_factories = {
        Optimizer.Adam: Adam,
        Optimizer.AdaGrad: AdaGrad,
        Optimizer.SAM_Adam: SAM_Adam,
        Optimizer.SAM_AdaGrad: SAM_AdaGrad,
    }

    # === Hyperparameter sweeps ===
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    rho_values = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0]

    sweeps = {
        Optimizer.Adam: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM_Adam: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
    }

    # Expand to concrete configurations
    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    # Run training
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=10_000,
        debug=True
    )

    # Plotting
    plot_all(
        results,
        experiment_name="adam_family_soudry"
    )

if __name__ == "__main__":
    main()
