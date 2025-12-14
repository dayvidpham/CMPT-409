"""Mini-batch SGD experiments on Soudry dataset with hyperparameter sweeps."""

import torch
import os

from engine import (
    run_training,
    LinearModel,
    DatasetSplit,
    Metric,
    Optimizer,
    Hyperparam,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_empirical_max_margin,
    get_error_rate,
    get_angle,
    get_direction_distance,
    expand_sweep_grid,
)
from engine.losses import LogisticLoss

from engine.optimizers import (
    step_sgd,
    step_loss_ngd,
    step_vec_ngd,
    step_sam_stable,
    step_sam_loss_ngd,
    step_sam_messy_loss_ngd,
    step_sam_vec_ngd,
    step_sam_messy_vec_ngd,
    make_optimizer_factory,
)
from engine.metrics import get_weight_norm, compute_update_norm, get_grad_norm
from engine.plotting import plot_all

cpu_count = os.cpu_count()
if cpu_count is not None:
    torch.set_num_threads(cpu_count // 2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Dataset ===
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=40)

    # === Model factory ===
    input_dim = X.shape[1]

    def model_factory():
        return LinearModel(input_dim, device=device)

    # === Loss function (configurable) ===
    # Choose which loss function to use
    # loss_fn = ExponentialLoss()  # Uncomment for default exponential loss
    loss_fn = LogisticLoss()  # Using LogisticLoss as example

    # === Metrics factory ===
    # Pass the loss function so metrics use the same loss as optimizers
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: loss_fn,  # Use configured loss function for metrics
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
                Metric.WeightNorm: get_weight_norm,
                Metric.GradNorm: get_grad_norm,
                Metric.UpdateNorm: compute_update_norm,
                Metric.GradLossRatio: loss_fn,  # Function not used, computed from grad_norm/loss
            },
            w_star=w_star,
        )

    # === Optimizer factories ===
    # All optimizers use the same configured loss function
    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_sgd, loss=loss_fn),
        Optimizer.LossNGD: make_optimizer_factory(step_loss_ngd, loss=loss_fn),
        Optimizer.VecNGD: make_optimizer_factory(step_vec_ngd, loss=loss_fn),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable, loss=loss_fn),
        Optimizer.SAM_LossNGD: make_optimizer_factory(step_sam_loss_ngd, loss=loss_fn),
        Optimizer.SAM_Messy_LossNGD: make_optimizer_factory(step_sam_messy_loss_ngd, loss=loss_fn),
        Optimizer.SAM_VecNGD: make_optimizer_factory(step_sam_vec_ngd, loss=loss_fn),
        Optimizer.SAM_Messy_VecNGD: make_optimizer_factory(step_sam_messy_vec_ngd, loss=loss_fn),
    }

    # === Hyperparameter sweeps ===
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]
    # rho_values = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0, 75.0, 100.0]

    learning_rates = [1.0, 5.0, 10.0]
    rho_values = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0, 75.0, 100.0]

    sweeps = {
        Optimizer.GD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.LossNGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.VecNGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_LossNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_Messy_LossNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_VecNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_Messy_VecNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
    }

    # Expand to concrete configurations
    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    # === Training ===
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        num_epochs=10_000,
        batch_size=32,
        drop_last=True,
        debug=True,
    )

    # === Plotting ===
    plot_all(
        results,
        experiment_name="sgd_soudry_messy__weird_idea",
        save_separate=False,
    )


if __name__ == "__main__":
    main()
