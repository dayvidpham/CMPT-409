"""Full-batch GD experiments on Soudry dataset."""

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
from engine.metrics import get_weight_norm, compute_update_norm
from engine.losses import LogisticLoss
from engine.optimizers import (
    step_gd,
    step_sam_stable,
    step_loss_ngd,
    step_vec_ngd,
    step_sam_loss_ngd,
    step_sam_vec_ngd,
    make_optimizer_factory,
)
from engine.plotting import plot_all

# Configure PyTorch to use half the CPU cores
cpu_count = os.cpu_count()
if cpu_count is not None:
    torch.set_num_threads(cpu_count // 2)


def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, margin=1.0, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=40)

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], device=device)

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
                Metric.GradNorm: get_weight_norm,  # Function not used, optimizer provides grad_norm
                Metric.UpdateNorm: compute_update_norm,  # Function not used, optimizer provides grad_norm
                Metric.GradLossRatio: loss_fn,  # Function not used, computed from grad_norm/loss
            },
            w_star=w_star,
        )

    # === Optimizer factories ===
    # All optimizers use the same configured loss function
    optimizer_logistic_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd, loss=loss_fn),
        Optimizer.LossNGD: make_optimizer_factory(step_loss_ngd, loss=loss_fn),
        Optimizer.VecNGD: make_optimizer_factory(step_vec_ngd, loss=loss_fn),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable, loss=loss_fn),
        Optimizer.SAM_LossNGD: make_optimizer_factory(step_sam_loss_ngd, loss=loss_fn),
        Optimizer.SAM_VecNGD: make_optimizer_factory(step_sam_vec_ngd, loss=loss_fn),
    }

    # === Hyperparameter sweeps ===
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    rho_values = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

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
        Optimizer.SAM_VecNGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
    }

    # Expand to concrete configurations
    optimizer_configs = expand_sweep_grid(optimizer_logistic_factories, sweeps)

    # ==== Run deterministic ====
    deterministic_results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=10_000,
        debug=True,
    )

    # Plotting
    plot_all(
        deterministic_results,
        experiment_name="soudry_linear_whole_schebang/deterministic",
        save_separate=False,    # slow af
        save_aggregated=False,  # deprecated
        save_combined=False,    # deprecated
    )



    # ==== Run stochastic ====
    stochastic_results = run_training(
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

    # Plotting
    plot_all(
        stochastic_results,
        experiment_name="soudry_linear_whole_schebang/stochastic",
        save_separate=False,    # slow af
        save_aggregated=False,  # deprecated
        save_combined=False,    # deprecated
    )


if __name__ == "__main__":
    main()
