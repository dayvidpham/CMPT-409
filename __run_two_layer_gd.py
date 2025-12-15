# ===============================================================
# run_ngd_experiments.py   (GD / NGD / SAM / SAM-NGD)
# ===============================================================
from engine import (
    run_training,
    TwoLayerModel,
    DatasetSplit,
    Metric,
    Optimizer,
    OptimizerConfig,
    Hyperparam,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_error_rate,
    expand_sweep_grid,
)
from engine.losses import ExponentialLoss, LogisticLoss
from engine.metrics import compute_update_norm, get_angle, get_direction_distance, get_empirical_max_margin, get_weight_norm
from engine.optimizers import make_stateful_optimizer_factory
from engine.optimizers.manual import (
    ManualGD,
    ManualLossNGD,
    ManualVecNGD,
    ManualSAM,
    ManualSAM_LossNGD,
    ManualSAM_VecNGD,
)
from engine.plotting import plot_all
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())


def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset
    N = 200
    D = 5000
    test_size = 40

    X, y, v_pop = make_soudry_dataset(n=N, d=D, margin=1.0, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=test_size)

    # ----------------------------------------------------------
    # 2-layer width
    # ----------------------------------------------------------

    k = 50

    # Model factory (Torch only now)
    def model_factory():
        return TwoLayerModel(D, k, device=device)

    # === Loss function (configurable) ===
    # Choose which loss function to use
    # loss_fn = ExponentialLoss()  # Uncomment for default exponential loss
    loss_fn = LogisticLoss()  # Using LogisticLoss as example

    # Metrics factory (NO w_star for two-layer)
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

    # ----------------------------------------------------------
    # Optimizer factories
    # ----------------------------------------------------------
    optimizer_factories = {
        Optimizer.GD: make_stateful_optimizer_factory(ManualGD, loss=loss_fn),
        Optimizer.LossNGD: make_stateful_optimizer_factory(ManualLossNGD, loss=loss_fn),
        Optimizer.VecNGD: make_stateful_optimizer_factory(ManualVecNGD, loss=loss_fn),
        Optimizer.SAM: make_stateful_optimizer_factory(ManualSAM, loss=loss_fn),
        Optimizer.SAM_LossNGD: make_stateful_optimizer_factory(ManualSAM_LossNGD, loss=loss_fn),
        Optimizer.SAM_VecNGD: make_stateful_optimizer_factory(ManualSAM_VecNGD, loss=loss_fn),
    }

    # ----------------------------------------------------------
    # Hyperparameter sweeps
    # ----------------------------------------------------------
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    rho_values = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    total_iters = 100


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
    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=total_iters,
        debug=True,
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    plot_all(
        results,
        experiment_name="soudry_twolayers_gd",
        save_separate=False,    # slow af
        save_aggregated=False,  # deprecated
        save_combined=False,    # deprecated
    )


if __name__ == "__main__":
    main()
