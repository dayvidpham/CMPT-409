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
    get_empirical_max_margin,
    get_error_rate,
    get_angle,
    get_direction_distance,
    expand_sweep_grid,
)
from engine.losses import LogisticLoss
from engine.metrics import get_weight_norm, compute_update_norm
from engine.optimizers import make_stateful_optimizer_factory
from engine.optimizers.manual import (
    ManualAdam,
    ManualAdaGrad,
    ManualSAM_Adam,
    ManualSAM_AdaGrad
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
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # Model factory
    # Note: Ensure TwoLayerModel is purely linear (Sequential(Linear, Linear))
    # to match the manual optimizer's gradient derivation.
    def model_factory():
        return TwoLayerModel(X.shape[1], 100, device=device)

    # === Loss function (configurable) ===
    loss_fn = LogisticLoss()

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: loss_fn,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
                Metric.WeightNorm: get_weight_norm,
                Metric.GradNorm: get_weight_norm,  # Function not used, optimizer provides grad_norm
                Metric.UpdateNorm: compute_update_norm,  # Function not used, optimizer provides update_norm
                Metric.GradLossRatio: loss_fn,  # Function not used, computed from grad_norm/loss
            },
            w_star=w_star
        )

    # Optimizer factories
    # Using the fused manual implementations for ~2x speedup on small models
    optimizer_factories = {
        Optimizer.Adam: make_stateful_optimizer_factory(ManualAdam, loss=loss_fn),
        Optimizer.AdaGrad: make_stateful_optimizer_factory(ManualAdaGrad, loss=loss_fn),
        Optimizer.SAM_Adam: make_stateful_optimizer_factory(ManualSAM_Adam, loss=loss_fn),
        Optimizer.SAM_AdaGrad: make_stateful_optimizer_factory(ManualSAM_AdaGrad, loss=loss_fn),
    }

    # Hyperparameter sweeps
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    rho_values = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

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
        total_iters=100_000,
        debug=True
    )

    # Plotting
    plot_all(
        results,
        experiment_name="2layers_adam_family_testing"
    )

if __name__ == "__main__":
    main()

