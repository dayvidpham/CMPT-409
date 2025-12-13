# import ctypes.util
# import os
#
# # NixOS compatibility fix: prevent ctypes from calling /sbin/ldconfig
# if not os.path.exists('/sbin/ldconfig'):
#     def _find_library_patched(name):
#         # On NixOS, we rely on LD_LIBRARY_PATH being set correctly by the shell wrapper
#         # so we just return the name and let dlopen find it.
#         return name
#     ctypes.util.find_library = _find_library_patched
#
from engine import (
    run_training,
    LinearModel,
    DatasetSplit,
    Metric,
    Optimizer,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_empirical_max_margin,
    exponential_loss,
    get_error_rate,
    get_angle,
    get_direction_distance,
)
from engine.optimizers import step_gd, step_sam_stable, step_ngd_stable, step_sam_ngd_stable
from engine.optimizers.base import make_optimizer
from engine.plotting import plot_all
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count() // 2)

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(f"Using device: {device}")

    # Generate dataset
    X, y, v_pop = make_soudry_dataset(n=1000, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=200, random_state=42)

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], device=device)

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
            },
            w_star=w_star
        )

    # Optimizers
    optimizers = {
        Optimizer.GD: make_optimizer(step_gd),
        Optimizer.SAM: make_optimizer(step_sam_stable),
        Optimizer.NGD: make_optimizer(step_ngd_stable),
        Optimizer.SAM_NGD: make_optimizer(step_sam_ngd_stable),
    }

    # Run training
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    #learning_rates = [1e-4]
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=100_000,
        debug=True
    )

    # Plotting
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="soudry"
    )

if __name__ == "__main__":
    main()
