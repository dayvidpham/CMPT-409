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
from engine.optimizers import Adam, AdaGrad, SAM_Adam, SAM_AdaGrad
from engine.plotting import plot_all
import numpy as np
import random
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset (Torch only now)
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)

    print("Angle(v, w*):", get_angle(v_pop, w_star))

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=SEED)

    # Model factory (Torch only now)
    def model_factory():
        return LinearModel(X.shape[1], device=device)

    # Metrics factory (includes Angle/Distance for linear model)
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

    # Optimizers (FIXED: no LR in names, moved outside loop)
    optimizers = {
        Optimizer.Adam: make_adaptive_optimizer(torch.optim.Adam, betas=(0.9, 0.999), eps=1e-8),
        Optimizer.AdaGrad: make_adaptive_optimizer(torch.optim.AdaGrad, eps=1e-8),
        Optimizer.SAM_Adam: make_sam_optimizer(torch.optim.Adam, rho=0.05, betas=(0.9, 0.999), eps=1e-8),
        Optimizer.SAM_AdaGrad: make_sam_optimizer(torch.optim.AdaGrad, rho=0.05, eps=1e-8),
    }

    # Run training (FIXED: moved outside loop)
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
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

    # Plotting (FIXED: moved outside loop)
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="adam_family"
    )

if __name__ == "__main__":
    main()
