# ===============================================================
# run_adam_experiments.py   (Adam / AdaGrad / SAM-Adam / SAM-AdaGrad)
# ===============================================================
from engine import (
    run_training,
    TwoLayerModel,
    DatasetSplit,
    Metric,
    Optimizer,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    exponential_loss,
    get_error_rate,
)
# Import the manual optimized factories
from engine.optimizers.manual import (
    ManualAdam, 
    ManualAdaGrad, 
    ManualSAM_Adam, 
    ManualSAM_AdaGrad
)
from engine.optimizers import make_adaptive_optimizer, make_sam_optimizer
from engine.plotting import plot_all
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

# ===============================================================
# Main experiment
# ===============================================================

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(N, D, device=device)
    print("Dataset ready:", X.shape, y.shape)

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------------------------
    # Width k
    # ----------------------------------------------------------
    k = 50

    # Model factory (Torch only now)
    def model_factory():
        return TwoLayerModel(D, k, device=device)

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
            },
            w_star=None
        )

    # ----------------------------------------------------------
    # Learning rates
    # ----------------------------------------------------------
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    total_iters = 10_000

    # ----------------------------------------------------------
    # Optimizer registry (FIXED: no LR in names, no duplicate declaration)
    # ----------------------------------------------------------
    optimizers = {
        Optimizer.Adam: ManualAdam(betas=(0.9, 0.999), eps=1e-8),
        Optimizer.AdaGrad: ManualAdaGrad(eps=1e-8),
        Optimizer.SAM_Adam: ManualSAM_Adam(rho=0.05, betas=(0.9, 0.999), eps=1e-8),
        Optimizer.SAM_AdaGrad: ManualSAM_AdaGrad(rho=0.05, eps=1e-8),
    }

    # ----------------------------------------------------------
    # Run training
    # ----------------------------------------------------------
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
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
