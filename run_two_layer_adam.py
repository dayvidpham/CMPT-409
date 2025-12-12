# ===============================================================
# run_adam_experiments.py   (Adam / Adagrad / SAM-Adam / SAM-Adagrad)
# ===============================================================
from engine import (
    run_training,
    TwoLayerModel,
    DatasetSplit,
    Metric,
    Optimizer,
    ComputeBackend,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    exponential_loss,
    get_error_rate,
)
from engine.optimizers import make_adaptive_optimizer, make_sam_optimizer
from engine.plotting import plot_all
import torch


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

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------------------------
    # Width k
    # ----------------------------------------------------------
    k = 50

    # Model factory
    def model_factory():
        return TwoLayerModel(D, k, backend=ComputeBackend.Torch, device="cpu")

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
        Optimizer.Adam: make_adaptive_optimizer(torch.optim.Adam, betas=(0.9, 0.999), eps=1e-8),
        Optimizer.AdaGrad: make_adaptive_optimizer(torch.optim.Adagrad, eps=1e-8),
        Optimizer.SAM_Adam: make_sam_optimizer(torch.optim.Adam, rho=0.05, betas=(0.9, 0.999), eps=1e-8),
        Optimizer.SAM_AdaGrad: make_sam_optimizer(torch.optim.Adagrad, rho=0.05, eps=1e-8),
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
