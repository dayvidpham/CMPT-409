import os
import numpy as np
import torch

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

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Experiment Configuration
    # -------------------------------------------------------------------------
    num_runs = 40
    
    # 1. Instantiate the Master Generator from OS entropy
    master_rng = np.random.default_rng()
    
    # 2. Generate 64-bit integer seeds for each run
    #    (np.random.default_rng supports arbitrary bit depth seeds, but 
    #     we produce int64s specifically to match Torch's manual_seed cap)
    #    We use the max value of int64 to utilize the full range.
    max_int64 = np.iinfo(np.int64).max
    seeds = master_rng.integers(low=0, high=max_int64, size=num_runs, dtype=np.int64)
    
    print(f"Generated {num_runs} random 64-bit seeds: {seeds}")

    # Dataset params
    N, D = 200, 5000
    total_iters = 100_000

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    rho_values = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

    # Define which optimizers to use (using same loss function)
    from engine.losses import LogisticLoss
    loss_fn = LogisticLoss()

    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd, loss=loss_fn),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable, loss=loss_fn),
        Optimizer.LossNGD: make_optimizer_factory(step_loss_ngd, loss=loss_fn),
        Optimizer.VecNGD: make_optimizer_factory(step_vec_ngd, loss=loss_fn),
        Optimizer.SAM_LossNGD: make_optimizer_factory(step_sam_loss_ngd, loss=loss_fn),
        Optimizer.SAM_VecNGD: make_optimizer_factory(step_sam_vec_ngd, loss=loss_fn),
    }

    # === Hyperparameter sweeps ===
    sweeps = {
        Optimizer.GD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.LossNGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.VecNGD: {
            Hyperparam.LearningRate: learning_rates,
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

    # Container: results[config] = [History_Run1, History_Run2, ...]
    aggregated_results = {config: [] for config in optimizer_configs}

    # -------------------------------------------------------------------------
    # Training Loop (Multiple Seeds)
    # -------------------------------------------------------------------------
    for run_idx, seed_val in enumerate(seeds):
        # Cast to Python int (int64) for compatibility with Torch
        seed = int(seed_val)
        print(f"\n=== Run {run_idx+1}/{num_runs} | Seed: {seed} ===")
        
        # A. Seed PyTorch (Supports int64)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # B. Create a specific RNG for this run's NumPy operations
        #    This Generator handles the 64-bit seed correctly.
        run_rng = np.random.default_rng(seed)

        # 1. Generate new dataset using the explicit RNG
        X, y, v_pop = make_soudry_dataset(n=N, d=D, device=device, rng=run_rng)
        w_star = get_empirical_max_margin(X, y)

        # 2. Split data using the explicit RNG
        datasets = split_train_test(X, y, test_size=0.2, rng=run_rng)

        # 3. Define Factories
        def model_factory():
            return LinearModel(X.shape[1], device=device)

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

        # 4. Run Training
        seed_results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizer_configs,
            metrics_collector_factory=metrics_factory,
            train_split=DatasetSplit.Train,
            total_iters=total_iters,
            debug=True
        )

        # 5. Collect Results
        for config in optimizer_configs:
            aggregated_results[config].append(seed_results[config])

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("\nGenerating Aggregated Plots...")
    plot_all(
        aggregated_results,
        experiment_name="soudry_aggregated",
        save_separate=False,
        save_aggregated=False,
        save_combined=False,
    )
    print("Done!")

if __name__ == "__main__":
    main()

