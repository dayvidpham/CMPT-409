import os
import numpy as np
import torch

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
    
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    
    optimizer_keys = [
        Optimizer.GD,
        Optimizer.SAM,
        Optimizer.NGD,
        Optimizer.SAM_NGD
    ]

    # Container: results[lr][opt] = [History_Run1, History_Run2, ...]
    aggregated_results = {lr: {opt: [] for opt in optimizer_keys} for lr in learning_rates}

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
                    Metric.Loss: exponential_loss,
                    Metric.Error: get_error_rate,
                    Metric.Angle: get_angle,
                    Metric.Distance: get_direction_distance,
                },
                w_star=w_star
            )

        # 4. Define Optimizers
        optimizers_map = {
            Optimizer.GD: make_optimizer(step_gd),
            Optimizer.SAM: make_optimizer(step_sam_stable),
            Optimizer.NGD: make_optimizer(step_ngd_stable),
            Optimizer.SAM_NGD: make_optimizer(step_sam_ngd_stable),
        }
        optimizers_map = {k: v for k, v in optimizers_map.items() if k in optimizer_keys}

        # 5. Run Training
        seed_results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizers_map,
            learning_rates=learning_rates,
            metrics_collector_factory=metrics_factory,
            train_split=DatasetSplit.Train,
            total_iters=total_iters,
            debug=True
        )

        # 6. Collect Results
        for lr in learning_rates:
            for opt in optimizer_keys:
                aggregated_results[lr][opt].append(seed_results[lr][opt])

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("\nGenerating Aggregated Plots...")
    plot_all(
        aggregated_results,
        learning_rates,
        optimizer_keys,
        experiment_name="soudry_aggregated",
        save_combined=True,
        save_separate=True,
        save_aggregated=True
    )
    print("Done!")

if __name__ == "__main__":
    main()

