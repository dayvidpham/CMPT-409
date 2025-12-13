#!/usr/bin/env python
"""
Comprehensive integration test that runs every optimizer with every model type for 5 iteration.
This ensures all combinations work correctly and catches integration issues early.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from engine import LinearModel, make_soudry_dataset, split_train_test, run_training
from engine import Metric, Optimizer, MetricsCollector
from engine import exponential_loss, get_error_rate, get_angle, get_direction_distance
from engine import get_empirical_max_margin
from engine.plotting import plot_all
from engine.optimizers import (
    step_gd, step_ngd_stable, step_sam_stable, step_sam_ngd_stable,
    Adam, AdaGrad, SAM_Adam, SAM_AdaGrad
)
from engine.optimizers.base import make_optimizer

NUM_SAMPLES = 200
NUM_TEST_SAMPLES = 40
NUM_HIDDEN_NEURONS_k = 5
DIM_PARAMS_D = 400
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_test_rng(seed=SEED) -> np.random.Generator:
    # Fresh rng
    return np.random.default_rng(seed)

def make_test_dataset(rng: np.random.Generator):
    # Fresh rng
    return make_soudry_dataset(n=NUM_SAMPLES, d=DIM_PARAMS_D, device=DEVICE, rng=rng)

def test_linear_model_optimizers():
    """Test all optimizers with LinearModel."""
    print("=" * 70)
    print("TESTING LINEAR MODEL WITH ALL OPTIMIZERS")
    print("=" * 70)

    RNG = make_test_rng()
    X, y, v_pop = make_test_dataset(RNG)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=NUM_TEST_SAMPLES, rng=RNG)
    device = DEVICE

    learning_rates = [1e-4, 1e-3]

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

    # All optimizers for linear model
    optimizers = {
        Optimizer.GD: make_optimizer(step_gd),
        Optimizer.NGD: make_optimizer(step_ngd_stable),
        Optimizer.SAM: make_optimizer(step_sam_stable),
        Optimizer.SAM_NGD: make_optimizer(step_sam_ngd_stable),
        Optimizer.Adam: Adam(),
        Optimizer.AdaGrad: AdaGrad(),
        Optimizer.SAM_Adam: SAM_Adam(),
        Optimizer.SAM_AdaGrad: SAM_AdaGrad(),
    }

    test_results = {}
    try:
        # Run 5 iterations for all optimizers
        results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizers,
            learning_rates=learning_rates,
            metrics_collector_factory=metrics_factory,
            total_iters=5,
            debug=True,
        )

        # Plotting
        plot_all(
            results,
            learning_rates,
            list(optimizers.keys()),
            experiment_name="check_engine/linear_models"
        )

        # Mark all optimizers as passed
        for opt_name in optimizers.keys():
            print(f"  ✓ {opt_name.name}")
            test_results[opt_name] = "PASS"

    except Exception as e:
        print(f"  ✗ Linear models failed: {e}")
        test_results["linear_models"] = f"FAIL: {e}"

    return test_results


def test_twolayer_model_optimizers():
    """Test all optimizers with TwoLayerModel."""
    print("\n" + "=" * 70)
    print("TESTING TWO-LAYER MODEL WITH ALL OPTIMIZERS")
    print("=" * 70)

    from engine import TwoLayerModel
    from engine.optimizers import (
        ManualGD, ManualNGD, ManualSAM, ManualSAM_NGD,
        ManualAdam, ManualAdaGrad, ManualSAM_Adam, ManualSAM_AdaGrad
    )

    RNG = make_test_rng()
    X, y, v_pop = make_test_dataset(RNG)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=NUM_TEST_SAMPLES, rng=RNG)
    D = DIM_PARAMS_D
    k = NUM_HIDDEN_NEURONS_k
    device = DEVICE

    learning_rates = [1e-4, 1e-3]

    def model_factory():
        return TwoLayerModel(D, k, device=device)

    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
            },
        )

    # All optimizers for two-layer model
    optimizers = {
        Optimizer.GD: ManualGD(),
        Optimizer.NGD: ManualNGD(),
        Optimizer.SAM: ManualSAM(),
        Optimizer.SAM_NGD: ManualSAM_NGD(),
        Optimizer.Adam: ManualAdam(),
        Optimizer.AdaGrad: ManualAdaGrad(),
        Optimizer.SAM_Adam: ManualSAM_Adam(),
        Optimizer.SAM_AdaGrad: ManualSAM_AdaGrad(),
    }

    test_results = {}
    try:
        # Run 5 iterations for all optimizers
        results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizers,
            learning_rates=learning_rates,
            metrics_collector_factory=metrics_factory,
            total_iters=5,
            debug=True,
        )

        # Plotting
        plot_all(
            results,
            learning_rates,
            list(optimizers.keys()),
            experiment_name="check_engine/twolayer_models"
        )

        # Mark all optimizers as passed
        for opt_name in optimizers.keys():
            print(f"  ✓ {opt_name.name}")
            test_results[opt_name] = "PASS"

    except Exception as e:
        print(f"  ✗ Two-layer models failed: {e}")
        test_results["twolayer_models"] = f"FAIL: {e}"

    return test_results


def test_basic_imports():
    """Quick test of basic module imports."""
    print("\n" + "=" * 70)
    print("TESTING BASIC MODULE IMPORTS")
    print("=" * 70)

    modules = [
        "engine.types",
        "engine.strategies",
        "engine.plotting",
        "engine.metrics",
        "engine.history",
        "engine.trainer",
        "engine.data",
        "engine.models",
        "engine.optimizers",
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append((module, e))

    return failed


def test_plotting_system():
    """Test the plotting system with default and custom strategies."""
    print("\n" + "=" * 70)
    print("TESTING PLOTTING SYSTEM")
    print("=" * 70)

    from engine import LinearModel, make_soudry_dataset, split_train_test
    from engine import Metric, Optimizer, MetricsCollector
    from engine import exponential_loss, get_error_rate, get_angle, get_direction_distance
    from engine import get_empirical_max_margin
    from engine.optimizers import step_gd, step_sam_stable
    from engine.optimizers.base import make_optimizer
    from engine.plotting import plot_all
    from engine.strategies import PlotStrategy, AxisScale, SafeLog, Scale, Clamp, LogLogStrategy, PercentageStrategy

    # Generate small dataset for quick plotting test
    print(f"\n  Generating dataset (n={NUM_SAMPLES}, d={DIM_PARAMS_D})...")

    RNG = make_test_rng()
    X, y, v_pop = make_test_dataset(RNG)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=40, rng=RNG)
    D = DIM_PARAMS_D
    k = NUM_HIDDEN_NEURONS_k
    device = DEVICE

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

    optimizers = {
        Optimizer.GD: make_optimizer(step_gd),
        Optimizer.SAM: make_optimizer(step_sam_stable),
    }

    print("  Running training (5 iterations)...")
    learning_rates = [0.1, 1.0]
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        total_iters=5,
        debug=True,
    )

    test_results = {}

    # Test 1: Default strategies
    print("\n  Test 1: Default strategies...")
    try:
        plot_all(
            results=results,
            learning_rates=learning_rates,
            optimizers=list(optimizers.keys()),
            experiment_name="check_engine/test_1_default_strategies",
        )
        print("    ✓ Default strategies")
        test_results["default_strategies"] = "PASS"
    except Exception as e:
        print(f"    ✗ Default strategies: {e}")
        test_results["default_strategies"] = f"FAIL: {e}"

    # Test 2: Custom error strategy (log-log)
    print("  Test 2: Custom error strategy (log-log)...")
    try:
        custom_error = PlotStrategy(
            transforms=[SafeLog(), Scale(100.0)],
            x_scale=AxisScale.Log,
            y_scale=AxisScale.Log,
            y_label_suffix=" (%) [log]",
        )
        plot_all(
            results=results,
            learning_rates=learning_rates,
            optimizers=list(optimizers.keys()),
            experiment_name="check_engine/test_2_custom_error",
            strategy_overrides={Metric.Error: custom_error},
        )
        print("    ✓ Custom error strategy")
        test_results["custom_error"] = "PASS"
    except Exception as e:
        print(f"    ✗ Custom error strategy: {e}")
        test_results["custom_error"] = f"FAIL: {e}"

    # Test 3: Clamped loss
    print("  Test 3: Clamped loss strategy...")
    try:
        loss_clamped = LogLogStrategy().pipe(Clamp(min_val=1e-10, max_val=1e2))
        plot_all(
            results=results,
            learning_rates=learning_rates,
            optimizers=list(optimizers.keys()),
            experiment_name="check_engine/test_3_clamped_loss",
            strategy_overrides={Metric.Loss: loss_clamped},
        )
        print("    ✓ Clamped loss strategy")
        test_results["clamped_loss"] = "PASS"
    except Exception as e:
        print(f"    ✗ Clamped loss strategy: {e}")
        test_results["clamped_loss"] = f"FAIL: {e}"

    # Test 4: Multiple overrides
    print("  Test 4: Multiple strategy overrides...")
    try:
        custom_error = PercentageStrategy()
        custom_loss = LogLogStrategy().pipe(Clamp(min_val=1e-12))
        plot_all(
            results=results,
            learning_rates=learning_rates,
            optimizers=list(optimizers.keys()),
            experiment_name="check_engine/test_4_multiple_overrides",
            strategy_overrides={
                Metric.Error: custom_error,
                Metric.Loss: custom_loss,
            },
        )
        print("    ✓ Multiple overrides")
        test_results["multiple_overrides"] = "PASS"
    except Exception as e:
        print(f"    ✗ Multiple overrides: {e}")
        test_results["multiple_overrides"] = f"FAIL: {e}"

    # Test 5: Symlog scale
    print("  Test 5: Symmetric log scale...")
    try:
        symlog_strategy = PlotStrategy(
            transforms=[],
            x_scale=AxisScale.Log,
            y_scale=AxisScale.Symlog,
            y_label_suffix=" [symlog]",
        )
        plot_all(
            results=results,
            learning_rates=learning_rates,
            optimizers=list(optimizers.keys()),
            experiment_name="check_engine/test_5_symlog",
            strategy_overrides={Metric.Loss: symlog_strategy},
        )
        print("    ✓ Symlog scale")
        test_results["symlog"] = "PASS"
    except Exception as e:
        print(f"    ✗ Symlog scale: {e}")
        test_results["symlog"] = f"FAIL: {e}"

    return test_results


def main():
    """Run all integration tests."""
    print("🔍 COMPREHENSIVE INTEGRATION TEST")
    print("Testing all optimizers with all model types\n")

    all_passed = True

    # Test basic imports
    import_failures = test_basic_imports()
    if import_failures:
        print("\n❌ Basic imports failed - skipping remaining tests")
        for module, error in import_failures:
            print(f"   {module}: {error}")
        return False

    # Test linear model
    linear_results = test_linear_model_optimizers()
    linear_failures = [k for k, v in linear_results.items() if v != "PASS"]

    # Test two-layer model
    twolayer_results = test_twolayer_model_optimizers()
    twolayer_failures = [k for k, v in twolayer_results.items() if v != "PASS"]

    # Test plotting system
    plot_results = test_plotting_system()
    plot_failures = [k for k, v in plot_results.items() if v != "PASS"]

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_linear = len(linear_results)
    total_twolayer = len(twolayer_results)
    total_plot = len(plot_results)
    total_tests = total_linear + total_twolayer + total_plot

    passed_linear = total_linear - len(linear_failures)
    passed_twolayer = total_twolayer - len(twolayer_failures)
    passed_plot = total_plot - len(plot_failures)
    total_passed = passed_linear + passed_twolayer + passed_plot

    total_failures = len(linear_failures) + len(twolayer_failures) + len(plot_failures)

    print(f"Linear Model:     {passed_linear}/{total_linear} passed")
    print(f"Two-Layer Model:  {passed_twolayer}/{total_twolayer} passed")
    print(f"Plotting System:  {passed_plot}/{total_plot} passed")
    print(f"Total:            {total_passed}/{total_tests} passed")

    if total_failures > 0:
        print("\n❌ FAILURES:")
        for opt in linear_failures:
            print(f"   LinearModel + {opt.name}: {linear_results[opt]}")
        for opt in twolayer_failures:
            print(f"   TwoLayerModel + {opt.name}: {twolayer_results[opt]}")
        for test in plot_failures:
            print(f"   Plotting/{test}: {plot_results[test]}")
        all_passed = False
    else:
        print("\n✅ ALL TESTS PASSED!")
        print(f"\nPlots saved to: experiments/check_engine/")

    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
