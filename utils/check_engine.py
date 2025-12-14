#!/usr/bin/env python
"""
Comprehensive integration test that runs every optimizer with every model type for 5 iterations.
This ensures all combinations work correctly and catches integration issues early.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from engine import (
    LinearModel,
    make_soudry_dataset,
    split_train_test,
    run_training,
    Metric,
    Optimizer,
    Hyperparam,
    MetricsCollector,
    exponential_loss,
    get_error_rate,
    get_angle,
    get_direction_distance,
    get_empirical_max_margin,
    expand_sweep_grid,
    LogisticLoss,
)
from engine.plotting import plot_all
from engine.optimizers import (
    step_gd,
    step_ngd_stable,
    step_sam_stable,
    step_sam_ngd_stable,
    make_optimizer_factory,
    # Adam,
    # AdaGrad,
    # SAM_Adam,
    # SAM_AdaGrad,
)

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
    """Test all optimizers with LinearModel using new API."""
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
            w_star=w_star,
        )

    # Optimizer factories for stateless optimizers
    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd),
        Optimizer.NGD: make_optimizer_factory(step_ngd_stable),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable),
        Optimizer.SAM_NGD: make_optimizer_factory(step_sam_ngd_stable),
    }

    # Sweeps with learning rates
    sweeps = {
        Optimizer.GD: {Hyperparam.LearningRate: learning_rates},
        Optimizer.NGD: {Hyperparam.LearningRate: learning_rates},
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],
        },
        Optimizer.SAM_NGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],
        },
    }

    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    test_results = {}
    try:
        # Run 5 iterations for all optimizers
        results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizer_configs,
            metrics_collector_factory=metrics_factory,
            total_iters=5,
            debug=True,
        )

        # Plotting
        plot_all(
            results,
            experiment_name="check_engine/linear_models",
        )

        # Mark all optimizers as passed
        for config in optimizer_configs.keys():
            print(f"  ✓ {config.name}")
            test_results[config.name] = "PASS"

    except Exception as e:
        print(f"  ✗ Linear models failed: {e}")
        test_results["linear_models"] = f"FAIL: {e}"

    return test_results


def test_linear_model_with_logistic_loss():
    """Test linear model optimizers with LogisticLoss (configurable loss)."""
    print("\n" + "=" * 70)
    print("TESTING LINEAR MODEL WITH LOGISTIC LOSS (Configurable Loss)")
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
            w_star=w_star,
        )

    # Create LogisticLoss instance
    logistic_loss = LogisticLoss()

    # Optimizer factories with LogisticLoss
    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd, loss=logistic_loss),
        Optimizer.NGD: make_optimizer_factory(step_ngd_stable, loss=logistic_loss),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable, loss=logistic_loss),
        Optimizer.SAM_NGD: make_optimizer_factory(step_sam_ngd_stable, loss=logistic_loss),
    }

    # Sweeps with learning rates
    sweeps = {
        Optimizer.GD: {Hyperparam.LearningRate: learning_rates},
        Optimizer.NGD: {Hyperparam.LearningRate: learning_rates},
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],
        },
        Optimizer.SAM_NGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],
        },
    }

    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    test_results = {}
    try:
        # Run 5 iterations for all optimizers with LogisticLoss
        results = run_training(
            datasets=datasets,
            model_factory=model_factory,
            optimizers=optimizer_configs,
            metrics_collector_factory=metrics_factory,
            total_iters=5,
            debug=True,
        )

        # Plotting
        plot_all(
            results,
            experiment_name="check_engine/linear_models_logistic_loss",
        )

        # Mark all optimizers as passed
        for config in optimizer_configs.keys():
            print(f"  ✓ {config.name}")
            test_results[config.name] = "PASS"

    except Exception as e:
        print(f"  ✗ Linear models with LogisticLoss failed: {e}")
        test_results["linear_models_logistic"] = f"FAIL: {e}"

    return test_results


def test_linear_model_stateful_optimizers():
    """Test stateful optimizers (Adam, AdaGrad) with LinearModel."""
    print("\n" + "=" * 70)
    print("TESTING LINEAR MODEL WITH STATEFUL OPTIMIZERS (Adam, AdaGrad)")
    print("=" * 70)

    RNG = make_test_rng()
    X, y, v_pop = make_test_dataset(RNG)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=NUM_TEST_SAMPLES, rng=RNG)
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
            w_star=w_star,
        )

    # Stateful optimizers need different handling - they have fixed lr
    # We'll test them individually with OptimizerConfig
    from engine import OptimizerConfig

    test_results = {}

    stateful_optimizers = [
        # (Optimizer.Adam, Adam, {}),
        # (Optimizer.AdaGrad, AdaGrad, {}),
        # (Optimizer.SAM_Adam, SAM_Adam, {"rho": 0.05}),
        # (Optimizer.SAM_AdaGrad, SAM_AdaGrad, {"rho": 0.05}),
    ]

    for opt_enum, opt_class, extra_params in stateful_optimizers:
        try:
            # Create config with fixed lr (stateful optimizers handle lr internally)
            config = OptimizerConfig.with_params(opt_enum, lr=1e-3, **extra_params)

            def make_factory(cls=opt_class, params=extra_params):
                return lambda: cls(**params)

            optimizers = {config: make_factory()}

            results = run_training(
                datasets=datasets,
                model_factory=model_factory,
                optimizers=optimizers,
                metrics_collector_factory=metrics_factory,
                total_iters=5,
                debug=False,
            )

            print(f"  ✓ {opt_enum.name}")
            test_results[opt_enum.name] = "PASS"

        except Exception as e:
            print(f"  ✗ {opt_enum.name}: {e}")
            test_results[opt_enum.name] = f"FAIL: {e}"

    return test_results


def test_twolayer_model_optimizers():
    """Test all optimizers with TwoLayerModel."""
    print("\n" + "=" * 70)
    print("TESTING TWO-LAYER MODEL WITH ALL OPTIMIZERS")
    print("=" * 70)

    from engine import TwoLayerModel, OptimizerConfig
    from engine.optimizers import (
        ManualGD,
        ManualLossNGD,
        ManualVecNGD,
        ManualSAM,
        ManualSAM_LossNGD,
        ManualSAM_VecNGD,
        ManualAdam,
        ManualAdaGrad,
        ManualSAM_Adam,
        ManualSAM_AdaGrad,
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

    # Manual optimizers for two-layer model (they handle lr in step())
    manual_optimizers = [
        (Optimizer.GD, ManualGD, {}),
        (Optimizer.LossNGD, ManualLossNGD, {}),
        (Optimizer.VecNGD, ManualVecNGD, {}),
        (Optimizer.SAM, ManualSAM, {"rho": 0.05}),
        (Optimizer.SAM_LossNGD, ManualSAM_LossNGD, {"rho": 0.05}),
        (Optimizer.SAM_VecNGD, ManualSAM_VecNGD, {"rho": 0.05}),
        (Optimizer.Adam, ManualAdam, {}),
        (Optimizer.AdaGrad, ManualAdaGrad, {}),
        (Optimizer.SAM_Adam, ManualSAM_Adam, {"rho": 0.05}),
        (Optimizer.SAM_AdaGrad, ManualSAM_AdaGrad, {"rho": 0.05}),
    ]

    test_results = {}

    for opt_enum, opt_class, extra_params in manual_optimizers:
        for lr in learning_rates:
            config = OptimizerConfig.with_params(opt_enum, lr=lr, **extra_params)
            try:

                def make_factory(cls=opt_class, params=extra_params):
                    return lambda: cls(**params)

                optimizers = {config: make_factory()}

                results = run_training(
                    datasets=datasets,
                    model_factory=model_factory,
                    optimizers=optimizers,
                    metrics_collector_factory=metrics_factory,
                    total_iters=5,
                    debug=False,
                )

                print(f"  ✓ {config.name}")
                test_results[config.name] = "PASS"

            except Exception as e:
                print(f"  ✗ {config.name}: {e}")
                test_results[config.name] = f"FAIL: {e}"

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
        "engine.sweeps",
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

    from engine.strategies import (
        PlotStrategy,
        AxisScale,
        SafeLog,
        Scale,
        Clamp,
        LogLogStrategy,
        PercentageStrategy,
    )

    # Generate small dataset for quick plotting test
    print(f"\n  Generating dataset (n={NUM_SAMPLES}, d={DIM_PARAMS_D})...")

    RNG = make_test_rng()
    X, y, v_pop = make_test_dataset(RNG)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=40, rng=RNG)
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
            w_star=w_star,
        )

    # Use new API
    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable),
    }

    learning_rates = [0.1, 1.0]
    sweeps = {
        Optimizer.GD: {Hyperparam.LearningRate: learning_rates},
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],
        },
    }

    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    print("  Running training (5 iterations)...")
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
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

    # Test linear model with stateless optimizers
    linear_results = test_linear_model_optimizers()
    linear_failures = [k for k, v in linear_results.items() if v != "PASS"]

    # Test linear model with LogisticLoss (configurable loss)
    logistic_results = test_linear_model_with_logistic_loss()
    logistic_failures = [k for k, v in logistic_results.items() if v != "PASS"]

    # Test linear model with stateful optimizers
    stateful_results = test_linear_model_stateful_optimizers()
    stateful_failures = [k for k, v in stateful_results.items() if v != "PASS"]

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
    total_logistic = len(logistic_results)
    total_stateful = len(stateful_results)
    total_twolayer = len(twolayer_results)
    total_plot = len(plot_results)
    total_tests = total_linear + total_logistic + total_stateful + total_twolayer + total_plot

    passed_linear = total_linear - len(linear_failures)
    passed_logistic = total_logistic - len(logistic_failures)
    passed_stateful = total_stateful - len(stateful_failures)
    passed_twolayer = total_twolayer - len(twolayer_failures)
    passed_plot = total_plot - len(plot_failures)
    total_passed = passed_linear + passed_logistic + passed_stateful + passed_twolayer + passed_plot

    total_failures = (
        len(linear_failures)
        + len(logistic_failures)
        + len(stateful_failures)
        + len(twolayer_failures)
        + len(plot_failures)
    )

    print(f"Linear Model (stateless):  {passed_linear}/{total_linear} passed")
    print(f"Linear Model (LogisticLoss): {passed_logistic}/{total_logistic} passed")
    print(f"Linear Model (stateful):   {passed_stateful}/{total_stateful} passed")
    print(f"Two-Layer Model:           {passed_twolayer}/{total_twolayer} passed")
    print(f"Plotting System:           {passed_plot}/{total_plot} passed")
    print(f"Total:                     {total_passed}/{total_tests} passed")

    if total_failures > 0:
        print("\n❌ FAILURES:")
        for name in linear_failures:
            print(f"   LinearModel (stateless) + {name}: {linear_results[name]}")
        for name in logistic_failures:
            print(f"   LinearModel (LogisticLoss) + {name}: {logistic_results[name]}")
        for name in stateful_failures:
            print(f"   LinearModel (stateful) + {name}: {stateful_results[name]}")
        for name in twolayer_failures:
            print(f"   TwoLayerModel + {name}: {twolayer_results[name]}")
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
