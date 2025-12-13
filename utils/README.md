# Utilities

This folder contains utility scripts for testing and experimentation.

## Scripts

### `check_compilation.py`
Checks Python syntax compilation for all engine modules.

```bash
python utils/check_compilation.py
```

**What it does:**
- Compiles all Python files in `engine/` using `py_compile`
- Catches syntax errors before runtime
- Returns exit code 0 if all files compile, 1 if any fail

### `check_imports.py`
Comprehensive integration test that runs every optimizer with every model type.

```bash
python utils/check_imports.py
```

**What it does:**
1. Tests basic module imports
2. Runs 1 iteration of every optimizer with LinearModel
3. Runs 1 iteration of every optimizer with TwoLayerModel
4. Tests the plotting system with 5 different strategy configurations:
   - Default strategies
   - Custom error strategy (log-log)
   - Clamped loss
   - Multiple strategy overrides
   - Symmetric log scale

**Output:**
- Test results printed to console
- Test plots saved to `experiments/check_engine/test_X/`

**Exit codes:**
- 0: All tests passed
- 1: One or more tests failed

### `strategy_scratchpad.py`
Interactive scratchpad for experimenting with custom plotting strategies.

```bash
python utils/strategy_scratchpad.py
```

**What it does:**
- Generates small dataset (n=200, d=2) with `make_soudry_dataset()`
- Runs short training (1000 iterations, 2 learning rates)
- Executes your custom experiment (modify `custom_experiment()` function)
- Optionally shows example strategies

**Use cases:**
- Quick iteration on new plotting strategies
- Testing different axis scales and transforms
- Prototyping visualizations before adding to main experiments

**Customization:**
Edit the `custom_experiment()` function to try your own strategies:

```python
def custom_experiment(results, learning_rates, optimizers):
    my_strategy = PlotStrategy(
        transforms=[SafeLog(), Scale(100.0)],
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Log,
        y_label_suffix=" (my custom label)",
    )

    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(optimizers.keys()),
        experiment_name="scratchpad/my_experiment",
        strategy_overrides={Metric.Error: my_strategy},
    )
```

## Quick Start

Run all checks before committing:

```bash
# Check compilation
python utils/check_compilation.py

# Run full integration tests (includes plotting)
python utils/check_imports.py

# Experiment with strategies
python utils/strategy_scratchpad.py
```

## When to Use

- **Before committing**: Run `check_compilation.py` and `check_imports.py`
- **After refactoring**: Run `check_imports.py` to ensure nothing broke
- **Developing new features**: Use `strategy_scratchpad.py` for rapid iteration
- **Adding new optimizers**: Check that `check_imports.py` passes
- **Adding new metrics**: Update strategies in `engine/types.py` and verify with scratchpad
