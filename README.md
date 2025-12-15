## Overview

This repository provides a complete experimental framework for studying implicit bias and margin dynamics of linear classifiers trained on separable data, following the setup of Soudry et al. (2018).
It supports both classic first-order optimizers and modern adaptive methods, including SAM variants.

## Setup

### Option 1: Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is Python's environment manager that supports official new standards. If you have `uv` installed:

```bash
# Creates virtual environment, syncs dependencies
uv sync
```

If you don't have `uv` installed, you can install it via:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Option 2: Using Traditional `venv`

If you prefer the standard Python virtual environment approach:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start: Configurable Runner (`run.py`)

The easiest way to run experiments is using the `run.py` script with presets:

```bash
# Using presets (recommended)
python run.py --preset soudry_gd       # Linear model, GD family, deterministic
python run.py --preset soudry_sgd      # Linear model, GD family, stochastic
python run.py --preset adam_gd         # Two-layer model, adaptive optimizers, deterministic
python run.py --preset adam_sgd        # Two-layer model, adaptive optimizers, stochastic

# Override preset output name
python run.py --preset soudry_gd --output custom_experiment_name
```

### `run.py` CLI Arguments

#### Using Presets (Other args optional):

```
--preset {soudry_gd, soudry_sgd, adam_gd, adam_sgd}
    Preset configuration that sets all other parameters to defaults.
    Other arguments can still override preset defaults.
```

#### Manual Configuration (All args required if no preset):

```
--optimizer-family {gd, adaptive}
    Optimizer family to use:
    - gd: Gradient Descent, Natural Gradient, SAM variants
    - adaptive: Adam, AdaGrad, SAM variants

--model {linear, twolayer}
    Model architecture:
    - linear: LinearModel (good for Soudry dataset)
    - twolayer: TwoLayerModel (for more complex patterns)

--output EXPERIMENT_NAME
    Name for the experiment (used for plotting and output directories)

--loss {logistic, exponential}
    Loss function to use

--deterministic {True, False}
    Training mode:
    - True: Full-batch deterministic training
    - False: Mini-batch stochastic training (SGD)
```

#### Examples:

```bash
# Using presets
python run.py --preset soudry_gd
python run.py --preset adam_sgd --output my_custom_name

# Manual configuration
python run.py --optimizer-family gd --model linear --output exp1 --loss logistic --deterministic True
python run.py --optimizer-family adaptive --model twolayer --output exp2 --loss logistic --deterministic False
```

### Default Values

All default hyperparameters, model configurations, loss functions, and run parameters are centralized in:

```
engine/default_run_params.py
```

Key defaults include:

- **Learning Rates**: `[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]` (7 values)
- **Rho Values** (SAM hyperparameter): `[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]` (6 values)
- **Dataset**: 200 samples, 5000 dimensions, 40 test samples
- **Loss Function**: LogisticLoss (used by all presets)
- **Deterministic Training**: 10,000 iterations (full-batch)
- **Stochastic Training**: 10,000 epochs, batch size 32

To customize defaults, modify `engine/default_run_params.py` directly or override via CLI arguments in `run.py`.

## Running the Core Experiments (GD / NGD / SAM / SAM–NGD)

To run the main Soudry-style experiment:

```bash
python run_gd_soudry.py
```
### What this script does

- Generates a synthetic **Soudry dataset** (two Gaussian clusters with margin separation)
- Computes the **empirical max-margin solution** using Linear SVM
- Trains using the following optimizers:
  - Gradient Descent (**GD**)
  - Natural Gradient Descent (**NGD**)
  - Sharpness-Aware Minimization (**SAM**)
  - **SAM + NGD**
- Logs metrics at **log-spaced iterations**

- Saves all results under:
```
experiments/<sourdry_GD>/<timestamp>/
```


## Running Adaptive Optimizer Experiments (Adam / AdaGrad / SAM Variants)

To run the adaptive optimizer suite:

```bash
python run_adam_family_soudry.py
```

- Generates the same Soudry-style dataset and evaluation pipeline
- Trains using the following adaptive optimizers:
  - **Adam**
  - **AdaGrad**
  - **Adam + SAM**
  - **AdaGrad + SAM**
- Records all metrics (loss, angle, distance, error) at **log-spaced intervals**
- Saves results using the same directory structure as the GD-family experiment
experiments/<sourdry_Adam_family>/<timestamp>/


4. Output Directory Structure

Each run produces the following hierarchy:

experiments/<experiment_name>/<timestamp>/
│
├── combined/
│     ├── distance.png
│     ├── angle.png
│     ├── loss.png
│     └── error.png
│
├── separate/
│     ├── optimizer_1/
│     │     ├── distance.png
│     │     ├── angle.png
│     │     ├── loss.png
│     │     └── error.png
│     ├── optimizer_2/
│     ├── optimizer_3/
│     └── optimizer_4/
│
└── results.npz     # contains all logged metrics for all optimizers and learning rates

5. Re-Generating Plots After Training

You can reconstruct all plots at any time using:

python test_plot.py --exp experiments/<experiment_name>/<timestamp>/

What this script does

Loads results.npz

Rebuilds all metric curves (distance, angle, loss, error)

Saves new plots to:

experiments/<experiment_name>/<timestamp>/test/plots/
    ├── combined/*.png
    └── separate/<optimizer>/*.png
