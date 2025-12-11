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
