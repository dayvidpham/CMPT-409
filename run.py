"""Configurable runner script that takes CLI arguments and retrieves experiment configuration.

Usage:
    # Using presets (other args optional):
    python run.py --preset soudry_gd
    python run.py --preset soudry_sgd
    python run.py --preset adam_gd --output custom_name
    python run.py --preset adam_sgd

    # Manual configuration:
    python run.py --optimizer-family gd --model linear --output my_experiment --loss logistic --deterministic True
    python run.py --optimizer-family adaptive --model twolayer --output my_experiment --loss logistic --deterministic False
"""

import argparse
import torch
import os

from engine import (
    run_training,
    split_train_test,
    get_empirical_max_margin,
)
from engine.losses import LogisticLoss, ExponentialLoss
from engine.default_run_params import (
    default_dataset_generation,
    default_metrics,
    default_model_linear,
    default_model_twolayer,
    default_optimizer_family_gd,
    default_optimizer_family_adaptive,
    default_deterministic_run,
    default_stochastic_run,
    default_plot_options,
)
from engine.plotting import plot_all
from engine import make_soudry_dataset


# Preset configurations mapping to existing scripts
PRESETS = {
    "soudry_gd": {
        "optimizer_family": "gd",
        "model": "linear",
        "loss": "logistic",
        "deterministic": "True",
        "default_output": "soudry_gd",
    },
    "soudry_sgd": {
        "optimizer_family": "gd",
        "model": "linear",
        "loss": "logistic",
        "deterministic": "False",
        "default_output": "soudry_sgd",
    },
    "adam_gd": {
        "optimizer_family": "adaptive",
        "model": "twolayer",
        "loss": "logistic",
        "deterministic": "True",
        "default_output": "adam_gd",
    },
    "adam_sgd": {
        "optimizer_family": "adaptive",
        "model": "twolayer",
        "loss": "logistic",
        "deterministic": "False",
        "default_output": "adam_sgd",
    },
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configurable experiment runner with CLI arguments or presets"
    )

    # Preset argument
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Preset configuration: 'soudry_gd', 'soudry_sgd', 'adam_gd', or 'adam_sgd'. "
        "When used, other arguments become optional (can override defaults).",
    )

    # Manual configuration arguments (optional if preset is used)
    parser.add_argument(
        "--optimizer-family",
        type=str,
        choices=["gd", "adaptive"],
        help="Optimizer family to use: 'gd' (GD, SAM, LossNGD, VecNGD, etc.) or 'adaptive' (Adam, AdaGrad, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "twolayer"],
        help="Model architecture: 'linear' or 'twolayer'",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Experiment name for output/plotting",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["logistic", "exponential"],
        help="Loss function: 'logistic' or 'exponential'",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        choices=["True", "False"],
        help="Whether to use deterministic (full-batch) training: 'True' or 'False'",
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Number of iterations/epochs to run (default: 10_000)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable debug output during training",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.preset:
        # Manual mode: all arguments required
        required_args = [
            ("optimizer_family", "--optimizer-family"),
            ("model", "--model"),
            ("output", "--output"),
            ("loss", "--loss"),
            ("deterministic", "--deterministic"),
        ]
        missing = [arg for arg, arg_flag in required_args if not getattr(args, arg)]
        if missing:
            parser.error(
                f"Either --preset must be specified, or all of these arguments are required: "
                f"{', '.join([arg_flag for arg, arg_flag in required_args if arg in missing])}"
            )

    return args


def apply_preset(args):
    """Apply preset configuration to args, allowing overrides.

    Args:
        args: Parsed command-line arguments

    Returns:
        Updated args with preset values filled in (respecting explicit overrides)
    """
    if args.preset:
        preset = PRESETS[args.preset]
        # Set defaults from preset, but allow explicit arguments to override
        if not args.optimizer_family:
            args.optimizer_family = preset["optimizer_family"]
        if not args.model:
            args.model = preset["model"]
        if not args.loss:
            args.loss = preset["loss"]
        if not args.deterministic:
            args.deterministic = preset["deterministic"]
        if not args.output:
            args.output = preset["default_output"]
    return args


def get_config(args):
    """Retrieve configuration from default_run_params based on CLI arguments.

    Args:
        args: Parsed command-line arguments (with preset already applied if used)

    Returns:
        Dictionary with all configuration needed to run the experiment
    """
    # Configure PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine loss function
    if args.loss == "logistic":
        loss_fn = LogisticLoss()
    else:
        loss_fn = ExponentialLoss()

    # Determine model factory
    if args.model == "linear":
        model_factory = default_model_linear(input_dim=5000, device=device)
        use_manual = False
    else:  # twolayer
        model_factory = default_model_twolayer(
            input_dim=5000, hidden_dim=50, device=device
        )
        use_manual = True

    # Get dataset generation config
    dataset_config = default_dataset_generation(
        n=200, d=5000, margin=1.0, test_size=40, device=device
    )

    # Get optimizer configuration
    if args.optimizer_family == "gd":
        optimizer_factories, sweeps = default_optimizer_family_gd(
            model=None, loss_fn=loss_fn, use_manual=use_manual
        )
    else:  # adaptive
        optimizer_factories, sweeps = default_optimizer_family_adaptive(
            model=None, loss_fn=loss_fn
        )

    # Expand sweep grid
    from engine import expand_sweep_grid

    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    # Get dataset - filter out test_size for make_soudry_dataset
    dataset_params = {k: v for k, v in dataset_config.items() if k != "test_size"}
    X, y, v_pop = make_soudry_dataset(**dataset_params)
    w_star = get_empirical_max_margin(X, y)
    datasets = split_train_test(X, y, test_size=dataset_config["test_size"])

    # Get metrics factory
    metrics_factory = default_metrics(w_star, loss_fn)

    # Determine run type (deterministic vs stochastic)
    deterministic = args.deterministic == "True"
    iters = args.iters if args.iters else 10_000
    debug = not args.quiet
    if deterministic:
        run_config = default_deterministic_run(total_iters=iters, debug=debug)
    else:
        run_config = default_stochastic_run(
            num_epochs=iters, batch_size=32, debug=debug
        )

    # Get plotting config
    plot_config = default_plot_options(experiment_name=args.output)

    return {
        "datasets": datasets,
        "model_factory": model_factory,
        "optimizer_configs": optimizer_configs,
        "metrics_factory": metrics_factory,
        "run_config": run_config,
        "plot_config": plot_config,
        "device": device,
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Show preset info if used
    if args.preset:
        print(f"Using preset: {args.preset}")
        print()

    # Apply preset to get final configuration
    args = apply_preset(args)

    print(f"Configuration:")
    print(f"  Optimizer family: {args.optimizer_family}")
    print(f"  Model: {args.model}")
    print(f"  Loss function: {args.loss}")
    print(f"  Experiment name: {args.output}")
    print(f"  Deterministic: {args.deterministic}")
    if args.iters:
        print(f"  Iterations/epochs: {args.iters}")
    if args.quiet:
        print(f"  Debug mode: False (quiet)")
    print()

    # Get configuration from default_run_params
    config = get_config(args)

    print("Configuration retrieved successfully!")
    print(f"  Number of configs: {len(config['optimizer_configs'])}")
    print()

    # Run training
    print("Running training...")
    results = run_training(
        datasets=config["datasets"],
        model_factory=config["model_factory"],
        optimizers=config["optimizer_configs"],
        metrics_collector_factory=config["metrics_factory"],
        **config["run_config"],
    )

    # Plot results
    print("Generating plots...")
    plot_all(results, **config["plot_config"])
    print("Done!")


if __name__ == "__main__":
    main()
