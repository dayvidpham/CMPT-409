#!/usr/bin/env python3
"""
Generate reproducible datasets from a seed and save in structured directories.

Usage:
    python utils/generate_data.py --seed 42
    python utils/generate_data.py --seed 42 --n 200 --d 5000
    python utils/generate_data.py --load experiments/generate_data/seed_42
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work from anywhere
script_dir = Path(__file__).resolve().parent  # utils/
project_root = script_dir.parent              # project root
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch

from engine import make_soudry_dataset, split_train_test, DatasetSplit


def generate_data(
    seed: int,
    n: int = 200,
    d: int = 5000,
    margin: float = 1.0,
    sigma: float = 0.3,
    test_size: float = 0.2,
    device: str = "cpu",
    base_dir: str = None,
    verbose: bool = True
):
    """
    Generate dataset from seed and save train/test splits in separate files.

    Args:
        seed: Random seed for reproducibility
        n: Number of samples
        d: Dimension
        margin: Separation margin
        sigma: Noise level
        test_size: Fraction of data for test set
        device: PyTorch device ('cpu' or 'cuda')
        base_dir: Base directory for saving data (default: project_root/experiments/generate_data)
        verbose: Print information about generated data
    """
    # Set default base_dir relative to project root
    if base_dir is None:
        base_dir = str(project_root / "experiments" / "generate_data")

    if verbose:
        print(f"Generating data with seed: {seed}")
        print(f"Parameters: n={n}, d={d}, margin={margin}, sigma={sigma}")
        print(f"Test split: {int(test_size*100)}%")

    # Seed PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Create NumPy RNG from seed
    rng = np.random.default_rng(seed)

    # Generate dataset
    X, y, _ = make_soudry_dataset(
        n=n,
        d=d,
        margin=margin,
        sigma=sigma,
        device=device,
        rng=rng
    )

    # Split into train/test
    datasets = split_train_test(X, y, test_size=test_size, rng=rng)
    X_train, y_train = datasets[DatasetSplit.Train]
    X_test, y_test = datasets[DatasetSplit.Test]

    # Create output directory
    output_dir = Path(base_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train split
    train_path = output_dir / "train.npz"
    np.savez(
        train_path,
        X=X_train.cpu().numpy(),
        y=y_train.cpu().numpy()
    )

    # Save test split
    test_path = output_dir / "test.npz"
    np.savez(
        test_path,
        X=X_test.cpu().numpy(),
        y=y_test.cpu().numpy()
    )

    # Save metadata (all generation parameters)
    metadata_path = output_dir / "metadata.npz"
    np.savez(
        metadata_path,
        seed=seed,
        n=n,
        d=d,
        margin=margin,
        sigma=sigma,
        test_size=test_size
    )

    if verbose:
        print(f"\nData saved to: {output_dir.absolute()}")
        print(f"  - train.npz:    {train_path.stat().st_size / 1024:.2f} KB")
        print(f"  - test.npz:     {test_path.stat().st_size / 1024:.2f} KB")
        print(f"  - metadata.npz: {metadata_path.stat().st_size / 1024:.2f} KB")
        print(f"\nSplit sizes:")
        print(f"  Train: {X_train.shape[0]} samples, shape {X_train.shape}")
        print(f"  Test:  {X_test.shape[0]} samples, shape {X_test.shape}")

    return output_dir


def load_data(directory: str, verbose: bool = True):
    """
    Load previously generated data from directory.

    Args:
        directory: Path to data directory (e.g., experiments/generate_data/seed_42)
        verbose: Print information about loaded data

    Returns:
        Dictionary containing train, test, and metadata
    """
    data_dir = Path(directory)

    if not data_dir.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")

    # Load files
    train_data = np.load(data_dir / "train.npz")
    test_data = np.load(data_dir / "test.npz")
    metadata = np.load(data_dir / "metadata.npz")

    if verbose:
        print(f"Loaded data from: {data_dir.absolute()}")
        print(f"Seed: {metadata['seed']}")
        print(f"Parameters: n={metadata['n']}, d={metadata['d']}, "
              f"margin={metadata['margin']}, sigma={metadata['sigma']}")
        print(f"\nSplit sizes:")
        print(f"  Train: {train_data['X'].shape[0]} samples, shape {train_data['X'].shape}")
        print(f"  Test:  {test_data['X'].shape[0]} samples, shape {test_data['X'].shape}")

    return {
        'train': {key: train_data[key] for key in train_data.files},
        'test': {key: test_data[key] for key in test_data.files},
        'metadata': {key: metadata[key] for key in metadata.files}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible datasets from seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data with seed 42
  python generate_data.py --seed 42

  # Generate with custom parameters
  python generate_data.py --seed 123 --n 500 --d 10000

  # Use custom base directory
  python generate_data.py --seed 42 --base-dir my_experiments

  # Load previously generated data
  python generate_data.py --load experiments/generate_data/seed_42

Output structure:
  experiments/generate_data/seed_{seed}/
    - train.npz     (X_train, y_train)
    - test.npz      (X_test, y_test)
    - metadata.npz  (seed, n, d, margin, sigma, test_size)
        """
    )

    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--seed', type=int, help='Random seed for data generation')
    action_group.add_argument('--load', type=str, help='Load and display data from directory')

    # Generation parameters
    parser.add_argument('--n', type=int, default=200, help='Number of samples (default: 200)')
    parser.add_argument('--d', type=int, default=5000, help='Dimension (default: 5000)')
    parser.add_argument('--margin', type=float, default=1.0, help='Separation margin (default: 1.0)')
    parser.add_argument('--sigma', type=float, default=0.3, help='Noise level (default: 0.3)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction (default: 0.2)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for tensor operations (default: cpu)')
    parser.add_argument('--base-dir', type=str, default='experiments/generate_data',
                       help='Base directory for saving data (default: experiments/generate_data)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output messages')

    args = parser.parse_args()

    verbose = not args.quiet

    if args.load:
        # Load existing data
        load_data(args.load, verbose=verbose)
    else:
        # Generate new data
        generate_data(
            seed=args.seed,
            n=args.n,
            d=args.d,
            margin=args.margin,
            sigma=args.sigma,
            test_size=args.test_size,
            device=args.device,
            base_dir=args.base_dir,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
