#!/usr/bin/env python3
"""
Test script to verify that generate_data.py produces identical data
to the method used in run_gd_soudry_repeated.py
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work from anywhere
script_dir = Path(__file__).resolve().parent  # utils/
project_root = script_dir.parent              # project root
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from engine import make_soudry_dataset, split_train_test, DatasetSplit


def test_generated_data(seed: int, base_dir: str | None = None):
    """
    Verify that saved data matches the experiment generation method.

    Args:
        seed: Seed to test
        base_dir: Base directory containing seed subdirectories (default: project_root/experiments/generate_data)
    """
    # Set default base_dir relative to project root
    if base_dir is None:
        base_dir = str(project_root / "experiments" / "generate_data")

    data_path = Path(base_dir) / f"seed_{seed}"

    if not data_path.exists():
        print(f"❌ Data directory does not exist: {data_path}")
        return False

    print(f"Testing data for seed: {seed}")
    print(f"Data directory: {data_path.absolute()}")
    print("=" * 70)

    # Generate using experiment method (from run_gd_soudry_repeated.py)
    torch.manual_seed(seed)
    run_rng = np.random.default_rng(seed)
    X, y, _ = make_soudry_dataset(n=200, d=5000, device='cpu', rng=run_rng)
    datasets = split_train_test(X, y, test_size=0.2, rng=run_rng)

    X_train_exp, y_train_exp = datasets[DatasetSplit.Train]
    X_test_exp, y_test_exp = datasets[DatasetSplit.Test]

    # Load from generated files
    train_data = np.load(data_path / "train.npz")
    test_data = np.load(data_path / "test.npz")
    metadata = np.load(data_path / "metadata.npz")

    X_train_gen = train_data['X']
    y_train_gen = train_data['y']
    X_test_gen = test_data['X']
    y_test_gen = test_data['y']

    # Verify metadata
    print("\nMetadata verification:")
    print(f"  Seed in metadata: {metadata['seed']}")
    print(f"  Expected seed:    {seed}")
    metadata_match = int(metadata['seed']) == seed
    print(f"  Seed matches: {'✓' if metadata_match else '✗'}")

    # Compare data
    print("\nData comparison:")
    train_x_match = np.allclose(X_train_gen, X_train_exp.cpu().numpy())
    train_y_match = np.allclose(y_train_gen, y_train_exp.cpu().numpy())
    test_x_match = np.allclose(X_test_gen, X_test_exp.cpu().numpy())
    test_y_match = np.allclose(y_test_gen, y_test_exp.cpu().numpy())

    print(f"  Train X matches: {'✓' if train_x_match else '✗'}")
    print(f"  Train y matches: {'✓' if train_y_match else '✗'}")
    print(f"  Test X matches:  {'✓' if test_x_match else '✗'}")
    print(f"  Test y matches:  {'✓' if test_y_match else '✗'}")

    # Check shapes
    print("\nShape verification:")
    print(f"  Train: {X_train_gen.shape} (generated) vs {X_train_exp.shape} (expected)")
    print(f"  Test:  {X_test_gen.shape} (generated) vs {X_test_exp.shape} (expected)")

    # Overall result
    all_match = all([metadata_match, train_x_match, train_y_match, test_x_match, test_y_match])

    print("\n" + "=" * 70)
    if all_match:
        print("✓ SUCCESS: Generated data is IDENTICAL to experiment method!")
    else:
        print("✗ FAILURE: Data mismatch detected!")

        if not all([train_x_match, train_y_match, test_x_match, test_y_match]):
            print("\nDifferences:")
            if not train_x_match:
                diff = np.abs(X_train_gen - X_train_exp.cpu().numpy()).max()
                print(f"  Max difference in train X: {diff}")
            if not train_y_match:
                diff = np.abs(y_train_gen - y_train_exp.cpu().numpy()).max()
                print(f"  Max difference in train y: {diff}")
            if not test_x_match:
                diff = np.abs(X_test_gen - X_test_exp.cpu().numpy()).max()
                print(f"  Max difference in test X: {diff}")
            if not test_y_match:
                diff = np.abs(y_test_gen - y_test_exp.cpu().numpy()).max()
                print(f"  Max difference in test y: {diff}")

    return all_match


def test_multiple_seeds(seeds: list[int], base_dir: str = None):
    """
    Test multiple seeds to ensure consistency.

    Args:
        seeds: List of seeds to test
        base_dir: Base directory containing seed subdirectories (default: project_root/experiments/generate_data)
    """
    # Set default base_dir relative to project root
    if base_dir is None:
        base_dir = str(project_root / "experiments" / "generate_data")
    print("\n" + "=" * 70)
    print("Testing multiple seeds")
    print("=" * 70)

    results = []
    for seed in seeds:
        print()
        passed = test_generated_data(seed, base_dir=base_dir)
        results.append((seed, passed))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for seed, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Seed {seed}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")

    return all_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test generated data against experiment method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single seed
  python test_generate_data.py --seed 42

  # Test multiple seeds
  python test_generate_data.py --seeds 42 123 456

  # Test with custom base directory
  python test_generate_data.py --seed 42 --base-dir my_experiments
        """
    )

    # Action group
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--seed', type=int, help='Test a single seed')
    action_group.add_argument('--seeds', type=int, nargs='+', help='Test multiple seeds')

    parser.add_argument('--base-dir', type=str, default='experiments/generate_data',
                       help='Base directory containing seed subdirectories (default: experiments/generate_data)')

    args = parser.parse_args()

    if args.seed:
        # Test single seed
        test_generated_data(args.seed, base_dir=args.base_dir)
    else:
        # Test multiple seeds
        test_multiple_seeds(args.seeds, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
