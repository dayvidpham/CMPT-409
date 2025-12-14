import numpy as np
import torch
from .types import DatasetSplit
from typing import Dict, Tuple, Optional


def make_soudry_dataset(
    n: int = 200,
    d: int = 5000,
    margin: float = 1,
    sigma: float = 0.3,
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
):
    """Generate Soudry-style linearly separable dataset using specific RNG.

    Args:
        n: Number of samples
        d: Dimension
        margin: Separation margin
        sigma: Noise level
        device: PyTorch device
        rng: numpy.random.Generator instance (optional). If None, creates a new default_rng.
    """
    if rng is None:
        rng = np.random.default_rng()

    v = np.ones(d) / np.sqrt(d)

    # Use new RNG interface: standard_normal replaces randn
    # shape (n//2, d)
    noise_pos = rng.standard_normal((n // 2, d)) * sigma
    noise_neg = rng.standard_normal((n // 2, d)) * sigma

    X_pos = margin * v[None, :] + noise_pos
    X_neg = -margin * v[None, :] + noise_neg

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])

    # Shuffle using new RNG
    idx = rng.permutation(n)

    # Convert to Torch tensors
    X_torch = torch.tensor(X[idx], dtype=torch.float64, device=device)
    y_torch = torch.tensor(y[idx], dtype=torch.float64, device=device)
    v_torch = torch.tensor(v, dtype=torch.float64, device=device)

    # Normalize by the maximum L2 norm across all data points
    max_norm = torch.linalg.norm(X_torch, dim=1).max()
    if max_norm > 0:
        X_rescaled = X_torch / max_norm
    else:
        X_rescaled = X_torch

    return X_rescaled, y_torch, v_torch


def split_train_test(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float | int = 0.2,
    rng: Optional[np.random.Generator] = None,
    random_state: Optional[int] = None,  # Kept for backward compat, used if rng is None
) -> Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split dataset using the provided RNG (avoids sklearn legacy constraints).
    """
    if rng is None:
        # Fallback to creating a generator from random_state or entropy
        rng = np.random.default_rng(random_state)

    n = X.shape[0]
    if isinstance(test_size, float):
        assert not np.isclose(0.0, test_size) and not np.isclose(1.0, test_size), (
            f"Expected: test_size must be in range 0.0 < test_size < 1.0, got: test_size is {test_size}"
        )
        n_test = int(n * test_size)
    elif isinstance(test_size, int):
        assert test_size > 0 and test_size < n, (
            f"test_size {test_size} needs to be >0 and <n, total dataset size {n}"
        )
        n_test = test_size

    # Manual shuffle with the generator to support 64-bit seeds implicitly via rng state
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    # Use torch indexing
    # Note: X, y are already on device, slicing keeps them there
    return {
        DatasetSplit.Train: (X[train_idx], y[train_idx]),
        DatasetSplit.Test: (X[test_idx], y[test_idx]),
    }
