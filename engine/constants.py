"""
Central location for numerical stability constants.

These constants are used consistently across optimizers, losses, and metrics.
"""

# Machine epsilon for float64 (Double Precision)
# This is the smallest representable positive number where 1.0 + EPS != 1.0
EPS = 2.2e-16

# Gradient/Loss tolerance for numerical stability
# Below this threshold, gradients and losses are treated as effectively zero.
#
# Why 1e-140?
# - float64 minimum positive normal: ~2.2e-308
# - With exponential loss: L(w) ≈ exp(-margin), loss decreases exponentially
# - At iteration t, margin ≈ log(t), so loss ≈ 1/t
# - 1e-140 allows margin ≈ 322, corresponding to ~10^140 iterations
# - This is safe: most training terminates long before hitting this limit
GRAD_TOL = 1e-140

# Clamping bounds for exponential loss gradient computation
# Prevents exp() overflow/underflow:
# - exp(710) ≈ inf (float64 overflow at ~709.78)
# - exp(-745) ≈ 0 (float64 underflow at ~-744.44)
# - We use ±300 for safety margin, giving exp(±300) ≈ finite values
CLAMP_MIN = -300
CLAMP_MAX = 300
