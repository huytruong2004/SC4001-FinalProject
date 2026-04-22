"""Paired bootstrap significance test for per-example accuracy differences.

Two-sided p-value via resampling under H0 (true mean difference = 0):
draw B indices with replacement, compute mean(a[idx]) - mean(b[idx])
for each, centre the resample distribution on zero (subtract the observed
mean difference), and report the fraction of centred draws whose magnitude
meets or exceeds |observed|.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_pvalue(
    correct_a: np.ndarray,  # (N,) bool — model A correct per example
    correct_b: np.ndarray,  # (N,) bool — model B correct per example (same examples)
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Returns (mean_diff, two_sided_p). When observed == 0.0 returns (0.0, 1.0) without resampling."""
    assert correct_a.shape == correct_b.shape
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    N = correct_a.shape[0]
    rng = np.random.default_rng(seed)
    observed = float(correct_a.mean() - correct_b.mean())

    # Resample indices with replacement, compute mean_diff for each.
    idx = rng.integers(0, N, size=(n_resamples, N))
    diffs = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)

    if observed == 0.0:
        return 0.0, 1.0
    # Null-recentred two-sided p: how often does a resample centred on the
    # null (true diff = 0) exceed |observed| in magnitude?
    diffs_centred = diffs - observed
    p = float((np.abs(diffs_centred) >= abs(observed)).mean())
    return observed, p
