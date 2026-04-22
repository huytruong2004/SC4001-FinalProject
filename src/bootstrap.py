"""Paired bootstrap significance test for per-example accuracy differences.

Two-sided p-value via resampling: draw B indices with replacement, compute
mean(a[idx]) - mean(b[idx]). The p-value is the fraction of resamples whose
sign is opposite to the observed mean difference, times 2 (two-sided).
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_pvalue(
    correct_a: np.ndarray,  # (N,) bool — model A correct per example
    correct_b: np.ndarray,  # (N,) bool — model B correct per example (same examples)
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Returns (mean_diff, two_sided_p)."""
    assert correct_a.shape == correct_b.shape
    N = correct_a.shape[0]
    rng = np.random.default_rng(seed)
    observed = float(correct_a.mean() - correct_b.mean())

    # Resample indices with replacement, compute mean_diff for each.
    idx = rng.integers(0, N, size=(n_resamples, N))
    diffs = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)

    if observed == 0.0:
        return 0.0, 1.0
    # Two-sided: how often does the resampled diff have the opposite sign of observed?
    if observed > 0:
        tail = (diffs <= 0).mean()
    else:
        tail = (diffs >= 0).mean()
    p = 2.0 * tail
    return observed, min(p, 1.0)
