"""Unit tests for src/bootstrap.py."""
import numpy as np
import torch

from src.bootstrap import paired_bootstrap_pvalue


def test_paired_bootstrap_no_difference():
    """Identical predictions -> p-value should be ~0.5 or undefined but in [0,1]."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 100).astype(bool)  # correctness of model A
    b = a.copy()
    mean_diff, p = paired_bootstrap_pvalue(a, b, n_resamples=1000, seed=0)
    assert mean_diff == 0.0
    assert 0.0 <= p <= 1.0


def test_paired_bootstrap_strong_difference():
    """A correct on every sample, B wrong on every sample -> p < 0.01."""
    a = np.ones(200, dtype=bool)
    b = np.zeros(200, dtype=bool)
    mean_diff, p = paired_bootstrap_pvalue(a, b, n_resamples=1000, seed=0)
    assert mean_diff == 1.0
    assert p < 0.01
