"""Unit tests for src/data.py — pure functions only."""
import numpy as np

from src.data import BG_RGB, trimap_to_binary


def test_trimap_to_binary_all_background():
    trimap = np.broadcast_to(BG_RGB, (10, 10, 3)).copy()
    mask = trimap_to_binary(trimap)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert mask.sum() == 0


def test_trimap_to_binary_all_foreground():
    trimap = np.full((10, 10, 3), 255, dtype=np.uint8)
    mask = trimap_to_binary(trimap)
    assert mask.sum() == 100


def test_trimap_to_binary_mixed():
    trimap = np.broadcast_to(BG_RGB, (4, 4, 3)).copy()
    trimap[1:3, 1:3] = [100, 50, 25]  # 2x2 foreground
    mask = trimap_to_binary(trimap)
    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)
    assert np.array_equal(mask, expected)


def test_trimap_to_binary_tolerant_to_near_blue():
    # trimap values very close to blue should still count as background.
    trimap = np.broadcast_to(BG_RGB, (3, 3, 3)).copy()
    trimap[1, 1] = [2, 3, 253]  # within threshold
    mask = trimap_to_binary(trimap)
    assert mask[1, 1] == 0
