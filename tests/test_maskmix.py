"""Unit tests for src/maskmix.py."""
import torch

from src.maskmix import maskmix_batch


def test_maskmix_output_shape():
    B, C, H, W = 4, 3, 8, 8
    x = torch.randn(B, C, H, W)
    m = torch.randint(0, 2, (B, 1, H, W)).float()
    y = torch.arange(B)
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=0)
    assert x_mix.shape == (B, C, H, W)
    assert m_mix.shape == (B, 1, H, W)
    assert y_mix.shape == (B,)


def test_maskmix_hard_label_no_interpolation():
    """When prob=1.0, every sample should be remixed and its label should
    equal the SOURCE sample's label (the one whose foreground we paste)."""
    torch.manual_seed(0)
    B, C, H, W = 4, 3, 8, 8
    x = torch.randn(B, C, H, W)
    m = torch.randint(0, 2, (B, 1, H, W)).float()
    y = torch.tensor([10, 20, 30, 40])
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=123)
    # Every output label must be one of the input labels (hard label, not a mix)
    for yy in y_mix.tolist():
        assert yy in {10, 20, 30, 40}


def test_maskmix_zero_prob_is_identity():
    B, C, H, W = 4, 3, 8, 8
    x = torch.randn(B, C, H, W)
    m = torch.randint(0, 2, (B, 1, H, W)).float()
    y = torch.arange(B)
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=0.0, seed=0)
    assert torch.allclose(x_mix, x)
    assert torch.equal(m_mix, m)
    assert torch.equal(y_mix, y)


def test_maskmix_composition_is_correct():
    """x_mix[i] where mask_A[i]=1 must equal x_A[i]; elsewhere x_B[i]."""
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 4, 4
    x = torch.zeros(B, C, H, W)
    x[0] = 1.0  # sample 0 is all 1s
    x[1] = 5.0  # sample 1 is all 5s
    m = torch.zeros(B, 1, H, W)
    m[0, 0, :2, :] = 1  # sample 0's foreground is the top half
    m[1, 0, :, :2] = 1  # sample 1's foreground is the left half (unused here)
    y = torch.tensor([100, 200])

    # Force pairing: sample 0 -> source, sample 1 -> background.
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=0,
                                        _force_source_index=torch.tensor([1, 0]))
    # For row 0: source is sample 1 (value 5). Its mask is left half.
    # Expected: left half = 5, right half = 1 (original sample 0 background).
    assert torch.allclose(x_mix[0, :, :, :2], torch.full_like(x_mix[0, :, :, :2], 5.0))
    assert torch.allclose(x_mix[0, :, :, 2:], torch.full_like(x_mix[0, :, :, 2:], 1.0))
    assert y_mix[0].item() == 200  # label of the foreground source


def test_maskmix_batch_size_one_returns_identity():
    """With B<2 there is no valid cross-sample source; return identity."""
    x = torch.randn(1, 3, 8, 8)
    m = torch.randint(0, 2, (1, 1, 8, 8)).float()
    y = torch.tensor([7])
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=0)
    assert torch.allclose(x_mix, x)
    assert torch.equal(m_mix, m)
    assert torch.equal(y_mix, y)


def test_maskmix_preserves_half_precision():
    """AMP path: fp16 input should yield fp16 output without casting."""
    B, C, H, W = 4, 3, 8, 8
    x = torch.randn(B, C, H, W).half()
    m = torch.randint(0, 2, (B, 1, H, W)).float()  # masks stay fp32
    y = torch.arange(B)
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=0)
    assert x_mix.dtype == torch.float16
    assert y_mix.dtype == y.dtype


def test_maskmix_mask_follows_label():
    """When MaskMix flips the label to the source, the returned mask must match the source mask."""
    B, C, H, W = 2, 3, 4, 4
    x = torch.zeros(B, C, H, W)
    m = torch.zeros(B, 1, H, W)
    m[0, 0, :2, :] = 1  # sample 0 FG = top half
    m[1, 0, :, :2] = 1  # sample 1 FG = left half
    y = torch.tensor([100, 200])
    # Force pair 0<-1, 1<-0; prob=1.0 forces apply everywhere.
    x_mix, m_mix, y_mix = maskmix_batch(x, m, y, prob=1.0, seed=0,
                                        _force_source_index=torch.tensor([1, 0]))
    # Row 0 should have taken source mask 1 (left half), label 200.
    assert torch.equal(m_mix[0], m[1])
    assert y_mix[0].item() == 200
    # Row 1 should have taken source mask 0 (top half), label 100.
    assert torch.equal(m_mix[1], m[0])
    assert y_mix[1].item() == 100
