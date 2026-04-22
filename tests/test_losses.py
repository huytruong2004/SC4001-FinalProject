"""Unit tests for src/losses.py."""
import torch
import torch.nn.functional as F

from src.losses import attn_kl_loss


def test_attn_kl_loss_zero_when_identical():
    """If attention map equals the mask (both normalized), KL(m||a) == 0."""
    B = 2
    N = 4  # 14*14 would be real; use 4 for quick math
    mask = torch.zeros(B, 1, N, N)
    mask[:, :, 1, 1] = 1.0  # single foreground pixel
    attn = mask.clone().view(B, N * N)

    loss = attn_kl_loss(attn, mask)
    assert loss.item() < 1e-6


def test_attn_kl_loss_positive_when_different():
    """Uniform attention vs. concentrated mask -> positive loss."""
    B = 2
    N = 4
    mask = torch.zeros(B, 1, N, N)
    mask[:, :, 1, 1] = 1.0
    attn = torch.full((B, N * N), 1.0 / (N * N))  # uniform

    loss = attn_kl_loss(attn, mask)
    assert loss.item() > 0.0


def test_attn_kl_loss_handles_empty_mask():
    """If a mask has zero foreground, the loss should be 0 (skipped), not nan."""
    B = 2
    N = 4
    mask = torch.zeros(B, 1, N, N)  # all background
    attn = torch.full((B, N * N), 1.0 / (N * N))

    loss = attn_kl_loss(attn, mask)
    assert torch.isfinite(loss).item()
    assert loss.item() == 0.0


def test_attn_kl_loss_batch_mean_not_sum():
    """Loss should be independent of batch size (reduced as mean)."""
    N = 4
    mask = torch.zeros(1, 1, N, N)
    mask[:, :, 1, 1] = 1.0
    attn = torch.full((1, N * N), 1.0 / (N * N))
    loss_1 = attn_kl_loss(attn, mask)

    mask4 = mask.repeat(4, 1, 1, 1)
    attn4 = attn.repeat(4, 1)
    loss_4 = attn_kl_loss(attn4, mask4)

    assert torch.allclose(loss_1, loss_4, atol=1e-6)
