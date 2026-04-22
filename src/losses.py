"""Auxiliary losses for mask-guided attention supervision.

The ViT [CLS]->patch attention map (reshaped to a 14x14 spatial grid) is
supervised against the downsampled ground-truth foreground mask via KL
divergence. Both sides are treated as probability distributions over the
196 patch locations.

Samples with all-zero masks (no foreground) contribute 0 to avoid NaNs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _downsample_mask(mask: torch.Tensor, side: int) -> torch.Tensor:
    """mask: (B, 1, H, W) -> (B, side*side)."""
    m = F.adaptive_avg_pool2d(mask, output_size=(side, side))
    m = m.flatten(1)  # (B, side*side)
    return m


def attn_kl_loss(
    attn: torch.Tensor,  # (B, N*N) — already flattened cls->patch attention
    mask: torch.Tensor,  # (B, 1, H, W)
) -> torch.Tensor:
    """Compute mean KL(p_mask || p_attn). Skips samples where the mask is empty.

    Returns a scalar tensor. Autograd flows through `attn` only — the mask is
    treated as a target distribution (no gradient needed).
    """
    B, N2 = attn.shape
    side = int(round(N2 ** 0.5))
    assert side * side == N2, f"attn length {N2} is not a perfect square"

    m_down = _downsample_mask(mask, side).detach()  # (B, N2)
    m_sum = m_down.sum(dim=1)  # (B,)
    valid = m_sum > 1e-6
    if not valid.any():
        return attn.new_zeros(())

    # Normalize to probability distributions (only over valid samples).
    p_mask = m_down[valid] / m_sum[valid].unsqueeze(1).clamp(min=1e-8)

    # Normalize attention (it may not sum to 1 exactly if we averaged over heads/layers).
    a = attn[valid]
    a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # KL(p_mask || p_attn) = sum p_mask * (log p_mask - log p_attn).
    # Use F.kl_div which computes sum target * (log target - input) when
    # input=log p, target=p, reduction='batchmean'.
    log_a = (a + 1e-12).log()
    kl = F.kl_div(log_a, p_mask, reduction="batchmean")
    return kl
