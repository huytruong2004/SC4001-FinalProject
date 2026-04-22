"""MaskMix: ground-truth-mask-conditioned CutMix variant.

Given a batch of (x, m, y), for each sample i with probability `prob`,
pick another sample j and composite the foreground of j (mask m_j) onto
the background of i:

    x_mix[i] = m_j ⊙ x_j + (1 - m_j) ⊙ x_i
    y_mix[i] = y_j                               (hard label)

The label is the foreground source's label — no interpolation.
"""
from __future__ import annotations

from typing import Optional

import torch


def maskmix_batch(
    x: torch.Tensor,           # (B, C, H, W)
    m: torch.Tensor,           # (B, 1, H, W), values in [0,1]
    y: torch.Tensor,           # (B,)
    prob: float = 0.5,
    seed: Optional[int] = None,
    _force_source_index: Optional[torch.Tensor] = None,  # test hook
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MaskMix to a batch. Returns (x_mix, y_mix) on the same device/dtype as x."""
    B = x.size(0)
    device = x.device

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Select which samples to remix.
    apply_mask = torch.rand(B, generator=gen) < prob  # (B,) bool on CPU
    apply_mask = apply_mask.to(device)

    # For each position i, pick a source index j != i.
    if _force_source_index is not None:
        src = _force_source_index.to(device)
    else:
        src = torch.randperm(B, generator=gen).to(device)
        # Avoid i -> i identity swaps (harmless but wasteful).
        same = (src == torch.arange(B, device=device))
        if same.any():
            src[same] = (src[same] + 1) % B

    x_src = x[src]      # (B, C, H, W)
    m_src = m[src]      # (B, 1, H, W)
    y_src = y[src]      # (B,)

    # Composite per-sample.
    x_mix = torch.where(m_src > 0.5, x_src, x)
    y_mix = torch.where(apply_mask, y_src, y)

    # Where apply_mask is False, restore original x.
    apply_mask_img = apply_mask.view(B, 1, 1, 1)
    x_mix = torch.where(apply_mask_img, x_mix, x)

    return x_mix, y_mix
