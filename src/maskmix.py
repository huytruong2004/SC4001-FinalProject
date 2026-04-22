"""MaskMix: ground-truth-mask-conditioned CutMix variant.

Given a batch of (x, m, y), for each sample i with probability `prob`,
pick another sample j and composite the foreground of j (thresholded at 0.5)
onto the background of i:

    x_mix[i] = x_j            where m_j > 0.5
    x_mix[i] = x_i            elsewhere
    y_mix[i] = y_j            (hard label; no interpolation)

The returned mask `m_mix` is remixed in lockstep with the composite image:
for samples where MaskMix was applied, `m_mix[i] = m_j` (the source mask),
otherwise `m_mix[i] = m[i]`. This keeps downstream consumers (in particular
attention supervision via `attn_kl_loss`) seeing a mask that is consistent
with the (possibly flipped) hard label.

If fewer than 2 samples are provided, the input is returned unchanged.
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MaskMix to a batch. Returns (x_mix, m_mix, y_mix) on the same device/dtype as x."""
    B = x.size(0)
    if B < 2:
        return x.clone(), m.clone(), y.clone()

    device = x.device

    # CPU-side generator keeps determinism independent of device RNG quirks.
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

    # Single-pass composite: paste x_src only where apply_mask AND m_src > 0.5.
    apply_img = apply_mask.view(B, 1, 1, 1)
    use_src = apply_img & (m_src > 0.5)
    x_mix = torch.where(use_src, x_src, x)
    m_mix = torch.where(apply_img, m_src, m)
    y_mix = torch.where(apply_mask, y_src, y)

    return x_mix, m_mix, y_mix
