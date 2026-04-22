"""VPT-Deep wrapping of a timm ViT-B/16.

Architecture:
 - timm ViT-B/16 (`vit_base_patch16_224.augreg_in21k`) with all backbone
   params frozen.
 - Per-block learnable prompt tokens (N=num_prompts per block) prepended to
   the sequence at each transformer block. After the block, prompts are
   discarded before the next block prepends its own prompts. This is the
   VPT-Deep protocol from Jia et al. 2022.
 - Linear classification head.
 - Optional attention capture from the last `capture_last_layers` blocks:
   mean over heads of the [CLS]->patch attention, averaged across the
   captured layers, returned as a (B, 196) flat tensor.
"""
from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class VPTDeepViT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224.augreg_in21k",
        num_prompts: int = 10,
        num_classes: int = 102,
        capture_last_layers: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.num_prompts = num_prompts
        self.capture_last_layers = capture_last_layers
        d = self.backbone.embed_dim
        L = len(self.backbone.blocks)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # VPT-Deep prompts: one tensor per block.
        # Init scale follows the VPT paper (uniform small or xavier).
        self.prompts = nn.Parameter(torch.empty(L, num_prompts, d))
        nn.init.uniform_(self.prompts, -0.1, 0.1)

        self.head = nn.Linear(d, num_classes)
        # head is always trainable

        self._attn_scores: list[torch.Tensor] = []
        self._capture_enabled: bool = False
        self._install_attn_hooks()

    def _install_attn_hooks(self) -> None:
        """Register forward hooks on the qkv proj of the last `capture_last_layers` blocks
        to reconstruct the attention map.

        timm's Attention module does `qkv = self.qkv(x)` then reshapes. We hook
        on qkv's output to snatch q, k, then compute our own attention in parallel.
        The hook short-circuits unless `self._capture_enabled` is True.
        It slices the [CLS]->patch row right here to avoid holding the full
        (B, H, N+197, N+197) tensor in memory.
        """
        self._hook_handles = []
        blocks = self.backbone.blocks
        for block in blocks[-self.capture_last_layers:]:
            attn_mod = block.attn
            num_heads = attn_mod.num_heads
            head_dim = attn_mod.head_dim if hasattr(attn_mod, "head_dim") else (block.attn.qkv.out_features // 3 // num_heads)
            scale = attn_mod.scale

            def make_hook(num_heads=num_heads, head_dim=head_dim, scale=scale):
                def hook(module, inp, out):
                    if not self._capture_enabled:
                        return
                    # out is the qkv output: (B, N, 3 * num_heads * head_dim).
                    B, N, _ = out.shape
                    qkv = out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]  # (B, H, N, D)
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)  # (B, H, N, N)
                    # Slice [CLS]->patch now to avoid keeping the full map.
                    # Sequence layout: [prompts(N_p)] [CLS] [patches(196)].
                    cls_idx = self.num_prompts
                    patch_start = self.num_prompts + 1
                    patch_end = self.num_prompts + 197
                    cls_to_patch = attn[:, :, cls_idx, patch_start:patch_end]  # (B, H, 196)
                    self._attn_scores.append(cls_to_patch)
                return hook

            self._hook_handles.append(attn_mod.qkv.register_forward_hook(make_hook()))

    def trainable_parameters(self):
        yield self.prompts
        yield from self.head.parameters()

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (logits, attn_flat) where attn_flat is (B, 196) or None."""
        self._attn_scores.clear()
        self._capture_enabled = bool(return_attn)

        B = x.size(0)
        # Patch + cls + pos
        x = self.backbone.patch_embed(x)  # (B, 196, d)
        cls = self.backbone.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)  # (B, 197, d)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # VPT-Deep: prepend prompts at each block, drop after the block.
        for i, block in enumerate(self.backbone.blocks):
            p = self.prompts[i].unsqueeze(0).expand(B, -1, -1)  # (B, N, d)
            x = torch.cat([p, x], dim=1)  # (B, N+197, d)
            x = block(x)
            x = x[:, self.num_prompts:]  # drop prompt tokens -> (B, 197, d)

        x = self.backbone.norm(x)
        cls_out = x[:, 0]  # (B, d)
        logits = self.head(cls_out)

        attn_flat = None
        if return_attn and self._attn_scores:
            # Each captured tensor is already (B, H, 196). Mean over heads, then
            # mean across the captured layers.
            per_layer = [a.mean(dim=1) for a in self._attn_scores]  # list of (B, 196)
            attn_flat = torch.stack(per_layer, dim=0).mean(dim=0)   # (B, 196)

        # Reset so subsequent forwards without return_attn don't waste compute.
        self._capture_enabled = False
        return logits, attn_flat
