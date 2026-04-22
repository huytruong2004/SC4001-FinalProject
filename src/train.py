"""Training entrypoint for one config.

Run from a notebook cell:
    from src.train import train_one_config
    result = train_one_config("configs/A1_baseline.yaml", seed=0,
                              data_root="data/flowers-102",
                              checkpoint_dir="checkpoints",
                              results_path="results/runs.jsonl")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import Flowers102WithMasks
from src.losses import attn_kl_loss
from src.maskmix import maskmix_batch
from src.model import VPTDeepViT
from src.utils import log_jsonl, seed_everything

torch.set_float32_matmul_precision("high")


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Merge base.yaml if specified.
    base_path = cfg.get("_base_")
    if base_path is not None:
        with open(Path(path).parent / base_path) as f:
            base = yaml.safe_load(f)
        merged = {**base, **{k: v for k, v in cfg.items() if k != "_base_"}}
        return merged
    return cfg


def _cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
                  prob: float = 0.5, seed: int | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference CutMix implementation for the Block C sanity row.

    Returns (x_mix, y_mix_hard). For the sanity row we use a HARD label
    (label of the larger area) rather than label interpolation, keeping the
    only difference between CutMix and MaskMix being the paste region.
    """
    B, C, H, W = x.shape
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    apply = torch.rand(B, generator=gen) < prob
    lam = torch.distributions.Beta(alpha, alpha).sample((1,)).item()
    cut_w = int(W * (1 - lam) ** 0.5)
    cut_h = int(H * (1 - lam) ** 0.5)
    cx = torch.randint(0, W, (1,), generator=gen).item()
    cy = torch.randint(0, H, (1,), generator=gen).item()
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    src = torch.randperm(B, generator=gen)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[src][:, :, y1:y2, x1:x2]
    # Hard label: if pasted area > 50% of image, use src label; else original.
    patch_frac = (x2 - x1) * (y2 - y1) / (H * W)
    use_src = apply & (patch_frac > 0.5)
    y_mix = torch.where(use_src, y[src], y)
    x_mix = torch.where(apply.view(B, 1, 1, 1), x_mix, x)
    return x_mix, y_mix


def train_one_config(
    config_path: str | Path,
    seed: int,
    data_root: str | Path,
    checkpoint_dir: str | Path,
    results_path: str | Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    cfg = _load_config(config_path)
    run_name = run_name or f"{Path(config_path).stem}_seed{seed}"
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA detected; training on CPU will be very slow.")
    if device == "cuda":
        print(f"flash_sdp: {torch.backends.cuda.flash_sdp_enabled()}")

    # Data ---------------------------------------------------------------
    train_ds = Flowers102WithMasks(
        root=data_root, split="train", image_size=cfg["image_size"],
        train_augment=True,
        subsample_k=cfg.get("subsample_k"),
        subsample_seed=cfg.get("subsample_seed", 0),
    )
    val_ds = Flowers102WithMasks(root=data_root, split="val",
                                 image_size=cfg["image_size"], train_augment=False)
    test_ds = Flowers102WithMasks(root=data_root, split="test",
                                  image_size=cfg["image_size"], train_augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 2), pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg.get("num_workers", 2), pin_memory=True,
                             persistent_workers=True)

    # Model --------------------------------------------------------------
    if cfg.get("adaptation") == "linear_probe":
        from src.model import VPTDeepViT as _M
        model = _M(num_prompts=1, num_classes=102, freeze_backbone=True)
        # Zero out prompts and freeze them for linear-probe semantics.
        model.prompts.data.zero_()
        model.prompts.requires_grad = False
        trainable = list(model.head.parameters())
    else:
        model = VPTDeepViT(
            num_prompts=cfg.get("num_prompts", 10),
            num_classes=102,
            capture_last_layers=cfg.get("capture_last_layers", 2),
        )
        trainable = [model.prompts] + list(model.head.parameters())
    model = model.to(device)
    if cfg.get("compile", False):
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # Training -----------------------------------------------------------
    ckpt_dir = Path(checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_ckpt = ckpt_dir / "best.pt"

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for step, (x, m, y) in enumerate(tqdm(train_loader, desc=f"ep{epoch}")):
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if cfg.get("augment") == "maskmix":
                x, m, y = maskmix_batch(x, m, y, prob=cfg.get("mix_prob", 0.5),
                                        seed=seed * 1000 + step)
            elif cfg.get("augment") == "cutmix":
                x, y = _cutmix_batch(x, y, alpha=1.0, prob=cfg.get("mix_prob", 0.5),
                                     seed=seed * 1000 + step)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, attn = model(x, return_attn=cfg.get("attn_lambda", 0.0) > 0)
                loss_ce = F.cross_entropy(logits, y)
                loss = loss_ce
                if cfg.get("attn_lambda", 0.0) > 0 and attn is not None:
                    loss_attn = attn_kl_loss(attn, m)
                    loss = loss + cfg["attn_lambda"] * loss_attn

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        # Val --------------------------------------------------------------
        val_top1 = _evaluate(model, val_loader, device)
        log_jsonl(results_path, {"run": run_name, "epoch": epoch,
                                 "train_loss": running / len(train_ds),
                                 "val_top1": val_top1})
        if val_top1 > best_val:
            best_val = val_top1
            unwrapped = getattr(model, "_orig_mod", model)
            torch.save({"model": unwrapped.state_dict(), "cfg": cfg, "epoch": epoch,
                        "val_top1": val_top1}, best_ckpt)

    # Final test with best checkpoint ------------------------------------
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model"])
    from src.eval import evaluate_full
    test_metrics = evaluate_full(model, test_loader, device, num_classes=102)
    result = {"run": run_name, "best_val_top1": best_val, **test_metrics,
              "config_path": str(config_path), "seed": seed}
    log_jsonl(results_path, {"final": True, **result})
    return result


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, _m, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(x, return_attn=False)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
