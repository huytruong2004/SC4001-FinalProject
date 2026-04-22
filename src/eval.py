"""Evaluation metrics and full-dataset evaluator."""
from __future__ import annotations

import torch


def top1_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def per_class_mean_accuracy(preds: torch.Tensor, labels: torch.Tensor,
                            num_classes: int) -> float:
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == c).float().mean().item())
    return float(sum(accs) / len(accs)) if accs else 0.0


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes: int) -> dict[str, float]:
    """Evaluate top-1 and per-class mean on `loader`; also return raw preds/labels."""
    model.eval()
    all_preds, all_labels = [], []
    for x, _m, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, _ = model(x, return_attn=False)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return {
        "test_top1": top1_accuracy(preds, labels),
        "test_per_class_mean": per_class_mean_accuracy(preds, labels, num_classes),
        "num_test": int(preds.numel()),
    }
