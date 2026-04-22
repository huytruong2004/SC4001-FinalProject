"""CLI that reproduces notebook Cells 8, 9, 11.

Builds the Block-A headline table, computes the paired-bootstrap p-value for
A4 vs A1 (seed 0), and writes the qualitative attention figure. Reads
per-run records from a JSONL file written by `src.train`, so it does not
depend on any per-block CSVs existing on disk.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.bootstrap import paired_bootstrap_pvalue
from src.data import Flowers102WithMasks
from src.model import VPTDeepViT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate Block-A results, run paired bootstrap, render qualitative attention figure.")
    p.add_argument("--runs-jsonl", type=Path, required=True, help="Path to results/runs.jsonl.")
    p.add_argument("--results-dir", type=Path, required=True, help="Directory for CSV/text outputs.")
    p.add_argument("--figures-dir", type=Path, required=True, help="Directory for figure outputs.")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing per-run best.pt files.")
    p.add_argument("--data-root", type=Path, required=True, help="Flowers-102 data root.")
    p.add_argument("--skip-figure", action="store_true", help="Skip the qualitative attention figure.")
    return p.parse_args()


def load_final_records(runs_jsonl: Path) -> pd.DataFrame:
    """Read runs.jsonl line-by-line and keep only final records, grouped by run."""
    rows = []
    with open(runs_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("final") is True:
                rows.append(rec)
    if not rows:
        raise RuntimeError(f"No final records found in {runs_jsonl}")
    df = pd.DataFrame(rows)
    # In case of re-runs, keep the last final record per run.
    df = df.drop_duplicates(subset=["run"], keep="last").reset_index(drop=True)
    return df


def build_headline_table(df_final: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Reproduce Cell 8: map Block-A run names to config labels, aggregate, write CSV."""
    pattern = re.compile(r"^A[1-4]_.*_seed\d+$")
    df_a = df_final[df_final["run"].apply(lambda r: bool(pattern.match(str(r))))].copy()
    config_name_map = {
        "A1_baseline_seed0": "A1 baseline", "A1_baseline_seed1": "A1 baseline", "A1_baseline_seed2": "A1 baseline",
        "A2_maskmix_seed0":  "A2 +MaskMix", "A2_maskmix_seed1":  "A2 +MaskMix", "A2_maskmix_seed2":  "A2 +MaskMix",
        "A3_attsup_seed0":   "A3 +AttSup",  "A3_attsup_seed1":   "A3 +AttSup",  "A3_attsup_seed2":   "A3 +AttSup",
        "A4_ours_seed0":     "A4 ours",     "A4_ours_seed1":     "A4 ours",     "A4_ours_seed2":     "A4 ours",
    }
    df_a["config"] = df_a["run"].map(config_name_map)
    headline = df_a.groupby("config").agg(
        top1_mean=("test_top1", "mean"),
        top1_std=("test_top1", "std"),
        per_class_mean=("test_per_class_mean", "mean"),
        per_class_std=("test_per_class_mean", "std"),
    ).round(4)
    out_path = results_dir / "headline_table.csv"
    headline.to_csv(out_path)
    print(f"[headline] wrote {out_path}")
    print(headline)
    return headline


def per_example_correct(ckpt_path: Path, data_root: Path) -> np.ndarray:
    """Run test-set inference and return boolean per-example correctness vector."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VPTDeepViT(
        num_prompts=cfg.get("num_prompts", 10),
        num_classes=102,
        capture_last_layers=cfg.get("capture_last_layers", 2),
    )
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    test_ds = Flowers102WithMasks(
        root=data_root, split="test", image_size=cfg["image_size"], train_augment=False
    )
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, _m, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, _ = model(x, return_attn=False)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return preds == labels


def run_significance_test(checkpoint_dir: Path, data_root: Path, results_dir: Path) -> None:
    """Reproduce Cell 9: paired bootstrap A4 vs A1 (seed 0)."""
    a1 = per_example_correct(checkpoint_dir / "A1_baseline_seed0" / "best.pt", data_root)
    a4 = per_example_correct(checkpoint_dir / "A4_ours_seed0" / "best.pt", data_root)
    mean_diff, p = paired_bootstrap_pvalue(a4, a1, n_resamples=5000, seed=0)
    print(f"A4 - A1 mean diff = {mean_diff:+.4f}, two-sided p = {p:.4f}")
    out_path = results_dir / "significance_A4_vs_A1.txt"
    with open(out_path, "w") as f:
        f.write(f"A4 - A1 mean diff = {mean_diff:+.4f}\n")
        f.write(f"two-sided paired bootstrap p = {p:.4f}\n")
        f.write(f"n_resamples = 5000, seed = 0\n")
    print(f"[significance] wrote {out_path}")


def load_model(ckpt_path: Path, device: str) -> VPTDeepViT:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    m = VPTDeepViT(
        num_prompts=cfg.get("num_prompts", 10),
        num_classes=102,
        capture_last_layers=cfg.get("capture_last_layers", 2),
    )
    m.load_state_dict(state["model"])
    return m.to(device).eval()


def render_qualitative_figure(checkpoint_dir: Path, data_root: Path, figures_dir: Path) -> None:
    """Reproduce Cell 11: 3x4 panel of image | GT mask | A1 attn | A4 attn."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_a1 = load_model(checkpoint_dir / "A1_baseline_seed0" / "best.pt", device)
    m_a4 = load_model(checkpoint_dir / "A4_ours_seed0" / "best.pt", device)

    TARGETS = [0, 53, 70]
    test_ds = Flowers102WithMasks(
        root=data_root, split="test", image_size=224, train_augment=False
    )

    def first_index_of_class(ds, c):
        for i in range(len(ds)):
            if ds[i][2] == c:
                return i
        raise ValueError(f"class {c} not found")

    fig, axes = plt.subplots(len(TARGETS), 4, figsize=(12, 3 * len(TARGETS)))
    for row, c in enumerate(TARGETS):
        idx = first_index_of_class(test_ds, c)
        img, mask, label = test_ds[idx]
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            _, attn_a1 = m_a1(x, return_attn=True)
            _, attn_a4 = m_a4(x, return_attn=True)

        def to_grid(a):
            return a.view(14, 14).cpu().numpy()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_show); axes[row, 0].set_title(f"class {label}")
        axes[row, 1].imshow(mask[0].numpy(), cmap="gray"); axes[row, 1].set_title("GT mask")
        axes[row, 2].imshow(img_show); axes[row, 2].imshow(to_grid(attn_a1[0]), alpha=0.6, cmap="hot")
        axes[row, 2].set_title("A1 baseline attn")
        axes[row, 3].imshow(img_show); axes[row, 3].imshow(to_grid(attn_a4[0]), alpha=0.6, cmap="hot")
        axes[row, 3].set_title("A4 ours attn")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir / "qualitative_attention.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_path}")


def main(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df_final = load_final_records(args.runs_jsonl)
    build_headline_table(df_final, args.results_dir)
    run_significance_test(args.checkpoint_dir, args.data_root, args.results_dir)
    if args.skip_figure:
        print("[figure] skipped (--skip-figure)")
    else:
        render_qualitative_figure(args.checkpoint_dir, args.data_root, args.figures_dir)


if __name__ == "__main__":
    main(parse_args())
