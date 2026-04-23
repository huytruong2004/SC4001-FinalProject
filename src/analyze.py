"""CLI to aggregate Block A/B/C/D results and render report figures.

All aggregate analysis (headline table, cross-seed significance tests,
learning-curve figure) runs purely from `results/runs.jsonl`, so the script
does not require checkpoints. The paired-bootstrap (per-example correctness
on the test set) and the qualitative attention figure do need checkpoints,
and are skipped gracefully when --checkpoint-dir is absent or empty.
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
from scipy import stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Block A/B/C/D results, run cross-seed t-tests, render figures."
    )
    p.add_argument("--runs-jsonl", type=Path, required=True, help="Path to results/runs.jsonl.")
    p.add_argument("--results-dir", type=Path, required=True, help="Directory for CSV/text outputs.")
    p.add_argument("--figures-dir", type=Path, required=True, help="Directory for figure outputs.")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Optional. Directory of per-run best.pt files. If present, also runs the per-example paired bootstrap and renders the qualitative attention figure.")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Optional. Flowers-102 data root. Required only if --checkpoint-dir is used.")
    p.add_argument("--skip-figure", action="store_true", help="Skip the qualitative attention figure even if checkpoints exist.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# runs.jsonl loading
# ---------------------------------------------------------------------------

def load_final_records(runs_jsonl: Path) -> pd.DataFrame:
    """Keep one 'final' record per run (last-write-wins)."""
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
    df = df.drop_duplicates(subset=["run"], keep="last").reset_index(drop=True)
    return df


def load_epoch_records(runs_jsonl: Path) -> pd.DataFrame:
    """Return one row per (run, epoch) from runs.jsonl."""
    rows = []
    with open(runs_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "epoch" in rec and rec.get("final") is not True:
                rows.append(rec)
    if not rows:
        raise RuntimeError(f"No per-epoch records found in {runs_jsonl}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# headline table (Blocks A + C + D + B)
# ---------------------------------------------------------------------------

_CONFIG_LABEL = {
    "A1_baseline": "A1 baseline",
    "A2_maskmix":  "A2 +MaskMix",
    "A3_attsup":   "A3 +AttSup",
    "A4_ours":     "A4 ours (MaskMix+AttSup)",
    "A5_cutmix_attsup": "A5 CutMix+AttSup",
    "D_linear_probe":   "D linear probe",
    "B_k1_baseline": "B k=1 baseline",
    "B_k1_ours":     "B k=1 ours",
    "B_k5_baseline": "B k=5 baseline",
    "B_k5_ours":     "B k=5 ours",
}

_RUN_RE = re.compile(r"^(?P<cfg>.+)_seed(?P<seed>\d+)$")


def _attach_cfg_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Add cfg_key and seed columns parsed from the 'run' string."""
    out = df.copy()
    extracted = out["run"].str.extract(_RUN_RE)
    out["cfg_key"] = extracted["cfg"]
    out["seed"] = pd.to_numeric(extracted["seed"], errors="coerce").astype("Int64")
    return out


def build_headline_table(df_final: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    df = _attach_cfg_seed(df_final)
    df = df[df["cfg_key"].isin(_CONFIG_LABEL)].copy()
    df["config"] = df["cfg_key"].map(_CONFIG_LABEL)

    ordered_cfgs = [_CONFIG_LABEL[k] for k in _CONFIG_LABEL if _CONFIG_LABEL[k] in set(df["config"])]
    headline = (
        df.groupby("config")
          .agg(
              n_seeds=("seed", "nunique"),
              top1_mean=("test_top1", "mean"),
              top1_std=("test_top1", "std"),
              per_class_mean=("test_per_class_mean", "mean"),
              per_class_std=("test_per_class_mean", "std"),
          )
          .reindex(ordered_cfgs)
          .round(4)
    )
    out_path = results_dir / "headline_table.csv"
    headline.to_csv(out_path)
    print(f"[headline] wrote {out_path}")
    print(headline)
    return headline


# ---------------------------------------------------------------------------
# cross-seed significance tests (Welch's t-test + Cohen's d)
# ---------------------------------------------------------------------------

# Each contrast: (treatment_key, reference_key, description)
_CONTRASTS = [
    ("A2_maskmix",       "A1_baseline", "A2 (+MaskMix) vs A1 (baseline)"),
    ("A3_attsup",        "A1_baseline", "A3 (+AttSup) vs A1 (baseline)"),
    ("A4_ours",          "A1_baseline", "A4 (ours) vs A1 (baseline)"),
    ("A5_cutmix_attsup", "A4_ours",     "A5 (CutMix+AttSup) vs A4 (MaskMix+AttSup)"),
]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled (unbiased) std. Positive means a > b."""
    na, nb = len(a), len(b)
    s2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    s = np.sqrt(s2)
    return float((np.mean(a) - np.mean(b)) / s) if s > 0 else float("nan")


def run_cross_seed_tests(df_final: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Welch's t-test per contrast, computed across seeds from runs.jsonl."""
    df = _attach_cfg_seed(df_final)

    rows = []
    for treat_key, ref_key, desc in _CONTRASTS:
        t = df[df["cfg_key"] == treat_key]["test_top1"].to_numpy()
        r = df[df["cfg_key"] == ref_key]["test_top1"].to_numpy()
        if len(t) < 2 or len(r) < 2:
            print(f"[significance] SKIP {desc}: need >=2 seeds per arm (have {len(t)}, {len(r)})")
            continue
        mean_diff = float(t.mean() - r.mean())
        welch = stats.ttest_ind(t, r, equal_var=False)
        d = _cohens_d(t, r)
        rows.append({
            "contrast": desc,
            "n_treat": len(t),
            "n_ref": len(r),
            "treat_mean": round(float(t.mean()), 4),
            "ref_mean": round(float(r.mean()), 4),
            "mean_diff": round(mean_diff, 4),
            "welch_t": round(float(welch.statistic), 3),
            "welch_p": round(float(welch.pvalue), 4),
            "cohens_d": round(d, 3),
        })

    sig_df = pd.DataFrame(rows)
    out_csv = results_dir / "significance.csv"
    sig_df.to_csv(out_csv, index=False)
    print(f"[significance] wrote {out_csv}")
    print(sig_df.to_string(index=False))

    # Human-readable companion file.
    out_txt = results_dir / "significance.txt"
    with open(out_txt, "w") as f:
        f.write("Cross-seed Welch's t-tests (two-sided), from results/runs.jsonl\n")
        f.write("=" * 64 + "\n\n")
        for _, row in sig_df.iterrows():
            f.write(f"{row['contrast']}\n")
            f.write(f"  means: treat={row['treat_mean']:.4f} (n={row['n_treat']}) "
                    f"ref={row['ref_mean']:.4f} (n={row['n_ref']})\n")
            f.write(f"  diff = {row['mean_diff']:+.4f}, Welch t = {row['welch_t']:+.3f}, "
                    f"p = {row['welch_p']:.4f}, Cohen's d = {row['cohens_d']:+.2f}\n\n")
    print(f"[significance] wrote {out_txt}")
    return sig_df


# ---------------------------------------------------------------------------
# learning-curves figure
# ---------------------------------------------------------------------------

def build_learning_curves(df_epoch: pd.DataFrame, figures_dir: Path) -> None:
    """val_top1 vs epoch, mean over seeds, one line per Block-A/C config."""
    df = _attach_cfg_seed(df_epoch)
    keep = ["A1_baseline", "A2_maskmix", "A3_attsup", "A4_ours", "A5_cutmix_attsup"]
    df = df[df["cfg_key"].isin(keep)].copy()

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1]})
    colors = {"A1_baseline": "C0", "A2_maskmix": "C1", "A3_attsup": "C2",
              "A4_ours": "C3", "A5_cutmix_attsup": "C4"}
    for cfg_key in keep:
        sub = df[df["cfg_key"] == cfg_key]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch")["val_top1"].agg(["mean", "std", "count"]).reset_index()
        mean = grouped["mean"].to_numpy()
        std = grouped["std"].fillna(0.0).to_numpy()
        ep = grouped["epoch"].to_numpy()
        label = _CONFIG_LABEL.get(cfg_key, cfg_key)
        for ax in (ax_full, ax_zoom):
            ax.plot(ep, mean, label=label, color=colors[cfg_key], linewidth=1.8)
            if grouped["count"].max() > 1:
                ax.fill_between(ep, mean - std, mean + std, color=colors[cfg_key], alpha=0.15, linewidth=0)

    ax_full.set_xlabel("epoch")
    ax_full.set_ylabel("val top-1")
    ax_full.set_title("full range")
    ax_full.legend(loc="lower right", fontsize=9)
    ax_full.grid(alpha=0.3)

    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_xlim(5, None)
    ax_zoom.set_ylim(0.95, 1.0)
    ax_zoom.set_title("zoom: epoch ≥ 5, val ∈ [0.95, 1.0]")
    ax_zoom.grid(alpha=0.3)

    fig.suptitle("Val top-1 vs epoch (mean ± std across seeds)", y=1.02)
    fig.tight_layout()
    out_path = figures_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_path}")


# ---------------------------------------------------------------------------
# optional: per-example paired bootstrap + qualitative attention figure
# (both require trained checkpoints)
# ---------------------------------------------------------------------------

def _checkpoints_available(checkpoint_dir: Path | None, required_runs: list[str]) -> bool:
    if checkpoint_dir is None:
        return False
    for run in required_runs:
        if not (checkpoint_dir / run / "best.pt").exists():
            return False
    return True


def run_paired_bootstrap_a4_vs_a1(checkpoint_dir: Path, data_root: Path, results_dir: Path) -> None:
    """Per-example paired-bootstrap on seed-0 A4 vs A1. Complements the cross-seed test."""
    import torch
    from torch.utils.data import DataLoader
    from src.bootstrap import paired_bootstrap_pvalue
    from src.data import Flowers102WithMasks
    from src.model import VPTDeepViT

    def per_example_correct(ckpt_path: Path) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = state["cfg"]
        model = VPTDeepViT(
            num_prompts=cfg.get("num_prompts", 10),
            num_classes=102,
            capture_last_layers=cfg.get("capture_last_layers", 2),
        ).to(device).eval()
        model.load_state_dict(state["model"])
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
        return (np.concatenate([p.numpy() for p in preds])
                == np.concatenate([y.numpy() for y in labels]))

    a1 = per_example_correct(checkpoint_dir / "A1_baseline_seed0" / "best.pt")
    a4 = per_example_correct(checkpoint_dir / "A4_ours_seed0" / "best.pt")
    mean_diff, p = paired_bootstrap_pvalue(a4, a1, n_resamples=5000, seed=0)
    print(f"[bootstrap] A4 - A1 per-example diff = {mean_diff:+.4f}, two-sided p = {p:.4f}")
    out_path = results_dir / "significance_A4_vs_A1_bootstrap.txt"
    with open(out_path, "w") as f:
        f.write(f"A4 - A1 per-example mean diff = {mean_diff:+.4f}\n")
        f.write(f"two-sided paired bootstrap p = {p:.4f}\n")
        f.write("n_resamples = 5000, seed = 0 (single-seed A1/A4 comparison over 6149 test images)\n")
    print(f"[bootstrap] wrote {out_path}")


def render_qualitative_figure(checkpoint_dir: Path, data_root: Path, figures_dir: Path) -> None:
    """3x4 panel: image | GT mask | A1 attn | A4 attn for 3 target classes."""
    import torch
    from src.data import Flowers102WithMasks
    from src.model import VPTDeepViT

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_a1 = load_model(checkpoint_dir / "A1_baseline_seed0" / "best.pt", device)
    m_a4 = load_model(checkpoint_dir / "A4_ours_seed0" / "best.pt", device)

    TARGETS = [0, 53, 70]
    test_ds = Flowers102WithMasks(root=data_root, split="test", image_size=224, train_augment=False)

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
        mean = __import__("torch").tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = __import__("torch").tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(img_show); axes[row, 0].set_title(f"class {label}")
        axes[row, 1].imshow(mask[0].numpy(), cmap="gray"); axes[row, 1].set_title("GT mask")
        axes[row, 2].imshow(img_show); axes[row, 2].imshow(attn_a1[0].view(14, 14).cpu().numpy(), alpha=0.6, cmap="hot")
        axes[row, 2].set_title("A1 baseline attn")
        axes[row, 3].imshow(img_show); axes[row, 3].imshow(attn_a4[0].view(14, 14).cpu().numpy(), alpha=0.6, cmap="hot")
        axes[row, 3].set_title("A4 ours attn")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    out_path = figures_dir / "qualitative_attention.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df_final = load_final_records(args.runs_jsonl)
    build_headline_table(df_final, args.results_dir)
    run_cross_seed_tests(df_final, args.results_dir)

    df_epoch = load_epoch_records(args.runs_jsonl)
    build_learning_curves(df_epoch, args.figures_dir)

    # Optional GPU/checkpoint-dependent outputs.
    bootstrap_runs = ["A1_baseline_seed0", "A4_ours_seed0"]
    if _checkpoints_available(args.checkpoint_dir, bootstrap_runs) and args.data_root is not None:
        run_paired_bootstrap_a4_vs_a1(args.checkpoint_dir, args.data_root, args.results_dir)
    else:
        print("[bootstrap] skipped (no --checkpoint-dir or required best.pt files missing).")

    if args.skip_figure:
        print("[figure] qualitative attention skipped (--skip-figure).")
    elif _checkpoints_available(args.checkpoint_dir, bootstrap_runs) and args.data_root is not None:
        render_qualitative_figure(args.checkpoint_dir, args.data_root, args.figures_dir)
    else:
        print("[figure] qualitative attention skipped (no checkpoints available locally).")


if __name__ == "__main__":
    main(parse_args())
