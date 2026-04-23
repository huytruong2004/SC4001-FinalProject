"""Generate figures/k_curve.png from headline numbers.

One-shot script used during the report rewrite. Plots A1 baseline and A4
(ours) test top-1 at k in {1, 5, 10} with a std band at k=10.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "figures" / "k_curve.png"


def main() -> None:
    # Per results/headline_table.csv. Hardcoded because the table has
    # heterogeneous rows (A-configs, B k-points, D linear probe) and we
    # only want the five values below.
    k_points = np.array([1, 5, 10])
    baseline = np.array([0.6499, 0.9543, 0.9888])
    ours = np.array([0.5720, 0.9419, 0.9787])
    baseline_std_at_k10 = 0.0023
    ours_std_at_k10 = 0.0029

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)

    ax.plot(k_points, baseline, marker="o", linestyle="--", color="#1f77b4",
            label="A1 baseline (VPT-Deep)")
    ax.plot(k_points, ours, marker="s", linestyle="-", color="#d62728",
            label="A4 ours (MaskMix + AttSup)")

    ax.errorbar(10, baseline[-1], yerr=baseline_std_at_k10, color="#1f77b4",
                capsize=3, linewidth=0, elinewidth=1.2)
    ax.errorbar(10, ours[-1], yerr=ours_std_at_k10, color="#d62728",
                capsize=3, linewidth=0, elinewidth=1.2)

    ax.annotate(f"gap = {baseline[0] - ours[0]:+.3f}",
                xy=(1, (baseline[0] + ours[0]) / 2),
                xytext=(1.7, (baseline[0] + ours[0]) / 2 - 0.02),
                fontsize=9, color="#555555")

    ax.set_xscale("log")
    ax.set_xticks(k_points)
    ax.set_xticklabels(["1", "5", "10"])
    ax.set_xlabel("training images per class (k)")
    ax.set_ylabel("test top-1 accuracy")
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
