# SC4001 Project F — Mask-Guided Adaptation for Oxford Flowers-102

Solo final project for NTU SC4001. 102-way fine-grained classification on
Oxford Flowers-102 (1020 train / 1020 val / 6149 test) using a frozen
ViT-B/16 backbone with VPT-Deep, augmented with two novelty components
that exploit the dataset's ground-truth segmentation masks:

1. **MaskMix** — a CutMix variant that uses pixel-accurate masks and a hard label.
2. **Mask-guided attention supervision** — a KL loss aligning `[CLS]→patch`
   attention to the downsampled GT mask.

See `docs/superpowers/specs/2026-04-22-flowers102-maskmix-vpt-design.md` for
the full design and `docs/superpowers/plans/2026-04-22-flowers102-maskmix-vpt.md`
for the implementation plan.

## Setup

Two surfaces:

1. **Local (Cursor / any IDE, CPU-only):** for editing `src/`, running unit tests,
   and authoring figures from saved CSVs.

   ```bash
   pip install -r requirements.txt
   python -m pytest tests/ -v
   ```

2. **Remote Colab T4 (free):** for GPU training. The `notebooks/flowers102_experiments.ipynb`
   notebook's first cell clones this repo onto the Colab VM, mounts Google Drive
   for checkpoint persistence, pip-installs dependencies, and downloads masks
   via `scripts/download_masks.py`. After experiments, a final cell commits and
   pushes `results/` and `figures/` back to GitHub (requires a PAT).

## Reproducing results

Open `notebooks/flowers102_experiments.ipynb` on a Colab-backed kernel and run
cells top to bottom. All experiments (Blocks A, B, C, D plus figures) finish in
~4 GPU-hours on a T4. Local workflow loop: edit `src/` → `pytest` → `git push`
→ re-run Cell 1 in the notebook (which `git pull`s) → run subsequent cells.

Outputs:
- `results/block_a.csv`, `results/block_b.csv`, `results/block_c.csv` — raw per-run metrics
- `results/headline_table.csv` — aggregated 2×2 table
- `results/significance_A4_vs_A1.txt` — paired-bootstrap p-value
- `figures/k_curve.png`, `figures/qualitative_attention.png`, `figures/sanity_mask_alignment.png`

## Tests

```bash
python -m pytest tests/ -v
```

Pure-function unit tests run on CPU in under 30 s and cover MaskMix, the
attention-supervision loss, accuracy metrics, and the bootstrap routine.

## Libraries

Cited in the report per course FAQ:
- PyTorch (Paszke et al. 2019)
- `timm` (Wightman 2019)
- torchvision Flowers102 dataset (Nilsback & Zisserman 2008)
