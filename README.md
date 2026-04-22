# SC4001 Project F: Mask-Guided Adaptation for Oxford Flowers-102

Solo final project for NTU SC4001. 102-way fine-grained classification on
Oxford Flowers-102 (1020 train / 1020 val / 6149 test) using a frozen
ViT-B/16 backbone with VPT-Deep, augmented with two novelty components
that exploit the dataset's ground-truth segmentation masks:

1. **MaskMix**, a CutMix variant that uses pixel-accurate masks and a hard label.
2. **Mask-guided attention supervision**, a KL loss aligning `[CLS]→patch`
   attention to the downsampled GT mask.

See `docs/superpowers/specs/2026-04-22-flowers102-maskmix-vpt-design.md` for
the full design and `docs/superpowers/plans/2026-04-22-flowers102-maskmix-vpt.md`
for the implementation plan.

## Setup

Three supported environments. The notebook and CLI both read three env vars
(`SC4001_REPO`, `SC4001_DATA`, `SC4001_CKPT`) so the same code runs anywhere.

### 1. Local (CPU, any IDE)

For editing `src/`, running unit tests, and authoring figures from saved CSVs.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

You can also launch Jupyter locally and open `notebooks/flowers102_experiments.ipynb`.
With no env vars set, Cell 1 falls back to the current working directory (or its
parent if launched from `notebooks/`), so `import src...` works out of the box.

### 2. Colab ephemeral

Clone and install inside a fresh Colab VM:

```bash
!git clone https://github.com/huytruong2004/SC4001-FinalProject.git
%cd SC4001-FinalProject
!pip install -r requirements.txt
```

Cell 1 no longer mounts Drive automatically. The CWD fallback handles the repo
root. If you want persistent data or checkpoints, mount Drive yourself and
export env vars before launching the notebook kernel, e.g.:

```python
import os
os.environ["SC4001_DATA"] = "/content/drive/MyDrive/sc4001_flowers102/data/flowers-102"
os.environ["SC4001_CKPT"] = "/content/drive/MyDrive/sc4001_flowers102/checkpoints"
```

Otherwise, checkpoints and the downloaded dataset live under `/workspace/...`
(which does not exist on Colab, so the bootstrap cell will create it on the
ephemeral VM disk; fine for one session, gone on disconnect).

### 3. vast.ai or remote SSH (H100 target)

After `ssh` into the box:

```bash
git clone https://github.com/huytruong2004/SC4001-FinalProject.git /workspace/SC4001-FinalProject
cd /workspace/SC4001-FinalProject
pip install -r requirements.txt

export SC4001_REPO=/workspace/SC4001-FinalProject
export SC4001_DATA=/workspace/data/flowers-102
export SC4001_CKPT=/workspace/checkpoints
```

Then run experiments directly from the CLI:

```bash
python -m src.train \
  --config configs/A1_baseline.yaml \
  --seed 0 \
  --data-root "$SC4001_DATA" \
  --checkpoint-dir "$SC4001_CKPT" \
  --results-path "$SC4001_REPO/results/runs.jsonl"
```

Commit 6 will add shell runners that wrap this across configs and seeds; the
CLI above already works standalone.

## Reproducing results

Two entry points:

1. **Notebook.** Open `notebooks/flowers102_experiments.ipynb` on any of the
   three environments and run cells top to bottom. On an H100, all experiments
   (Blocks A, B, C, D plus figures) target a ~7 GPU-hour budget.
2. **CLI / shell runners.** Invoke `python -m src.train` as shown above.
   Useful for headless SSH sessions, batched runs across configs, or tmux loops.

Local workflow loop: edit `src/` with tests (`pytest`), push to GitHub, pull on
the remote, re-run the relevant cells or CLI invocations.

## Outputs

- `results/block_a.csv`, `results/block_b.csv`, `results/block_c.csv`: raw per-run metrics
- `results/headline_table.csv`: aggregated 2x2 table
- `results/significance_A4_vs_A1.txt`: paired-bootstrap p-value
- `figures/k_curve.png`, `figures/qualitative_attention.png`, `figures/sanity_mask_alignment.png`

## Tests

```bash
pytest tests/ -v
```

Pure-function unit tests run on CPU in under 30 s and cover MaskMix, the
attention-supervision loss, accuracy metrics, and the bootstrap routine.

## Libraries

Cited in the report per course FAQ:

- PyTorch (Paszke et al. 2019)
- `timm` (Wightman 2019)
- torchvision Flowers102 dataset (Nilsback & Zisserman 2008)
