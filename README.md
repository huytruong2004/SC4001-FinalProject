# SC4001 Project F: Mask-Guided Adaptation for Oxford Flowers-102

Solo final project for NTU SC4001. 102-way fine-grained classification on
Oxford Flowers-102 (1020 train / 1020 val / 6149 test) using a frozen
ViT-B/16 backbone with VPT-Deep, studied under a mechanistic ablation of
two mask-guided components plus a paste-shape control:

1. **MaskMix** (A2): a CutMix variant that pastes a pixel-accurate
   foreground using the dataset's ground-truth segmentation mask and
   adopts the source image's hard label.
2. **Mask-guided attention supervision** (A3): a KL loss aligning the
   mean `[CLS]→patch` attention of the last two transformer blocks with
   the downsampled GT mask.
3. **Rectangular CutMix + AttSup** (A5): a paste-shape control that
   isolates the effect of the pixel-accurate mask region from the act of
   mixing or the auxiliary loss.

The full write-up is at [`report/sc4001-report.pdf`](report/sc4001-report.pdf).

## Headline finding

On strongly-pretrained ViT-B/16 with VPT-Deep, mask-guided supervision
does *not* help. Test top-1 on the 6149-image test set, three seeds per
row (mean ± std):

| Config                        | MaskMix | `L_attn` | Top-1 (%)     |
|-------------------------------|---------|----------|---------------|
| A1 baseline                   | off     | off      | 98.88 ± 0.23  |
| A2 +MaskMix                   | on      | off      | 97.96 ± 0.39  |
| A3 +AttSup                    | off     | on       | 98.62 ± 0.25  |
| A4 MaskMix + AttSup           | on      | on       | 97.87 ± 0.29  |
| A5 CutMix + AttSup            | rect    | on       | 98.75 ± 0.18  |

Cross-seed Welch's t-tests (`n = 3` per arm, two-sided):

- A2 vs A1: `p = 0.034`, `d = -2.87` (MaskMix alone significantly worse)
- A3 vs A1: `p = 0.262`, `d = -1.07` (AttSup neutral)
- A4 vs A1: `p = 0.010`, `d = -3.86` (combined method significantly worse)
- A5 vs A4: `p = 0.017`, `d = +3.67` (CutMix recovers baseline)

The pixel-accurate paste shape is the harmful ingredient. The gap widens
rather than narrows under data scarcity: at `k = 1` image per class,
baseline 64.99% vs A4 57.20%. See the report for the mechanism discussion.

## Reproducing the training runs (GPU)

Training needs a CUDA GPU. All committed results were produced on one
NVIDIA H100 NVL via vast.ai in bf16 at batch size 128, total wall-clock
about two hours for all four blocks.

### One-time setup on the GPU box

```bash
git clone https://github.com/huytruong2004/SC4001-FinalProject.git /workspace/SC4001-FinalProject
cd /workspace/SC4001-FinalProject
pip install -r requirements.txt

export SC4001_REPO=/workspace/SC4001-FinalProject
export SC4001_DATA=/workspace/data/flowers-102
export SC4001_CKPT=/workspace/checkpoints

python scripts/download_masks.py --data-dir "$SC4001_DATA"
```

The `download_masks.py` step pulls `102segmentations.tgz` (~450 MB) from
the VGG host once and extracts the 8189 per-image trimaps into
`$SC4001_DATA/segmim/`. It is idempotent. The torchvision `Flowers102`
loader downloads the images and labels on first use of the dataset, so
no separate image-download step is needed.

### Run the four experiment blocks

Each `run_block_*.sh` loops `python -m src.train` over its configs and
seeds and appends one `"final": true` record per run to
`results/runs.jsonl`.

```bash
bash scripts/run_block_d.sh    # linear-probe floor, 1 run
bash scripts/run_block_a.sh    # mechanistic ablation, 12 runs (A1–A4 × 3 seeds)
bash scripts/run_block_b.sh    # k-curve, 4 runs (k=1,5 × {baseline, ours})
bash scripts/run_block_c.sh    # paste-shape control, 3 runs (A5 × 3 seeds)
bash scripts/run_analysis.sh   # aggregate CSVs + figures
```

Or invoke a single config directly:

```bash
python -m src.train \
  --config configs/A1_baseline.yaml \
  --seed 0 \
  --data-root "$SC4001_DATA" \
  --checkpoint-dir "$SC4001_CKPT" \
  --results-path "$SC4001_REPO/results/runs.jsonl"
```

### Gotcha: non-interactive SSH does not activate the venv

`ssh host 'nohup bash scripts/run_block_a.sh ...'` does not source
`.bashrc`, so `python` resolves to the system binary and the training
import will fail. Prepend the venv explicitly:

```bash
ssh host 'cd /workspace/SC4001-FinalProject && \
  PATH=/venv/main/bin:$PATH nohup bash scripts/run_block_a.sh > run.log 2>&1 &'
```

## Regenerating analysis locally (no GPU)

All aggregate tables, significance tests, and the learning-curves figure
can be rebuilt on CPU from the committed `results/runs.jsonl`. No
checkpoints required.

```bash
pip install -r requirements.txt
pytest tests/ -v
python -m src.analyze \
  --runs-jsonl results/runs.jsonl \
  --results-dir results \
  --figures-dir figures
python scripts/make_k_curve.py    # rebuilds figures/k_curve.png
```

The optional `--checkpoint-dir` flag unlocks two further outputs that
need trained weights:

- `results/significance_A4_vs_A1_bootstrap.txt`: pre-registered
  per-example paired-bootstrap p-value on the seed-0 A1 vs A4 pair.
- `figures/qualitative_attention.png`: A1 vs A4 attention maps on three
  confusion-pair classes.

The committed `figures/qualitative_attention.png` was rendered from
checkpoints that no longer exist (the training instance was destroyed),
so this flag will only produce new output if you re-train first.

## Outputs

Committed artifacts (regenerable from `results/runs.jsonl`):

- `results/runs.jsonl` — one line per epoch plus one `"final": true`
  record per run; 20 finals across Blocks A/B/C/D.
- `results/headline_table.csv` — 10-config aggregate (A1–A5, D, and
  Block B k-curve) with mean/std over seeds.
- `results/significance.csv` / `significance.txt` — cross-seed Welch's
  t-test for the four contrasts carrying the analysis.
- `figures/learning_curves.png` — val top-1 vs epoch, full and zoomed
  panels; A2 and A4 curves sit visibly below A1/A3/A5.
- `figures/k_curve.png` — test top-1 at k ∈ {1, 5, 10} for A1 vs A4.
- `figures/qualitative_attention.png` — A1 vs A4 attention maps on three
  confusion-pair classes from Nilsback & Zisserman 2008.
- `results/run_logs/run_block_{a,b,b_k1_fix,c,d}.log` — per-block
  training stdout for audit and the appendix.
- `results/runs.jsonl.bak` — pre-`drop_last`-fix audit trail preserving
  the broken k=1 rows from the first Block-B attempt.

## Notebook (legacy)

`notebooks/flowers102_experiments.ipynb` was the early-prototyping entry
point on a Colab T4 kernel. The committed results were produced by the
shell runners above on an H100 NVL box, not by the notebook. The
notebook still works if you prefer cell-by-cell exploration, but it is
not the canonical reproduction path.

## Tests

```bash
pytest tests/ -v
```

23 tests, pure-function and CPU-only, run in under 30 s. Cover MaskMix,
the attention-supervision loss, accuracy metrics, and the paired-bootstrap
routine.

## Libraries

Cited in the report per course FAQ:

- PyTorch (Paszke et al. 2019)
- `timm` (Wightman 2019)
- torchvision Flowers102 dataset (Nilsback & Zisserman 2008)
- scipy (Welch's t-test for the cross-seed significance comparisons)
