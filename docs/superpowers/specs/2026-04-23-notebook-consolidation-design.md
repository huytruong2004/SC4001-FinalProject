# Notebook Consolidation + Runtime Reduction Design

**Status:** design approved (brainstorm 2026-04-23)
**Date:** 2026-04-23
**Author:** Truong Huy (solo project)
**Course:** NTU SC4001 Neural Networks & Deep Learning
**Supersedes execution surface of:** `docs/superpowers/specs/2026-04-22-flowers102-maskmix-vpt-design.md`

This spec consolidates the Flowers-102 MaskMix + VPT project into a single Jupyter notebook executed on a Colab-bridged kernel, and tightens the runtime so the experimental grid completes comfortably inside one Colab session with safety margin for re-runs. The underlying method (MaskMix, mask-guided attention supervision, VPT-Deep on frozen ViT-B/16) is unchanged. Only the execution surface and the runtime budget move.

## 1. Motivation

### Structural

Current repo: a small Python package (`src/`), a one-shot script (`scripts/download_masks.py`), eleven YAML configs (`configs/`), and a thin driver notebook that orchestrates them. The driver used `git clone / git pull / git push` to shuttle code between the local machine and Colab. Cursor's Colab-bridged remote kernel now writes notebook changes to the local disk directly, so the git-sync workflow is dead weight. The current notebook also contains a final cell that pushes results back to GitHub, which is no longer needed.

### Runtime

The prior experimental run hit approximately 5-6 hours and was terminated by Colab before completion. The spec budgeted 20 runs at 10 minutes each (approximately 3.6 hours of training plus 3.4 hours of buffer). Reality significantly exceeded this. Root causes, ranked by impact:
1. Fixed 50-epoch schedule with no early stopping on val top-1. VPT-Deep on frozen ViT-B/16 with 1020 images plateaus well before epoch 50.
2. Batch size 32 on a T4 with 15 GB VRAM. Under-utilizing available memory.
3. Full per-epoch validation over all 1020 val images regardless of improvement trajectory. Not a major cost individually, but compounds over 50 * 20 evaluations.
4. No run-level resume. A Colab disconnect at any point loses the entire session's progress.

## 2. Scope

### In scope
- Replace the `src/` + `scripts/` + `configs/` + driver-notebook layout with a single self-contained notebook.
- Add runtime-reduction levers: shorter epoch cap, early stopping, larger batch size, run-level skip markers, smoke test.
- Rewrite the 23 pytest unit tests to import from the notebook via a small shim, preserving all correctness invariants.
- Delete obsolete files and update `README.md`.

### Out of scope
- The novelty method itself. MaskMix, the attention-supervision loss, VPT-Deep, the choice of backbone, and the evaluation protocol are unchanged.
- The experimental grid structure. Four Block A rows, three Block B k-values (k=1, 5, 10 with k=10 reused from Block A), one Block C row, one Block D row. Seeds remain at 3 per row.
- The report itself. No report changes.
- GPU procurement. Still free Colab T4.
- `docs/`, `figures/`, `results/` directories. Untouched.

### Explicitly not attempted
- Step-level checkpoint resume within a single run. Run-level skip is sufficient given a shorter per-run wall time.
- Learning-rate scaling for the new batch size. Trainable parameter count is ~170K and the existing LR was validated empirically; re-tuning is a confound.
- Linear probe of a non-ViT backbone. Out of novelty scope.

## 3. Runtime reduction package

Five levers, applied together. Each is individually reversible if something breaks.

| Lever | Current | New | Rationale |
|---|---|---|---|
| Epoch cap | 50, no early stopping | 30 max, early-stop patience=5 on val top-1, no min-delta | VPT-Deep on 1020 images typically plateaus around epoch 20 |
| Batch size | 32 | 64 | T4 has sufficient headroom under AMP for ViT-B/16 at B=64; approximately halves steps per epoch |
| Learning rate | 1e-3 | 1e-3 (unchanged) | Trainable is 170K params; linear scaling unnecessary, avoids confound with previous runs |
| Checkpoint body | full model state (~340 MB) | trainable subset (prompts + head, ~700 KB) plus cfg snapshot | Drops Drive I/O latency per best-epoch save; reload reconstructs frozen backbone from timm |
| Run-level skip | none | per-run `final.json` marker; runner cells skip completed runs on re-execution | Colab disconnect recovery without losing completed runs |

### Expected timing (T4, fp16 AMP, B=64)

Per-step forward + backward on frozen ViT-B/16 with VPT trainable: approximately 0.45 s. Train epoch (16 steps): approximately 7 s. Val epoch (1020 images, forward only, no attention capture): approximately 4 s. Expected mean early-stop epoch: approximately 20. Mean per-run wall time: approximately 4-5 minutes.

Block totals:
- Block D (linear probe, 1 run): approximately 3 minutes
- Block A (2x2 ablation, 12 runs): approximately 55 minutes
- Block C (CutMix sanity, 3 runs): approximately 15 minutes
- Block B (k-curve, 4 runs): approximately 20 minutes
- Experiments total: approximately 95 minutes
- Plus mask download, sanity check, smoke test, aggregation, figures: approximately 25 minutes
- **Grand total: approximately 2 hours.** Target ceiling: 4-4.5 hours. Comfortable margin.

If the per-step estimate is 50% optimistic, the grand total doubles to approximately 4 hours, still inside ceiling.

## 4. Notebook architecture

One notebook, `notebooks/flowers102_experiments.ipynb`, 28 cells (indices 0 through 27), all manipulation via the `NotebookEdit` tool.

### Cell sequence

The exact ordering. Every code cell only uses names defined in a previous cell, so top-to-bottom execution works.

| # | Type | Name | Content |
|---|---|---|---|
| 0 | md | Title + TOC | Project title, one-paragraph intent, anchored TOC linking every section |
| 1 | md | `## Setup` | divider |
| 2 | code | `setup` | Mount Drive; set `DATA_ROOT`, `CKPT_ROOT`, `FIGURES_DIR`, `RESULTS_DIR`, `DEVICE`; print torch + CUDA summary. No git. |
| 3 | code | `download_masks` | Idempotent 102segmentations.tgz download with tqdm, truncated-tgz guard, `filter="data"` |
| 4 | md | `## Data` | divider |
| 5 | code | `data` | Seed helpers, `Flowers102WithMasks` dataset (joint hflip, NEAREST mask resize, ImageNet norm, `subsample_k`), mask-alignment sanity plot |
| 6 | md | `## Augmentation (MaskMix)` | divider |
| 7 | code | `maskmix` | `maskmix_batch(x, m, y) -> (x_mix, m_mix, y_mix)`, `_cutmix_batch` reference for Block C |
| 8 | md | `## Loss (attention supervision)` | divider |
| 9 | code | `losses` | `attn_kl_loss` with fp32 upcast and empty-mask masking |
| 10 | md | `## Model (VPT-Deep)` | divider |
| 11 | code | `model` | `VPTDeepViT` with gated attention hooks that slice `[CLS]->patch` in-hook |
| 12 | md | `## Training entrypoint` | divider |
| 13 | code | `train` | `train_one_config(cfg: dict, seed, run_name, data_root, ckpt_root, results_path) -> dict`; early-stop patience=5; trainable-only checkpoint body |
| 14 | md | `## Evaluation + bootstrap` | divider |
| 15 | code | `eval` | `top1_accuracy`, `per_class_mean_accuracy`, `evaluate_full`, `paired_bootstrap_pvalue` (null-recentred two-sided) |
| 16 | md | `## Runners` | divider |
| 17 | code | `configs` | Single `CONFIGS: dict[str, dict]` with all configurations inlined. No YAML. |
| 18 | code | `smoke_test` | Flag-gated (`RUN_SMOKE = True`) 1-config x 1-epoch x 1-seed pipeline integrity check, ~90 s |
| 19 | code | `runner_D` | Block D linear probe, 1 run |
| 20 | code | `runner_A` | Block A 2x2, 12 runs, writes `block_a.jsonl` + `block_a.csv` |
| 21 | code | `runner_C` | Block C CutMix sanity, 3 runs, writes `block_c.csv` |
| 22 | code | `runner_B` | Block B k-curve (k=1 x 2 configs, k=5 x 2 configs = 4 runs; k=10 reused from Block A), writes `block_b.csv` |
| 23 | md | `## Aggregation` | divider |
| 24 | code | `aggregate` | Headline 2x2 table, std bars, bootstrap p-value for A4 vs A1 |
| 25 | md | `## Figures` | divider |
| 26 | code | `figure_kcurve` | k-curve PNG |
| 27 | code | `figure_attention` | 3x3 qualitative attention figure on confusion-pair classes |

### Runner cell contract

Every runner cell follows this exact pattern:

```python
# [runner_A]: Block A 2x2 ablation, 12 runs
BLOCK_A = [("A1_baseline", "baseline", s) for s in [0,1,2]] \
       + [("A2_maskmix",  "maskmix",  s) for s in [0,1,2]] \
       + [("A3_attsup",   "attsup",   s) for s in [0,1,2]] \
       + [("A4_ours",     "ours",     s) for s in [0,1,2]]

results = []
for run_name, config_key, seed in BLOCK_A:
    full_name = f"{run_name}_seed{seed}"
    done = CKPT_ROOT / full_name / "final.json"
    if done.exists():
        results.append(json.loads(done.read_text()))
        print(f"skip {full_name}")
        continue
    result = train_one_config(
        cfg=CONFIGS[config_key], seed=seed, run_name=full_name,
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR/"block_a.jsonl",
    )
    (CKPT_ROOT / full_name).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result))
    results.append(result)

pd.DataFrame(results).to_csv(RESULTS_DIR/"block_a.csv", index=False)
```

Properties:
- Run-level skip via `final.json`. Safe to re-execute any runner cell after a Colab disconnect.
- Per-epoch metrics to `.jsonl` (debug trail). Final metrics to `.csv` (aggregation input).
- `train_one_config` takes a `cfg` dict, not a path.

### Early-stop logic inside `train_one_config`

```python
best_val = -1.0
best_epoch = -1
patience = 5
for epoch in range(cfg["epochs_max"]):          # epochs_max = 30
    train_one_epoch(...)
    val = evaluate_val(...)
    log_jsonl(..., {"epoch": epoch, "val_top1": val})
    if val > best_val:
        best_val = val
        best_epoch = epoch
        torch.save(trainable_state_and_cfg, ckpt_dir/"best.pt")
    if epoch - best_epoch >= patience:
        break
```

No min-delta threshold. A noisy val epoch that fails to exceed the current best keeps the patience counter running. Hard cap at 30 epochs.

### Smoke test cell

```python
# [smoke_test]: pipeline integrity check, ~90 s
RUN_SMOKE = True
if RUN_SMOKE:
    cfg = {**CONFIGS["ours"], "epochs_max": 1, "subsample_k": 2}
    result = train_one_config(cfg=cfg, seed=99, run_name="_smoke",
                              data_root=DATA_ROOT, ckpt_root=CKPT_ROOT/"_smoke",
                              results_path=RESULTS_DIR/"_smoke.jsonl")
    assert result["best_val_top1"] > 0.0, "smoke: val top-1 is zero"
    assert math.isfinite(result["best_val_top1"]), "smoke: non-finite val"
    shutil.rmtree(CKPT_ROOT/"_smoke")
    print("smoke ok")
```

Catches broken mask paths, attention hook shape mismatches, NaN loss, MaskMix tuple unpacking regressions.

## 5. Disk layout on Drive

```
/content/drive/MyDrive/sc4001_flowers102/
    data/                              # DATA_ROOT
        flowers-102/                   # TorchVision layout (jpg/, imagelabels.mat, setid.mat)
        102segmentations/              # downloaded once, reused across sessions
    checkpoints/                       # CKPT_ROOT
        <run_name>/
            best.pt                    # trainable state + cfg; ~700 KB
            final.json                 # skip marker, written at run end
    results/                           # RESULTS_DIR
        block_{a,b,c,d}.jsonl          # per-epoch trace
        block_{a,b,c,d}.csv            # per-run final metrics
    figures/                           # FIGURES_DIR
        k_curve.png
        qualitative_attention.png
        sanity_mask_alignment.png
```

No `/content/sc4001_flowers102_repo/`. No `git clone` anywhere.

### Persistence semantics

- **Mask download:** idempotent, with size and member-count assertion before extraction.
- **Checkpoints:** trainable-only state (`prompts` + `head`) plus cfg snapshot. Reload reconstructs frozen backbone from timm. Per-best-epoch save, not per-epoch.
- **`final.json`:** ~200 bytes, single write at run end. The skip marker.
- **jsonl logs:** append-only, line-delimited. Partial last line tolerated (aggregation ignores unparseable lines).
- **CSVs:** rewritten at runner-cell end from the union of `final.json` markers and newly completed runs.

## 6. Device, dtype, RNG

- `DEVICE = "cuda"` with CPU fallback warning (tests instantiate the model on CPU).
- AMP on for GPU, off for CPU. Trainable params kept in fp32; autocast handles mixed precision.
- `seed_everything(seed)` at the top of each run.
- MaskMix and CutMix use `torch.Generator(device="cpu")` seeded by `seed * 1000 + step`.
- DataLoader `shuffle=True` without an explicit generator, relying on the torch seed set by `seed_everything`. Unchanged from existing behavior.

## 7. Tests infrastructure

### The shim

`tests/_nb_import.py`, approximately 25 lines. Parses the notebook, execs non-side-effectful definition cells in a fresh `types.ModuleType`, exposes the result as `nb`.

Skip markers (cells whose source begins with these are not exec'd at test time):
- `# [setup]`
- `# [download_masks]`
- `# [smoke_test]`
- `# [runner_...]`
- `# [aggregate]`
- `# [figure_...]`

Everything else is a pure definition cell and is safe to exec.

### Test-file edits

Each of the five test files (`test_bootstrap.py`, `test_data.py`, `test_eval.py`, `test_losses.py`, `test_maskmix.py`) replaces `from src.X import Y` with:

```python
from tests._nb_import import nb
maskmix_batch = nb.maskmix_batch
attn_kl_loss  = nb.attn_kl_loss
VPTDeepViT    = nb.VPTDeepViT
# etc
```

Bodies unchanged. All 23 tests continue to pass, on CPU, in under 30 seconds. `pytest.ini` unchanged.

### Correctness invariants the tests must continue to guard

1. `maskmix_batch` returns a 3-tuple `(x_mix, m_mix, y_mix)`.
2. `attn_kl_loss` internally upcasts to fp32.
3. `VPTDeepViT.forward` attention capture is gated on `return_attn`.
4. `paired_bootstrap_pvalue` uses the null-recentred formula `p = mean(|diffs - observed| >= |observed|)`.
5. Data augmentation uses NEAREST interpolation on the mask.

If any of these break during consolidation, it is a bug in the consolidation and must be fixed before proceeding.

## 8. Code style rules (applied inside the notebook)

1. **Single responsibility per function.** Preserved from existing `src/` factoring.
2. **Explicit args, no hidden globals.** The only read-only globals are `DATA_ROOT`, `CKPT_ROOT`, `FIGURES_DIR`, `RESULTS_DIR`, `DEVICE`.
3. **Type hints preserved** in the form `-> tuple[torch.Tensor, Optional[torch.Tensor]]`.
4. **Top-down dependency order strictly enforced.** No forward references.
5. **Comments only for non-obvious invariants,** one short line each:
   - `attn_kl_loss`: fp32 upcast rationale (AMP passes fp16; naive KL NaNs).
   - `VPTDeepViT.forward`: gate + slice in-hook rationale (avoids full attention tensor).
   - Data augmentation: NEAREST rationale (mask stays binary after resize).
   - `maskmix_batch`: 3-tuple return rationale (m_mix matches flipped hard label).
   - `paired_bootstrap_pvalue`: null-recentred two-sided formula note.
6. No commented-out code. No multi-paragraph docstrings. No dead imports.
7. Determinism preserved.
8. Error handling at boundaries only. Training loop trusts its inputs.
9. Logging via tqdm and `log_jsonl`, not print spam. Setup diagnostics are the only print.
10. **Per-cell imports.** Each code cell imports what it needs, even if a previous cell already imported the same module. Makes each cell independently readable when debugging on Colab.
11. No `autoreload`. Only `%matplotlib inline` in figure cells if needed.
12. PEP 8-ish, line length <= 100, no star imports, no bare except.

No em dashes in comments.

## 9. Implementation rules

- **All notebook manipulation via `NotebookEdit`.** No `nbformat` write-whole-file scripts, no raw JSON edits. Cell-by-cell edits via replace / insert / delete by cell index.
- **No git commands in any cell.** The push-to-GitHub cell is removed entirely. Commit discipline is a local concern.
- **No imports from `src.*` or `scripts.*` anywhere.**
- **Configs inlined as Python dicts** in the `CONFIGS` cell. No YAML at runtime.
- **Notebook committed unrun.** Outputs not saved in the committed state.

## 10. File deletions

```
src/                       # all 9 files: __init__.py, utils.py, data.py, maskmix.py,
                           #              losses.py, model.py, eval.py, bootstrap.py, train.py
scripts/                   # download_masks.py
configs/                   # base.yaml + A1..A5 + B_k1/k5 x baseline/ours + D_linear_probe
```

Files to edit:
- `notebooks/flowers102_experiments.ipynb` (rebuilt cell by cell)
- `tests/_nb_import.py` (new)
- `tests/test_data.py`, `test_maskmix.py`, `test_losses.py`, `test_eval.py`, `test_bootstrap.py` (import rewrite only)
- `README.md` (reflect single-notebook workflow)

Files untouched: `docs/`, `figures/`, `results/`, `pytest.ini`, `requirements.txt`, `CLAUDE.md`.

## 11. Commit granularity

Four commits on `main`, each self-contained and individually revertable. No `Co-Authored-By: Claude` trailer; subject lines only.

1. **`chore: add tests/_nb_import.py shim`**. Shim only. Tests still pass against `src/`.
2. **`refactor: consolidate pipeline into single notebook`**. Rebuild the notebook (28 cells) via `NotebookEdit`. No deletions yet. Notebook committed unrun.
3. **`refactor: rewrite tests to import from notebook`**. Rewrite five test files. `pytest` must pass 23/23 against the notebook. `src/` still present but unreferenced.
4. **`chore: remove obsolete src/, scripts/, configs/` + `docs: README single-notebook workflow`**. Deletions plus README update.

Rationale: after commit 2 the notebook runs but tests still hit `src/`. After commit 3 tests hit the notebook but `src/` is dead weight. After commit 4 the repo is clean. Each intermediate state is runnable.

## 12. Rollout order for implementation

1. Write `tests/_nb_import.py`.
2. Build the new notebook cell by cell via `NotebookEdit`, deleting old cells and inserting new ones in the canonical order from Section 4.
3. Rewrite test imports. Run `pytest` locally. Confirm all 23 pass.
4. Delete `src/`, `scripts/`, `configs/`.
5. Update `README.md`.
6. Commit in the four-commit sequence from Section 11.
7. Report back. No push unless explicitly requested.

## 13. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Test failure after import rewrite due to missing definition in the notebook | Medium | Incremental commit sequence: commit 2 lands notebook first without touching tests, so test breakage is isolated to commit 3 |
| Batch size 64 OOMs on T4 under AMP with attention capture | Low | Smoke test catches before full runs; fallback is B=48 |
| Early stopping triggers too aggressively on a noisy val curve | Low | Patience=5 is lenient; hard cap at 30 preserves a ceiling |
| `NotebookEdit` cell index drift during incremental construction | Medium | Build strictly in-order: delete all existing cells first, then insert new cells from index 0 upward |
| Trainable-only checkpoint reload breaks existing aggregation path | Low | Aggregation reads CSVs, not checkpoints; checkpoint format changes are internal to `train_one_config` |
| Colab session drop mid-session | High | Run-level skip via `final.json` recovers within approximately 5 minutes of lost work |

## 14. What this spec does not change

- The MaskMix augmentation formula or its hard-label semantics.
- The attention supervision loss formula or the last-L=2 capture depth.
- The VPT-Deep protocol or the N=10 prompt count.
- The frozen `vit_base_patch16_224.augreg_in21k` backbone.
- The experimental grid (Blocks A, B, C, D) or the seed count (3 seeds for Block A and C).
- The evaluation protocol (top-1 + per-class mean accuracy, paired bootstrap on A4 vs A1).
- The report structure or page budget.

This is an execution-surface refactor plus a runtime tightening. The science is unchanged.
