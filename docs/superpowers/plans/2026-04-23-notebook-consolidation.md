# Notebook Consolidation + Runtime Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the SC4001 Flowers-102 pipeline into a single runnable notebook (no `src/`, no `scripts/`, no `configs/`, no git-sync cells), apply the five-lever runtime-reduction package (epoch cap, early stopping, batch 64, trainable-only checkpoints, run-level skip markers), and keep all 23 pytest unit tests green against the notebook via a small import shim.

**Architecture:** One 28-cell notebook at `notebooks/flowers102_experiments.ipynb`. All notebook manipulation via the `NotebookEdit` tool. Tests continue to run on CPU, importing functions from the notebook through `tests/_nb_import.py` (a ~25-line shim that execs non-side-effectful definition cells in a fresh module namespace). Four self-contained commits in order: shim, notebook, test rewrites, cleanup.

**Tech Stack:** PyTorch + timm (ViT-B/16 backbone), torchvision Flowers102 dataset, pandas for aggregation, matplotlib for figures, pytest for local unit tests, `nbformat` for the shim only (not for notebook edits).

**Design spec:** `docs/superpowers/specs/2026-04-23-notebook-consolidation-design.md` (commit `871347e`, fixup `19d52fe`).

---

## File structure (target end state)

### Created
- `tests/_nb_import.py` — shim module (~25 lines)
- `docs/superpowers/plans/2026-04-23-notebook-consolidation.md` — this file (already written)

### Modified
- `notebooks/flowers102_experiments.ipynb` — rebuilt to 28 cells via `NotebookEdit`
- `tests/test_bootstrap.py`, `test_data.py`, `test_eval.py`, `test_losses.py`, `test_maskmix.py` — import rewrites only
- `README.md` — reflect single-notebook workflow

### Deleted
- `src/` (9 files: `__init__.py`, `utils.py`, `data.py`, `maskmix.py`, `losses.py`, `model.py`, `eval.py`, `bootstrap.py`, `train.py`)
- `scripts/download_masks.py` (whole directory)
- `configs/` (11 YAML files: `base.yaml`, `A1_baseline.yaml`, `A2_maskmix.yaml`, `A3_attsup.yaml`, `A4_ours.yaml`, `A5_cutmix_attsup.yaml`, `B_k1_baseline.yaml`, `B_k1_ours.yaml`, `B_k5_baseline.yaml`, `B_k5_ours.yaml`, `D_linear_probe.yaml`)

### Untouched
- `docs/` (except the spec, which was already committed)
- `figures/`, `results/` (existing run artifacts from the prior session)
- `pytest.ini`, `requirements.txt`, `CLAUDE.md`

---

## Notebook cell inventory (the target)

Every code cell starts with `# [<name>]: <one-line purpose>`. The `<name>` doubles as the test-shim skip marker.

| Index | Type | Name | Purpose |
|---|---|---|---|
| 0 | md | — | Title, one-paragraph intent, anchored TOC |
| 1 | md | — | `## Setup` divider |
| 2 | code | `setup` | Drive mount, path globals, device, torch summary |
| 3 | code | `download_masks` | Idempotent 102segmentations.tgz download |
| 4 | md | — | `## Data` divider |
| 5 | code | `data` | `trimap_to_binary`, `Flowers102WithMasks`, alignment sanity |
| 6 | md | — | `## Augmentation (MaskMix)` divider |
| 7 | code | `maskmix` | `maskmix_batch`, `_cutmix_batch` |
| 8 | md | — | `## Loss (attention supervision)` divider |
| 9 | code | `losses` | `_downsample_mask`, `attn_kl_loss` |
| 10 | md | — | `## Model (VPT-Deep)` divider |
| 11 | code | `model` | `VPTDeepViT` |
| 12 | md | — | `## Training entrypoint` divider |
| 13 | code | `train` | `seed_everything`, `log_jsonl`, `train_one_config` with early-stop + trainable-only ckpt |
| 14 | md | — | `## Evaluation + bootstrap` divider |
| 15 | code | `eval` | `top1_accuracy`, `per_class_mean_accuracy`, `evaluate_full`, `paired_bootstrap_pvalue` |
| 16 | md | — | `## Runners` divider |
| 17 | code | `configs` | `CONFIGS: dict[str, dict]` with all inlined configs |
| 18 | code | `smoke_test` | Flag-gated pipeline integrity check |
| 19 | code | `runner_D` | Block D linear probe (1 run) |
| 20 | code | `runner_A` | Block A 2x2 ablation (12 runs) |
| 21 | code | `runner_C` | Block C CutMix sanity (3 runs) |
| 22 | code | `runner_B` | Block B k-curve (4 runs) |
| 23 | md | — | `## Aggregation` divider |
| 24 | code | `aggregate` | Headline table + bootstrap p-value |
| 25 | md | — | `## Figures` divider |
| 26 | code | `figure_kcurve` | k-curve PNG |
| 27 | code | `figure_attention` | 3x3 qualitative attention figure |

---

## Phase 1 — Add the notebook-import shim

### Task 1: Create `tests/_nb_import.py`

**Files:**
- Create: `tests/_nb_import.py`

- [ ] **Step 1.1: Write the shim file**

```python
"""Import notebook definitions into a test-only module namespace.

Execs the non-side-effectful definition cells of the project notebook and
exposes their top-level names as attributes of a `types.ModuleType` called
`nb`. Tests do `from tests._nb_import import nb` and then
`maskmix_batch = nb.maskmix_batch`.

Cells whose source begins with one of SKIP_MARKERS are *not* exec'd at test
time (Drive mounts, downloads, smoke tests, runners, aggregation, figures).
Everything else is a pure definition cell and is safe to exec on CPU.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "notebooks" / "flowers102_experiments.ipynb"

SKIP_MARKERS = (
    "# [setup]",
    "# [download_masks]",
    "# [smoke_test]",
    "# [runner_",
    "# [aggregate]",
    "# [figure_",
)


def _load() -> types.ModuleType:
    nb = json.loads(NB_PATH.read_text())
    mod = types.ModuleType("flowers102_nb")
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(src.lstrip().startswith(m) for m in SKIP_MARKERS):
            continue
        exec(compile(src, NB_PATH.name, "exec"), mod.__dict__)
    return mod


nb = _load()
```

- [ ] **Step 1.2: Confirm the existing test suite is unaffected**

Run: `pytest tests/ -q`
Expected: 23 passed in under 30 seconds. The shim is imported only by nothing yet; all tests still go through `from src.X import Y`.

- [ ] **Step 1.3: Commit**

```bash
git add tests/_nb_import.py
git commit -m "chore: add tests/_nb_import.py shim"
```

Expected: commit lands; `git log --oneline -1` shows the message.

---

## Phase 2 — Rebuild the notebook cell by cell

All edits in this phase go through `NotebookEdit`. No `nbformat` writes, no raw JSON edits, no `cat > file`.

**Strategy:** delete the existing 13 cells from the tail, then insert 28 new cells starting at index 0. Do not keep any partial state from the old notebook.

### Task 2: Clear the existing notebook

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 2.1: Inspect current cell count**

Run: `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']))"`
Expected: `13`.

- [ ] **Step 2.2: Delete cells from the tail until 1 remains**

`NotebookEdit` rejects deletion of the last cell on most implementations (a notebook needs at least one cell). Delete cells at index 12, 11, ..., 1 by repeatedly calling `NotebookEdit(edit_mode="delete", cell_id=<index-as-string>)` with the current tail index. After each delete re-inspect the count to confirm.

After twelve deletes: `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']))"` should print `1`.

- [ ] **Step 2.3: Replace the remaining cell with the title markdown (becomes cell 0)**

Use `NotebookEdit(edit_mode="replace", cell_id="0", cell_type="markdown", new_source=<title_md>)` where `<title_md>` is:

````markdown
# SC4001 Flowers-102 — MaskMix + VPT-Deep Experiments

**Course:** NTU SC4001 Neural Networks & Deep Learning (solo project)
**Backbone:** frozen `timm/vit_base_patch16_224.augreg_in21k`, VPT-Deep (N=10)
**Novelty:** MaskMix (GT-mask-guided CutMix, hard label) + mask-guided `[CLS]→patch` attention supervision on the last L=2 blocks.
**Execution surface:** Colab T4 via the Cursor remote-kernel bridge. No `git` commands in any cell. Drive-mounted artifacts persist across sessions.

## Table of contents

1. [Setup](#Setup)
2. [Data](#Data)
3. [Augmentation (MaskMix)](#Augmentation-(MaskMix))
4. [Loss (attention supervision)](#Loss-(attention-supervision))
5. [Model (VPT-Deep)](#Model-(VPT-Deep))
6. [Training entrypoint](#Training-entrypoint)
7. [Evaluation + bootstrap](#Evaluation-+-bootstrap)
8. [Runners](#Runners) — Block D, A, C, B
9. [Aggregation](#Aggregation)
10. [Figures](#Figures) — k-curve, qualitative attention

Run top-to-bottom. Runner cells are safe to re-execute after a Colab disconnect; completed runs are skipped via `final.json` markers.
````

- [ ] **Step 2.4: Confirm state**

Run: `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']), nb['cells'][0]['cell_type'])"`
Expected: `1 markdown`.

### Task 3: Insert the Setup section (cells 1-3)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 3.1: Insert cell 1 — `## Setup` divider (markdown)**

```markdown
## Setup
```

Invoke: `NotebookEdit(edit_mode="insert", cell_id="0", cell_type="markdown", new_source="## Setup")` (inserts after cell 0, i.e. becomes cell 1).

- [ ] **Step 3.2: Insert cell 2 — `setup` (code)**

```python
# [setup]: mount Drive, set path globals, device, torch summary
import os
import shutil
from pathlib import Path

import torch

try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/sc4001_flowers102")
except ImportError:
    DRIVE_ROOT = Path.home() / "sc4001_flowers102"

DATA_ROOT    = DRIVE_ROOT / "data" / "flowers-102"
CKPT_ROOT    = DRIVE_ROOT / "checkpoints"
RESULTS_DIR  = DRIVE_ROOT / "results"
FIGURES_DIR  = DRIVE_ROOT / "figures"
for p in (DATA_ROOT, CKPT_ROOT, RESULTS_DIR, FIGURES_DIR):
    p.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")
if DEVICE == "cuda":
    print(f"device: {torch.cuda.get_device_name(0)}, vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("WARNING: no CUDA; runners will be very slow.")
print(f"DRIVE_ROOT={DRIVE_ROOT}")
```

Invoke: `NotebookEdit(edit_mode="insert", cell_id="1", cell_type="code", new_source=<above>)`.

- [ ] **Step 3.3: Insert cell 3 — `download_masks` (code)**

Port verbatim from `scripts/download_masks.py` with these adjustments:
- Remove `from __future__ import annotations`, `argparse`, and the `if __name__ == "__main__"` block.
- Call `download(DATA_ROOT)` at the bottom of the cell so a top-to-bottom run downloads masks immediately.
- Prefix the cell with `# [download_masks]: idempotent 102segmentations.tgz download into DATA_ROOT/segmim/`.
- Keep imports local (`import tarfile`, `import urllib.request`, `from pathlib import Path`, `from tqdm.auto import tqdm`).

Full cell body:

```python
# [download_masks]: idempotent 102segmentations.tgz download into DATA_ROOT/segmim/
import tarfile
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz"
EXPECTED_COUNT = 8189


def _download_with_progress(url: str, dest: Path) -> None:
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name) as bar:
        def hook(blocks: int, block_size: int, total_size: int) -> None:
            if total_size > 0 and bar.total is None:
                bar.total = total_size
            bar.update(blocks * block_size - bar.n)
        urllib.request.urlretrieve(url, dest, reporthook=hook)


def download_masks(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = data_dir / "segmim"
    if mask_dir.exists() and len(list(mask_dir.glob("segmim_*.jpg"))) == EXPECTED_COUNT:
        print(f"Masks already present at {mask_dir} ({EXPECTED_COUNT} files). Skipping.")
        return

    tgz_path = data_dir / "102segmentations.tgz"
    if not tgz_path.exists():
        print(f"Downloading {URL} -> {tgz_path}")
        _download_with_progress(URL, tgz_path)

    print(f"Extracting {tgz_path} -> {data_dir}")
    try:
        with tarfile.open(tgz_path) as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc="extracting", unit="file"):
                tf.extract(member, data_dir, filter="data")
    except (tarfile.ReadError, EOFError) as e:
        tgz_path.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupt archive at {tgz_path} (removed). Re-run to re-download.") from e

    count = len(list(mask_dir.glob("segmim_*.jpg")))
    if count != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} masks, got {count}")
    print(f"OK: {count} masks at {mask_dir}")


download_masks(DATA_ROOT)
```

Invoke: `NotebookEdit(edit_mode="insert", cell_id="2", cell_type="code", new_source=<above>)`.

- [ ] **Step 3.4: Confirm state**

Run: `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']))"`
Expected: `4`.

### Task 4: Insert the Data section (cells 4-5)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 4.1: Insert cell 4 — `## Data` divider (markdown)**

Source: `## Data`

Invoke `NotebookEdit(edit_mode="insert", cell_id="3", cell_type="markdown", new_source="## Data")`.

- [ ] **Step 4.2: Insert cell 5 — `data` (code)**

Port from `src/data.py` (full body) with these adjustments:
- Drop `from __future__ import annotations`.
- Keep all imports local at the top of the cell (`numpy`, `torch`, `PIL.Image`, `torch.utils.data.Dataset`, `torchvision.transforms as T`, `torchvision.datasets.Flowers102`).
- Prefix the cell with `# [data]: Flowers102WithMasks dataset + mask-alignment sanity plot`.
- After the `Flowers102WithMasks` class definition, add a sanity-plot block guarded by `if __name__ != "__main__":` — no, simpler: just run the sanity plot unconditionally inside the cell. This runs once when the cell is executed top-to-bottom; the shim skips this cell via `# [data]` not being in SKIP_MARKERS — wait, the cell is `# [data]:` but that is *not* a skip marker (only `# [setup]`, `# [download_masks]`, `# [smoke_test]`, `# [runner_`, `# [aggregate]`, `# [figure_` are skipped). So the sanity plot would execute during test import, which tries to load real data from DATA_ROOT and will fail on CI. Solution: wrap the sanity plot in `if False:` or keep it in a separate function `_sanity_plot()` that the cell does *not* call, instead document "call `_sanity_plot()` manually to verify". The simpler route is a second code cell for the sanity plot, marked with `# [sanity_mask_alignment]:` and added to SKIP_MARKERS in the shim... but that changes Section 7 of the spec. Keep cell count at 28: embed the sanity call behind a flag `SHOW_MASK_SANITY = False` that the user toggles to `True` when running the notebook live. Tests exec with `SHOW_MASK_SANITY = False` and skip the plot block.

Full cell body:

```python
# [data]: Flowers102WithMasks dataset + mask-alignment sanity plot (gated)
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import Flowers102

BG_RGB = np.array([0, 0, 254], dtype=np.uint8)


def trimap_to_binary(trimap_rgb: np.ndarray) -> np.ndarray:
    diff = np.abs(trimap_rgb.astype(np.int16) - BG_RGB.astype(np.int16)).sum(axis=2)
    return (diff > 10).astype(np.uint8)  # 1 = foreground


class Flowers102WithMasks(Dataset):
    """Returns (image[3,H,W] float, mask[1,H,W] float in {0,1}, label int)."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, split, image_size=224, train_augment=False,
                 subsample_k=None, subsample_seed=0):
        self.root = Path(root)
        self._ds = Flowers102(root=str(self.root), split=split, download=True)
        self._mask_dir = self.root / "segmim"
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Masks not found at {self._mask_dir}; run the download_masks cell.")
        self.image_size = image_size
        self.train_augment = train_augment

        # _image_files / _labels are torchvision Flowers102 private attrs (verified on 0.17).
        self._image_ids = [int(Path(p).stem.split("_")[1]) for p in self._ds._image_files]

        if subsample_k is not None:
            self._subsample(subsample_k, subsample_seed)

        resize_short = int(self.image_size * 256 / 224)
        self._image_resize = T.Compose([T.Resize(resize_short), T.CenterCrop(self.image_size)])
        # NEAREST so mask stays binary after resize.
        self._mask_resize = T.Compose([T.Resize(resize_short, interpolation=T.InterpolationMode.NEAREST),
                                        T.CenterCrop(self.image_size)])
        self._normalize = T.Compose([T.ToTensor(), T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])
        self._train_photo = T.Compose([T.RandAugment(num_ops=2, magnitude=9),
                                        T.ToTensor(),
                                        T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])

    def _subsample(self, k, seed):
        rng = np.random.default_rng(seed)
        labels = np.array(self._ds._labels)
        keep = []
        for c in np.unique(labels):
            idxs = np.where(labels == c)[0]
            rng.shuffle(idxs)
            keep.extend(idxs[:k].tolist())
        keep.sort()
        self._ds._image_files = [self._ds._image_files[i] for i in keep]
        self._ds._labels = [self._ds._labels[i] for i in keep]
        self._image_ids = [self._image_ids[i] for i in keep]

    def __len__(self):
        return len(self._ds)

    def _load_mask(self, image_id):
        with Image.open(self._mask_dir / f"segmim_{image_id:05d}.jpg") as im:
            rgb = np.array(im.convert("RGB"))
        return trimap_to_binary(rgb)

    def __getitem__(self, i):
        pil_img, label = self._ds[i]
        image_id = self._image_ids[i]
        pil_img = self._image_resize(pil_img)
        mask_np = self._load_mask(image_id)
        mask_pil = self._mask_resize(Image.fromarray(mask_np * 255).convert("L"))
        if self.train_augment:
            if torch.rand(()) < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_tensor = self._train_photo(pil_img)
        else:
            img_tensor = self._normalize(pil_img)
        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.uint8)).float().unsqueeze(0) / 255.0
        return img_tensor, mask_tensor, int(label)


SHOW_MASK_SANITY = False  # set True at runtime for the Colab alignment check
if SHOW_MASK_SANITY:
    import matplotlib.pyplot as plt
    ds = Flowers102WithMasks(root=DATA_ROOT, split="train", train_augment=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, idx in enumerate(np.random.RandomState(0).choice(len(ds), size=5, replace=False)):
        img, mask, y = ds[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array(Flowers102WithMasks.IMAGENET_STD)
                  + np.array(Flowers102WithMasks.IMAGENET_MEAN)).clip(0, 1)
        axes[0, i].imshow(img_np); axes[0, i].axis("off"); axes[0, i].set_title(f"cls {y}")
        axes[1, i].imshow(mask.squeeze().numpy(), cmap="gray"); axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sanity_mask_alignment.png", dpi=120)
    plt.show()
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="4", cell_type="code", new_source=<above>)`.

- [ ] **Step 4.3: Confirm state**

`python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']))"` → `6`.

### Task 5: Insert the Augmentation section (cells 6-7)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 5.1: Insert cell 6 — `## Augmentation (MaskMix)` divider (markdown)**

Source: `## Augmentation (MaskMix)`

Invoke `NotebookEdit(edit_mode="insert", cell_id="5", cell_type="markdown", new_source="## Augmentation (MaskMix)")`.

- [ ] **Step 5.2: Insert cell 7 — `maskmix` (code)**

Port from `src/maskmix.py` verbatim (drop `from __future__`), plus the `_cutmix_batch` function from `src/train.py:41-70`. Full body:

```python
# [maskmix]: GT-mask CutMix variant; returns (x_mix, m_mix, y_mix) with hard labels
from typing import Optional

import torch


def maskmix_batch(
    x: torch.Tensor,           # (B, C, H, W)
    m: torch.Tensor,           # (B, 1, H, W), values in [0,1]
    y: torch.Tensor,           # (B,)
    prob: float = 0.5,
    seed: Optional[int] = None,
    _force_source_index: Optional[torch.Tensor] = None,  # test hook
):
    """Returns (x_mix, m_mix, y_mix); m_mix matches the flipped hard label so attn sup stays consistent."""
    B = x.size(0)
    if B < 2:
        return x.clone(), m.clone(), y.clone()
    device = x.device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    apply_mask = torch.rand(B, generator=gen) < prob
    apply_mask = apply_mask.to(device)

    if _force_source_index is not None:
        src = _force_source_index.to(device)
    else:
        src = torch.randperm(B, generator=gen).to(device)
        same = (src == torch.arange(B, device=device))
        if same.any():
            src[same] = (src[same] + 1) % B

    x_src = x[src]; m_src = m[src]; y_src = y[src]
    apply_img = apply_mask.view(B, 1, 1, 1)
    use_src = apply_img & (m_src > 0.5)
    x_mix = torch.where(use_src, x_src, x)
    m_mix = torch.where(apply_img, m_src, m)
    y_mix = torch.where(apply_mask, y_src, y)
    return x_mix, m_mix, y_mix


def _cutmix_batch(x, y, alpha=1.0, prob=0.5, seed=None):
    """Reference CutMix for Block C sanity row. Hard label: source if pasted area > 50% of image."""
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
    patch_frac = (x2 - x1) * (y2 - y1) / (H * W)
    use_src = apply & (patch_frac > 0.5)
    y_mix = torch.where(use_src, y[src], y)
    x_mix = torch.where(apply.view(B, 1, 1, 1), x_mix, x)
    return x_mix, y_mix
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="6", cell_type="code", new_source=<above>)`.

- [ ] **Step 5.3: Confirm state**

Expected cell count after this task: `8`.

### Task 6: Insert the Loss section (cells 8-9)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 6.1: Insert cell 8 — `## Loss (attention supervision)` divider (markdown)**

Source: `## Loss (attention supervision)`

Invoke `NotebookEdit(edit_mode="insert", cell_id="7", cell_type="markdown", new_source="## Loss (attention supervision)")`.

- [ ] **Step 6.2: Insert cell 9 — `losses` (code)**

Port from `src/losses.py` verbatim (drop `from __future__`). Full body:

```python
# [losses]: mask-guided attention supervision loss
import torch
import torch.nn.functional as F


def _downsample_mask(mask, side):
    m = F.adaptive_avg_pool2d(mask, output_size=(side, side))
    return m.flatten(1)


def attn_kl_loss(attn, mask):
    """Mean KL(p_mask || p_attn). Empty-mask samples contribute 0. Upcasts attn to fp32 (AMP passes fp16)."""
    _, N2 = attn.shape
    side = int(round(N2 ** 0.5))
    assert side * side == N2, f"attn length {N2} is not a perfect square"

    m_down = _downsample_mask(mask, side).detach()
    m_sum = m_down.sum(dim=1)
    valid = m_sum > 1e-6
    if not valid.any():
        return attn.new_zeros(())

    p_mask = m_down[valid] / m_sum[valid].unsqueeze(1).clamp(min=1e-8)
    # upcast to fp32; AMP passes fp16 and a naive KL will NaN.
    a = attn[valid].float()
    a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_a = (a + 1e-12).log()
    return F.kl_div(log_a, p_mask, reduction="batchmean")
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="8", cell_type="code", new_source=<above>)`.

- [ ] **Step 6.3: Confirm state**

Expected cell count: `10`.

### Task 7: Insert the Model section (cells 10-11)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 7.1: Insert cell 10 — `## Model (VPT-Deep)` divider (markdown)**

Source: `## Model (VPT-Deep)`

Invoke `NotebookEdit(edit_mode="insert", cell_id="9", cell_type="markdown", new_source="## Model (VPT-Deep)")`.

- [ ] **Step 7.2: Insert cell 11 — `model` (code)**

Port from `src/model.py` verbatim (drop `from __future__`). Full body:

```python
# [model]: VPT-Deep wrapping timm ViT-B/16 with gated [CLS]->patch attention capture
from typing import Optional

import timm
import torch
import torch.nn as nn


class VPTDeepViT(nn.Module):
    def __init__(self,
                 backbone_name: str = "vit_base_patch16_224.augreg_in21k",
                 num_prompts: int = 10,
                 num_classes: int = 102,
                 capture_last_layers: int = 2,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.num_prompts = num_prompts
        self.capture_last_layers = capture_last_layers
        d = self.backbone.embed_dim
        L = len(self.backbone.blocks)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.prompts = nn.Parameter(torch.empty(L, num_prompts, d))
        nn.init.uniform_(self.prompts, -0.1, 0.1)
        self.head = nn.Linear(d, num_classes)

        self._attn_scores: list = []
        self._capture_enabled: bool = False
        self._install_attn_hooks()

    def _install_attn_hooks(self):
        self._hook_handles = []
        for block in self.backbone.blocks[-self.capture_last_layers:]:
            attn_mod = block.attn
            num_heads = attn_mod.num_heads
            head_dim = getattr(attn_mod, "head_dim",
                               block.attn.qkv.out_features // 3 // num_heads)
            scale = attn_mod.scale

            def make_hook(num_heads=num_heads, head_dim=head_dim, scale=scale):
                def hook(module, inp, out):
                    # gate + slice in-hook: avoids holding (B, H, N+197, N+197) for baseline runs.
                    if not self._capture_enabled:
                        return
                    B, N, _ = out.shape
                    qkv = out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    cls_idx = self.num_prompts
                    patch_start = self.num_prompts + 1
                    patch_end = self.num_prompts + 197
                    self._attn_scores.append(attn[:, :, cls_idx, patch_start:patch_end])
                return hook
            self._hook_handles.append(attn_mod.qkv.register_forward_hook(make_hook()))

    def trainable_parameters(self):
        yield self.prompts
        yield from self.head.parameters()

    def forward(self, x, return_attn: bool = False):
        self._attn_scores.clear()
        self._capture_enabled = bool(return_attn)
        B = x.size(0)
        x = self.backbone.patch_embed(x)
        cls = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        for i, block in enumerate(self.backbone.blocks):
            p = self.prompts[i].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([p, x], dim=1)
            x = block(x)
            x = x[:, self.num_prompts:]
        x = self.backbone.norm(x)
        logits = self.head(x[:, 0])
        attn_flat = None
        if return_attn and self._attn_scores:
            per_layer = [a.mean(dim=1) for a in self._attn_scores]
            attn_flat = torch.stack(per_layer, dim=0).mean(dim=0)
        self._capture_enabled = False
        return logits, attn_flat
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="10", cell_type="code", new_source=<above>)`.

- [ ] **Step 7.3: Confirm state**

Expected cell count: `12`.

### Task 8: Insert the Training-entrypoint section (cells 12-13)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 8.1: Insert cell 12 — `## Training entrypoint` divider (markdown)**

Source: `## Training entrypoint`

Invoke `NotebookEdit(edit_mode="insert", cell_id="11", cell_type="markdown", new_source="## Training entrypoint")`.

- [ ] **Step 8.2: Insert cell 13 — `train` (code)**

New training entrypoint with four changes vs `src/train.py`:
1. Takes `cfg: dict` directly (no YAML path loader).
2. Reads `cfg["epochs_max"]` (default 30) with early-stop patience=5, no min-delta.
3. Saves trainable-only state (`prompts.data`, `head.state_dict()`) plus cfg snapshot in `best.pt`.
4. Accepts `ckpt_root` and builds `ckpt_root/run_name/best.pt`.

Full body:

```python
# [train]: train_one_config(cfg, seed, ...) with early-stop patience=5 and trainable-only checkpoint
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _build_model(cfg):
    if cfg.get("adaptation") == "linear_probe":
        m = VPTDeepViT(num_prompts=1, num_classes=102, freeze_backbone=True)
        m.prompts.data.zero_()
        m.prompts.requires_grad = False
        trainable = list(m.head.parameters())
    else:
        m = VPTDeepViT(
            num_prompts=cfg.get("num_prompts", 10),
            num_classes=102,
            capture_last_layers=cfg.get("capture_last_layers", 2),
        )
        trainable = [m.prompts] + list(m.head.parameters())
    return m, trainable


def _save_trainable(model, cfg, val_top1, epoch, path):
    state = {
        "prompts": model.prompts.detach().cpu(),
        "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        "cfg": cfg, "epoch": epoch, "val_top1": val_top1,
    }
    torch.save(state, path)


def _load_trainable(model, path, device):
    st = torch.load(path, map_location=device)
    model.prompts.data.copy_(st["prompts"].to(device))
    model.head.load_state_dict({k: v.to(device) for k, v in st["head"].items()})
    return st


@torch.no_grad()
def _val_top1(model, loader, device):
    model.eval()
    correct = total = 0
    for x, _m, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, _ = model(x, return_attn=False)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_config(cfg, seed, run_name, data_root, ckpt_root, results_path):
    """Train one run. Returns dict with best_val_top1, test_top1, test_per_class_mean, etc."""
    seed_everything(seed)
    device = DEVICE
    train_ds = Flowers102WithMasks(root=data_root, split="train", image_size=cfg["image_size"],
                                    train_augment=True,
                                    subsample_k=cfg.get("subsample_k"),
                                    subsample_seed=cfg.get("subsample_seed", 0))
    val_ds = Flowers102WithMasks(root=data_root, split="val",
                                  image_size=cfg["image_size"], train_augment=False)
    test_ds = Flowers102WithMasks(root=data_root, split="test",
                                   image_size=cfg["image_size"], train_augment=False)

    bs = cfg["batch_size"]; nw = cfg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model, trainable = _build_model(cfg)
    model = model.to(device)
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs_max"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_dir = Path(ckpt_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0; best_epoch = -1
    patience = cfg.get("patience", 5)

    for epoch in range(cfg["epochs_max"]):
        model.train()
        running = 0.0
        for step, (x, m, y) in enumerate(tqdm(train_loader, desc=f"{run_name} ep{epoch}", leave=False)):
            x = x.to(device, non_blocking=True); m = m.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if cfg.get("augment") == "maskmix":
                x, m, y = maskmix_batch(x, m, y, prob=cfg.get("mix_prob", 0.5),
                                         seed=seed * 1000 + step)
            elif cfg.get("augment") == "cutmix":
                x, y = _cutmix_batch(x, y, alpha=1.0, prob=cfg.get("mix_prob", 0.5),
                                      seed=seed * 1000 + step)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, attn = model(x, return_attn=cfg.get("attn_lambda", 0.0) > 0)
                loss = F.cross_entropy(logits, y)
                if cfg.get("attn_lambda", 0.0) > 0 and attn is not None:
                    loss = loss + cfg["attn_lambda"] * attn_kl_loss(attn, m)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        val = _val_top1(model, val_loader, device)
        log_jsonl(results_path, {"run": run_name, "epoch": epoch,
                                 "train_loss": running / len(train_ds), "val_top1": val})
        if val > best_val:
            best_val = val; best_epoch = epoch
            _save_trainable(model, cfg, val, epoch, ckpt_dir / "best.pt")
        if epoch - best_epoch >= patience:
            break

    # Reload best and run test.
    _load_trainable(model, ckpt_dir / "best.pt", device)
    test_metrics = evaluate_full(model, test_loader, device, num_classes=102)
    result = {"run": run_name, "best_val_top1": best_val, **test_metrics,
              "seed": seed, "cfg": cfg}
    log_jsonl(results_path, {"final": True, **result})
    return result
```

Note: `train_one_config` references `VPTDeepViT`, `Flowers102WithMasks`, `maskmix_batch`, `_cutmix_batch`, `attn_kl_loss`, `evaluate_full`, and `DEVICE`. All are defined in earlier cells or will be in the next cell (`evaluate_full` is in cell 15).

Invoke `NotebookEdit(edit_mode="insert", cell_id="12", cell_type="code", new_source=<above>)`.

- [ ] **Step 8.3: Confirm state**

Expected cell count: `14`.

### Task 9: Insert the Evaluation+bootstrap section (cells 14-15)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 9.1: Insert cell 14 — `## Evaluation + bootstrap` divider (markdown)**

Source: `## Evaluation + bootstrap`

Invoke `NotebookEdit(edit_mode="insert", cell_id="13", cell_type="markdown", new_source="## Evaluation + bootstrap")`.

- [ ] **Step 9.2: Insert cell 15 — `eval` (code)**

Port from `src/eval.py` + `src/bootstrap.py` (drop `from __future__`). Full body:

```python
# [eval]: top1/per-class mean accuracy, full-set evaluator, paired bootstrap (null-recentred)
import numpy as np
import torch


def top1_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def per_class_mean_accuracy(preds, labels, num_classes):
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == c).float().mean().item())
    return float(sum(accs) / len(accs)) if accs else 0.0


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    for x, _m, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, _ = model(x, return_attn=False)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds); labels = torch.cat(all_labels)
    return {"test_top1": top1_accuracy(preds, labels),
            "test_per_class_mean": per_class_mean_accuracy(preds, labels, num_classes),
            "num_test": int(preds.numel()),
            "_preds": preds.numpy(), "_labels": labels.numpy()}


def paired_bootstrap_pvalue(correct_a, correct_b, n_resamples=5000, seed=0):
    """Null-recentred two-sided: p = mean(|diffs - observed| >= |observed|)."""
    assert correct_a.shape == correct_b.shape
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    N = correct_a.shape[0]
    rng = np.random.default_rng(seed)
    observed = float(correct_a.mean() - correct_b.mean())
    idx = rng.integers(0, N, size=(n_resamples, N))
    diffs = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
    if observed == 0.0:
        return 0.0, 1.0
    return observed, float((np.abs(diffs - observed) >= abs(observed)).mean())
```

Note: `evaluate_full` now returns `_preds` and `_labels` (as numpy arrays) so the bootstrap cell can use them. Existing tests for `evaluate_full` do not assert against these keys, so the addition is backward-compatible.

Invoke `NotebookEdit(edit_mode="insert", cell_id="14", cell_type="code", new_source=<above>)`.

- [ ] **Step 9.3: Confirm state**

Expected cell count: `16`.

### Task 10: Insert the Runners section (cells 16-22)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 10.1: Insert cell 16 — `## Runners` divider (markdown)**

Source: `## Runners`

Invoke `NotebookEdit(edit_mode="insert", cell_id="15", cell_type="markdown", new_source="## Runners")`.

- [ ] **Step 10.2: Insert cell 17 — `configs` (code)**

Single source of truth for all configurations. Every field that base.yaml / A*.yaml / B*.yaml / D_linear_probe.yaml set becomes a key here.

```python
# [configs]: single inlined CONFIGS dict replacing configs/*.yaml
BASE_CFG = {
    "image_size": 224,
    "batch_size": 64,          # was 32; T4 headroom under AMP
    "num_workers": 2,
    "epochs_max": 30,          # was 50; early-stop patience=5 almost always trips first
    "patience": 5,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "num_prompts": 10,
    "capture_last_layers": 2,
    "adaptation": "vpt_deep",
    "augment": "none",         # none | maskmix | cutmix
    "mix_prob": 0.5,
    "attn_lambda": 0.0,
    "subsample_k": None,
    "subsample_seed": 0,
}

def _with(**overrides):
    return {**BASE_CFG, **overrides}

CONFIGS = {
    # Block A (2x2)
    "baseline":   _with(),
    "maskmix":    _with(augment="maskmix"),
    "attsup":     _with(attn_lambda=0.3),
    "ours":       _with(augment="maskmix", attn_lambda=0.3),
    # Block C (CutMix sanity)
    "cutmix_attsup": _with(augment="cutmix", attn_lambda=0.3),
    # Block B (k-curve)
    "k1_baseline":   _with(subsample_k=1),
    "k1_ours":       _with(subsample_k=1, augment="maskmix", attn_lambda=0.3),
    "k5_baseline":   _with(subsample_k=5),
    "k5_ours":       _with(subsample_k=5, augment="maskmix", attn_lambda=0.3),
    # Block D (linear probe floor)
    "linear_probe":  _with(adaptation="linear_probe"),
}
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="16", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.3: Insert cell 18 — `smoke_test` (code)**

```python
# [smoke_test]: 1 config x 1 epoch x k=2 subsample, asserts pipeline integrity (~90s)
import math
import shutil

RUN_SMOKE = True
if RUN_SMOKE:
    cfg = {**CONFIGS["ours"], "epochs_max": 1, "subsample_k": 2, "patience": 2}
    result = train_one_config(
        cfg=cfg, seed=99, run_name="_smoke",
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR / "_smoke.jsonl",
    )
    assert result["best_val_top1"] > 0.0, "smoke: val top-1 is zero"
    assert math.isfinite(result["best_val_top1"]), "smoke: non-finite val"
    shutil.rmtree(CKPT_ROOT / "_smoke", ignore_errors=True)
    (RESULTS_DIR / "_smoke.jsonl").unlink(missing_ok=True)
    print("smoke ok")
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="17", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.4: Insert cell 19 — `runner_D` (code)**

```python
# [runner_D]: Block D linear probe, 1 run
import json

import pandas as pd

run_name = "D_linear_probe_seed0"
done = CKPT_ROOT / run_name / "final.json"
if done.exists():
    block_d_result = json.loads(done.read_text())
    print(f"skip {run_name}")
else:
    block_d_result = train_one_config(
        cfg=CONFIGS["linear_probe"], seed=0, run_name=run_name,
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR / "block_d.jsonl",
    )
    (CKPT_ROOT / run_name).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({k: v for k, v in block_d_result.items() if k not in ("_preds", "_labels")}))

pd.DataFrame([{k: v for k, v in block_d_result.items() if k not in ("_preds", "_labels")}]).to_csv(
    RESULTS_DIR / "block_d.csv", index=False)
```

Note: `final.json` and the CSV exclude `_preds` / `_labels` (numpy arrays from `evaluate_full`). The in-memory dict keeps them for the bootstrap cell.

Invoke `NotebookEdit(edit_mode="insert", cell_id="18", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.5: Insert cell 20 — `runner_A` (code)**

```python
# [runner_A]: Block A 2x2 ablation, 12 runs (4 configs x 3 seeds)
import json

import pandas as pd

BLOCK_A = (
    [("A1_baseline", "baseline", s) for s in [0, 1, 2]]
    + [("A2_maskmix", "maskmix", s) for s in [0, 1, 2]]
    + [("A3_attsup",  "attsup",  s) for s in [0, 1, 2]]
    + [("A4_ours",    "ours",    s) for s in [0, 1, 2]]
)

block_a_results = []
for base_name, config_key, seed in BLOCK_A:
    full_name = f"{base_name}_seed{seed}"
    done = CKPT_ROOT / full_name / "final.json"
    if done.exists():
        block_a_results.append(json.loads(done.read_text()))
        print(f"skip {full_name}")
        continue
    result = train_one_config(
        cfg=CONFIGS[config_key], seed=seed, run_name=full_name,
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR / "block_a.jsonl",
    )
    (CKPT_ROOT / full_name).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({k: v for k, v in result.items() if k not in ("_preds", "_labels")}))
    block_a_results.append(result)

pd.DataFrame([{k: v for k, v in r.items() if k not in ("_preds", "_labels")} for r in block_a_results]) \
  .to_csv(RESULTS_DIR / "block_a.csv", index=False)
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="19", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.6: Insert cell 21 — `runner_C` (code)**

```python
# [runner_C]: Block C CutMix sanity, 3 runs (1 config x 3 seeds)
import json

import pandas as pd

BLOCK_C = [("A5_cutmix_attsup", "cutmix_attsup", s) for s in [0, 1, 2]]

block_c_results = []
for base_name, config_key, seed in BLOCK_C:
    full_name = f"{base_name}_seed{seed}"
    done = CKPT_ROOT / full_name / "final.json"
    if done.exists():
        block_c_results.append(json.loads(done.read_text()))
        print(f"skip {full_name}")
        continue
    result = train_one_config(
        cfg=CONFIGS[config_key], seed=seed, run_name=full_name,
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR / "block_c.jsonl",
    )
    (CKPT_ROOT / full_name).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({k: v for k, v in result.items() if k not in ("_preds", "_labels")}))
    block_c_results.append(result)

pd.DataFrame([{k: v for k, v in r.items() if k not in ("_preds", "_labels")} for r in block_c_results]) \
  .to_csv(RESULTS_DIR / "block_c.csv", index=False)
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="20", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.7: Insert cell 22 — `runner_B` (code)**

```python
# [runner_B]: Block B k-curve, 4 runs (k=1 baseline+ours, k=5 baseline+ours; k=10 reused from Block A)
import json

import pandas as pd

BLOCK_B = [
    ("B_k1_baseline", "k1_baseline", 0),
    ("B_k1_ours",     "k1_ours",     0),
    ("B_k5_baseline", "k5_baseline", 0),
    ("B_k5_ours",     "k5_ours",     0),
]

block_b_results = []
for base_name, config_key, seed in BLOCK_B:
    full_name = f"{base_name}_seed{seed}"
    done = CKPT_ROOT / full_name / "final.json"
    if done.exists():
        block_b_results.append(json.loads(done.read_text()))
        print(f"skip {full_name}")
        continue
    result = train_one_config(
        cfg=CONFIGS[config_key], seed=seed, run_name=full_name,
        data_root=DATA_ROOT, ckpt_root=CKPT_ROOT,
        results_path=RESULTS_DIR / "block_b.jsonl",
    )
    (CKPT_ROOT / full_name).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({k: v for k, v in result.items() if k not in ("_preds", "_labels")}))
    block_b_results.append(result)

pd.DataFrame([{k: v for k, v in r.items() if k not in ("_preds", "_labels")} for r in block_b_results]) \
  .to_csv(RESULTS_DIR / "block_b.csv", index=False)
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="21", cell_type="code", new_source=<above>)`.

- [ ] **Step 10.8: Confirm state**

Expected cell count: `23`.

### Task 11: Insert the Aggregation section (cells 23-24)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 11.1: Insert cell 23 — `## Aggregation` divider (markdown)**

Source: `## Aggregation`

Invoke `NotebookEdit(edit_mode="insert", cell_id="22", cell_type="markdown", new_source="## Aggregation")`.

- [ ] **Step 11.2: Insert cell 24 — `aggregate` (code)**

```python
# [aggregate]: headline 2x2 table (mean +/- std over seeds) + A4-vs-A1 paired bootstrap
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

df_a = pd.read_csv(RESULTS_DIR / "block_a.csv")
# Recover config key from "<name>_seed<k>".
df_a["config"] = df_a["run"].str.extract(r"^(A[1-4]_[^_]+(?:_[^_]+)?)_seed")
summary = df_a.groupby("config")[["test_top1", "test_per_class_mean"]].agg(["mean", "std"]).round(4)
print(summary)
summary.to_csv(RESULTS_DIR / "block_a_summary.csv")

# A4 ours vs A1 baseline paired bootstrap, seed-0 run of each.
test_ds = Flowers102WithMasks(root=DATA_ROOT, split="test", image_size=224, train_augment=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

def _load_and_eval(run_name):
    model = VPTDeepViT(num_prompts=10, num_classes=102, capture_last_layers=2).to(DEVICE)
    _load_trainable(model, CKPT_ROOT / run_name / "best.pt", DEVICE)
    r = evaluate_full(model, test_loader, DEVICE, num_classes=102)
    return r["_preds"], r["_labels"]

preds_a4, labels_a4 = _load_and_eval("A4_ours_seed0")
preds_a1, labels_a1 = _load_and_eval("A1_baseline_seed0")
assert np.array_equal(labels_a4, labels_a1), "test-set order must match across runs"

correct_a4 = (preds_a4 == labels_a4)
correct_a1 = (preds_a1 == labels_a1)
mean_diff, p = paired_bootstrap_pvalue(correct_a4, correct_a1, n_resamples=5000, seed=0)
print(f"A4 vs A1: mean_diff={mean_diff:+.4f}, two-sided p={p:.4f}")
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="23", cell_type="code", new_source=<above>)`.

- [ ] **Step 11.3: Confirm state**

Expected cell count: `25`.

### Task 12: Insert the Figures section (cells 25-27)

**Files:**
- Modify: `notebooks/flowers102_experiments.ipynb`

- [ ] **Step 12.1: Insert cell 25 — `## Figures` divider (markdown)**

Source: `## Figures`

Invoke `NotebookEdit(edit_mode="insert", cell_id="24", cell_type="markdown", new_source="## Figures")`.

- [ ] **Step 12.2: Insert cell 26 — `figure_kcurve` (code)**

```python
# [figure_kcurve]: test accuracy vs k for baseline vs ours; std shading at k=10 from Block A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_b = pd.read_csv(RESULTS_DIR / "block_b.csv")
df_a = pd.read_csv(RESULTS_DIR / "block_a.csv")

# k=10 mean and std from Block A (A1 and A4, 3 seeds each).
df_a["config"] = df_a["run"].str.extract(r"^(A[1-4]_[^_]+(?:_[^_]+)?)_seed")
k10_base_mean = df_a.loc[df_a["config"] == "A1_baseline", "test_top1"].mean()
k10_base_std  = df_a.loc[df_a["config"] == "A1_baseline", "test_top1"].std()
k10_ours_mean = df_a.loc[df_a["config"] == "A4_ours",     "test_top1"].mean()
k10_ours_std  = df_a.loc[df_a["config"] == "A4_ours",     "test_top1"].std()

# k=1, k=5 from Block B (1 seed each).
df_b["config"] = df_b["run"].str.extract(r"^(B_k[15]_[^_]+)_seed")
k1_base = df_b.loc[df_b["config"] == "B_k1_baseline", "test_top1"].values[0]
k1_ours = df_b.loc[df_b["config"] == "B_k1_ours",     "test_top1"].values[0]
k5_base = df_b.loc[df_b["config"] == "B_k5_baseline", "test_top1"].values[0]
k5_ours = df_b.loc[df_b["config"] == "B_k5_ours",     "test_top1"].values[0]

ks = np.array([1, 5, 10])
base_y = np.array([k1_base, k5_base, k10_base_mean])
ours_y = np.array([k1_ours, k5_ours, k10_ours_mean])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ks, base_y, "o-", label="baseline (A1)")
ax.plot(ks, ours_y, "s-", label="ours (A4)")
ax.errorbar([10], [k10_base_mean], yerr=[k10_base_std], fmt="o", color="C0", capsize=3)
ax.errorbar([10], [k10_ours_mean], yerr=[k10_ours_std], fmt="s", color="C1", capsize=3)
ax.set_xlabel("k (training images per class)")
ax.set_ylabel("test top-1")
ax.set_xticks(ks)
ax.set_title("k-curve: gap between ours and baseline widens as k shrinks")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "k_curve.png", dpi=160)
plt.show()
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="25", cell_type="code", new_source=<above>)`.

- [ ] **Step 12.3: Insert cell 27 — `figure_attention` (code)**

```python
# [figure_attention]: 3x3 qualitative attention grid on confusion-pair classes
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Confusion pairs per Nilsback 2008: tree mallow vs pink primrose, globe thistle vs thorn apple.
# Class ids for Flowers102 (torchvision's label space is 0-indexed, matches VGG's imagelabels.mat 1-102).
CONFUSION_CLASSES = [62, 64, 10, 27, 5, 33]  # six visually tricky classes; adjust at runtime.

ds = Flowers102WithMasks(root=DATA_ROOT, split="test", image_size=224, train_augment=False)

model_base = VPTDeepViT(num_prompts=10, num_classes=102, capture_last_layers=2).to(DEVICE)
_load_trainable(model_base, CKPT_ROOT / "A1_baseline_seed0" / "best.pt", DEVICE)
model_ours = VPTDeepViT(num_prompts=10, num_classes=102, capture_last_layers=2).to(DEVICE)
_load_trainable(model_ours, CKPT_ROOT / "A4_ours_seed0" / "best.pt", DEVICE)

# Pick one sample per class.
picked = {}
for i, (_, _, y) in enumerate(ds):
    if y in CONFUSION_CLASSES and y not in picked:
        picked[y] = i
    if len(picked) == len(CONFUSION_CLASSES):
        break

fig, axes = plt.subplots(len(CONFUSION_CLASSES), 4, figsize=(12, 3 * len(CONFUSION_CLASSES)))
for row, c in enumerate(CONFUSION_CLASSES):
    img, mask, y = ds[picked[c]]
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * np.array(Flowers102WithMasks.IMAGENET_STD)
              + np.array(Flowers102WithMasks.IMAGENET_MEAN)).clip(0, 1)

    x = img.unsqueeze(0).to(DEVICE)
    model_base.eval(); model_ours.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        _, attn_b = model_base(x, return_attn=True)
        _, attn_o = model_ours(x, return_attn=True)
    attn_b = attn_b[0].float().cpu().numpy().reshape(14, 14)
    attn_o = attn_o[0].float().cpu().numpy().reshape(14, 14)

    axes[row, 0].imshow(img_np); axes[row, 0].set_title(f"cls {c}"); axes[row, 0].axis("off")
    axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray"); axes[row, 1].set_title("GT mask"); axes[row, 1].axis("off")
    axes[row, 2].imshow(attn_b, cmap="hot"); axes[row, 2].set_title("baseline attn"); axes[row, 2].axis("off")
    axes[row, 3].imshow(attn_o, cmap="hot"); axes[row, 3].set_title("ours attn"); axes[row, 3].axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "qualitative_attention.png", dpi=150)
plt.show()
```

Invoke `NotebookEdit(edit_mode="insert", cell_id="26", cell_type="code", new_source=<above>)`.

- [ ] **Step 12.4: Confirm state**

Expected cell count: `28`.

### Task 13: Commit the rebuilt notebook

- [ ] **Step 13.1: Verify the notebook is committed unrun (no outputs)**

Run: `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print({c.get('outputs') for c in nb['cells']})"`
Expected: `{None, []}` (no cells have outputs; markdown cells have no outputs key).

- [ ] **Step 13.2: Commit**

```bash
git add notebooks/flowers102_experiments.ipynb
git commit -m "refactor: consolidate pipeline into single notebook"
```

Expected: one file changed, commit lands. Tests still pass against `src/` (unchanged).

---

## Phase 3 — Rewrite tests to import from the notebook

All five test files drop `from src.X import Y` and pull from the shim instead. Bodies unchanged.

### Task 14: Rewrite `tests/test_bootstrap.py`

**Files:**
- Modify: `tests/test_bootstrap.py:1-5`

- [ ] **Step 14.1: Replace the import block**

Replace lines 1-5 (`"""docstring"""` through `from src.bootstrap import paired_bootstrap_pvalue`) with:

```python
"""Unit tests for paired_bootstrap_pvalue (now lives in the notebook)."""
import numpy as np
import torch

from tests._nb_import import nb

paired_bootstrap_pvalue = nb.paired_bootstrap_pvalue
```

Bodies unchanged.

- [ ] **Step 14.2: Run the file's tests**

Run: `pytest tests/test_bootstrap.py -q`
Expected: all tests in that file pass.

### Task 15: Rewrite `tests/test_data.py`

**Files:**
- Modify: `tests/test_data.py:1-5`

- [ ] **Step 15.1: Replace the import block**

Replace lines 1-5 with:

```python
"""Unit tests for data.py pure functions (now live in the notebook)."""
import numpy as np

from tests._nb_import import nb

BG_RGB = nb.BG_RGB
trimap_to_binary = nb.trimap_to_binary
```

- [ ] **Step 15.2: Run the file's tests**

Run: `pytest tests/test_data.py -q`
Expected: all tests pass.

### Task 16: Rewrite `tests/test_eval.py`

**Files:**
- Modify: `tests/test_eval.py:1-6`

- [ ] **Step 16.1: Replace the import block**

Replace lines 1-6 with:

```python
"""Unit tests for eval.py (now lives in the notebook)."""
import numpy as np
import torch

from tests._nb_import import nb

top1_accuracy = nb.top1_accuracy
per_class_mean_accuracy = nb.per_class_mean_accuracy
```

- [ ] **Step 16.2: Run the file's tests**

Run: `pytest tests/test_eval.py -q`
Expected: all tests pass.

### Task 17: Rewrite `tests/test_losses.py`

**Files:**
- Modify: `tests/test_losses.py:1-5`

- [ ] **Step 17.1: Replace the import block**

Replace lines 1-5 with:

```python
"""Unit tests for losses.py (now lives in the notebook)."""
import torch

from tests._nb_import import nb

attn_kl_loss = nb.attn_kl_loss
```

- [ ] **Step 17.2: Run the file's tests**

Run: `pytest tests/test_losses.py -q`
Expected: all tests pass.

### Task 18: Rewrite `tests/test_maskmix.py`

**Files:**
- Modify: `tests/test_maskmix.py:1-5`

- [ ] **Step 18.1: Replace the import block**

Replace lines 1-5 with:

```python
"""Unit tests for maskmix.py (now lives in the notebook)."""
import torch

from tests._nb_import import nb

maskmix_batch = nb.maskmix_batch
```

- [ ] **Step 18.2: Run the file's tests**

Run: `pytest tests/test_maskmix.py -q`
Expected: all tests pass.

### Task 19: Run the whole suite and commit

- [ ] **Step 19.1: Full suite**

Run: `pytest tests/ -q`
Expected: `23 passed` in under 30 seconds.

- [ ] **Step 19.2: Commit**

```bash
git add tests/test_bootstrap.py tests/test_data.py tests/test_eval.py tests/test_losses.py tests/test_maskmix.py
git commit -m "refactor: rewrite tests to import from notebook"
```

Expected: five files changed. `src/` still on disk but unreferenced.

---

## Phase 4 — Delete obsolete files and update README

### Task 20: Delete `src/`, `scripts/`, `configs/`

**Files:**
- Delete: `src/` (entire directory, 9 files + `__pycache__`)
- Delete: `scripts/` (entire directory, 1 file)
- Delete: `configs/` (entire directory, 11 YAML files)

- [ ] **Step 20.1: Delete the three directories**

```bash
git rm -r src/ scripts/ configs/
```

Expected: `git status` shows 9 + 1 + 11 = 21 deletions staged, plus any `__pycache__` that was tracked (unlikely; `.gitignore` should cover it).

- [ ] **Step 20.2: Confirm tests still pass (no regression from deletion)**

Run: `pytest tests/ -q`
Expected: `23 passed`. If anything breaks, a test file still has a stray `from src.X` or `sys.path` hack.

### Task 21: Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 21.1: Rewrite the reproduction section**

Read the current README:

```bash
cat README.md
```

Replace the "Reproduction" section (and any `scripts/` / `src/` / `configs/` references elsewhere) with:

```markdown
## Reproduction

All experiments run from a single notebook: `notebooks/flowers102_experiments.ipynb`.

**Requirements:** Colab T4 (or equivalent), Drive mounted, `timm`, `torchvision`, `tqdm`, `pandas`, `matplotlib` (all in `requirements.txt`).

**Steps:**
1. Open the notebook on a Colab-bridged kernel (Cursor remote kernel, standalone Colab, or local Jupyter).
2. Run top to bottom. The first cells download ~450 MB of segmentation masks into `/content/drive/MyDrive/sc4001_flowers102/data/flowers-102/segmim/` (one-off per Drive; skipped on re-run).
3. The `smoke_test` cell (~90 s) runs one mini-epoch to catch pipeline breaks before the main runners.
4. Runner cells (Block D, A, C, B) produce `results/block_{a,b,c,d}.csv` and checkpoints under `checkpoints/<run_name>/best.pt`. Completed runs are skipped on re-execution via `final.json` markers.
5. Aggregation and figure cells read the CSVs and write `figures/k_curve.png` and `figures/qualitative_attention.png`.

**Expected total runtime:** approximately 2-4 hours on a free-tier T4 with batch 64 and early-stop patience 5.

**Testing:** `pytest tests/ -q` runs 23 unit tests on CPU in under 30 seconds, importing definitions directly from the notebook via `tests/_nb_import.py`.
```

Also remove any line that mentions `scripts/download_masks.py`, `src/train.py`, `configs/A1_baseline.yaml`, etc.

- [ ] **Step 21.2: Commit**

```bash
git add -- README.md src/ scripts/ configs/
git commit -m "chore: remove obsolete src/, scripts/, configs/; update README for single-notebook workflow"
```

**Note on `git add`:** list the deletions explicitly (`src/`, `scripts/`, `configs/` stay in `git add` because the directories are gone but the deletions are still staged from Task 20.1). Do *not* use `git add -u` or `git add -A` — they sweep in pre-existing dirty files like `docs/sc4001-project-brief.md`.

Expected: single commit, 22 files changed (21 deletions + README).

---

## Self-review checklist

After all tasks complete, verify:

- [ ] `pytest tests/ -q` → 23 passed.
- [ ] `ls src/ scripts/ configs/` → three "No such file or directory" errors.
- [ ] `python3 -c "import json; nb = json.load(open('notebooks/flowers102_experiments.ipynb')); print(len(nb['cells']))"` → `28`.
- [ ] `grep -rn "from src\." tests/ notebooks/ 2>/dev/null` → no output.
- [ ] `grep -rn "import scripts" tests/ notebooks/ 2>/dev/null` → no output.
- [ ] `grep -n "git clone\|git pull\|git push" notebooks/flowers102_experiments.ipynb` → no output.
- [ ] `git log --oneline -4` shows the four commits in order: shim, notebook, test-rewrites, cleanup+README. No `Co-Authored-By` trailer in any.
- [ ] Cell 0 is the title markdown; cell 1 is `## Setup`; cell 2 starts with `# [setup]:`; cell 27 starts with `# [figure_attention]:`.
- [ ] `git status` shows no staged changes beyond what the plan intended. In particular, `docs/sc4001-project-brief.md` remains unstaged (pre-existing dirty file).

If any item fails, fix it and recommit in a small follow-up commit (do not amend a published commit).

---

## Rollback

If Phase 2 produces a broken notebook and you need to back out cleanly:

```bash
git reset --hard <commit-before-Phase-2>    # e.g., the shim commit
```

Phases 1 and 2 are independent; the shim survives on its own. Phase 3 depends on Phase 2; if you revert Phase 2 you must also revert Phase 3.

Deleting `src/` in Phase 4 is the point of no return without revert. Before running Task 20, confirm Phase 3 tests pass.
