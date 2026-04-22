"""Oxford Flowers-102 dataset with aligned foreground masks.

The stock torchvision Flowers102 class downloads images, labels, and splits
but not segmentation masks. We wrap it and load masks from data/flowers-102/segmim/
which scripts/download_masks.py populates.

Mask convention (per Nilsback & Zisserman 2008): the segmim_*.jpg trimap is
blue (RGB [0,0,254]) where the pixel is background and some other color
(usually the original flower color, sometimes black for hard background)
where the pixel is foreground. We binarise: mask == 1 iff NOT blue.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import Flowers102


BG_RGB = np.array([0, 0, 254], dtype=np.uint8)


def trimap_to_binary(trimap_rgb: np.ndarray) -> np.ndarray:
    """Convert a HxWx3 uint8 trimap to an HxW uint8 binary foreground mask."""
    diff = np.abs(trimap_rgb.astype(np.int16) - BG_RGB.astype(np.int16)).sum(axis=2)
    return (diff > 10).astype(np.uint8)  # 1 = foreground


class Flowers102WithMasks(Dataset):
    """Returns (image: FloatTensor[3,H,W], mask: FloatTensor[1,H,W], label: int).

    Image is normalised with ImageNet mean/std. Mask is 0/1 float, same H,W.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str | Path,
        split: str,  # "train" | "val" | "test"
        image_size: int = 224,
        train_augment: bool = False,
        subsample_k: int | None = None,
        subsample_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self._ds = Flowers102(root=str(self.root), split=split, download=True)
        self._mask_dir = self.root / "segmim"
        if not self._mask_dir.exists():
            raise FileNotFoundError(
                f"Masks not found at {self._mask_dir}. "
                f"Run scripts/download_masks.py --data-dir {self.root}."
            )
        self.image_size = image_size
        self.train_augment = train_augment

        # torchvision stores the 1-based dataset-wide image ids in self._ds._image_files;
        # each path ends in .../image_XXXXX.jpg. Parse ids for mask lookup.
        self._image_ids: list[int] = []
        for p in self._ds._image_files:
            stem = Path(p).stem  # "image_06734"
            self._image_ids.append(int(stem.split("_")[1]))

        if subsample_k is not None:
            self._subsample(subsample_k, subsample_seed)

    def _subsample(self, k: int, seed: int) -> None:
        """Keep only k samples per class (deterministic by (class, seed))."""
        rng = np.random.default_rng(seed)
        labels = np.array(self._ds._labels)
        keep: list[int] = []
        for c in np.unique(labels):
            idxs = np.where(labels == c)[0]
            rng.shuffle(idxs)
            keep.extend(idxs[:k].tolist())
        keep.sort()
        self._ds._image_files = [self._ds._image_files[i] for i in keep]
        self._ds._labels = [self._ds._labels[i] for i in keep]
        self._image_ids = [self._image_ids[i] for i in keep]

    def __len__(self) -> int:
        return len(self._ds)

    def _load_mask(self, image_id: int) -> np.ndarray:
        mask_path = self._mask_dir / f"segmim_{image_id:05d}.jpg"
        with Image.open(mask_path) as im:
            rgb = np.array(im.convert("RGB"))
        return trimap_to_binary(rgb)  # HxW uint8

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pil_img, label = self._ds[i]
        image_id = self._image_ids[i]

        # Resize + center-crop both image and mask to image_size (preserves alignment).
        base_resize = T.Compose([
            T.Resize(int(self.image_size * 256 / 224)),
            T.CenterCrop(self.image_size),
        ])
        pil_img = base_resize(pil_img)
        mask_np = self._load_mask(image_id)
        mask_pil = base_resize(Image.fromarray(mask_np * 255).convert("L"))

        if self.train_augment:
            # Random horizontal flip applied jointly.
            if torch.rand(()) < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            # Photometric aug on image only.
            photo = T.Compose([T.RandAugment(num_ops=2, magnitude=9), T.ToTensor(),
                               T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])
            img_tensor = photo(pil_img)
        else:
            img_tensor = T.Compose([T.ToTensor(),
                                    T.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)])(pil_img)

        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.uint8)).float().unsqueeze(0) / 255.0
        return img_tensor, mask_tensor, int(label)
