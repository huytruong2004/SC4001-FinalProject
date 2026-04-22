"""Download and extract Oxford Flowers-102 segmentation masks.

Usage:
    python scripts/download_masks.py --data-dir data/flowers-102

Idempotent: re-running is a no-op if the masks directory already has 8189 files.
"""
from __future__ import annotations

import argparse
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


def download(data_dir: Path) -> None:
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
        raise RuntimeError(
            f"Corrupt archive at {tgz_path} (removed). Re-run to re-download."
        ) from e

    count = len(list(mask_dir.glob("segmim_*.jpg")))
    if count != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} masks, got {count}")
    print(f"OK: {count} masks at {mask_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/flowers-102"))
    args = ap.parse_args()
    download(args.data_dir)
