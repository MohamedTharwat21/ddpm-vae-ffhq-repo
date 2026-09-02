"""FFHQ dataset discovery, deterministic 60k/10k split, and the shared Dataset class.


No path here is hardcoded to ``/kaggle/...`` - ``search_root`` /
``data_root`` / ``split_file`` are all parameters, sourced from the active
YAML config (see :mod:`ffhq_repo.config`), so the same code runs unchanged
on Kaggle, a local machine, or any other environment; pass Kaggle-style
paths in the config if that's where the data actually lives.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

__all__ = [
    "IMAGE_EXTENSIONS", "count_images_recursive", "find_data_root",
    "discover_all_images", "stable_seed", "generate_or_load_split",
    "verify_split", "ManifestImageDataset",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def count_images_recursive(dirpath: str) -> int:
    total = 0
    for _, _, filenames in os.walk(dirpath):
        total += sum(1 for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
    return total


def find_data_root(search_root: str, min_images: int = 50000) -> str:
    """Return the shallowest directory under search_root whose subtree
    (recursively) contains at least min_images image files."""
    all_dirs = sorted({search_root} | {dp for dp, _, _ in os.walk(search_root)},
                       key=lambda d: d.count(os.sep))
    for d in all_dirs:
        if count_images_recursive(d) >= min_images:
            return d
    raise FileNotFoundError(
        f"Could not find a directory with >= {min_images} image files (recursively) under "
        f"{search_root}. Point config.data.data_root at the FFHQ dataset directly, or make "
        "sure it is mounted/downloaded under that search root."
    )


def discover_all_images(data_root: str):
    """Recursively find every image file under data_root, sorted deterministically."""
    records = []
    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                records.append(os.path.relpath(os.path.join(dirpath, fname), data_root))
    return sorted(records)


def stable_seed(*parts, mod: int = 2 ** 32) -> int:
    """Deterministic, process/machine-independent integer seed from any parts."""
    s = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


def generate_or_load_split(split_file: str, all_images, train_count: int, val_count: int, seed):
    """Load split_file if it exists; otherwise generate a deterministic split once and save it.

    Determinism: all_images is already sorted, so the only randomness is a
    single numpy RandomState seeded via stable_seed(seed) that permutes
    that sorted list; the first train_count indices go to train, the next
    val_count go to val. Never regenerates an existing split_file - this is
    what guarantees the VAE and DDPM/LDM experiments share byte-identical
    train/val sets.
    """
    if os.path.exists(split_file):
        with open(split_file) as f:
            split = json.load(f)
        print(f"Loaded existing split from {split_file} - NOT regenerating "
              f"(train={len(split['train'])}, val={len(split['val'])}).")
        return split

    if len(all_images) < train_count + val_count:
        raise ValueError(
            f"Only {len(all_images)} images discovered, but train_count + val_count = "
            f"{train_count + val_count}. Check config.data.data_root."
        )

    rng = np.random.RandomState(stable_seed(seed, "ffhq_split"))
    perm = rng.permutation(len(all_images))
    train_files = sorted(all_images[i] for i in perm[:train_count])
    val_files = sorted(all_images[i] for i in perm[train_count:train_count + val_count])

    split = dict(
        train=train_files, val=val_files, seed=seed,
        train_count=len(train_files), val_count=len(val_files),
        total_discovered=len(all_images),
    )
    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f)
    print(f"Generated new split and saved to {split_file} "
          f"(train={len(train_files)}, val={len(val_files)}).")
    return split


def verify_split(split: dict, all_images) -> None:
    train_set, val_set = set(split["train"]), set(split["val"])
    overlap = train_set & val_set
    union = train_set | val_set
    all_set = set(all_images)

    print("=" * 70)
    print("DATASET SPLIT VERIFICATION")
    print("=" * 70)
    print(f"Total: {len(all_images):,}")
    print(f"Train: {len(train_set):,}")
    print(f"Validation: {len(val_set):,}")
    print(f"train ∩ val overlap: {len(overlap)} (must be 0)")
    print(f"train ∪ val == all discovered images: {union == all_set}")
    print("=" * 70)

    assert len(overlap) == 0, "OVERLAP DETECTED between train and val splits!"
    assert len(train_set) == split["train_count"] and len(val_set) == split["val_count"], (
        "split file's recorded counts don't match its own file lists"
    )


class ManifestImageDataset(Dataset):
    """Resize -> CenterCrop -> ToTensor, optionally rescaled to [-1, 1].

    ``normalize_to_unit_range=False`` (the VAE convention): tensors stay in
    [0, 1].
    ``normalize_to_unit_range=True`` (the DDPM/LDM convention): tensors are
    additionally rescaled to [-1, 1] via ``Normalize([0.5]*3, [0.5]*3)``, so
    the network's target has zero mean, matching Ho et al. 2020.

    Corrupt-image handling: substitute a neighboring sample rather than crash.
    """

    def __init__(self, manifest_df: pd.DataFrame, data_root: str, split: str, img_size: int,
                 normalize_to_unit_range: bool = False, return_index: bool = False):
        self.df = manifest_df[manifest_df["vae_split"] == split].reset_index(drop=True)
        self.data_root = data_root
        self.return_index = return_index
        tfs = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # -> [0, 1]
        ]
        if normalize_to_unit_range:
            tfs.append(transforms.Normalize([0.5] * 3, [0.5] * 3))  # -> [-1, 1]
        self.transform = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_root, row["filepath"])
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] failed to load {path} ({e}); substituting a neighboring sample")
            return self.__getitem__((idx + 1) % len(self))
        img = self.transform(img)
        if self.return_index:
            return img, idx
        return img
