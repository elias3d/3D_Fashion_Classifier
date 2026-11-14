"""Read the Maya-exported dataset (manifest.csv + .npz meshes)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .classes import CLASS_NAMES
from .preprocess import GRID, N_POINTS, mesh_to_voxels

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"   # shipped subset, works without Maya


def read_manifest(split, data_dir=DATA_DIR):
    """Rows of data/<split>/manifest.csv, or of samples/<split>/ when the full dataset isn't there."""
    path = Path(data_dir) / split / "manifest.csv"
    if not path.exists() and Path(data_dir) == DATA_DIR:
        data_dir = SAMPLES_DIR
        path = SAMPLES_DIR / split / "manifest.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        rows = list({r["asset_id"]: r for r in csv.DictReader(fh)}.values())   # ignore duplicate rows
    for r in rows:
        r["path"] = str(Path(data_dir) / r["mesh"])
    return [r for r in rows if r["label"] in CLASS_NAMES and Path(r["path"]).exists()]


class VoxelDataset(Dataset):
    """Loads every mesh once, then voxelizes on the fly (with random augmentation when training)."""

    def __init__(self, rows, augment=False, grid=GRID, n_points=N_POINTS, seed=0):
        self.rows = rows
        self.augment = augment
        self.grid, self.n_points = grid, n_points
        self.meshes = []
        for r in rows:
            d = np.load(r["path"])
            self.meshes.append((d["vertices"].astype(np.float32), d["faces"].astype(np.int64)))
        self.labels = [CLASS_NAMES.index(r["label"]) for r in rows]
        self.rng = np.random.default_rng(seed)
        # Validation/test voxels never change, so compute them once.
        self.cache = None if augment else [self._voxels(i, np.random.default_rng(i)) for i in range(len(rows))]

    def _voxels(self, i, rng):
        v, f = self.meshes[i]
        vox, _, _ = mesh_to_voxels(v, f, self.grid, self.n_points, rng=rng, augment=self.augment)
        return vox

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        vox = self.cache[i] if self.cache is not None else self._voxels(i, self.rng)
        return torch.from_numpy(vox)[None], self.labels[i]
