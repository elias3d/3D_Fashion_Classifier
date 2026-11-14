"""Export labelled Maya assets into the training dataset.

Layout (labels = folder name + manifest + Maya attribute, all kept in sync):

    data/<split>/<label>/<asset_id>.fbx   the 3D asset (for artists / other DCCs)
    data/<split>/<label>/<asset_id>.npz   triangles used by the ML pipeline
    data/<split>/manifest.csv             one row per asset
"""
from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path

import maya.cmds as cmds
import maya.mel as mel
import numpy as np

from fashion3d.classes import CLASS_NAMES

from . import PROJECT_ROOT
from .garments import build_garment
from .maya_mesh import get_label, get_triangles

DATA_DIR = PROJECT_ROOT / "data"
FIELDS = ["asset_id", "label", "split", "source", "style", "fbx", "mesh", "n_vertices", "n_faces", "seed"]


def ensure_fbx():
    if not cmds.pluginInfo("fbxmaya", query=True, loaded=True):
        cmds.loadPlugin("fbxmaya", quiet=True)


def export_fbx(node, path):
    ensure_fbx()
    cmds.select(node, replace=True)
    mel.eval("FBXResetExport")
    mel.eval("FBXExportSmoothingGroups -v true")
    cmds.file(str(path).replace("\\", "/"), force=True, options="v=0;", type="FBX export",
              preserveReferences=True, exportSelected=True)
    cmds.select(clear=True)


def safe_id(text):
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "asset"


def export_asset(node, split, label=None, asset_id=None, source="manual", style="", seed="",
                 data_dir=DATA_DIR, write_fbx=True):
    """Write one asset (FBX + npz) and return its manifest row."""
    label = label or get_label(node)
    if label not in CLASS_NAMES:
        raise ValueError(f"{node}: label must be one of {CLASS_NAMES}, got {label!r}. Tag it first.")
    asset_id = asset_id or safe_id(node.split("|")[-1]) + "_" + time.strftime("%H%M%S")
    folder = Path(data_dir) / split / label
    folder.mkdir(parents=True, exist_ok=True)
    v, f = get_triangles(node)
    mesh_path = folder / f"{asset_id}.npz"
    np.savez_compressed(mesh_path, vertices=v.astype(np.float32), faces=f.astype(np.int32))
    fbx_path = folder / f"{asset_id}.fbx"
    if write_fbx:
        export_fbx(node, fbx_path)
    rel = lambda p: p.relative_to(Path(data_dir)).as_posix()
    return {
        "asset_id": asset_id, "label": label, "split": split, "source": source, "style": style,
        "fbx": rel(fbx_path) if write_fbx else "", "mesh": rel(mesh_path),
        "n_vertices": len(v), "n_faces": len(f), "seed": seed,
    }


def append_manifest(rows, split, data_dir=DATA_DIR):
    """Add rows to the split's manifest. A row with an asset_id that is already listed replaces the
    old one, so re-generating or re-exporting an asset never creates duplicates."""
    path = Path(data_dir) / split / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if path.exists():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                merged[r["asset_id"]] = r
    for r in rows:
        merged[r["asset_id"]] = r
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged.values())
    return path


def generate_split(split, per_class, seed, data_dir=DATA_DIR, write_fbx=True, progress=None,
                   variant=None, style=""):
    """Build, label and export `per_class` garments of every type.

    `variant(node, rng)` can modify each garment before export (used for the
    test styles). `progress(done, total, message)` returning False cancels.
    """
    rows, total, done = [], per_class * len(CLASS_NAMES), 0
    master = random.Random(seed)
    for label in CLASS_NAMES:
        for i in range(per_class):
            s = master.randrange(10 ** 9)
            rng = random.Random(s)
            asset_id = f"{label}_{style + '_' if style else ''}{seed}_{i:04d}"
            detail = 1.0
            if variant is not None and getattr(variant, "detail", None):
                detail = variant.detail
            node = build_garment(label, seed=s, name=asset_id, detail=detail)
            if variant is not None:
                node = variant(node, rng) or node
            rows.append(export_asset(node, split, label, asset_id, source="generated", style=style,
                                     seed=s, data_dir=data_dir, write_fbx=write_fbx))
            cmds.delete(node)
            done += 1
            if progress and progress(done, total, asset_id) is False:
                append_manifest(rows, split, data_dir)
                return rows
    append_manifest(rows, split, data_dir)
    return rows


def dataset_counts(data_dir=DATA_DIR):
    out = {}
    for manifest in Path(data_dir).glob("*/manifest.csv"):
        with open(manifest, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["split"], row["label"])
                out[key] = out.get(key, 0) + 1
    return out
