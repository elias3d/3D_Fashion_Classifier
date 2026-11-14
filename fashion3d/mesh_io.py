"""Load meshes outside Maya.

.npz  -> written by our Maya tools (vertices + faces)
.obj/.stl/.ply/.glb/.gltf/.off -> read with trimesh
.fbx  -> converted by a headless Maya (mayapy) because FBX needs Autodesk's SDK
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIMESH_EXT = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["vertices"].astype(np.float32), d["faces"].astype(np.int64)


def load_trimesh(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def find_mayapy() -> str | None:
    env = os.environ.get("MAYAPY")
    if env and Path(env).exists():
        return env
    found = sorted(glob.glob(r"C:\Program Files\Autodesk\Maya20*\bin\mayapy.exe"))
    found = [p for p in found if int(Path(p).parts[-3][4:8]) >= 2025] or found
    return found[-1] if found else None


def fbx_to_npz(fbx_path, timeout=240) -> Path:
    mayapy = find_mayapy()
    if not mayapy:
        raise RuntimeError("FBX needs Maya. mayapy.exe was not found (set the MAYAPY env var).")
    out = Path(tempfile.mkdtemp()) / (Path(fbx_path).stem + ".npz")
    script = PROJECT_ROOT / "scripts" / "maya_fbx_to_npz.py"
    r = subprocess.run([mayapy, str(script), str(fbx_path), str(out)],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"mayapy FBX conversion failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return out


def load_mesh(path):
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        return load_npz(path)
    if ext in TRIMESH_EXT:
        return load_trimesh(path)
    if ext == ".fbx":
        return load_npz(fbx_to_npz(path))
    raise ValueError(f"Unsupported 3D format: {ext}")
