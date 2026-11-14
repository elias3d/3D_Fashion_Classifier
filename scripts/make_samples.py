"""Copy a small Maya-free sample set into samples/ (shipped with the project).

    python scripts/make_samples.py

samples/test, samples/styles  -> .npz meshes + manifest.csv (no FBX), used by the app's "Test assets"
samples/obj                   -> one .obj per fashion type, to try the app's upload without Maya
"""
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trimesh  # noqa: E402

from fashion3d.classes import CLASS_NAMES  # noqa: E402
from fashion3d.mesh_io import load_npz  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "samples"

for split in ("test", "styles"):
    src = DATA / split / "manifest.csv"
    if not src.exists():
        sys.exit(f"{src} missing: generate the dataset with Maya first.")
    rows = list(csv.DictReader(open(src, newline="")))
    (OUT / split).mkdir(parents=True, exist_ok=True)
    for r in rows:
        dst = OUT / r["mesh"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA / r["mesh"], dst)
        r["fbx"] = ""
    with open(OUT / split / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{split}: {len(rows)} meshes")

(OUT / "obj").mkdir(parents=True, exist_ok=True)
for label in CLASS_NAMES:
    npz = sorted((DATA / "test" / label).glob("*.npz"))[0]
    v, f = load_npz(npz)
    trimesh.Trimesh(v, f, process=False).export(OUT / "obj" / f"{label}.obj")
print("obj: one example per type")
