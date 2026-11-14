"""Generate the labelled 3D fashion dataset with headless Maya.

    "C:\\Program Files\\Autodesk\\Maya2025\\bin\\mayapy.exe" scripts/maya_generate_dataset.py
    ... --train-per-class 100 --test-per-class 25 --styles-per-class 5

The same functions are behind the "Generate dataset" button in the Maya panel.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import maya.standalone  # noqa: E402

maya.standalone.initialize(name="python")

from maya_tools.dataset_tools import DATA_DIR, ensure_fbx, generate_split  # noqa: E402
from maya_tools.styles import STYLES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=25)
    ap.add_argument("--styles-per-class", type=int, default=5)
    ap.add_argument("--no-fbx", action="store_true", help="skip FBX files (faster, npz only)")
    ap.add_argument("--train-fbx", action="store_true",
                    help="also write FBX for the train split (~110 KB each; off to keep the folder small)")
    ap.add_argument("--clean", action="store_true", help="delete the existing data/ folder first")
    args = ap.parse_args()

    if args.clean and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    ensure_fbx()
    t0 = time.time()

    def progress(done, total, msg):
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}  {msg}  ({time.time() - t0:.0f}s)", flush=True)

    jobs = [("train", args.train_per_class, 1, None, ""), ("test", args.test_per_class, 999, None, "")]
    jobs += [("styles", args.styles_per_class, 5000 + i, style, name) for i, (name, style) in enumerate(STYLES.items())]
    for split, n, seed, variant, style in jobs:
        if n <= 0:
            continue
        print(f"[{split}{' / ' + style if style else ''}] {n} per class", flush=True)
        fbx = not args.no_fbx and (split != "train" or args.train_fbx)
        generate_split(split, n, seed, write_fbx=fbx, progress=progress, variant=variant, style=style)
    print(f"Done in {time.time() - t0:.0f}s -> {DATA_DIR}")


if __name__ == "__main__":
    main()
    maya.standalone.uninitialize()
