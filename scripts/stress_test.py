"""Extra experiment: how does the model cope with things it was never trained on?

    python scripts/stress_test.py

- lying down: test garments rotated 90 degrees around X (the model assumes Y-up)
- upside down: rotated 180 degrees around X
- TTA: the normal test set with 8-view rotation test-time augmentation
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fashion3d.dataset import read_manifest  # noqa: E402
from fashion3d.inference import FashionClassifier  # noqa: E402
from fashion3d.mesh_io import load_npz  # noqa: E402


def rot_x(v, deg):
    a = np.radians(deg)
    r = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)
    return v @ r.T


def main():
    clf = FashionClassifier()
    rows = read_manifest("test")
    out = {}
    for name, deg in [("upright", 0), ("lying down (90 deg X)", 90), ("upside down (180 deg X)", 180)]:
        correct, confs, mistakes = 0, [], {}
        for r in rows:
            v, f = load_npz(r["path"])
            res = clf.predict_mesh(rot_x(v, deg), f)
            correct += res["label"] == r["label"]
            confs.append(res["confidence"])
            if res["label"] != r["label"]:
                key = f"{r['label']}->{res['label']}"
                mistakes[key] = mistakes.get(key, 0) + 1
        top = dict(sorted(mistakes.items(), key=lambda kv: -kv[1])[:5])
        out[name] = {"accuracy": correct / len(rows), "mean_confidence": float(np.mean(confs)), "top_mistakes": top}
        print(f"{name:26s} acc={correct / len(rows):.3f} conf={np.mean(confs):.3f} top mistakes={top}", flush=True)
    (ROOT / "models" / "stress_test.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
