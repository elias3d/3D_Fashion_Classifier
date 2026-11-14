"""Load the trained VoxelCNN and classify meshes (used by Streamlit, the Maya server and tests)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .classes import CLASS_NAMES, display
from .model import VoxelCNN, pick_device
from .preprocess import GRID, N_POINTS, mesh_to_voxels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "voxel_cnn.pt"


class FashionClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, device: str = "auto"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {self.model_path}. Train first: python scripts/train.py"
            )
        self.device = pick_device(device)
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.class_names = cfg.get("class_names", CLASS_NAMES)
        self.grid = cfg.get("grid", GRID)
        self.n_points = cfg.get("n_points", N_POINTS)
        self.model = VoxelCNN(len(self.class_names), self.grid).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.meta = {k: v for k, v in ckpt.items() if k not in ("state_dict",)}

    @torch.no_grad()
    def predict_voxels(self, vox_batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(vox_batch, dtype=np.float32)).to(self.device)
        if x.ndim == 3:
            x = x[None]
        x = x[:, None]                                    # add channel dim -> (B,1,G,G,G)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def predict_mesh(self, vertices, faces, tta: int = 1, seed: int = 0) -> dict:
        """Classify one mesh.

        tta > 1 = test-time augmentation: the mesh is also seen rotated around
        Y (tta evenly spaced angles) and the probabilities are averaged.
        """
        rng = np.random.default_rng(seed)
        vox, info, norm = mesh_to_voxels(vertices, faces, self.grid, self.n_points, rng=rng)
        batch = [vox]
        for k in range(1, max(1, tta)):
            v_k, _, _ = mesh_to_voxels(vertices, faces, self.grid, self.n_points,
                                       rng=np.random.default_rng(seed + k), yaw=360.0 * k / tta)
            batch.append(v_k)
        probs = self.predict_voxels(np.stack(batch)).mean(0)
        return self._package(probs, vox, info, norm)

    def _package(self, probs, vox, info, norm) -> dict:
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        return {
            "label": self.class_names[best],
            "display": display(self.class_names[best]),
            "confidence": float(probs[best]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
            "top3": [{"label": self.class_names[i], "display": display(self.class_names[i]),
                      "probability": float(probs[i])} for i in order[:3]],
            "voxels": vox,
            "points": norm,
            "info": info,
        }


def load_metrics() -> dict | None:
    p = PROJECT_ROOT / "models" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None
