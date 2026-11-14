"""Talk to scripts/serve.py from inside Maya (standard library only)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

SERVER = "http://127.0.0.1:8765"


def _call(path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            raise RuntimeError(json.loads(body).get("error", body)) from None
        except ValueError:
            raise RuntimeError(body) from None


def health(timeout=1.5):
    try:
        return _call("/health", timeout=timeout)
    except Exception as e:
        return {"ok": False, "offline": True, "error": str(e)}


def reload_model():
    return _call("/reload", {})


def predict(name, vertices, faces, tta=1, want_voxels=False, scene="", true_label=None):
    return _call("/predict", {
        "name": name,
        "vertices": [round(float(x), 5) for x in vertices.reshape(-1)],
        "faces": [int(i) for i in faces.reshape(-1)],
        "tta": tta, "want_voxels": want_voxels, "scene": scene, "true_label": true_label,
    })
