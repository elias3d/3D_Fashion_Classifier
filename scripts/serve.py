"""Local inference server: Maya sends a mesh, gets a prediction back.

    python scripts/serve.py            # http://127.0.0.1:8765

Why a server? Maya ships its own Python, and installing PyTorch + CUDA into it
is fragile. The server runs in the project's normal Python (with the GPU),
and Maya talks to it with plain HTTP using only the standard library.

Endpoints
    GET  /health    model status
    POST /predict   {"name", "vertices": [x,y,z,...], "faces": [i,j,k,...], "tta": 1, "want_voxels": true}
    POST /reload    reload models/voxel_cnn.pt after re-training

Every Maya prediction is also saved to runs/maya_inbox/ so the Streamlit app can show it.
"""
import argparse
import base64
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fashion3d.inference import FashionClassifier, load_metrics  # noqa: E402
from fashion3d.render import png_bytes, render_thumbnail  # noqa: E402

INBOX = ROOT / "runs" / "maya_inbox"
STATE = {"clf": None, "error": None, "device": "auto"}
LOCK = threading.Lock()


def load_model():
    try:
        STATE["clf"] = FashionClassifier(device=STATE["device"])
        STATE["error"] = None
    except Exception as e:  # keep serving /health so Maya can show the problem
        STATE["clf"], STATE["error"] = None, str(e)


def predict(payload):
    clf = STATE["clf"]
    if clf is None:
        raise RuntimeError(STATE["error"] or "Model not loaded")
    v = np.asarray(payload["vertices"], np.float32).reshape(-1, 3)
    f = np.asarray(payload["faces"], np.int64).reshape(-1, 3)
    tta = int(payload.get("tta", 1))
    with LOCK:
        res = clf.predict_mesh(v, f, tta=tta)
    thumb = png_bytes(render_thumbnail(v, f, size=192))
    vox = res.pop("voxels")
    res.pop("points")
    out = {k: res[k] for k in ("label", "display", "confidence", "probabilities", "top3", "info")}
    out["name"] = payload.get("name", "asset")
    out["tta"] = tta
    out["thumbnail_png"] = base64.b64encode(thumb).decode()
    if payload.get("want_voxels"):
        idx = np.argwhere(vox > 0)
        out["grid"] = int(vox.shape[0])
        out["voxels"] = [[int(i), int(j), int(k), round(float(vox[i, j, k]), 3)] for i, j, k in idx]

    # Save for the Streamlit "Live from Maya" page
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    folder = INBOX / f"{stamp}_{re.sub(r'[^A-Za-z0-9_]+', '_', out['name'])[:40]}"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "mesh.npz", vertices=v, faces=f)
    (folder / "thumb.png").write_bytes(thumb)
    meta = {k: out[k] for k in ("name", "label", "display", "confidence", "probabilities", "top3", "tta")}
    meta.update(received=time.strftime("%Y-%m-%d %H:%M:%S"), scene=payload.get("scene", ""),
                true_label=payload.get("true_label"))
    (folder / "result.json").write_text(json.dumps(meta, indent=2))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            clf = STATE["clf"]
            m = load_metrics() or {}
            self._send(200, {
                "ok": clf is not None, "error": STATE["error"],
                "device": str(clf.device) if clf else None,
                "classes": clf.class_names if clf else None,
                "test_acc": m.get("test_acc"), "trained_at": m.get("trained_at"),
            })
        else:
            self._send(404, {"error": "unknown endpoint"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/predict":
                self._send(200, predict(payload))
            elif self.path == "/reload":
                with LOCK:
                    load_model()
                self._send(200, {"ok": STATE["clf"] is not None, "error": STATE["error"]})
            else:
                self._send(404, {"error": "unknown endpoint"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print("[serve]", fmt % args, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    STATE["device"] = args.device
    load_model()
    print(f"Fashion3D server on http://127.0.0.1:{args.port}  model={'OK' if STATE['clf'] else STATE['error']}",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
