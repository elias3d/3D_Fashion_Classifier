"""Streamlit side of the bridge: drive a running Maya through its command port.

In Maya (the panel has a button for this):
    cmds.commandPort(name=":7001", sourceType="python")
"""
from __future__ import annotations

import json
import socket
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = Path(tempfile.gettempdir()) / "fashion3d_bridge"


def port_open(port=7001, host="127.0.0.1", timeout=0.3) -> bool:
    """Quick check: is a Maya command port listening? (No Maya = refused instantly.)"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class MayaBridge:
    def __init__(self, host="127.0.0.1", port=7001, timeout=30.0):
        self.host, self.port, self.timeout = host, port, timeout
        BRIDGE_DIR.mkdir(exist_ok=True)

    def _send_line(self, code):
        with socket.create_connection((self.host, self.port), timeout=3) as s:
            s.sendall(code.encode())
            s.settimeout(0.5)
            try:
                s.recv(4096)       # Maya echoes a result; we only need the file
            except (socket.timeout, OSError):
                pass

    def call(self, cmd, **kwargs):
        req = BRIDGE_DIR / f"{uuid.uuid4().hex}.json"
        req.write_text(json.dumps({"cmd": cmd, **kwargs}))
        resp = req.with_suffix(".response.json")
        root = str(PROJECT_ROOT).replace("\\", "/")
        try:
            self._send_line(
                f"import sys; sys.path.insert(0, r'{root}') if r'{root}' not in sys.path else None; "
                f"import maya_tools.bridge as _fb; _fb.run(r'{req.as_posix()}')"
            )
        except OSError:
            req.unlink(missing_ok=True)      # Maya not reachable: don't leave the request behind
            raise
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if resp.exists():
                try:
                    data = json.loads(resp.read_text())
                except ValueError:       # still being written
                    time.sleep(0.05)
                    continue
                req.unlink(missing_ok=True)
                resp.unlink(missing_ok=True)
                if not data.get("ok"):
                    raise RuntimeError(data.get("error", "Maya bridge error"))
                return data["data"]
            time.sleep(0.05)
        req.unlink(missing_ok=True)
        raise TimeoutError("Maya did not answer. Is Maya busy, or is the command port closed?")

    def ping(self):
        return self.call("ping")

    def list_assets(self):
        return self.call("list_assets")

    def export_mesh(self, node):
        out = BRIDGE_DIR / f"{uuid.uuid4().hex}.npz"
        self.call("export_mesh", node=node, out=out.as_posix())
        return out
