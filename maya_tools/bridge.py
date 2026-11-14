"""Maya side of the Streamlit <-> Maya command-port bridge.

Streamlit writes a request JSON file, then sends ONE line through the Maya
command port:  import maya_tools.bridge as b; b.run(r"<request.json>")
This module does the work and writes <request>.response.json.
Data goes through files, so large meshes never travel through the socket.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import maya.cmds as cmds
import numpy as np

from .actions import build_voxel_twin, list_scene_assets, scene_name
from .maya_mesh import CONF_ATTR, PRED_ATTR, get_triangles, set_float_attr, set_label, set_string_attr


def _handle(req):
    cmd = req["cmd"]
    if cmd == "ping":
        return {"maya": cmds.about(version=True), "scene": scene_name()}
    if cmd == "list_assets":
        return {"assets": list_scene_assets(), "scene": scene_name()}
    if cmd == "export_mesh":
        v, f = get_triangles(req["node"])
        np.savez_compressed(req["out"], vertices=v, faces=f)
        return {"path": req["out"], "n_vertices": int(len(v)), "n_faces": int(len(f))}
    if cmd == "write_prediction":
        set_string_attr(req["node"], PRED_ATTR, req["label"])
        set_float_attr(req["node"], CONF_ATTR, float(req["confidence"]))
        return {"ok": True}
    if cmd == "set_label":
        set_label(req["node"], req["label"], source="streamlit")
        return {"ok": True}
    if cmd == "select":
        cmds.select(req["node"], replace=True)
        cmds.viewFit()
        return {"ok": True}
    if cmd == "voxel_twin":
        res = json.loads(Path(req["result"]).read_text())
        return {"node": build_voxel_twin(req["node"], res)}
    raise ValueError(f"unknown bridge command {cmd!r}")


_HANDLED = set()


def run(request_path):
    """Handle one request. Safe to call twice: Maya's command port can deliver the same line more than once."""
    request_path = Path(request_path)
    out = request_path.with_suffix(".response.json")
    key = str(request_path)
    if key in _HANDLED or not request_path.exists():
        return str(out)
    _HANDLED.add(key)
    try:
        resp = {"ok": True, "data": _handle(json.loads(request_path.read_text()))}
    except Exception as e:
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    out.write_text(json.dumps(resp))
    return str(out)
