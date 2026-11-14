"""Maya actions shared by the panel, the shelf and the Streamlit command-port bridge."""
from __future__ import annotations

import os
from pathlib import Path

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np

from fashion3d.preprocess import MARGIN

from . import PROJECT_ROOT, client
from .dataset_tools import ensure_fbx, safe_id
from .maya_mesh import (CONF_ATTR, PRED_ATTR, get_label, get_triangles, set_float_attr, set_string_attr,
                        top_level_assets)

TWIN_SUFFIX = "_modelView"


def find_project_python():
    """The Python that has PyTorch (the project's .venv), searched upward from the project folder."""
    env = os.environ.get("FASHION3D_PYTHON")
    if env and Path(env).exists():
        return env
    if cmds.optionVar(exists="fashion3d_python"):
        p = cmds.optionVar(query="fashion3d_python")
        if p and Path(p).exists():
            return p
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        for rel in (".venv/Scripts/python.exe", "venv/Scripts/python.exe", ".venv/bin/python"):
            if (parent / rel).exists():
                return str(parent / rel)
    # environment created by run_windows.bat
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Fashion3D" / "venv" / "Scripts" / "python.exe"
    if local.exists():
        return str(local)
    return "python"


def scene_name():
    return cmds.file(query=True, sceneName=True, shortName=True) or "untitled"


def classify_nodes(nodes=None, tta=1, write_attrs=True, show_voxels=False):
    """Classify each selected asset. Returns [(node, result_dict), ...]."""
    nodes = top_level_assets(nodes)
    if not nodes:
        raise RuntimeError("Select one or more polygon assets first.")
    results = []
    for node in nodes:
        if node.split("|")[-1].endswith(TWIN_SUFFIX):
            continue
        v, f = get_triangles(node)
        res = client.predict(node.split("|")[-1], v, f, tta=tta, want_voxels=show_voxels,
                             scene=scene_name(), true_label=get_label(node))
        if write_attrs:
            set_string_attr(node, PRED_ATTR, res["label"])
            set_float_attr(node, CONF_ATTR, round(res["confidence"], 4))
        if show_voxels:
            build_voxel_twin(node, res)
        results.append((node, res))
    return results


def build_voxel_twin(node, res, color=(0.95, 0.55, 0.2)):
    """Create a blocky mesh next to the asset showing exactly what the model sees (the 32^3 grid)."""
    name = node.split("|")[-1] + TWIN_SUFFIX
    if cmds.objExists(name):
        cmds.delete(name)
    g = res["grid"]
    center = np.array(res["info"]["center"])
    half = res["info"]["half_extent"]
    cell = 2.0 / g / MARGIN * half
    bb = cmds.exactWorldBoundingBox(node)
    shift = np.array([(bb[3] - bb[0]) * 1.25, 0, 0])
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) - 0.5
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    pts, counts, conns = [], [], []
    for i, j, k, _val in res["voxels"]:
        c = ((np.array([i, j, k]) + 0.5) / g * 2 - 1) / MARGIN * half + center + shift
        base = len(pts)
        pts.extend(c + corners * cell * 0.9)
        for q in quads:
            counts.append(4)
            conns.extend(base + idx for idx in q)
    mesh = om.MFnMesh()
    obj = mesh.create(om.MPointArray([om.MPoint(*p) for p in pts]), counts, conns)
    transform = om.MFnDagNode(obj).fullPathName()
    transform = cmds.rename(transform, name)
    shader = "fashion3d_voxel_mat"
    if not cmds.objExists(shader):
        shader = cmds.shadingNode("lambert", asShader=True, name=shader)
        cmds.setAttr(shader + ".color", *color, type="double3")
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shader + "SG")
        cmds.connectAttr(shader + ".outColor", sg + ".surfaceShader", force=True)
    cmds.sets(transform, edit=True, forceElement=shader + "SG")
    return transform


def import_fbx_files(paths):
    """Import each FBX into its own group so it can be labelled and exported as one asset."""
    ensure_fbx()
    roots = []
    for p in paths:
        before = set(cmds.ls(assemblies=True, long=True))
        ns = safe_id(Path(p).stem)
        cmds.file(str(p).replace("\\", "/"), i=True, type="FBX", ignoreVersion=True, namespace=ns,
                  mergeNamespacesOnClash=False)
        new = [n for n in cmds.ls(assemblies=True, long=True) if n not in before]
        new = [n for n in new if top_level_assets([n])]
        if not new:
            continue
        root = new[0] if len(new) == 1 else cmds.group(new, name=ns + "_asset")
        set_string_attr(root, "fashionSourceFile", str(p))
        roots.append(root)
    return roots


def list_scene_assets():
    """Top-level mesh assets in the scene (used by Streamlit through the command port)."""
    out = []
    for n in cmds.ls(assemblies=True, long=True):
        if n.split("|")[-1].endswith(TWIN_SUFFIX):
            continue
        for a in top_level_assets([n]) or []:
            out.append({"node": a, "name": a.split("|")[-1], "label": get_label(a)})
        kids = cmds.listRelatives(n, children=True, type="transform", fullPath=True) or []
        if not cmds.listRelatives(n, shapes=True) and kids and len(kids) > 1:
            # a group of separate garments (e.g. the showcase): list children instead
            out = [o for o in out if o["node"] != n]
            for k in kids:
                if top_level_assets([k]):
                    out.append({"node": k, "name": k.split("|")[-1], "label": get_label(k)})
    return out
