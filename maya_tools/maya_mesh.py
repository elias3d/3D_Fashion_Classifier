"""Read geometry and labels out of Maya nodes."""
from __future__ import annotations

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np

LABEL_ATTR = "fashionType"
SOURCE_ATTR = "fashionSource"
PRED_ATTR = "fashionPredicted"
CONF_ATTR = "fashionConfidence"


def mesh_shapes(node):
    """All non-intermediate mesh shapes under a node (works for groups from FBX imports)."""
    if cmds.nodeType(node) == "mesh":
        shapes = [node]
    else:
        shapes = cmds.listRelatives(node, allDescendents=True, type="mesh", fullPath=True) or []
    return [s for s in shapes if not cmds.getAttr(s + ".intermediateObject")]


def get_triangles(node):
    """World-space triangles of everything under `node` -> (vertices (V,3), faces (F,3))."""
    verts, faces, offset = [], [], 0
    for shape in mesh_shapes(node):
        sel = om.MSelectionList()
        sel.add(shape)
        fn = om.MFnMesh(sel.getDagPath(0))
        pts = fn.getPoints(om.MSpace.kWorld)
        v = np.array([(p.x, p.y, p.z) for p in pts], np.float32)
        _, tri_verts = fn.getTriangles()
        f = np.array(tri_verts, np.int32).reshape(-1, 3) + offset
        verts.append(v)
        faces.append(f)
        offset += len(v)
    if not verts:
        raise ValueError(f"{node} has no polygon mesh.")
    return np.concatenate(verts), np.concatenate(faces)


def top_level_assets(nodes=None):
    """Selected (or given) transforms that contain meshes, without duplicates of children."""
    nodes = nodes if nodes is not None else (cmds.ls(selection=True, long=True, transforms=True) or [])
    out = []
    for n in nodes:
        if mesh_shapes(n) and not any(n.startswith(o + "|") for o in out):
            out.append(n)
    return out


def set_string_attr(node, attr, value):
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, dataType="string")
    cmds.setAttr(f"{node}.{attr}", value, type="string")


def set_float_attr(node, attr, value):
    if not cmds.attributeQuery(attr, node=node, exists=True):
        cmds.addAttr(node, longName=attr, attributeType="double")
    cmds.setAttr(f"{node}.{attr}", value)


def get_label(node):
    if cmds.attributeQuery(LABEL_ATTR, node=node, exists=True):
        return cmds.getAttr(f"{node}.{LABEL_ATTR}") or None
    return None


def set_label(node, label, source="manual"):
    set_string_attr(node, LABEL_ATTR, label)
    set_string_attr(node, SOURCE_ATTR, source)
