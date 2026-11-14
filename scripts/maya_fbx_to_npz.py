"""Headless Maya: import an FBX and save its triangles as .npz (used by the Streamlit upload).

    mayapy scripts/maya_fbx_to_npz.py input.fbx output.npz
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import maya.standalone  # noqa: E402

maya.standalone.initialize(name="python")

import maya.cmds as cmds  # noqa: E402
import numpy as np  # noqa: E402

from maya_tools.dataset_tools import ensure_fbx  # noqa: E402
from maya_tools.maya_mesh import get_triangles  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
ensure_fbx()
cmds.file(new=True, force=True)
cmds.file(src.replace("\\", "/"), i=True, type="FBX", ignoreVersion=True, mergeNamespacesOnClash=False,
          namespace="fbx")
meshes = [m for m in cmds.ls(type="mesh", long=True) if not cmds.getAttr(m + ".intermediateObject")]
if not meshes:
    sys.exit("No meshes found in FBX")
vs, fs, off = [], [], 0
for m in meshes:
    v, f = get_triangles(m)
    vs.append(v)
    fs.append(f + off)
    off += len(v)
np.savez_compressed(dst, vertices=np.concatenate(vs), faces=np.concatenate(fs))
print(f"OK {len(meshes)} meshes -> {dst}")
maya.standalone.uninitialize()
