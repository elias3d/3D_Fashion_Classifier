"""Drag and drop this file into the Maya viewport (Maya 2025+).

It adds a "Fashion3D" shelf button and opens the panel.
"""
import sys
from pathlib import Path

import maya.cmds as cmds
import maya.mel as mel


def onMayaDroppedPythonFile(*_):
    root = Path(__file__).resolve().parent.as_posix()
    if root not in sys.path:
        sys.path.insert(0, root)
    command = (
        "import sys\n"
        f"sys.path.insert(0, r'{root}') if r'{root}' not in sys.path else None\n"
        "import maya_tools.panel as fashion3d_panel\n"
        "fashion3d_panel.show()\n"
    )
    shelf_top = mel.eval("$tmp=$gShelfTopLevel")
    shelf = cmds.tabLayout(shelf_top, query=True, selectTab=True)
    for b in cmds.shelfLayout(shelf, query=True, childArray=True) or []:
        if cmds.shelfButton(b, query=True, exists=True) and cmds.shelfButton(b, query=True, label=True) == "Fashion3D":
            cmds.deleteUI(b)
    cmds.shelfButton(parent=shelf, label="Fashion3D", annotation="Fashion3D classifier panel",
                     imageOverlayLabel="F3D", image1="pythonFamily.png", sourceType="python", command=command)
    import maya_tools.panel as fashion3d_panel
    fashion3d_panel.show()
    print("Fashion3D installed on shelf '%s'" % shelf)
