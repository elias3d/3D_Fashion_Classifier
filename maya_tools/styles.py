"""The 5 test styles from the assignment, translated from 2D drawings to 3D assets.

    centered  -> garment at the world origin, normal proportions
    corner    -> garment moved far from the origin and spun around (like a digit drawn in a corner)
    thin      -> low-poly and slim (like a thin pen stroke)
    thick     -> smoothed, high-poly and bulky (like a thick marker)
    messy     -> noisy surface, holes, a stray object and a tilt (like a scribbled digit)
"""
from __future__ import annotations

import maya.cmds as cmds
import numpy as np

from .garments import deform


class Style:
    detail = 1.0

    def __call__(self, node, rng):
        return node


class Centered(Style):
    pass


class Corner(Style):
    def __call__(self, node, rng):
        cmds.xform(node, translation=(rng.uniform(-900, 900), rng.uniform(-200, 400), rng.uniform(-900, 900)),
                   rotation=(0, rng.uniform(0, 360), 0), worldSpace=True)
        return node


class Thin(Style):
    detail = 0.4                       # fewer polygons around every tube

    def __call__(self, node, rng):
        s = rng.uniform(0.5, 0.65)
        cmds.scale(s, 1, s, node, relative=True)
        cmds.makeIdentity(node, apply=True, scale=True)
        return node


class Thick(Style):
    def __call__(self, node, rng):
        cmds.polySmooth(node, divisions=1, constructionHistory=False)
        s = rng.uniform(1.4, 1.7)
        cmds.scale(s, 1, s, node, relative=True)
        cmds.makeIdentity(node, apply=True, scale=True)
        return node


class Messy(Style):
    def __call__(self, node, rng):
        np_rng = np.random.default_rng(rng.randrange(2 ** 31))

        def noisy(p):
            diag = float(np.linalg.norm(np.ptp(p, axis=0)))
            return p + np_rng.normal(0, 0.012 * diag, size=p.shape)

        deform(node, noisy)
        n_faces = cmds.polyEvaluate(node, face=True)
        holes = np_rng.choice(n_faces, size=int(n_faces * 0.15), replace=False)
        cmds.delete([f"{node}.f[{i}]" for i in sorted(holes)])
        bb = cmds.exactWorldBoundingBox(node)
        size = max(bb[3] - bb[0], bb[4] - bb[1])
        stray = cmds.polySphere(radius=size * rng.uniform(0.08, 0.15), constructionHistory=False)[0]
        cmds.move(rng.uniform(bb[0], bb[3]), rng.uniform(bb[1], bb[4]), bb[5], stray)
        name = node.split("|")[-1]
        node = cmds.polyUnite(node, stray, constructionHistory=False, name=name + "_m")[0]
        for leftover in (name, stray):
            if cmds.objExists(leftover):
                cmds.delete(leftover)
        cmds.rotate(rng.uniform(-18, 18), 0, rng.uniform(-18, 18), node, relative=True)
        return node


STYLES = {"centered": Centered(), "corner": Corner(), "thin": Thin(), "thick": Thick(), "messy": Messy()}
