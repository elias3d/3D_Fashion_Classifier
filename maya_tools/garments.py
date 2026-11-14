"""Procedural placeholder garments built with Maya polygon primitives.

Every garment is assembled from polyPipe (hollow tubes = torso, sleeves,
legs, skirts), polyTorus (collars, cuffs, waistbands), polyCube (soles,
lapels, plackets) and polySphere (shoes, hoods, buttons). Random
proportions make every asset different, so the model has to learn the
*shape* of each fashion type instead of memorising one mesh.

All garments are Y-up and face +Z, with units in centimetres (Maya default).
"""
from __future__ import annotations

import random

import maya.api.OpenMaya as om
import maya.cmds as cmds
import numpy as np

from fashion3d.classes import CLASS_NAMES

from .maya_mesh import set_label, set_string_attr


# ----------------------------------------------------------------- low level
def _points(node):
    sel = om.MSelectionList()
    sel.add(node)
    dp = sel.getDagPath(0)
    dp.extendToShape()
    fn = om.MFnMesh(dp)
    pts = fn.getPoints(om.MSpace.kObject)
    return fn, np.array([(p.x, p.y, p.z) for p in pts], np.float64)


def deform(node, fn):
    """Apply a numpy function (N,3)->(N,3) to every vertex of a mesh."""
    mfn, p = _points(node)
    q = fn(p)
    mfn.setPoints(om.MPointArray([om.MPoint(*row) for row in q]), om.MSpace.kObject)


def place(node, rx=0.0, ry=0.0, rz=0.0, t=(0.0, 0.0, 0.0)):
    """Rotate around the world origin, then move, then freeze transforms."""
    cmds.xform(node, worldSpace=True, pivots=(0, 0, 0))
    cmds.xform(node, rotation=(rx, ry, rz), translation=t, worldSpace=True)
    cmds.makeIdentity(node, apply=True, translate=True, rotate=True, scale=True, normal=False)
    return node


class Builder:
    """Collects primitive parts for one garment."""

    def __init__(self, rng: random.Random, detail: float = 1.0):
        self.rng = rng
        self.detail = detail
        self.parts = []

    def u(self, a, b):
        return self.rng.uniform(a, b)

    def chance(self, p):
        return self.rng.random() < p

    def sa(self, base=16):
        return max(5, int(round(base * self.detail)))

    # ------------------------------------------------------------- primitives
    def pipe(self, r_bottom, r_top, height, depth=1.0, thick=0.08, sh=6, profile=None,
             ripple=(0.0, 0), keep=True):
        """Hollow tube from y=0 (radius r_bottom) to y=height (radius r_top)."""
        node = cmds.polyPipe(radius=1, height=1, thickness=thick, subdivisionsAxis=self.sa(18),
                             subdivisionsHeight=max(2, int(sh * self.detail)), subdivisionsCaps=1,
                             constructionHistory=False)[0]

        def f(p):
            y = p[:, 1]
            t = (y - y.min()) / max(np.ptp(y), 1e-9)
            r = r_bottom + (r_top - r_bottom) * t
            if profile is not None:
                r = r * profile(t)
            amp, k = ripple
            if amp:
                # pleats: strongest at the hem (t=0), fading out towards the waist
                r = r * (1 + amp * np.sin(k * np.arctan2(p[:, 2], p[:, 0])) * (1 - t))
            return np.stack([p[:, 0] * r, t * height, p[:, 2] * r * depth], 1)

        deform(node, f)
        if keep:
            self.parts.append(node)
        return node

    def ring(self, radius, section, depth=1.0, y=0.0, keep=True):
        node = cmds.polyTorus(radius=radius, sectionRadius=section, subdivisionsAxis=self.sa(20),
                              subdivisionsHeight=max(4, int(8 * self.detail)),
                              constructionHistory=False)[0]
        deform(node, lambda p: p * np.array([1.0, 1.0, depth]) + np.array([0.0, y, 0.0]))
        if keep:
            self.parts.append(node)
        return node

    def box(self, w, h, d, t=(0, 0, 0), r=(0, 0, 0), keep=True):
        node = cmds.polyCube(width=w, height=h, depth=d, constructionHistory=False)[0]
        place(node, *r, t=t)
        if keep:
            self.parts.append(node)
        return node

    def ellipsoid(self, rx, ry, rz, t=(0, 0, 0), keep=True):
        node = cmds.polySphere(radius=1, subdivisionsAxis=self.sa(16),
                               subdivisionsHeight=max(4, int(12 * self.detail)),
                               constructionHistory=False)[0]
        deform(node, lambda p: p * np.array([rx, ry, rz]) + np.array(t))
        if keep:
            self.parts.append(node)
        return node

    # --------------------------------------------------------------- garment bits
    @staticmethod
    def shoulder_profile(t):
        return 1.0 - 0.18 * np.clip((t - 0.82) / 0.18, 0, 1) ** 2

    def torso(self, W, H, D, y0=0.0, flare=1.0):
        node = self.pipe(W / 2 * flare, W / 2 * 0.95, H, depth=D, profile=self.shoulder_profile)
        place(node, t=(0, y0, 0))
        return node

    def limb(self, length, r0, r1, angle_z, origin, cuff=None, angle_y=0.0):
        """Sleeve or leg: tube along +Y, then rotated (angle_z) and moved to origin."""
        nodes = [self.pipe(r0, r1, length, sh=5)]
        if cuff:
            c = self.ring(r1 * cuff[0], cuff[1], y=length)
            nodes.append(c)
        for n in nodes:
            place(n, ry=angle_y, rz=angle_z, t=origin)
        return nodes

    def sleeves(self, W, H, sr, L, taper, droop, cuff=None):
        for side in (1, -1):
            self.limb(L, sr, sr * taper, -side * (90 + droop), (side * W / 2 * 0.86, H - sr * 1.05, 0), cuff)


# ------------------------------------------------------------------ garments
def tshirt(b: Builder):
    W, H, D = b.u(44, 56), b.u(58, 72), b.u(0.55, 0.7)
    b.torso(W, H, D, flare=b.u(1.0, 1.08))
    b.sleeves(W, H, sr=b.u(8, 11), L=b.u(14, 24), taper=b.u(0.85, 1.0), droop=b.u(15, 45))
    b.ring(b.u(8, 10.5), 1.2, depth=1.2, y=H - 0.5)


def shirt(b: Builder):
    W, H, D = b.u(42, 52), b.u(70, 82), b.u(0.55, 0.68)
    b.torso(W, H, D, flare=b.u(1.0, 1.05))
    sr = b.u(7, 9)
    b.sleeves(W, H, sr=sr, L=b.u(55, 65), taper=b.u(0.68, 0.8), droop=b.u(55, 80), cuff=(1.05, 1.3))
    front = W / 2 * D
    stand = b.pipe(b.u(7.5, 9), b.u(7, 8.5), b.u(3, 5), depth=1.1)
    place(stand, t=(0, H - 1.5, 0))
    for side in (1, -1):                                   # pointed collar wings
        b.box(b.u(7, 9), 0.8, b.u(5, 6.5), t=(side * 4.5, H + 0.6, front * 0.55), r=(35, 0, -side * 20))
    b.box(3.0, H * 0.85, 0.8, t=(0, H * 0.46, front + 0.3))  # button placket
    for i in range(5):
        b.ellipsoid(0.7, 0.7, 0.5, t=(0, H * (0.15 + 0.15 * i), front + 0.9))


def pullover(b: Builder):
    W, H, D = b.u(52, 62), b.u(58, 70), b.u(0.65, 0.8)
    b.torso(W, H, D, flare=b.u(0.95, 1.02))
    hem_h = b.u(5, 8)
    hem = b.pipe(W / 2 * 0.95, W / 2 * 0.97, hem_h, depth=D, thick=0.15)
    place(hem, t=(0, -hem_h * 0.6, 0))
    b.sleeves(W, H, sr=b.u(9, 12), L=b.u(55, 65), taper=b.u(0.65, 0.8), droop=b.u(50, 75), cuff=(1.0, 2.0))
    neck = b.pipe(b.u(8, 10), b.u(7.5, 9.5), b.u(4, 14), depth=0.95, thick=0.15)
    place(neck, t=(0, H - 1.5, 0))


def coat(b: Builder):
    W, H, D = b.u(48, 58), b.u(95, 120), b.u(0.65, 0.8)
    flare = b.u(1.15, 1.4)
    b.torso(W, H, D, flare=flare)
    b.sleeves(W, H, sr=b.u(9, 11), L=b.u(58, 68), taper=b.u(0.75, 0.9), droop=b.u(55, 78), cuff=(1.05, 1.6))
    front = W / 2 * D
    for side in (1, -1):
        b.box(b.u(8, 11), b.u(18, 26), 1.2, t=(side * W * 0.16, H * 0.82, front + 0.6), r=(0, 0, side * b.u(15, 25)))
    if b.chance(0.5):
        t = 0.55
        r = (W / 2 * flare) + (W / 2 * 0.95 - W / 2 * flare) * t
        b.ring(r * 1.02, 1.5, depth=D, y=H * t)
    if b.chance(0.35):
        b.ellipsoid(W * 0.22, b.u(14, 18), W * 0.2, t=(0, H + 5, -front * 0.45))
    else:
        b.ring(b.u(10, 12), 2.0, depth=1.1, y=H)
    for i in range(b.rng.randint(3, 5)):
        b.ellipsoid(1.0, 1.0, 0.6, t=(W * 0.07, H * (0.35 + 0.11 * i), front + 0.9))


def dress(b: Builder):
    Wb, Hb, D = b.u(32, 40), b.u(28, 38), b.u(0.55, 0.65)
    L = b.u(50, 90)
    rt = Wb / 2 * 0.95
    ripple = (b.u(0.03, 0.08), b.rng.randint(6, 14)) if b.chance(0.5) else (0.0, 0)
    b.pipe(rt * b.u(1.5, 2.6), rt, L, depth=D * 1.15, ripple=ripple)
    bod = b.pipe(Wb / 2, Wb / 2 * 0.92, Hb, depth=D, profile=Builder.shoulder_profile)
    place(bod, t=(0, L, 0))
    top = L + Hb
    if b.chance(0.6):
        for side in (1, -1):
            strap = b.pipe(1.0, 1.0, b.u(10, 15), sh=2)
            place(strap, rz=side * 8, t=(side * Wb * 0.22, top - 2, 0))
    else:
        b.sleeves(Wb, top, sr=b.u(6, 8), L=b.u(8, 13), taper=1.0, droop=b.u(20, 40))
    if b.chance(0.4):
        b.ring(rt * 1.04, 1.2, depth=D * 1.1, y=L)


def skirt(b: Builder):
    r, L = b.u(15, 20), b.u(35, 70)
    D = b.u(0.7, 0.85)
    ripple = (b.u(0.03, 0.09), b.rng.randint(6, 16)) if b.chance(0.5) else (0.0, 0)
    b.pipe(r * b.u(1.3, 2.4), r, L, depth=D, ripple=ripple)
    band = b.pipe(r * 1.02, r * 1.02, b.u(3, 5), depth=D, thick=0.15)
    place(band, t=(0, L - 1.5, 0))


def pants(b: Builder):
    rh, hh, D = b.u(17, 21), b.u(18, 26), b.u(0.65, 0.75)
    L = b.u(75, 100)
    r0 = b.u(9.5, 12.5)
    r1 = r0 * b.u(0.6, 0.95)
    splay = b.u(1, 6)
    cuff = (1.05, 1.2) if b.chance(0.3) else None
    for side in (1, -1):
        b.limb(L, r0, r1, 180 + side * splay, (side * rh * 0.48, L, 0), cuff)
    hips = b.pipe(rh, rh * 0.97, hh, depth=D)
    place(hips, t=(0, L - 3, 0))
    b.ring(rh * 0.98, 1.5, depth=D, y=L - 3 + hh)


def shoes(b: Builder):
    Ls, Ws, Hs = b.u(26, 31), b.u(9, 11), b.u(2, 3.5)
    boot = b.chance(0.4)
    heel = b.chance(0.25)
    shaft_h = b.u(10, 24)

    def one(x_off, yaw):
        start = len(b.parts)
        b.box(Ws, Hs, Ls, t=(0, Hs / 2, 0))
        b.ellipsoid(Ws * 0.5, b.u(5, 8), Ls * 0.42, t=(0, Hs + 2, -Ls * 0.06))
        b.ellipsoid(Ws * 0.45, b.u(3, 4.5), Ls * 0.2, t=(0, Hs + 1.5, Ls * 0.28))
        if boot:
            shaft = b.pipe(b.u(4.5, 6.5), b.u(5, 7), shaft_h, depth=1.1, sh=3)
            place(shaft, t=(0, Hs + 4, -Ls * 0.22))
        if heel:
            hh = b.u(4, 8)
            b.box(Ws * 0.6, hh, Ws * 0.6, t=(0, -hh / 2, -Ls * 0.38))
        for n in b.parts[start:]:
            place(n, ry=yaw, t=(x_off, 0, 0))

    if b.chance(0.55):
        gap = b.u(6, 9)
        one(gap, b.u(-10, 0))
        one(-gap, b.u(0, 10))
    else:
        one(0, 0)


BUILDERS = {
    "tshirt": tshirt, "pants": pants, "pullover": pullover, "dress": dress,
    "coat": coat, "shirt": shirt, "skirt": skirt, "shoes": shoes,
}
assert set(BUILDERS) == set(CLASS_NAMES)


# --------------------------------------------------------------- public API
def build_garment(label, seed=None, name=None, detail=1.0, noise=None, lean=None):
    """Build one labelled garment in the current scene and return its transform."""
    rng = random.Random(seed)
    b = Builder(rng, detail=detail)
    BUILDERS[label](b)
    node = cmds.polyUnite(b.parts, constructionHistory=False, name=name or f"{label}_garment")[0]
    for p in b.parts:
        if cmds.objExists(p):
            cmds.delete(p)

    noise = rng.uniform(0.002, 0.006) if noise is None else noise
    lx, lz = (rng.uniform(-0.04, 0.04), rng.uniform(-0.03, 0.03)) if lean is None else lean
    jitter = np.random.default_rng(rng.randrange(2 ** 31))

    def post(p):
        diag = float(np.linalg.norm(np.ptp(p, axis=0)))
        q = p + jitter.normal(0, noise * diag, size=p.shape)
        q[:, 0] += q[:, 1] * lx
        q[:, 2] += q[:, 1] * lz
        return q

    deform(node, post)
    # make all face normals point the same way, so the viewport has no dark patches
    # (doesn't change the geometry the model sees: surface sampling ignores normals)
    cmds.polyNormal(node, normalMode=2, userNormalMode=0, constructionHistory=False)
    cmds.xform(node, centerPivots=True)
    set_label(node, label, source="generated")
    set_string_attr(node, "fashionSeed", str(seed))
    return node


def build_showcase(spacing=110.0):
    """One garment of every type in a row: handy for screenshots and testing in the GUI."""
    nodes = []
    grp = cmds.group(empty=True, name="fashion_showcase")
    for i, label in enumerate(CLASS_NAMES):
        n = build_garment(label, seed=1000 + i, name=f"showcase_{label}")
        cmds.move((i - (len(CLASS_NAMES) - 1) / 2) * spacing, 0, 0, n, relative=True)
        cmds.parent(n, grp)
        nodes.append(n)
    return grp, nodes
