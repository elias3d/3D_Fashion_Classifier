"""Garment builder for the browser: the 3D version of the MNIST drawing canvas.

Sliders -> parameters -> a triangle mesh built with numpy (no Maya needed).
The shapes follow the same recipe as the Maya generator (maya_tools/garments.py):
tubes for torso / sleeves / legs / skirts, rings for collars and cuffs, boxes and
ellipsoids for details and shoes. Units: centimetres, Y up, garment faces +Z.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- presets
PRESETS = {
    "tshirt":   dict(mode="clothing", body_len=65, body_width=50, body_depth=0.62, flare=1.04,
                     sleeve_len=19, sleeve_width=9.5, sleeve_angle=30, neck="round", details=[],
                     lower="none", skirt_len=60, skirt_flare=1.8, pleats=0.0,
                     leg_len=87, leg_width=11.0, leg_taper=0.78),
    "shirt":    dict(mode="clothing", body_len=76, body_width=47, body_depth=0.60, flare=1.02,
                     sleeve_len=60, sleeve_width=8.0, sleeve_angle=68, neck="shirt collar",
                     details=["Cuffs", "Button placket"], lower="none", skirt_len=60, skirt_flare=1.8,
                     pleats=0.0, leg_len=87, leg_width=11.0, leg_taper=0.78),
    "pullover": dict(mode="clothing", body_len=64, body_width=57, body_depth=0.72, flare=0.98,
                     sleeve_len=60, sleeve_width=10.5, sleeve_angle=62, neck="high",
                     details=["Cuffs", "Ribbed hem"], lower="none", skirt_len=60, skirt_flare=1.8,
                     pleats=0.0, leg_len=87, leg_width=11.0, leg_taper=0.78),
    "coat":     dict(mode="clothing", body_len=108, body_width=53, body_depth=0.72, flare=1.27,
                     sleeve_len=63, sleeve_width=10.0, sleeve_angle=66, neck="round",
                     details=["Cuffs", "Lapels", "Belt", "Buttons"], lower="none", skirt_len=60,
                     skirt_flare=1.8, pleats=0.0, leg_len=87, leg_width=11.0, leg_taper=0.78),
    "dress":    dict(mode="clothing", body_len=33, body_width=36, body_depth=0.60, flare=1.0,
                     sleeve_len=0, sleeve_width=7.0, sleeve_angle=30, neck="none", details=["Straps"],
                     lower="skirt", skirt_len=70, skirt_flare=2.0, pleats=0.05,
                     leg_len=87, leg_width=11.0, leg_taper=0.78),
    "skirt":    dict(mode="clothing", body_len=0, body_width=36, body_depth=0.78, flare=1.0,
                     sleeve_len=0, sleeve_width=7.0, sleeve_angle=30, neck="none", details=[],
                     lower="skirt", skirt_len=52, skirt_flare=1.8, pleats=0.05,
                     leg_len=87, leg_width=11.0, leg_taper=0.78),
    "pants":    dict(mode="clothing", body_len=0, body_width=38, body_depth=0.70, flare=1.0,
                     sleeve_len=0, sleeve_width=7.0, sleeve_angle=30, neck="none", details=[],
                     lower="legs", skirt_len=60, skirt_flare=1.8, pleats=0.0,
                     leg_len=87, leg_width=11.0, leg_taper=0.78),
    "shoes":    dict(mode="footwear", shoe_len=28, shoe_height=6.5, boot_shaft=0, heel=False, pair=True),
}
NECKS = ["none", "round", "high", "shirt collar"]
DETAILS = ["Cuffs", "Button placket", "Buttons", "Lapels", "Belt", "Hood", "Ribbed hem", "Straps"]
LOWER = ["none", "skirt", "legs"]


# ----------------------------------------------------------------------------- primitives
class Mesh:
    def __init__(self):
        self.v, self.f, self.n = [], [], 0

    def add(self, v, f):
        self.v.append(np.asarray(v, np.float32))
        self.f.append(np.asarray(f, np.int64) + self.n)
        self.n += len(v)

    def arrays(self):
        return np.concatenate(self.v), np.concatenate(self.f)


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _grid_faces(rows, cols, wrap=True):
    f = []
    for i in range(rows - 1):
        for j in range(cols if wrap else cols - 1):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c, d = a + cols, b + cols
            f += [(a, b, d), (a, d, c)]
    return f


def tube(m, r0, r1, length, depth=1.0, rz=0.0, t=(0, 0, 0), profile=None, pleats=0.0, k=10, seg=18, rings=6):
    """Open tube along +Y (radius r0 -> r1), then rotated around Z by rz degrees and moved to t."""
    ts = np.linspace(0, 1, rings)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    T, A = np.meshgrid(ts, ang, indexing="ij")
    r = r0 + (r1 - r0) * T
    if profile is not None:
        r = r * profile(T)
    if pleats:
        r = r * (1 + pleats * np.sin(k * A) * (1 - T))
    v = np.stack([np.cos(A) * r, T * length, np.sin(A) * r * depth], -1).reshape(-1, 3)
    v = v @ _rot_z(rz).T + np.asarray(t, float)
    m.add(v, _grid_faces(rings, seg))


def torus(m, R, r, depth=1.0, y=0.0, rz=0.0, t=(0, 0, 0), seg=20, sides=8):
    u = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    w = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    U, W = np.meshgrid(u, w, indexing="ij")
    v = np.stack([(R + r * np.cos(W)) * np.cos(U), r * np.sin(W) + y, (R + r * np.cos(W)) * np.sin(U) * depth], -1)
    v = v.reshape(-1, 3) @ _rot_z(rz).T + np.asarray(t, float)
    f = []
    for i in range(seg):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = ((i + 1) % seg) * sides + j, ((i + 1) % seg) * sides + (j + 1) % sides
            f += [(a, b, d), (a, d, c)]
    m.add(v, f)


def box(m, w, h, d, t=(0, 0, 0), rot=(0, 0, 0)):
    c = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * [w, h, d]
    rx, ry, rz = np.radians(rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    c = c @ (_rot_z(np.degrees(rz)) @ _rot_y(np.degrees(ry)) @ Rx).T + np.asarray(t, float)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    m.add(c, f)


def ellipsoid(m, rx, ry, rz, t=(0, 0, 0), seg=16, rings=10):
    th = np.linspace(0, np.pi, rings)
    ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(TH) * np.cos(PH) * rx, np.cos(TH) * ry, np.sin(TH) * np.sin(PH) * rz], -1)
    m.add(v.reshape(-1, 3) + np.asarray(t, float), _grid_faces(rings, seg))


def shoulder_profile(t):
    return 1.0 - 0.18 * np.clip((t - 0.82) / 0.18, 0, 1) ** 2


# ----------------------------------------------------------------------------- garments
def _shoe(m, L, H, shaft, heel, x=0.0, yaw=0.0):
    sub = Mesh()
    W, Hs = L * 0.36, 2.8
    box(sub, W, Hs, L, t=(0, Hs / 2, 0))
    ellipsoid(sub, W * 0.5, H, L * 0.42, t=(0, Hs + 2, -L * 0.06))
    ellipsoid(sub, W * 0.45, H * 0.6, L * 0.2, t=(0, Hs + 1.5, L * 0.28))
    if shaft > 0:
        tube(sub, 5.5, 6.0, shaft, depth=1.1, t=(0, Hs + 4, -L * 0.22), rings=3)
    if heel:
        box(sub, W * 0.6, 6, W * 0.6, t=(0, -3, -L * 0.38))
    v, f = sub.arrays()
    m.add(v @ _rot_y(yaw).T + [x, 0, 0], f)


def build(p: dict):
    """Parameters (see PRESETS) -> (vertices (N,3), faces (M,3))."""
    m = Mesh()
    if p.get("mode") == "footwear":
        L, H, shaft, heel = p["shoe_len"], p["shoe_height"], p["boot_shaft"], p["heel"]
        if p["pair"]:
            _shoe(m, L, H, shaft, heel, x=7.5, yaw=-5)
            _shoe(m, L, H, shaft, heel, x=-7.5, yaw=5)
        else:
            _shoe(m, L, H, shaft, heel)
        return m.arrays()

    W, D, H = p["body_width"], p["body_depth"], p["body_len"]
    details = set(p.get("details", []))
    lower = p["lower"]

    # lower part
    top_of_lower = 0.0
    if lower == "skirt":
        S = p["skirt_len"]
        rt = W / 2 * 0.95
        tube(m, rt * p["skirt_flare"], rt, S, depth=D * (1.15 if H > 0 else 1.0), pleats=p["pleats"])
        if H <= 0:                                         # stand-alone skirt: waistband
            tube(m, rt * 1.02, rt * 1.02, 4, depth=D, t=(0, S - 1.5, 0), rings=2)
        top_of_lower = S
    elif lower == "legs":
        L, r0 = p["leg_len"], p["leg_width"]
        rh = W / 2
        for side in (1, -1):
            tube(m, r0, r0 * p["leg_taper"], L, rz=180 + side * 3.5, t=(side * rh * 0.48, L, 0))
            if "Cuffs" in details:
                tip = np.array([side * rh * 0.48, L, 0]) + _rot_z(180 + side * 3.5) @ [0, L, 0]
                torus(m, r0 * p["leg_taper"] * 1.05, 1.2, t=tip)
        hh = 22.0
        tube(m, rh, rh * 0.97, hh, depth=D, t=(0, L - 3, 0))
        if H <= 0:
            torus(m, rh * 0.98, 1.5, depth=D, y=L - 3 + hh)
        top_of_lower = L - 3 + hh

    # upper body
    if H > 0:
        y0 = top_of_lower
        tube(m, W / 2 * p["flare"], W / 2 * 0.95, H, depth=D, t=(0, y0, 0), profile=shoulder_profile)
        top = y0 + H
        front = W / 2 * D
        knit = "Ribbed hem" in details                     # knitted garment: thick hem, cuffs and neck
        if knit:
            rh = W / 2 * p["flare"]
            for k in range(3):
                torus(m, rh * 0.96, 1.6, depth=D, y=y0 - 4 + k * 2.2)
        sl, sr = p["sleeve_len"], p["sleeve_width"]
        if sl > 0:
            d = p["sleeve_angle"]
            taper = 0.95 if sl < 30 else 0.75
            for side in (1, -1):
                rz = -side * (90 + d)
                origin = (side * W / 2 * 0.86, top - sr * 1.05, 0)
                tube(m, sr, sr * taper, sl, rz=rz, t=origin, rings=5)
                if "Cuffs" in details:
                    torus(m, sr * taper * (1.0 if knit else 1.05), 2.0 if knit else 1.4, y=sl, rz=rz, t=origin)
        if "Straps" in details:
            for side in (1, -1):
                tube(m, 1.0, 1.0, 12, rz=side * 8, t=(side * W * 0.22, top - 2, 0), seg=8, rings=2)
        neck = p["neck"]
        if neck == "round":
            torus(m, min(10.5, W * 0.2), 1.6, depth=1.15, y=top - 0.5)
        elif neck == "high":
            rn = min(9.5, W * 0.18)
            tube(m, rn, rn * 0.95, 10, depth=0.95, t=(0, top - 1.5, 0), rings=3)
            tube(m, rn * 0.85, rn * 0.8, 10, depth=0.95, t=(0, top - 1.5, 0), rings=3)   # thick knit wall
        elif neck == "shirt collar":
            tube(m, 8.3, 7.8, 4, depth=1.1, t=(0, top - 1.5, 0), rings=2)
            for side in (1, -1):
                box(m, 8, 0.8, 5.8, t=(side * 4.5, top + 0.6, front * 0.55), rot=(35, 0, -side * 20))
        if "Button placket" in details:
            box(m, 3.0, H * 0.85, 0.8, t=(0, y0 + H * 0.46, front + 0.3))
            for i in range(5):
                ellipsoid(m, 0.7, 0.7, 0.5, t=(0, y0 + H * (0.15 + 0.15 * i), front + 0.9), seg=8, rings=5)
        if "Buttons" in details:
            for i in range(4):
                ellipsoid(m, 1.0, 1.0, 0.6, t=(W * 0.07, y0 + H * (0.35 + 0.11 * i), front + 0.9), seg=8, rings=5)
        if "Lapels" in details:
            for side in (1, -1):
                box(m, 9.5, 22, 1.2, t=(side * W * 0.16, y0 + H * 0.82, front + 0.6), rot=(0, 0, side * 20))
        if "Belt" in details:
            r_mid = W / 2 * p["flare"] + (W / 2 * 0.95 - W / 2 * p["flare"]) * 0.55
            torus(m, r_mid * 1.02, 1.5, depth=D, y=y0 + H * 0.55)
        if "Hood" in details:
            ellipsoid(m, W * 0.22, 16, W * 0.2, t=(0, top + 5, -front * 0.45))

    if m.n == 0:
        raise ValueError("Nothing to build: add a body, a skirt or legs.")
    return m.arrays()
