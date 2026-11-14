"""3D preprocessing: triangle mesh -> normalized 32x32x32 voxel grid.

This is the 3D version of the MNIST pipeline:

    MNIST:  drawing -> grayscale -> resize 28x28      -> pixels / 255  -> model
    Here :  mesh    -> surface points -> center+scale -> voxelize 32^3 -> 0..1 -> model

Only numpy is used, so the exact same code runs in Maya (mayapy), in the
training script and in the Streamlit app.
"""
from __future__ import annotations

import numpy as np

GRID = 32          # voxel grid resolution (32 x 32 x 32 = 32,768 "3D pixels")
N_POINTS = 6000    # surface points sampled from the mesh
MARGIN = 0.96      # keep the shape slightly inside the grid border


def clean_mesh(vertices, faces):
    """Return float32 (V,3) vertices and int64 (F,3) faces, dropping bad triangles."""
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        raise ValueError("Mesh has no vertices or faces.")
    f = f[(f >= 0).all(1) & (f < len(v)).all(1)]
    if not np.isfinite(v).all():
        raise ValueError("Mesh has NaN/inf vertex positions.")
    return v, f


def sample_surface(vertices, faces, n=N_POINTS, rng=None):
    """Sample n points uniformly over the mesh surface (area-weighted).

    Using the surface (not the vertices) makes the result independent of how
    many polygons the artist used: a low-poly and a high-poly coat give the
    same point cloud.
    """
    rng = rng or np.random.default_rng(0)
    tri = vertices[faces]                                   # (F,3,3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        return vertices[rng.integers(0, len(vertices), n)].astype(np.float32)
    idx = rng.choice(len(faces), size=n, p=area / total)
    r1 = np.sqrt(rng.random(n, dtype=np.float32))
    r2 = rng.random(n, dtype=np.float32)
    t = tri[idx]
    pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
    return pts.astype(np.float32)


SCALE_RANGE = (0.55, 1.6)   # v2. v1 used (0.85, 1.15) and failed on slim / bulky garments


def augment_points(points, rng, scale_range=SCALE_RANGE):
    """Random changes used only while training, so the model generalises.

    - any rotation around the up (Y) axis: a coat is a coat whichever way it faces
    - a small tilt (+-8 degrees)
    - stretch width/depth by scale_range (relative to the height), so slim and bulky
      versions of a garment are seen during training
    - tiny jitter
    """
    a = rng.uniform(0, 2 * np.pi)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    tx, tz = np.radians(rng.uniform(-8, 8, size=2))
    rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]], np.float32)
    rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]], np.float32)
    p = points @ (ry @ rx @ rz).T
    lo, hi = scale_range
    # log-uniform so shrinking and growing are equally likely; one shared factor for X/Z
    # (garments get slimmer/bulkier as a whole) plus a small independent wobble per axis
    s = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    p = p * (np.array([s, 1.0, s]) * rng.uniform(0.9, 1.1, size=3)).astype(np.float32)
    span = float(np.ptp(p, axis=0).max()) or 1.0
    p = p + rng.normal(0, 0.004 * span, size=p.shape).astype(np.float32)
    return p


def rotate_y(points, degrees):
    a = np.radians(degrees)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    return points @ ry.T


def normalize_points(points):
    """Center the shape and scale it so its longest side spans [-1, 1].

    This is why a garment far from the world origin, or modelled in metres
    instead of centimetres, looks identical to the model.
    """
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    half = float((hi - lo).max()) / 2 or 1.0
    norm = (points - center) / half * MARGIN
    return norm.astype(np.float32), {"center": center.tolist(), "half_extent": half}


def voxelize(norm_points, grid=GRID):
    """Turn normalized points into a grid of values between 0 and 1.

    Each cell counts how many surface points fall inside it. Counts are
    log-scaled and divided by the maximum, so the busiest cell = 1.0 and
    empty space = 0.0 (the 3D equivalent of dividing pixels by 255).
    """
    idx = np.floor((norm_points + 1) / 2 * grid).astype(np.int64).clip(0, grid - 1)
    counts = np.zeros((grid, grid, grid), np.float32)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    vox = np.log1p(counts)
    m = vox.max()
    if m > 0:
        vox /= m
    return vox


def mesh_to_voxels(vertices, faces, grid=GRID, n_points=N_POINTS, rng=None, augment=False, yaw=0.0,
                   scale_range=SCALE_RANGE):
    """Full pipeline. Returns (voxels[grid,grid,grid] float32 in 0..1, info dict, normalized points)."""
    v, f = clean_mesh(vertices, faces)
    rng = rng or np.random.default_rng(0)
    pts = sample_surface(v, f, n_points, rng)
    if augment:
        pts = augment_points(pts, rng, scale_range)
    elif yaw:
        pts = rotate_y(pts, yaw)
    norm, info = normalize_points(pts)
    vox = voxelize(norm, grid)
    info.update(
        n_vertices=int(len(v)),
        n_faces=int(len(f)),
        n_points=int(n_points),
        occupied=int((vox > 0).sum()),
        value_min=float(vox.min()),
        value_max=float(vox.max()),
    )
    return vox, info, norm


def projections(vox):
    """Three 2D 'X-ray' views (max over one axis) for display.

    Returned as images with Y pointing up: front (x,y), side (z,y), top (x,z).
    """
    front = vox.max(axis=2).T[::-1]   # look along Z
    side = vox.max(axis=0)[::-1]      # look along X: (y, z) -> rows = y
    top = vox.max(axis=1).T           # look down Y
    return {"front": front, "side": side, "top": top}
