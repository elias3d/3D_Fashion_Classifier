"""Tiny software renderer (numpy + Pillow) for asset thumbnails.

Only used for the user interface (Maya panel and Streamlit gallery).
The model never sees these images: it only sees the voxel grid.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def render_thumbnail(vertices, faces, size=256, yaw=-35.0, pitch=18.0,
                     color=(124, 140, 220), bg=(246, 247, 250), max_faces=20000,
                     face_colors=None, width=None) -> Image.Image:
    v = np.asarray(vertices, np.float32).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    fc = None if face_colors is None else np.asarray(face_colors, np.float32)
    if len(f) > max_faces:
        keep = np.random.default_rng(0).choice(len(f), max_faces, replace=False)
        f = f[keep]
        fc = None if fc is None else fc[keep]
    v = v - (v.min(0) + v.max(0)) / 2
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    p = v @ (rx @ ry).T
    w = width or size
    span_x, span_y = float(np.ptp(p[:, 0])) or 1.0, float(np.ptp(p[:, 1])) or 1.0
    scale = min(w * 0.92 / span_x, size * 0.86 / span_y)
    xy = np.stack([p[:, 0] * scale + w / 2, -p[:, 1] * scale + size / 2], 1)

    tri = p[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.35, 0.55, 0.75]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)            # two-sided lighting
    order = np.argsort(tri[:, :, 2].mean(1))           # painter's algorithm: far first

    img = Image.new("RGB", (w, size), bg)
    draw = ImageDraw.Draw(img)
    c = np.array(color, np.float32)
    for i in order:
        pts = [tuple(xy[j]) for j in f[i]]
        base = c if fc is None else fc[i]
        col = tuple(int(x) for x in np.clip(base * shade[i], 0, 255))
        draw.polygon(pts, fill=col)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
