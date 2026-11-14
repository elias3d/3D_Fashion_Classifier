"""Streamlit helpers: cached model, 3D figures and charts."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from .classes import CLASS_NAMES, display
from .inference import DEFAULT_MODEL, PROJECT_ROOT, FashionClassifier, load_metrics
from .preprocess import projections

BLUE = "#2a78d6"        # categorical slot 1 (prediction / train)
ORANGE = "#eb6834"      # categorical slot 2 (validation)
MUTED = "#c3c2b7"       # non-highlighted bars
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INBOX = PROJECT_ROOT / "runs" / "maya_inbox"


@st.cache_resource(max_entries=2)
def get_classifier(device: str, model_mtime: float) -> FashionClassifier:
    return FashionClassifier(DEFAULT_MODEL, device=device)


def classifier(device: str) -> FashionClassifier | None:
    if not DEFAULT_MODEL.exists():
        return None
    return get_classifier(device, DEFAULT_MODEL.stat().st_mtime)


@st.cache_data(ttl=60)
def metrics() -> dict | None:
    return load_metrics()


# ------------------------------------------------------------------ 3D figures
def _y_up(p):
    """Maya is Y-up, Plotly is Z-up: rotate so garments stand upright."""
    p = np.asarray(p)
    return p[:, 0], -p[:, 2], p[:, 1]


def _scene(fig, height=420):
    axis = dict(visible=False)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=-1.9, z=0.7))),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def mesh_figure(vertices, faces, max_faces=40000, height=420):
    f = np.asarray(faces)
    if len(f) > max_faces:
        f = f[np.random.default_rng(0).choice(len(f), max_faces, replace=False)]
    x, y, z = _y_up(vertices)
    fig = go.Figure(go.Mesh3d(x=x, y=y, z=z, i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#6da7ec", opacity=1.0,
                              flatshading=True,
                              lighting=dict(ambient=0.8, diffuse=0.45, specular=0.0, roughness=1.0,
                                            facenormalsepsilon=0),
                              lightposition=dict(x=50000, y=-50000, z=100000),
                              hoverinfo="skip"))
    return _scene(fig, height)


def voxel_figure(vox, height=420):
    idx = np.argwhere(vox > 0)
    vals = vox[vox > 0]
    x, y, z = _y_up(idx)
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=3.2, symbol="square", color=vals, cmin=0, cmax=1,
                    colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
                    colorbar=dict(title="value", thickness=10, len=0.6)),
        hovertemplate="voxel (%{x}, %{z}, %{y})<br>value %{marker.color:.2f}<extra></extra>",
    ))
    return _scene(fig, height)


def points_figure(points, height=420, n=2500):
    p = np.asarray(points)[:n]
    x, y, z = _y_up(p)
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                 marker=dict(size=1.8, color=BLUE), hoverinfo="skip"))
    return _scene(fig, height)


def projection_images(vox, scale=6):
    """Front/side/top 'X-ray' views of the grid, upscaled with nearest-neighbour so pixels stay crisp."""
    out = {}
    for name, img in projections(vox).items():
        arr = (255 - np.clip(img, 0, 1) * 255).astype(np.uint8)
        out[name] = Image.fromarray(arr).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    return out


# ------------------------------------------------------------------ charts
def probability_chart(probabilities: dict, predicted: str):
    df = pd.DataFrame({
        "type": [display(c) for c in CLASS_NAMES],
        "probability": [probabilities[c] for c in CLASS_NAMES],
        "predicted": [c == predicted for c in CLASS_NAMES],
    })
    base = alt.Chart(df).encode(
        y=alt.Y("type:N", sort=None, title=None),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True),
                title="Probability"),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.condition("datum.predicted", alt.value(BLUE), alt.value(MUTED)),
        tooltip=[alt.Tooltip("type:N", title="Fashion type"),
                 alt.Tooltip("probability:Q", format=".2%", title="Probability")],
    )
    labels = base.mark_text(align="left", dx=4, color="#52514e").encode(
        text=alt.Text("probability:Q", format=".1%"))
    return (bars + labels).properties(height=8 * 30)


def history_chart(history, metric="acc"):
    df = pd.DataFrame(history)
    cols = {"acc": ("train_acc", "val_acc"), "loss": ("train_loss", "val_loss")}[metric]
    long = df.melt(id_vars="epoch", value_vars=list(cols), var_name="series", value_name="value")
    long["series"] = long["series"].map({cols[0]: "Train", cols[1]: "Validation"})
    fmt = "%" if metric == "acc" else ".2f"
    return alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("epoch:Q", title="Epoch"),
        y=alt.Y("value:Q", title="Accuracy" if metric == "acc" else "Loss", axis=alt.Axis(format=fmt),
                scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Train", "Validation"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["epoch:Q", "series:N", alt.Tooltip("value:Q", format=".3f")],
    ).properties(height=260)


def confusion_chart(cm):
    cm = np.asarray(cm)
    rows = []
    for i, t in enumerate(CLASS_NAMES):
        total = max(1, cm[i].sum())
        for j, p in enumerate(CLASS_NAMES):
            rows.append({"true": display(t), "predicted": display(p), "count": int(cm[i, j]),
                         "share": cm[i, j] / total})
    df = pd.DataFrame(rows)
    order = [display(c) for c in CLASS_NAMES]
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", sort=order, title="Predicted type", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("true:N", sort=order, title="True type"),
    )
    cells = base.mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=BLUE_RAMP), legend=None),
        tooltip=["true:N", "predicted:N", "count:Q", alt.Tooltip("share:Q", format=".0%")],
    )
    text = base.mark_text(fontSize=12).encode(
        text=alt.condition("datum.count > 0", "count:Q", alt.value("")),
        color=alt.condition("datum.share > 0.5", alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (cells + text).properties(height=380)


def slice_chart(vox, axis_name, index):
    sl = {"front (z)": vox[:, :, index].T, "side (x)": vox[index, :, :], "top (y)": vox[:, index, :].T}[axis_name]
    g = sl.shape[0]
    rows, cols = np.mgrid[0:g, 0:g]
    df = pd.DataFrame({"col": cols.ravel(), "row": rows.ravel(), "value": sl.ravel()})
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", sort="descending", axis=None),
        color=alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], range=["#fcfcfb"] + BLUE_RAMP), legend=None),
        tooltip=["col:O", "row:O", alt.Tooltip("value:Q", format=".2f")],
    ).properties(height=300, width=300)


# ------------------------------------------------------------------ Maya inbox
def inbox_entries(limit=24):
    if not INBOX.exists():
        return []
    out = []
    for d in sorted(INBOX.iterdir(), reverse=True)[:limit]:
        r = d / "result.json"
        if r.exists():
            meta = json.loads(r.read_text())
            meta["folder"] = str(d)
            out.append(meta)
    return out
