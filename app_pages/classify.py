import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fashion3d import ui
from fashion3d.classes import CLASS_NAMES, display
from fashion3d.dataset import DATA_DIR, read_manifest
from fashion3d.maya_bridge import MayaBridge
from fashion3d.mesh_io import load_mesh, load_npz
from fashion3d.render import render_thumbnail

st.title("Fashion3D: what type of garment is this 3D asset?")
st.markdown(
    "Pick a 3D asset (live from **Maya**, an uploaded file, or a test asset). The app cleans it up "
    "into a **32 × 32 × 32 voxel grid** with values between 0 and 1, sends it to a **3D convolutional "
    "neural network** trained on garments generated in Maya, and shows the predicted fashion type, "
    "how confident the model is, and the full probability breakdown for all 8 types."
)

device = st.session_state.get("device") or "cpu"
tta = 8 if st.session_state.get("tta") else 1
clf = ui.classifier(device)
if clf is None:
    st.error("No trained model yet. Train it from the Maya panel (Train tab) or run "
             "`python scripts/train.py`.", icon=":material/error:")
    st.stop()

ss = st.session_state
ss.setdefault("current", None)
ss.setdefault("uploader_key", 0)


def set_current(name, vertices, faces, source, true_label=None, node=None):
    ss.current = {"name": name, "v": vertices, "f": faces, "source": source,
                  "true_label": true_label, "node": node}


@st.cache_data(max_entries=300)
def cached_npz(path):
    return load_npz(path)


@st.cache_data(max_entries=64)
def cached_thumb(path):
    v, f = load_npz(path)
    return render_thumbnail(v, f, size=160)


# ------------------------------------------------------------------ pick an asset
maya_live = bool(ss.get("maya_live"))
has_mayapy = bool(ss.get("mayapy"))
source = st.segmented_control(
    "3D asset source",
    ["Test assets", "Maya scene (live)", "Upload a file", "Maya panel history"],
    default="Test assets", key="source",
    format_func=lambda s: f":material/lock: {s}" if s == "Maya scene (live)" and not maya_live else s,
)
source = source or "Test assets"
if ss.get("last_source") != source:     # switching source starts fresh
    ss.current = None
    ss.last_source = source

with st.container(border=True):
    if source == "Test assets":
        st.caption("Garments the model has never seen: the held-out test set plus the 5 assignment test styles.")
        rows = read_manifest("test") + read_manifest("styles")
        if not rows:
            st.info("No test assets found (neither data/ nor samples/).")
        else:
            groups = {"test set": [r for r in rows if r["split"] == "test"]}
            for s in ("centered", "corner", "thin", "thick", "messy"):
                groups[f"style: {s}"] = [r for r in rows if r.get("style") == s]
            groups = {k: v for k, v in groups.items() if v}
            c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
            group = c1.selectbox("Group", list(groups))
            label = c2.selectbox("Fashion type", CLASS_NAMES, format_func=display)
            items = [r for r in groups[group] if r["label"] == label]
            if c3.button("Random asset", icon=":material/casino:", width="stretch"):
                ss.pick = random.randrange(len(items))
            idx = ss.get("pick", 0) % max(1, len(items))
            if items:
                row = items[idx]
                v, f = cached_npz(row["path"])
                set_current(row["asset_id"], v, f, f"Test assets / {group}", true_label=row["label"])

    elif source == "Maya scene (live)" and not maya_live:
        st.info("**This feature needs Maya.** It reads garments straight from your open Maya scene. "
                "To turn it on: open Maya → Fashion3D panel → Classify tab → **Open command port :7001**, "
                "then click **Check again** in the sidebar. Everything else in the app works without Maya.",
                icon=":material/lock:")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        c1.selectbox("Asset in the Maya scene", ["Maya is not connected"], disabled=True)
        c2.button("Classify from Maya", icon=":material/sync:", width="stretch", disabled=True)

    elif source == "Maya scene (live)":
        bridge = MayaBridge(port=int(ss.get("maya_port", 7001)), timeout=30)
        try:
            info = bridge.list_assets()
        except Exception as e:
            info = None
            st.warning(f"Maya's port is open but it didn't answer (is Maya busy?). Details: {e}",
                       icon=":material/link_off:")
        if info:
            assets = info["assets"]
            st.caption(f"Connected to Maya, scene **{info['scene']}**: {len(assets)} asset(s).")
            if not assets:
                st.info("The scene has no polygon assets. Try the panel's *Build showcase* button.")
            else:
                c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
                names = [a["node"] for a in assets]
                labels = {a["node"]: a["label"] for a in assets}
                node = c1.selectbox("Asset in the Maya scene", names,
                                    format_func=lambda n: n.split("|")[-1] + (f"  (tagged {display(labels[n])})"
                                                                              if labels[n] else ""))
                if c2.button("Classify from Maya", type="primary", icon=":material/sync:", width="stretch"):
                    with st.spinner("Exporting triangles from Maya..."):
                        path = bridge.export_mesh(node)
                        v, f = load_npz(path)
                    set_current(node.split("|")[-1], v, f, "Maya scene (live)", labels[node], node=node)

    elif source == "Upload a file":
        types = ["obj", "stl", "ply", "glb", "gltf", "off", "npz"] + (["fbx"] if has_mayapy else [])
        up = st.file_uploader("3D file", type=types, key=f"upload_{ss.uploader_key}",
                              help="FBX is converted by a headless Maya (mayapy), which takes ~10-20 s."
                              if has_mayapy else "FBX needs Maya installed. Export your asset as OBJ instead.")
        if not has_mayapy:
            st.caption(":material/lock: FBX is turned off because Maya isn't installed on this computer. "
                       "OBJ / STL / PLY / GLB work. Example files are in `samples/obj/`.")
        else:
            st.caption("Example files to try are in `samples/obj/`.")
        if up is not None and (ss.current is None or ss.current.get("upload_id") != up.file_id):
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getvalue())
            try:
                with st.spinner("Reading the mesh" + (" with headless Maya..." if tmp.suffix.lower() == ".fbx"
                                                      else "...")):
                    v, f = load_mesh(tmp)
                set_current(up.name, v, f, "Upload")
                ss.current["upload_id"] = up.file_id
            except Exception as e:
                st.error(f"Could not read this file: {e}", icon=":material/error:")

    else:  # Maya panel history
        entries = ui.inbox_entries()
        if not entries:
            st.info("Nothing yet. Classify something with the Maya panel and it shows up here.")
        else:
            st.caption("Every asset classified from the Maya panel is saved here (newest first).")
            labels = [f"{e['received']}  ·  {e['name']}  ->  {e['display']} {e['confidence']:.0%}" for e in entries]
            i = st.selectbox("Prediction from Maya", range(len(entries)), format_func=lambda k: labels[k])
            e = entries[i]
            v, f = cached_npz(str(Path(e["folder"]) / "mesh.npz"))
            set_current(e["name"], v, f, f"Maya panel ({e.get('scene') or 'scene'})", e.get("true_label"))

cur = ss.current
if cur is None:
    st.stop()

# ------------------------------------------------------------------ predict
t0 = time.perf_counter()
res = clf.predict_mesh(cur["v"], cur["f"], tta=tta)
ms = (time.perf_counter() - t0) * 1000
vox, info = res["voxels"], res["info"]

head = st.container(horizontal=True, vertical_alignment="center")
head.subheader(cur["name"])
if cur.get("true_label"):
    ok = cur["true_label"] == res["label"]
    head.badge(f"Label: {display(cur['true_label'])}", icon=":material/sell:", color="gray")
    head.badge("Correct" if ok else "Wrong", icon=":material/check:" if ok else ":material/close:",
               color="green" if ok else "red")
if head.button("Clear", icon=":material/restart_alt:"):
    ss.current = None
    ss.uploader_key += 1
    st.rerun()
st.caption(f"Source: {cur['source']} · {info['n_vertices']:,} vertices · {info['n_faces']:,} triangles · "
           f"inference {ms:.0f} ms on {device}{' with 8-view TTA' if tta > 1 else ''}")

left, right = st.columns([1.1, 1])
with left:
    with st.container(border=True):
        st.markdown("**Original 3D asset**")
        st.plotly_chart(ui.mesh_figure(cur["v"], cur["f"]), key="mesh3d")
with right:
    with st.container(border=True):
        st.markdown("**Prediction**")
        m1, m2 = st.columns(2)
        m1.metric("Predicted fashion type", res["display"])
        m2.metric("Confidence", f"{res['confidence']:.1%}",
                  delta=f"{res['confidence'] - res['top3'][1]['probability']:.1%} ahead of #2",
                  delta_color="off")
        st.markdown("Top 3: " + "  ·  ".join(f"**{t['display']}** {t['probability']:.1%}" for t in res["top3"]))
        st.altair_chart(ui.probability_chart(res["probabilities"], res["label"]))
        if res["confidence"] < 0.6:
            st.warning("Low confidence: the model is unsure between several types.", icon=":material/help:")

    if cur["source"] == "Maya scene (live)" and cur.get("node"):
        bridge = MayaBridge(port=int(ss.get("maya_port", 7001)))
        b = st.container(horizontal=True)
        off = not maya_live
        if b.button("Write result to Maya", icon=":material/save:", disabled=off):
            bridge.call("write_prediction", node=cur["node"], label=res["label"], confidence=res["confidence"])
            st.toast("Saved fashionPredicted / fashionConfidence on the Maya node", icon=":material/check:")
        if b.button("Tag in Maya with this type", icon=":material/sell:", disabled=off):
            bridge.call("set_label", node=cur["node"], label=res["label"])
            st.toast(f"Tagged {cur['name']} as {res['display']}", icon=":material/check:")
        if b.button("Show model view in Maya", icon=":material/view_in_ar:", disabled=off):
            idx = np.argwhere(vox > 0)
            payload = {"grid": int(vox.shape[0]), "info": {"center": info["center"],
                                                           "half_extent": info["half_extent"]},
                       "voxels": [[int(i), int(j), int(k), float(vox[i, j, k])] for i, j, k in idx]}
            p = Path(tempfile.mkdtemp()) / "result.json"
            p.write_text(json.dumps(payload))
            node = bridge.call("voxel_twin", node=cur["node"], result=str(p))["node"]
            st.toast(f"Created {node} next to the asset", icon=":material/check:")

# ------------------------------------------------------------------ what the model sees
st.subheader("What the model sees")
st.markdown(
    f"The mesh is turned into **{info['n_points']:,} surface points**, centered and scaled so its longest side "
    f"fits in [-1, 1], then counted into a **{vox.shape[0]}³ voxel grid**. Each value is between "
    f"**{info['value_min']:.0f} and {info['value_max']:.0f}** (log point density ÷ max). "
    f"{info['occupied']:,} of {vox.size:,} voxels are filled. Model input shape: `(1, 1, {vox.shape[0]}, "
    f"{vox.shape[1]}, {vox.shape[2]})`."
)
c1, c2 = st.columns([1.1, 1])
with c1:
    with st.container(border=True):
        st.markdown("**32 × 32 × 32 voxel grid** (drag to rotate)")
        st.plotly_chart(ui.voxel_figure(vox), key="voxels3d")
with c2:
    with st.container(border=True):
        st.markdown("**X-ray views** (each pixel = max value through the grid)")
        imgs = ui.projection_images(vox)
        cols = st.columns(3)
        for col, name in zip(cols, ("front", "side", "top")):
            col.image(imgs[name], caption=name, alt=f"{name} projection of the voxel grid")
    with st.container(border=True):
        st.markdown("**Raw voxel values**: one 32 × 32 slice")
        a1, a2 = st.columns([1, 2])
        axis = a1.selectbox("Slice direction", ["front (z)", "side (x)", "top (y)"])
        k = a2.slider("Slice index", 0, vox.shape[0] - 1, vox.shape[0] // 2)
        st.altair_chart(ui.slice_chart(vox, axis, k))

with st.expander("Show the raw 32 × 32 number matrix for this slice"):
    sl = {"front (z)": vox[:, :, k].T[::-1], "side (x)": vox[k, :, :][::-1], "top (y)": vox[:, k, :].T}[axis]
    st.dataframe(pd.DataFrame(np.round(sl, 2)), height=320)

with st.expander("Preprocessing step by step"):
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Step 1-3**: sampled surface points after centering and scaling to [-1, 1]")
        st.plotly_chart(ui.points_figure(res["points"], height=320), key="points3d")
    with s2:
        st.table(pd.DataFrame({
            "Step": ["1. Read triangles", "2. Sample surface", "3. Center + scale",
                     "4. Voxelize", "5. Normalize", "6. Model input"],
            "Result": [f"{info['n_vertices']:,} vertices, {info['n_faces']:,} triangles",
                       f"{info['n_points']:,} points, area weighted",
                       f"center {np.round(info['center'], 1).tolist()}, half size {info['half_extent']:.1f}",
                       f"{vox.shape[0]}³ grid, {info['occupied']:,} filled voxels",
                       f"values {info['value_min']:.2f} to {info['value_max']:.2f}",
                       f"tensor (1, 1, {vox.shape[0]}, {vox.shape[0]}, {vox.shape[0]})"],
        }).set_index("Step"))
