import pandas as pd
import streamlit as st

from fashion3d import ui
from fashion3d.classes import CLASS_NAMES, display
from fashion3d.dataset import read_manifest
from fashion3d.mesh_io import load_npz
from fashion3d.render import render_thumbnail

st.title("Model and dataset")

m = ui.metrics()
if not m:
    st.info("No training run yet. Train from the Maya panel or run `python scripts/train.py`.")
    st.stop()

with st.container(horizontal=True):
    st.metric("Test accuracy", f"{m.get('test_acc', 0):.1%}", border=True,
              help=f"On {m.get('n_test', 0)} garments generated with a different seed, never seen in training")
    st.metric("Best validation accuracy", f"{m['best_val_acc']:.1%}", border=True)
    st.metric("Training time", f"{m['train_seconds'] / 60:.1f} min", border=True,
              help=f"{m['epochs']} epochs on {m['device_name']}")
    st.metric("Parameters", f"{m['parameters'] / 1e6:.2f} M", border=True)
st.caption(f"Trained {m['trained_at']} on {m['n_train']} assets (+{m['n_val']} validation). "
           "The app loads the saved model from models/voxel_cnn.pt, so it never retrains.")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**Accuracy per epoch**")
        st.altair_chart(ui.history_chart(m["history"], "acc"))
with c2:
    with st.container(border=True):
        st.markdown("**Loss per epoch**")
        st.altair_chart(ui.history_chart(m["history"], "loss"))

c3, c4 = st.columns([1.3, 1])
with c3:
    with st.container(border=True):
        st.markdown("**Confusion matrix** on the test set (rows = true type, columns = prediction)")
        if "confusion_matrix" in m:
            st.altair_chart(ui.confusion_chart(m["confusion_matrix"]))
with c4:
    with st.container(border=True):
        st.markdown("**Accuracy per fashion type**")
        pc = m.get("per_class_acc", {})
        st.dataframe(
            pd.DataFrame({"Type": [display(c) for c in CLASS_NAMES], "Accuracy": [pc.get(c, 0) for c in CLASS_NAMES]}),
            column_config={"Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
            hide_index=True,
        )

if m.get("styles"):
    st.subheader("Assignment test styles")
    st.caption("5 unseen garments per type for each style, generated in Maya. These are the 3D versions of "
               "the centered / corner / thin / thick / messy digit tests.")
    rows = [{"Style": s, "Assets": d["n"], "Accuracy": d["accuracy"], "Mean confidence": d["mean_confidence"],
             "Mistakes": ", ".join(d["mistakes"]) or "none"} for s, d in m["styles"].items()]
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Mean confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
        hide_index=True,
    )


v1_path = ui.PROJECT_ROOT / "models" / "metrics_v1_baseline.json"
stress_path = ui.PROJECT_ROOT / "models" / "stress_test.json"
if v1_path.exists() and m.get("styles"):
    import json

    v1 = json.loads(v1_path.read_text())
    st.subheader("Experiment: improving the augmentation")
    st.caption("v1 stretched garments only 0.85-1.15x during training and failed on slim and bulky garments. "
               "v2 stretches width/depth 0.55-1.6x. Same network, same data, same 30 epochs.")
    comp = pd.DataFrame({
        "Style": list(m["styles"]),
        "v1 accuracy": [v1["styles"][s]["accuracy"] for s in m["styles"]],
        "v2 accuracy": [m["styles"][s]["accuracy"] for s in m["styles"]],
    })
    st.dataframe(comp, hide_index=True, column_config={
        "v1 accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "v2 accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })
if stress_path.exists():
    import json

    stress = json.loads(stress_path.read_text())
    st.subheader("Stress test: garments that are not standing up")
    st.caption("The 240 test garments rotated around X. The model was only trained on upright (Y-up) garments.")
    st.dataframe(pd.DataFrame([
        {"Pose": k, "Accuracy": d["accuracy"], "Mean confidence": d["mean_confidence"],
         "Most common mistakes": ", ".join(f"{a} ×{n}" for a, n in d["top_mistakes"].items()) or "none"}
        for k, d in stress.items()]), hide_index=True, column_config={
        "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Mean confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })


@st.cache_data(max_entries=64)
def thumb(path):
    v, f = load_npz(path)
    return render_thumbnail(v, f, size=150)


st.subheader("Dataset generated in Maya")
counts = {}
for split in ("train", "test", "styles"):
    for r in read_manifest(split):
        counts.setdefault(display(r["label"]), {}).setdefault(split, 0)
        counts[display(r["label"])][split] += 1
st.dataframe(pd.DataFrame(counts).T.fillna(0).astype(int), alt="Assets per fashion type and split")
if not read_manifest("train"):
    st.caption("Showing the shipped sample set (test + styles). The full training set is generated with Maya: "
               "`mayapy scripts/maya_generate_dataset.py`.")

if st.toggle("Show a gallery of training assets"):
    rows = read_manifest("train")
    for c in CLASS_NAMES:
        items = [r for r in rows if r["label"] == c][:6]
        st.markdown(f"**{display(c)}**")
        cols = st.columns(6)
        for col, r in zip(cols, items):
            col.image(thumb(r["path"]), caption=r["asset_id"], alt=f"{display(c)} training asset")
