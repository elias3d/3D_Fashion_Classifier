"""Fashion3D: a Streamlit app that classifies 3D garments from Maya.

    streamlit run app.py
"""
import streamlit as st
import torch

from fashion3d.maya_bridge import port_open
from fashion3d.mesh_io import find_mayapy

st.set_page_config(page_title="Fashion3D classifier", page_icon=":material/checkroom:", layout="wide")

with st.sidebar:
    st.subheader("Settings")
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    st.segmented_control("Inference device", devices, default=devices[0], key="device",
                         help="cuda = your NVIDIA GPU, cpu = processor. Results are the same, GPU is faster.")
    st.toggle("Rotation TTA (8 views)", key="tta",
              help="Test-time augmentation: the asset is also classified rotated 8 times around Y "
                   "and the probabilities are averaged. Slower but more robust.")
    st.number_input("Maya command port", value=7001, step=1, key="maya_port",
                    help="Open it from the Maya panel (Classify tab) or with "
                         "cmds.commandPort(name=':7001', sourceType='python')")

    # Maya is optional: check what is available and turn off only the features that need it
    st.subheader("Maya")
    st.session_state["maya_live"] = port_open(st.session_state["maya_port"])
    st.session_state["mayapy"] = find_mayapy()
    status = st.container(horizontal=True)
    if st.session_state["maya_live"]:
        status.badge("Live link: on", icon=":material/link:", color="green")
    else:
        status.badge("Live link: off", icon=":material/link_off:", color="gray")
    if st.session_state["mayapy"]:
        status.badge("Maya installed", icon=":material/check:", color="green")
    else:
        status.badge("Maya not installed", icon=":material/close:", color="gray")
    if not st.session_state["maya_live"]:
        st.caption("The app works without Maya. Only *Maya scene (live)* is turned off. To turn it on, open the "
                   "Fashion3D panel in Maya and click **Open command port :7001**.")
    if not st.session_state["mayapy"]:
        st.caption("FBX upload is turned off (it needs Maya installed). OBJ, STL, PLY, GLB work.")
    st.button("Check again", icon=":material/refresh:", width="stretch")

page = st.navigation([
    st.Page("app_pages/classify.py", title="Classify", icon=":material/checkroom:", default=True),
    st.Page("app_pages/model.py", title="Model and dataset", icon=":material/monitoring:"),
    st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/school:"),
])
page.run()
