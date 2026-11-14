"""Fashion3D dockable panel for Maya 2025+.

    import sys; sys.path.insert(0, r"<project folder>")
    import maya_tools.panel as p; p.show()

Tabs: 1 Dataset (generate) | 2 Label | 3 Train | 4 Classify
"""
from __future__ import annotations

import base64
import os
import re
import webbrowser
from pathlib import Path

import maya.cmds as cmds
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # Maya 2024 and older
    from PySide2 import QtCore, QtGui, QtWidgets

from fashion3d.classes import CLASS_NAMES, display

from . import PROJECT_ROOT, client
from .actions import classify_nodes, find_project_python, import_fbx_files
from .dataset_tools import DATA_DIR, append_manifest, dataset_counts, export_asset, generate_split
from .garments import build_showcase
from .maya_mesh import get_label, set_label, top_level_assets

WORKSPACE = "fashion3dPanelWorkspaceControl"
COMMAND_PORT = ":7001"
STREAMLIT_PORT = 8517
_PROCS = {}          # keep server / training / streamlit processes alive across panel reloads
_panel = None


def _button(text, slot, tip=None, primary=False):
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    if primary:
        b.setStyleSheet("QPushButton{background:#5865c9;color:white;font-weight:bold;padding:6px;}")
    return b


class FashionPanel(MayaQWidgetDockableMixin, QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fashion3dPanel")
        self.setWindowTitle("Fashion3D classifier")
        self.results = {}
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel("<b>Fashion3D</b> - classify 3D garments with a voxel CNN")
        lay.addWidget(head)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._dataset_tab(), "1 Dataset")
        self.tabs.addTab(self._label_tab(), "2 Label")
        self.tabs.addTab(self._train_tab(), "3 Train")
        self.tabs.addTab(self._classify_tab(), "4 Classify")
        lay.addWidget(self.tabs)
        self.status = QtWidgets.QLabel()
        lay.addWidget(self.status)
        self._job = cmds.scriptJob(event=["SelectionChanged", self._refresh_selection], killWithScene=False)
        self.destroyed.connect(lambda *_: cmds.scriptJob(kill=self._job, force=True)
                               if cmds.scriptJob(exists=self._job) else None)
        self._refresh_counts()
        self._check_server()

    # ------------------------------------------------------------------ dataset
    def _dataset_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.train_n = QtWidgets.QSpinBox(minimum=1, maximum=2000, value=150)
        self.test_n = QtWidgets.QSpinBox(minimum=0, maximum=500, value=30)
        self.seed = QtWidgets.QSpinBox(minimum=0, maximum=99999, value=1)
        self.train_fbx = QtWidgets.QCheckBox("Also write FBX for train split (~110 KB each)")
        f.addRow("Train assets per type", self.train_n)
        f.addRow("Test assets per type", self.test_n)
        f.addRow("Random seed", self.seed)
        f.addRow(self.train_fbx)
        f.addRow(_button("Generate labelled dataset", self._generate, primary=True))
        f.addRow(_button("Build showcase (one of each type) in scene", self._showcase))
        f.addRow(_button("Open data folder", lambda: os.startfile(str(DATA_DIR)) if DATA_DIR.exists() else None))
        self.counts = QtWidgets.QLabel()
        self.counts.setWordWrap(True)
        f.addRow(self.counts)
        return w

    def _refresh_counts(self):
        c = dataset_counts()
        if not c:
            self.counts.setText("No dataset yet.")
            return
        lines = []
        for split in sorted({s for s, _ in c}):
            parts = [f"{display(l)} {c.get((split, l), 0)}" for l in CLASS_NAMES]
            lines.append(f"<b>{split}</b>: " + ", ".join(parts))
        self.counts.setText("<br>".join(lines))

    def _generate(self):
        n_train, n_test = self.train_n.value(), self.test_n.value()
        total = (n_train + n_test) * len(CLASS_NAMES)
        dlg = QtWidgets.QProgressDialog("Building garments...", "Cancel", 0, total, self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        offset = {"n": 0}

        def progress(done, _tot, msg):
            dlg.setValue(offset["n"] + done)
            dlg.setLabelText(msg)
            QtWidgets.QApplication.processEvents()
            return not dlg.wasCanceled()

        cmds.undoInfo(stateWithoutFlush=False)
        try:
            generate_split("train", n_train, self.seed.value(), write_fbx=self.train_fbx.isChecked(),
                           progress=progress)
            offset["n"] = n_train * len(CLASS_NAMES)
            if n_test and not dlg.wasCanceled():
                generate_split("test", n_test, self.seed.value() + 998, progress=progress)
        finally:
            cmds.undoInfo(stateWithoutFlush=True)
            dlg.close()
        self._refresh_counts()
        self.status.setText(f"Dataset written to {DATA_DIR}")

    def _showcase(self):
        grp, _ = build_showcase()
        cmds.select(grp)
        cmds.viewFit()

    # ------------------------------------------------------------------ label
    def _label_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.sel_info = QtWidgets.QLabel("Select assets in the viewport.")
        self.sel_info.setWordWrap(True)
        f.addRow(self.sel_info)
        self.label_combo = QtWidgets.QComboBox()
        for c in CLASS_NAMES:
            self.label_combo.addItem(display(c), c)
        f.addRow("Fashion type", self.label_combo)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("Tag selection", self._tag, primary=True))
        row.addWidget(_button("Suggest from model", self._suggest, "Pre-fill the type with the model's prediction"))
        f.addRow(row)
        f.addRow(_button("Import FBX files...", self._import_fbx))
        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(["train", "test"])
        f.addRow("Dataset split", self.split_combo)
        f.addRow(_button("Export tagged selection to dataset", self._export_tagged))
        return w

    def _refresh_selection(self):
        try:
            nodes = top_level_assets()
        except Exception:
            return
        if not nodes:
            self.sel_info.setText("Select assets in the viewport.")
            return
        bits = [f"{n.split('|')[-1]}: <b>{display(get_label(n)) if get_label(n) else 'unlabelled'}</b>"
                for n in nodes[:8]]
        self.sel_info.setText("<br>".join(bits) + ("<br>..." if len(nodes) > 8 else ""))

    def _tag(self):
        label = self.label_combo.currentData()
        nodes = top_level_assets()
        for n in nodes:
            set_label(n, label)
        self._refresh_selection()
        self.status.setText(f"Tagged {len(nodes)} asset(s) as {display(label)}")

    def _suggest(self):
        try:
            res = classify_nodes(top_level_assets()[:1], write_attrs=False)
        except Exception as e:
            return self._error(e)
        if res:
            r = res[0][1]
            self.label_combo.setCurrentIndex(CLASS_NAMES.index(r["label"]))
            self.status.setText(f"Model suggests {r['display']} ({r['confidence']:.0%}). Check it, then Tag.")

    def _import_fbx(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Import FBX", "", "FBX (*.fbx)")
        if paths:
            roots = import_fbx_files(paths)
            cmds.select(roots)
            self.status.setText(f"Imported {len(roots)} asset(s). Tag them, then export.")

    def _export_tagged(self):
        nodes = top_level_assets()
        split = self.split_combo.currentText()
        rows, skipped = [], []
        for n in nodes:
            if get_label(n) in CLASS_NAMES:
                rows.append(export_asset(n, split, source="manual"))
            else:
                skipped.append(n.split("|")[-1])
        if rows:
            append_manifest(rows, split)
        self._refresh_counts()
        msg = f"Exported {len(rows)} asset(s) to {split}."
        if skipped:
            msg += f" Skipped unlabelled: {', '.join(skipped[:5])}"
        self.status.setText(msg)

    # ------------------------------------------------------------------ train
    def _train_tab(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        self.python = QtWidgets.QLineEdit(find_project_python())
        self.python.editingFinished.connect(
            lambda: cmds.optionVar(stringValue=("fashion3d_python", self.python.text())))
        f.addRow("Python with PyTorch", self.python)
        self.epochs = QtWidgets.QSpinBox(minimum=1, maximum=500, value=30)
        self.batch = QtWidgets.QSpinBox(minimum=4, maximum=256, value=32)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setToolTip("auto = NVIDIA GPU if PyTorch can see one, otherwise CPU")
        f.addRow("Epochs", self.epochs)
        f.addRow("Batch size", self.batch)
        f.addRow("Device", self.device)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("Train model", self._train, primary=True))
        row.addWidget(_button("Stop", self._stop_train))
        f.addRow(row)
        self.train_bar = QtWidgets.QProgressBar()
        f.addRow(self.train_bar)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMinimumHeight(180)
        f.addRow(self.log)
        return w

    def _train(self):
        if _PROCS.get("train") and _PROCS["train"].state() != QtCore.QProcess.NotRunning:
            return self.status.setText("Training is already running.")
        p = QtCore.QProcess()
        p.setWorkingDirectory(str(PROJECT_ROOT))
        p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._train_output(p))
        p.finished.connect(lambda *_: self._train_done(p))
        self.log.clear()
        self.train_bar.setRange(0, self.epochs.value())
        self.train_bar.setValue(0)
        p.start(self.python.text(), ["-u", "scripts/train.py", "--epochs", str(self.epochs.value()),
                                     "--batch", str(self.batch.value()), "--device", self.device.currentText()])
        _PROCS["train"] = p
        self.status.setText("Training...")

    def _train_output(self, p):
        text = bytes(p.readAllStandardOutput()).decode(errors="replace")
        self.log.appendPlainText(text.rstrip())
        for m in re.finditer(r"EPOCH (\d+)/(\d+)", text):
            self.train_bar.setValue(int(m.group(1)))

    def _train_done(self, p):
        ok = p.exitCode() == 0
        self.status.setText("Training finished." if ok else "Training failed - see log.")
        if ok and not client.health().get("offline"):
            try:
                client.reload_model()
                self.status.setText("Training finished. Server reloaded the new model.")
            except Exception:
                pass
        self._check_server()

    def _stop_train(self):
        p = _PROCS.get("train")
        if p:
            p.kill()

    # ------------------------------------------------------------------ classify
    def _classify_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.server_lbl = QtWidgets.QLabel("Server: ?")
        row.addWidget(self.server_lbl, 1)
        row.addWidget(_button("Start server", self._start_server))
        row.addWidget(_button("Check", self._check_server))
        v.addLayout(row)
        opts = QtWidgets.QHBoxLayout()
        self.tta = QtWidgets.QCheckBox("Rotation TTA (8 views)")
        self.tta.setToolTip("Average the prediction over 8 rotations around Y: slower, more robust")
        self.write_attrs = QtWidgets.QCheckBox("Write result to attributes")
        self.write_attrs.setChecked(True)
        self.voxels = QtWidgets.QCheckBox("Show model view (voxels)")
        for c in (self.tta, self.write_attrs, self.voxels):
            opts.addWidget(c)
        v.addLayout(opts)
        v.addWidget(_button("Classify selection", self._classify, primary=True))
        self.result_list = QtWidgets.QListWidget()
        self.result_list.setIconSize(QtCore.QSize(72, 72))
        self.result_list.currentItemChanged.connect(self._show_result)
        v.addWidget(self.result_list, 1)
        self.bars = {}
        grid = QtWidgets.QGridLayout()
        for i, c in enumerate(CLASS_NAMES):
            grid.addWidget(QtWidgets.QLabel(display(c)), i, 0)
            bar = QtWidgets.QProgressBar(maximum=1000, textVisible=True)
            bar.setFormat("0%")
            grid.addWidget(bar, i, 1)
            self.bars[c] = bar
        v.addLayout(grid)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(_button("Open command port :7001", self._command_port,
                               "Lets the Streamlit app list, classify and label assets in this scene"))
        row2.addWidget(_button("Open Streamlit app", self._streamlit))
        v.addLayout(row2)
        return w

    def _check_server(self):
        h = client.health()
        if h.get("ok"):
            acc = f", test acc {h['test_acc']:.1%}" if h.get("test_acc") is not None else ""
            self.server_lbl.setText(f"Server: <span style='color:#3a3'>online</span> ({h['device']}{acc})")
        elif h.get("offline"):
            self.server_lbl.setText("Server: <span style='color:#c33'>offline</span>")
        else:
            self.server_lbl.setText(f"Server: no model - {h.get('error', '')[:60]}")
        return h

    def _start_server(self):
        if not client.health().get("offline"):
            return self._check_server()
        p = QtCore.QProcess()
        p.setWorkingDirectory(str(PROJECT_ROOT))
        p.start(self.python.text(), ["-u", "scripts/serve.py", "--device", self.device.currentText()])
        _PROCS["server"] = p
        self.server_lbl.setText("Server: starting...")
        QtCore.QTimer.singleShot(6000, self._check_server)

    def _classify(self):
        try:
            results = classify_nodes(tta=8 if self.tta.isChecked() else 1,
                                     write_attrs=self.write_attrs.isChecked(),
                                     show_voxels=self.voxels.isChecked())
        except Exception as e:
            return self._error(e)
        for node, r in results:
            pm = QtGui.QPixmap()
            pm.loadFromData(base64.b64decode(r["thumbnail_png"]))
            second = r["top3"][1]
            truth = get_label(node)
            mark = "" if not truth else ("  [correct]" if truth == r["label"] else f"  [label: {display(truth)}]")
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f"{node.split('|')[-1]}\n{r['display']}  "
                                             f"{r['confidence']:.1%}   (2nd: {second['display']} "
                                             f"{second['probability']:.1%}){mark}")
            item.setData(QtCore.Qt.UserRole, node)
            self.results[node] = r
            self.result_list.insertItem(0, item)
        self.result_list.setCurrentRow(0)
        self.status.setText(f"Classified {len(results)} asset(s).")

    def _show_result(self, item, _prev=None):
        if not item:
            return
        node = item.data(QtCore.Qt.UserRole)
        r = self.results.get(node)
        if not r:
            return
        for c, bar in self.bars.items():
            p = r["probabilities"][c]
            bar.setValue(int(p * 1000))
            bar.setFormat(f"{p:.1%}")
        if cmds.objExists(node):
            cmds.select(node, replace=True)

    def _command_port(self):
        if not cmds.commandPort(COMMAND_PORT, query=True):
            cmds.commandPort(name=COMMAND_PORT, sourceType="python")
        self.status.setText("Command port :7001 open - Streamlit can now talk to this Maya.")

    def _streamlit(self):
        if not (_PROCS.get("streamlit") and _PROCS["streamlit"].state() != QtCore.QProcess.NotRunning):
            p = QtCore.QProcess()
            p.setWorkingDirectory(str(PROJECT_ROOT))
            p.start(self.python.text(), ["-m", "streamlit", "run", "app.py", "--server.headless", "true",
                                         "--server.port", str(STREAMLIT_PORT)])
            _PROCS["streamlit"] = p
            QtCore.QTimer.singleShot(5000, lambda: webbrowser.open(f"http://localhost:{STREAMLIT_PORT}"))
        else:
            webbrowser.open(f"http://localhost:{STREAMLIT_PORT}")

    def _error(self, e):
        self.status.setText(f"<span style='color:#c33'>{e}</span>")
        cmds.warning(str(e))


def show():
    global _panel
    if cmds.workspaceControl(WORKSPACE, exists=True):
        cmds.deleteUI(WORKSPACE)
    _panel = FashionPanel()
    _panel.show(dockable=True, area="right", floating=False)
    return _panel
