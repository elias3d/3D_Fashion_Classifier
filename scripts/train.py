"""Train the VoxelCNN on the Maya-generated dataset.

    python scripts/train.py                      # GPU if available, else CPU
    python scripts/train.py --device cpu --epochs 15

Writes:
    models/voxel_cnn.pt     trained weights + config (loaded by the app, so it never retrains)
    models/metrics.json     training curves, test accuracy, confusion matrix, style-test results

The Maya panel's "Train" tab runs this same script and reads the EPOCH lines for its progress bar.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fashion3d.classes import CLASS_NAMES  # noqa: E402
from fashion3d.dataset import VoxelDataset, read_manifest  # noqa: E402
from fashion3d.model import VoxelCNN, count_parameters, pick_device  # noqa: E402
from fashion3d.preprocess import GRID, N_POINTS, SCALE_RANGE  # noqa: E402


def split_train_val(rows, val_frac, seed):
    rng = random.Random(seed)
    train, val = [], []
    for c in CLASS_NAMES:
        items = [r for r in rows if r["label"] == c]
        rng.shuffle(items)
        k = max(1, int(len(items) * val_frac))
        val += items[:k]
        train += items[k:]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy() if probs else np.zeros((0, len(CLASS_NAMES)))
    labels = torch.cat(labels).numpy() if labels else np.zeros(0, int)
    return probs, labels


def confusion(labels, preds, n):
    cm = np.zeros((n, n), int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "models" / "voxel_cnn.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({gpu})", flush=True)

    all_train = read_manifest("train")
    if not all_train:
        sys.exit("No training data. Generate it in Maya first (panel > Dataset, or scripts/maya_generate_dataset.py).")
    train_rows, val_rows = split_train_val(all_train, args.val_frac, args.seed)
    test_rows = read_manifest("test")
    style_rows = read_manifest("styles")
    print(f"Assets: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} styles={len(style_rows)}",
          flush=True)

    train_ds = VoxelDataset(train_rows, augment=True, seed=args.seed)
    val_dl = DataLoader(VoxelDataset(val_rows), batch_size=64)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = VoxelCNN(len(CLASS_NAMES), GRID).to(device)
    print(f"Model: VoxelCNN, {count_parameters(model):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best_acc, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, correct, seen = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        vp, vl = evaluate(model, val_dl, device)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(vp) + 1e-9), torch.from_numpy(vl)))
        val_acc = float((vp.argmax(1) == vl).mean())
        row = {"epoch": epoch, "train_loss": tot_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": val_acc, "seconds": time.time() - t0}
        history.append(row)
        if val_acc >= best_acc:
            best_acc, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"EPOCH {epoch}/{args.epochs} loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} time={row['seconds']:.0f}s", flush=True)

    model.load_state_dict(best_state)
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device), "device_name": gpu, "epochs": args.epochs,
        "train_seconds": time.time() - t0, "parameters": count_parameters(model),
        "class_names": CLASS_NAMES, "history": history, "best_val_acc": best_acc,
        "n_train": len(train_rows), "n_val": len(val_rows), "augment_scale_range": list(SCALE_RANGE),
    }

    if test_rows:
        tp, tl = evaluate(model, DataLoader(VoxelDataset(test_rows), batch_size=64), device)
        pred = tp.argmax(1)
        cm = confusion(tl, pred, len(CLASS_NAMES))
        metrics.update(n_test=len(test_rows), test_acc=float((pred == tl).mean()), confusion_matrix=cm.tolist(),
                       per_class_acc={c: float(cm[i, i] / max(1, cm[i].sum())) for i, c in enumerate(CLASS_NAMES)})
        print(f"TEST accuracy: {metrics['test_acc']:.3f} on {len(test_rows)} unseen assets", flush=True)

    if style_rows:
        sp, sl = evaluate(model, DataLoader(VoxelDataset(style_rows), batch_size=64), device)
        per_asset = []
        for r, p, t in zip(style_rows, sp, sl):
            per_asset.append({"asset_id": r["asset_id"], "style": r["style"], "label": r["label"],
                              "pred": CLASS_NAMES[int(p.argmax())], "confidence": float(p.max()),
                              "true_prob": float(p[t])})
        styles = {}
        for s in dict.fromkeys(r["style"] for r in style_rows):
            items = [a for a in per_asset if a["style"] == s]
            wrong = [a for a in items if a["pred"] != a["label"]]
            styles[s] = {"n": len(items), "accuracy": 1 - len(wrong) / len(items),
                         "mean_confidence": float(np.mean([a["confidence"] for a in items])),
                         "mistakes": [f"{a['label']}->{a['pred']} ({a['confidence']:.0%})" for a in wrong]}
            print(f"STYLE {s:9s} acc={styles[s]['accuracy']:.3f} conf={styles[s]['mean_confidence']:.3f} "
                  f"mistakes={styles[s]['mistakes']}", flush=True)
        metrics["styles"] = styles
        metrics["style_predictions"] = per_asset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"class_names": CLASS_NAMES, "grid": GRID, "n_points": N_POINTS, "arch": "VoxelCNN"},
                "best_val_acc": best_acc, "trained_at": metrics["trained_at"]}, out)
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
