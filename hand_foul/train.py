"""Training loop: `hand-foul train --data data/`.

Outputs
  <out>                       best weights (default weights/best.pt)
  runs/<run_name>/best.pt     copy of the best weights
  runs/<run_name>/split.json  exactly which files went to train/val/test
  runs/<run_name>/history.json per-epoch train/val loss & accuracy
  runs/<run_name>/metrics.json test accuracy, confusion matrix, per-class P/R
  runs/<run_name>/confusion_matrix.png
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    CLASSES,
    HandFoulDataset,
    discover_splits,
    eval_transforms,
    save_split,
    seed_everything,
    train_transforms,
)
from .model import DEFAULT_BACKBONE, DEFAULT_IMG_SIZE, build_model, pick_device, save_checkpoint


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def confusion_matrix(labels: list[int], preds: list[int], n: int = len(CLASSES)) -> list[list[int]]:
    cm = [[0] * n for _ in range(n)]
    for y, p in zip(labels, preds):
        cm[y][p] += 1
    return cm


def per_class_metrics(cm: list[list[int]]) -> dict[str, dict[str, float]]:
    out = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(cm))) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def format_confusion(cm: list[list[int]]) -> str:
    w = max(8, max(len(c) for c in CLASSES) + 2)
    head = " " * (w + 6) + "".join(f"pred {c:<{w}}" for c in CLASSES)
    rows = [head]
    for i, cls in enumerate(CLASSES):
        rows.append(f"true {cls:<{w}}" + "".join(f"{v:<{w + 5}d}" for v in cm[i]))
    return "\n".join(rows)


def plot_confusion(cm: list[list[int]], path: Path, title: str = "Confusion matrix (test)") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    vmax = max(max(r) for r in cm) or 1
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > vmax / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Loops
# ----------------------------------------------------------------------------
def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds += out.argmax(1).tolist()
            labels += y.tolist()
    n = max(len(labels), 1)
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / n
    return total_loss / n, acc, preds, labels


def train(
    data_dir: str | Path = "data",
    out: str | Path = "weights/best.pt",
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    backbone: str = DEFAULT_BACKBONE,
    img_size: int = DEFAULT_IMG_SIZE,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    pretrained: bool = True,
    device: str | None = None,
    num_workers: int | None = None,
    cache: bool = True,
    decode_workers: int = 1,
    log=print,
) -> dict:
    """Train a classifier and evaluate it on the test split. Returns metrics."""
    t0 = time.time()
    seed_everything(seed)
    dev = pick_device(device)
    if num_workers is None:
        num_workers = 0 if sys.platform.startswith("win") else 4

    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- data ----------------------------------------------------------------
    splits, info = discover_splits(data_dir, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    save_split(splits, info, run_dir / "split.json")
    cache_size = img_size if cache else None
    cache_dir = Path(data_dir) / ".cache" if cache else None
    ds_train = HandFoulDataset(splits["train"], train_transforms(img_size), cache_size, cache_dir)
    ds_val = HandFoulDataset(splits["val"], eval_transforms(img_size), cache_size, cache_dir)
    ds_test = HandFoulDataset(splits["test"], eval_transforms(img_size), cache_size, cache_dir)
    pin = dev.type == "cuda"
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    counts = ds_train.class_counts()
    log(f"device: {dev} | backbone: {backbone} | img_size: {img_size} | layout: {info['layout']}")
    log(f"train {len(ds_train)} {dict(zip(CLASSES, counts))} | val {len(ds_val)} | test {len(ds_test)}")
    if cache:
        n_all = len(ds_train) + len(ds_val) + len(ds_test)
        log(f"decoding {n_all} photos once (24 MP HEIC ~1 s each; shrunk copies cached in {cache_dir}) ...")
        tc = time.time()
        for ds in (ds_train, ds_val, ds_test):
            ds.preload(workers=decode_workers, log=log)
        log(f"  done in {time.time() - tc:.0f}s")

    # -- model ---------------------------------------------------------------
    model = build_model(backbone, pretrained=pretrained).to(dev)
    total = sum(counts)
    class_w = torch.tensor([total / (len(CLASSES) * max(c, 1)) for c in counts], dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # -- train ---------------------------------------------------------------
    best = {"epoch": 0, "val_acc": -1.0, "val_loss": float("inf")}
    best_path = run_dir / "best.pt"
    history = []
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        te = time.time()
        if epoch == 1:
            log("training ...")
        tr_loss, tr_acc, _, _ = _run_epoch(model, dl_train, criterion, dev, optimizer)
        va_loss, va_acc, _, _ = _run_epoch(model, dl_val, criterion, dev)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

        improved = (va_acc > best["val_acc"]) or (va_acc == best["val_acc"] and va_loss < best["val_loss"])
        if improved:
            best = {"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss}
            save_checkpoint(best_path, model, backbone, img_size, extra={"epoch": epoch, "val_acc": va_acc, "run": run_name})
            bad_epochs = 0
        else:
            bad_epochs += 1
        log(f"epoch {epoch:>3}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - te:.1f}s{'  *best' if improved else ''}")
        if patience and bad_epochs >= patience:
            log(f"early stop: no val improvement for {patience} epochs")
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # -- test ----------------------------------------------------------------
    state = torch.load(best_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state)
    te_loss, te_acc, preds, labels = _run_epoch(model, dl_test, criterion, dev)
    cm = confusion_matrix(labels, preds)
    metrics = {
        "run": run_name,
        "backbone": backbone,
        "img_size": img_size,
        "best_epoch": best["epoch"],
        "val_acc": best["val_acc"],
        "test_acc": te_acc,
        "test_loss": te_loss,
        "confusion_matrix": cm,
        "classes": list(CLASSES),
        "per_class": per_class_metrics(cm),
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "seconds": round(time.time() - t0, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion(cm, run_dir / "confusion_matrix.png")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_path, out)
    metrics["weights"] = str(out)
    metrics["run_dir"] = str(run_dir)

    log("")
    log(f"best epoch {best['epoch']} (val acc {best['val_acc']:.3f})")
    log(f"TEST accuracy: {te_acc:.3f}  ({len(ds_test)} images)")
    log(format_confusion(cm))
    for cls, m in metrics["per_class"].items():
        log(f"  {cls:<7} precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}  (n={m['support']})")
    log(f"weights -> {out}")
    log(f"run dir -> {run_dir}  (confusion_matrix.png, metrics.json, split.json)")
    return metrics
