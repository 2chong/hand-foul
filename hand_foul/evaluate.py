"""Evaluate trained weights on a labelled folder: `hand-foul eval tests_set/`.

Expected layout (same as the training data):
    tests_set/normal/*.jpg
    tests_set/foul/*.jpg

Prints accuracy, confusion matrix and per-class precision / recall, lists every
misclassified photo, and writes annotated copies of the wrong ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import CLASSES, list_images
from .predict import Classifier
from .train import confusion_matrix, format_confusion, per_class_metrics, plot_confusion


def evaluate(
    data_dir: str | Path,
    weights: str | Path = "weights/best.pt",
    out_dir: str | Path | None = "eval",
    device: str | None = None,
    save_all: bool = False,
    log=print,
) -> dict:
    data_dir = Path(data_dir)
    paths: list[Path] = []
    labels: list[int] = []
    for i, cls in enumerate(CLASSES):
        found = list_images(data_dir / cls)
        paths += found
        labels += [i] * len(found)
    if not paths:
        raise FileNotFoundError(
            f"no labelled images under {data_dir}. Expected {data_dir}/normal/ and {data_dir}/foul/ "
            f"(sort them by hand, or run `hand-foul label --data {data_dir}` with photos in {data_dir}/inbox/)."
        )
    for cls in CLASSES:
        if not list_images(data_dir / cls):
            log(f"warning: {data_dir / cls} is empty")

    clf = Classifier.from_pretrained(weights, device=device)
    results = clf.predict_many(paths)
    preds = [clf.classes.index(r["label"]) for r in results]
    cm = confusion_matrix(labels, preds)
    n = len(paths)
    acc = sum(int(p == y) for p, y in zip(preds, labels)) / n

    out = Path(out_dir) if out_dir else None
    rows = []
    wrong = []
    for p, y, r in zip(paths, labels, results):
        ok = r["label"] == CLASSES[y]
        row = {"path": str(p), "true": CLASSES[y], "pred": r["label"], "confidence": r["confidence"], "correct": ok}
        rows.append(row)
        if not ok:
            wrong.append(row)
        if out and (save_all or not ok):
            sub = out / ("correct" if ok else "wrong")
            clf.show(p, save=sub / f"{p.stem}_true-{CLASSES[y]}_pred-{r['label']}.jpg", window=False, result=r)

    metrics = {
        "data_dir": str(data_dir),
        "weights": str(weights),
        "n": n,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": list(CLASSES),
        "per_class": per_class_metrics(cm),
        "wrong": wrong,
    }

    log(f"weights: {weights}  |  {n} images from {data_dir}")
    log(f"ACCURACY: {acc:.3f}  ({n - len(wrong)}/{n} correct)")
    log(format_confusion(cm))
    for cls, m in metrics["per_class"].items():
        log(f"  {cls:<7} precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}  (n={m['support']})")
    if wrong:
        log(f"\n{len(wrong)} misclassified:")
        for w in sorted(wrong, key=lambda x: -x["confidence"]):
            log(f"  true {w['true']:<7} pred {w['pred']:<7} {w['confidence'] * 100:5.1f}%  {w['path']}")
    else:
        log("\nno mistakes")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        plot_confusion(cm, out / "confusion_matrix.png", title=f"Confusion matrix ({data_dir.name})")
        log(f"\nresults.json, metrics.json, confusion_matrix.png{', wrong/*.jpg' if wrong else ''} -> {out}/")
    return metrics
