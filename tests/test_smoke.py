"""Smoke test: dummy images -> train -> predict -> annotated image, no real photos needed."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from hand_foul import CLASSES, Classifier
from hand_foul.cli import app
from hand_foul.label import LabelSession
from hand_foul.train import train

SKIN = (222, 184, 150)


def make_dummy_image(path: Path, label: str, size: int = 96, rng: random.Random = random.Random(0)) -> None:
    """Grey 'blanket', black pad with a red border, skin-coloured blob inside (normal) or outside (foul)."""
    img = Image.new("RGB", (size, size), (120, 120, 130))
    d = ImageDraw.Draw(img)
    m = size // 6
    d.rectangle([m, m, size - m, size - m], fill=(10, 10, 10), outline=(220, 30, 30), width=3)
    r = size // 10
    if label == "normal":
        cx = rng.randint(m + r + 4, size - m - r - 4)
        cy = rng.randint(m + r + 4, size - m - r - 4)
    else:  # touching / outside the pad
        cx = rng.choice([rng.randint(0, m), rng.randint(size - m, size - 1)])
        cy = rng.randint(0, size - 1)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SKIN)
    img.save(path, quality=90)


def make_dummy_dataset(root: Path, n_per_class: int = 12) -> None:
    rng = random.Random(1)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            make_dummy_image(root / cls / f"{cls}_{i:02d}.jpg", cls, rng=rng)


@pytest.fixture(scope="module")
def dummy_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    make_dummy_dataset(root)
    return root


def test_train_predict_show_end_to_end(dummy_data: Path, tmp_path: Path):
    weights = tmp_path / "weights" / "best.pt"
    metrics = train(
        data_dir=dummy_data,
        out=weights,
        runs_dir=tmp_path / "runs",
        run_name="smoke",
        backbone="resnet18",
        img_size=64,
        epochs=2,
        batch_size=8,
        pretrained=False,  # no network needed
        device="cpu",
        num_workers=0,
        log=lambda *_: None,
    )

    # training artefacts
    assert weights.is_file()
    run_dir = tmp_path / "runs" / "smoke"
    for name in ("best.pt", "split.json", "history.json", "metrics.json", "confusion_matrix.png"):
        assert (run_dir / name).is_file(), name
    assert 0.0 <= metrics["test_acc"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and all(len(r) == 2 for r in cm)
    assert sum(map(sum, cm)) == metrics["n_test"] > 0

    # python API
    clf = Classifier.from_pretrained(weights, device="cpu")
    sample = next((dummy_data / "foul").glob("*.jpg"))
    res = clf.predict(sample)
    assert res["label"] in CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert set(res["probs"]) == set(CLASSES)
    assert abs(sum(res["probs"].values()) - 1.0) < 1e-3

    many = clf.predict_folder(dummy_data / "normal")
    assert len(many) == 12 and all(r["label"] in CLASSES for _, r in many)

    # visual output
    out_img = tmp_path / "viz" / "pred.jpg"
    saved = clf.show(sample, save=out_img, window=False)
    assert saved == out_img and out_img.is_file()
    with Image.open(out_img) as im:
        assert im.size == Image.open(sample).size

    # evaluation on a labelled folder
    from hand_foul.evaluate import evaluate

    ev = evaluate(dummy_data, weights=weights, out_dir=tmp_path / "eval", device="cpu", log=lambda *_: None)
    assert ev["n"] == 24 and 0.0 <= ev["accuracy"] <= 1.0
    assert sum(map(sum, ev["confusion_matrix"])) == 24
    assert (tmp_path / "eval" / "confusion_matrix.png").is_file()

    # CLI
    runner = CliRunner()
    r = runner.invoke(app, ["predict", str(dummy_data / "foul"), "-w", str(weights), "-o", str(tmp_path / "preds"), "--no-show", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "preds" / "results.json").is_file()
    assert len(list((tmp_path / "preds").glob("*_pred.jpg"))) == 12


def test_label_session_move_and_undo(tmp_path: Path):
    data = tmp_path / "data"
    inbox = data / "inbox"
    inbox.mkdir(parents=True)
    for i in range(3):
        make_dummy_image(inbox / f"p{i}.jpg", "normal")
    (data / "foul" / "p1.jpg").parent.mkdir(parents=True)
    make_dummy_image(data / "foul" / "p1.jpg", "foul")  # name collision target

    s = LabelSession(data)
    assert s.remaining == 3 and s.current.name == "p0.jpg"

    dest = s.assign("normal")
    assert dest == data / "normal" / "p0.jpg" and dest.is_file()
    assert s.remaining == 2 and s.current.name == "p1.jpg"

    dest = s.assign("foul")  # collides with existing foul/p1.jpg -> suffixed
    assert dest.name == "p1_1.jpg" and dest.is_file()
    assert s.counts() == {"normal": 1, "foul": 2, "skip": 0}

    back = s.undo()
    assert back == inbox / "p1.jpg" and back.is_file() and not dest.exists()
    assert s.current == back and s.remaining == 2

    s.assign("skip")
    assert (data / "skip" / "p1.jpg").is_file()

    # resuming picks up what is left in the inbox
    s2 = LabelSession(data)
    assert [p.name for p in s2.queue] == ["p2.jpg"]
    assert s2.undo() is None


def test_live_overlay_draws_red_border_on_foul():
    import numpy as np
    from hand_foul.live import draw_overlay

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    warn = draw_overlay(frame, foul_prob=0.9, warning=True, fps=12.0, tick=0.3)
    ok = draw_overlay(frame, foul_prob=0.1, warning=False, fps=12.0, tick=0.3)
    assert warn.shape == ok.shape == frame.shape
    b, g, r = warn[120, 2]  # left border pixel, BGR
    assert r > 150 and g < 100  # red border when warning
    b, g, r = ok[120, 2]
    assert g > 120 and r < 120  # green border when normal
