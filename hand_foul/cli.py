"""`hand-foul` command line interface (typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .model import BACKBONES, DEFAULT_BACKBONE, DEFAULT_IMG_SIZE

app = typer.Typer(
    help="normal / foul hand-position classifier.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool):
    if value:
        typer.echo(f"hand-foul {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Optional[bool] = typer.Option(None, "--version", callback=_version, is_eager=True, help="Print version and exit."),
):
    pass


@app.command()
def label(
    data: Path = typer.Option(Path("data"), "--data", "-d", help="Data root containing inbox/ (normal/, foul/, skip/ are created)."),
):
    """Open the keyboard labelling window (1 normal, 2 foul, s skip, z undo, q quit)."""
    from .label import run_label_tool

    run_label_tool(data)


@app.command()
def train(
    data: Path = typer.Option(Path("data"), "--data", "-d", help="data/{normal,foul}/ or data/{train,val,test}/<class>/"),
    out: Path = typer.Option(Path("weights/best.pt"), "--out", "-o", help="Where to write the best weights."),
    runs_dir: Path = typer.Option(Path("runs"), help="Per-run artefacts (metrics, confusion matrix, split)."),
    run_name: Optional[str] = typer.Option(None, help="Run folder name (default: timestamp)."),
    backbone: str = typer.Option(DEFAULT_BACKBONE, help=f"One of {', '.join(BACKBONES)}."),
    img_size: int = typer.Option(DEFAULT_IMG_SIZE, help="Square input size after letterboxing."),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(16),
    lr: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(1e-4),
    patience: int = typer.Option(6, help="Early-stop after N epochs without val improvement (0 = off)."),
    seed: int = typer.Option(42),
    val_ratio: float = typer.Option(0.15, help="Only for the flat layout."),
    test_ratio: float = typer.Option(0.15, help="Only for the flat layout."),
    pretrained: bool = typer.Option(True, help="Start from ImageNet weights (downloaded once by torchvision)."),
    device: Optional[str] = typer.Option(None, help="cuda / cpu (default: auto)."),
    num_workers: Optional[int] = typer.Option(None, help="DataLoader workers (default: 0 on Windows, 4 elsewhere)."),
    cache: bool = typer.Option(True, help="Decode every photo once; keep shrunk copies in memory and in data/.cache/ (fast re-runs)."),
    decode_workers: int = typer.Option(1, help="Threads for the one-off decoding (HEIC decoding is already multi-threaded)."),
):
    """Train on labelled photos, then report test accuracy + confusion matrix."""
    from .train import train as _train

    _train(
        data_dir=data, out=out, runs_dir=runs_dir, run_name=run_name, backbone=backbone,
        img_size=img_size, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        patience=patience, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio,
        pretrained=pretrained, device=device, num_workers=num_workers, cache=cache,
        decode_workers=decode_workers, log=typer.echo,
    )


@app.command()
def predict(
    inputs: list[Path] = typer.Argument(..., help="Image file(s) and/or folder(s)."),
    weights: str = typer.Option("weights/best.pt", "--weights", "-w", help="Checkpoint path or http(s) URL."),
    out: Optional[Path] = typer.Option(Path("predictions"), "--out", "-o", help="Folder for annotated images (use --no-save to skip)."),
    save: bool = typer.Option(True, "--save/--no-save", help="Save annotated images to --out."),
    show: Optional[bool] = typer.Option(None, "--show/--no-show", help="Open a window per image (default: on for a single image)."),
    device: Optional[str] = typer.Option(None, help="cuda / cpu (default: auto)."),
):
    """Classify photo(s) and write/show images with the label + confidence drawn on top."""
    from .predict import Classifier, predict_paths

    clf = Classifier.from_pretrained(weights, device=device)
    if show is None:
        show = len(inputs) == 1 and inputs[0].is_file()
    try:
        predict_paths(clf, inputs, out_dir=out if save else None, window=show, log=typer.echo)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd(
    data: Path = typer.Argument(Path("tests_set"), help="Folder with <data>/normal/ and <data>/foul/ (the answers)."),
    weights: str = typer.Option("weights/best.pt", "--weights", "-w", help="Checkpoint path or http(s) URL."),
    out: Optional[Path] = typer.Option(Path("eval"), "--out", "-o", help="Where to write metrics, confusion matrix and wrong/*.jpg."),
    save_all: bool = typer.Option(False, help="Also save annotated copies of the correct ones (eval/correct/)."),
    device: Optional[str] = typer.Option(None, help="cuda / cpu (default: auto)."),
):
    """Score the model on a hand-labelled folder: accuracy, confusion matrix, list of mistakes."""
    from .evaluate import evaluate

    try:
        evaluate(data, weights=weights, out_dir=out, device=device, save_all=save_all, log=typer.echo)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def live(
    source: str = typer.Option("0", "--source", "-s", help="Webcam index (0, 1, ...) or stream URL (http://<phone-ip>:8080/video)."),
    weights: str = typer.Option("weights/best.pt", "--weights", "-w", help="Checkpoint path or http(s) URL."),
    threshold: float = typer.Option(0.5, help="Smoothed foul probability above which the WARNING is shown."),
    smoothing: float = typer.Option(0.6, help="0 = react instantly, 0.9 = very smooth (less flicker, more lag)."),
    every: int = typer.Option(1, help="Run the model every N frames (raise on a slow CPU)."),
    width: Optional[int] = typer.Option(None, help="Request this capture width from the camera."),
    mirror: bool = typer.Option(False, help="Flip the image horizontally."),
    save_dir: Path = typer.Option(Path("live_captures"), help="Where the s key saves frames."),
    device: Optional[str] = typer.Option(None, help="cuda / cpu (default: auto)."),
):
    """Real-time foul detection from a webcam or phone camera (red border + WARNING on foul)."""
    from .live import run_live

    try:
        run_live(source=source, weights=weights, threshold=threshold, smoothing=smoothing, every=every,
                 width=width, mirror=mirror, save_dir=save_dir, device=device, log=typer.echo)
    except (RuntimeError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":  # python -m hand_foul.cli
    app()
