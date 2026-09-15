"""Inference API: `Classifier` plus annotated-image output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import IMG_EXTS, eval_transforms, list_images, load_image
from .draw import annotate_prediction, fit_within, pil_to_bgr
from .model import load_checkpoint, pick_device

ImageLike = "str | Path | Image.Image | np.ndarray"


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):  # assume BGR (OpenCV) if 3 channels
        arr = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
        return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")
    return load_image(image)


def _resolve_weights(weights: str | Path) -> Path:
    """Accept a local path or an http(s) URL (downloaded once to ~/.cache/hand_foul)."""
    w = str(weights)
    if w.startswith(("http://", "https://")):
        cache = Path.home() / ".cache" / "hand_foul"
        cache.mkdir(parents=True, exist_ok=True)
        dst = cache / (w.rsplit("/", 1)[-1] or "best.pt")
        if not dst.exists():
            print(f"downloading weights -> {dst}")
            torch.hub.download_url_to_file(w, str(dst), progress=True)
        return dst
    return Path(w)


class Classifier:
    """normal / foul classifier.

    >>> clf = Classifier.from_pretrained("weights/best.pt")
    >>> clf.predict("photo.jpg")
    {'label': 'foul', 'confidence': 0.93, 'probs': {'normal': 0.07, 'foul': 0.93}}
    """

    def __init__(self, model: torch.nn.Module, meta: dict, device: torch.device):
        self.model = model.eval()
        self.meta = meta
        self.device = device
        self.classes: list[str] = list(meta["classes"])
        self.img_size: int = int(meta["img_size"])
        self.transform = eval_transforms(self.img_size)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_pretrained(cls, weights: str | Path = "weights/best.pt", device: str | None = None) -> "Classifier":
        dev = pick_device(device)
        model, meta = load_checkpoint(_resolve_weights(weights), dev)
        return cls(model, meta, dev)

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def predict_probs(self, images: list) -> np.ndarray:
        """Softmax probabilities, shape (N, num_classes)."""
        if not images:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        batch = torch.stack([self.transform(_to_pil(im)) for im in images]).to(self.device)
        return torch.softmax(self.model(batch), dim=1).cpu().numpy()

    def _result(self, probs: np.ndarray) -> dict:
        idx = int(probs.argmax())
        conf = float(probs[idx])
        return {
            "label": self.classes[idx],
            "confidence": round(conf, 4),
            "probs": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
        }

    def predict(self, image) -> dict:
        """Classify one image (path, PIL image or OpenCV BGR array)."""
        return self._result(self.predict_probs([image])[0])

    def predict_many(self, images: list, batch_size: int = 16) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(images), batch_size):
            for p in self.predict_probs(images[i : i + batch_size]):
                out.append(self._result(p))
        return out

    def predict_folder(self, folder: str | Path, batch_size: int = 16) -> list[tuple[Path, dict]]:
        paths = list_images(folder)
        return list(zip(paths, self.predict_many(paths, batch_size=batch_size)))

    # -- visualisation -------------------------------------------------------
    def annotate(self, image, result: dict | None = None) -> Image.Image:
        """Return the image with a coloured frame + 'LABEL 93%' banner."""
        pil = _to_pil(image)
        result = result or self.predict(pil)
        return annotate_prediction(pil, result["label"], result["confidence"])

    def show(self, image, save: str | Path | None = None, window: bool | None = None, result: dict | None = None) -> Path | None:
        """Annotate an image; save it to `save` and/or open a window.

        window=None -> open a window only when nothing is being saved.
        Returns the saved path (or None).
        """
        annotated = self.annotate(image, result)
        saved = None
        if save is not None:
            saved = Path(save)
            saved.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(saved, quality=92)
        if window is None:
            window = save is None
        if window:
            show_window(annotated, title=f"hand-foul: {getattr(image, 'name', image) if isinstance(image, (str, Path)) else 'image'}")
        return saved


def show_window(img: Image.Image, title: str = "hand-foul", max_w: int = 1280, max_h: int = 800) -> None:
    """Blocking OpenCV window; press any key to close."""
    import cv2

    cv2.imshow(title, pil_to_bgr(fit_within(img, max_w, max_h)))
    cv2.waitKey(0)
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass


def predict_paths(
    clf: Classifier,
    inputs: list[str | Path],
    out_dir: str | Path | None = "predictions",
    window: bool = False,
    log=print,
) -> list[dict]:
    """Shared by the CLI: predict files/folders, save annotated copies, print a summary."""
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths += list_images(p)
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
        else:
            log(f"skipping {p} (not an image or folder)")
    if not paths:
        raise FileNotFoundError("no images found")

    out = Path(out_dir) if out_dir else None
    rows = []
    results = clf.predict_many(paths)
    for p, r in zip(paths, results):
        save = out / f"{p.stem}_pred.jpg" if out else None
        clf.show(p, save=save, window=window, result=r)
        log(f"{r['label']:<7} {r['confidence'] * 100:5.1f}%  {p}")
        rows.append({"path": str(p), **r, "annotated": str(save) if save else None})

    counts = {c: sum(r["label"] == c for r in results) for c in clf.classes}
    log(f"\n{len(paths)} images: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if out:
        (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"annotated images + results.json -> {out}/")
    return rows
