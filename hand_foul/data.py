"""Image loading, transforms and dataset discovery / splitting."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES: tuple[str, ...] = ("normal", "foul")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

try:  # iPhone photos: HEIC needs pillow-heif to be registered with Pillow
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = (114, 114, 114)


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------
def load_image(path: str | Path) -> Image.Image:
    """Open an image, apply EXIF orientation (phone photos!) and return RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def shrink_to(img: Image.Image, long_side: int) -> Image.Image:
    """Downscale (never upscale) so the longer side == long_side, keeping aspect."""
    w, h = img.size
    if max(w, h) <= long_side:
        return img
    s = long_side / max(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
class Letterbox:
    """Resize so the longer side == size, keep aspect ratio, pad to a square.

    Nothing is cropped, so the mouse pad border never leaves the frame.
    """

    def __init__(self, size: int, fill=PAD_FILL):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"Letterbox(size={self.size})"


class RandomRot90:
    """Lossless 0/90/180/270 degree rotation (top-down photos have no 'up')."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k, expand=True) if k else img

    def __repr__(self):
        return "RandomRot90()"


def train_transforms(img_size: int) -> T.Compose:
    # Deliberately *no* RandomResizedCrop / translate: cropping the pad edge
    # would change what the label means. Scale only shrinks (adds margin) and
    # rotation is small so at most the extreme corners of the photo are lost.
    return T.Compose(
        [
            Letterbox(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRot90(),
            T.RandomAffine(degrees=8, scale=(0.75, 1.0), fill=PAD_FILL),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transforms(img_size: int) -> T.Compose:
    return T.Compose(
        [
            Letterbox(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    path: Path
    label: int

    @property
    def class_name(self) -> str:
        return CLASSES[self.label]


class HandFoulDataset(Dataset):
    """Dataset over labelled samples.

    Phone photos are huge (24 MP HEIC takes ~1 s to decode), so `preload()`
    decodes every image once, shrinks it so its longer side == `cache_size`
    (the letterbox size used by the transforms, so nothing is lost) and keeps
    the small copy in memory. After that each epoch is just augmentation.
    """

    def __init__(
        self,
        samples: list[Sample],
        transform,
        cache_size: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.samples = list(samples)
        self.transform = transform
        self.cache_size = cache_size
        self.cache_dir = Path(cache_dir) / str(cache_size) if (cache_dir and cache_size) else None
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _disk_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        src = self.samples[idx].path
        key = hashlib.md5(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{src.stem}_{key}.jpg"

    def _load_small(self, idx: int) -> tuple[Image.Image, bool]:
        """Return (shrunk image, came_from_disk_cache)."""
        dp = self._disk_path(idx)
        src = self.samples[idx].path
        if dp is not None and dp.is_file() and dp.stat().st_mtime >= src.stat().st_mtime:
            with Image.open(dp) as im:
                return im.convert("RGB"), True
        img = load_image(src)
        if self.cache_size:
            img = shrink_to(img, self.cache_size)
        if dp is not None:
            dp.parent.mkdir(parents=True, exist_ok=True)
            img.save(dp, quality=95)
        return img, False

    def preload(self, workers: int = 1, log=None) -> None:
        """Decode + shrink all images once and keep them in memory.

        With `cache_dir` the shrunk copies are also written to disk, so the
        next training run starts in seconds instead of minutes. HEIC decoding
        is already multi-threaded inside libheif, so workers=1 is usually best.
        """
        if not self.cache_size:
            return
        from concurrent.futures import ThreadPoolExecutor

        todo = [i for i in range(len(self.samples)) if i not in self._cache]
        if not todo:
            return
        done = hits = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, (img, hit) in zip(todo, ex.map(self._load_small, todo)):
                self._cache[i] = img
                done += 1
                hits += hit
                if log and (done % 20 == 0 or done == len(todo)):
                    log(f"  {done}/{len(todo)}  (from disk cache: {hits})")

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._cache.get(idx)
        if img is None:
            img, _ = self._load_small(idx)
            if self.cache_size:
                self._cache[idx] = img
        return self.transform(img), s.label

    def class_counts(self) -> list[int]:
        counts = [0] * len(CLASSES)
        for s in self.samples:
            counts[s.label] += 1
        return counts


# ----------------------------------------------------------------------------
# Discovery & split
# ----------------------------------------------------------------------------
def _samples_in(class_root: Path) -> list[Sample]:
    out: list[Sample] = []
    for cls in CLASSES:
        for p in list_images(class_root / cls):
            out.append(Sample(p, CLASS_TO_IDX[cls]))
    return out


def discover_splits(
    data_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[Sample]], dict]:
    """Return ({"train": [...], "val": [...], "test": [...]}, info).

    Two layouts are supported:

    A) pre-split (used as-is):
        data/train/{normal,foul}/  data/val/{normal,foul}/  data/test/{normal,foul}/
    B) flat (random stratified split with a fixed seed):
        data/{normal,foul}/
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    if (data_dir / "train").is_dir():
        splits = {
            "train": _samples_in(data_dir / "train"),
            "val": _samples_in(data_dir / "val"),
            "test": _samples_in(data_dir / "test"),
        }
        if not splits["val"]:
            splits["val"] = splits["test"]
        info = {"layout": "pre-split", "data_dir": str(data_dir)}
    else:
        rng = random.Random(seed)
        splits = {"train": [], "val": [], "test": []}
        for cls in CLASSES:
            items = [Sample(p, CLASS_TO_IDX[cls]) for p in list_images(data_dir / cls)]
            rng.shuffle(items)
            n = len(items)
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            if n >= 3:  # make sure every split gets at least one of each class
                n_test = max(1, n_test)
                n_val = max(1, n_val)
            splits["test"] += items[:n_test]
            splits["val"] += items[n_test : n_test + n_val]
            splits["train"] += items[n_test + n_val :]
        info = {
            "layout": "flat-random",
            "data_dir": str(data_dir),
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }

    for name in ("train", "val", "test"):
        if not splits[name]:
            raise RuntimeError(
                f"split '{name}' is empty. Put labelled images in "
                f"{data_dir}/normal and {data_dir}/foul (at least ~3 per class), "
                f"or use the pre-split layout {data_dir}/train|val|test/<class>/."
            )
    return splits, info


def save_split(splits: dict[str, list[Sample]], info: dict, path: str | Path) -> None:
    payload = {
        "info": info,
        "classes": list(CLASSES),
        **{k: [{"path": str(s.path), "label": s.class_name} for s in v] for k, v in splits.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
