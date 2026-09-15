"""Backbone construction and checkpoint (de)serialisation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .data import CLASSES

DEFAULT_BACKBONE = "resnet18"
DEFAULT_IMG_SIZE = 512
BACKBONES = ("resnet18", "resnet34", "resnet50", "mobilenet_v3_large", "efficientnet_b0")
CHECKPOINT_FORMAT = 1


def build_model(backbone: str = DEFAULT_BACKBONE, num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {BACKBONES}")
    weights = "DEFAULT" if pretrained else None
    model = models.get_model(backbone, weights=weights)

    if backbone.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone.startswith("mobilenet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def save_checkpoint(path: str | Path, model: nn.Module, backbone: str, img_size: int, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "backbone": backbone,
        "img_size": int(img_size),
        "classes": list(CLASSES),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"weights not found: {path}\n"
            "  - download best.pt from the GitHub Release and put it at weights/best.pt, or\n"
            "  - train your own with `hand-foul train --data data/`"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    model = build_model(meta["backbone"], num_classes=len(meta["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, meta


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
