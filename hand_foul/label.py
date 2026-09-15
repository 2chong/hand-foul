"""Keyboard labelling tool: `hand-foul label`.

Put every photo into data/inbox/, run the tool, press one key per photo.
Files are *moved* to data/normal, data/foul or data/skip, so the tool can be
closed at any time and resumed later from whatever is left in the inbox.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import CLASSES, list_images, load_image
from .draw import (
    AMBER,
    DARK,
    GREEN,
    RED,
    WHITE,
    fit_within,
    get_font,
    korean_font_available,
    pil_to_bgr,
)

TARGETS = (*CLASSES, "skip")
KEYMAP = {"1": "normal", "2": "foul", "s": "skip"}


# ----------------------------------------------------------------------------
# Headless session logic (unit-testable, no window)
# ----------------------------------------------------------------------------
@dataclass
class LabelSession:
    data_dir: Path
    queue: list[Path] = field(default_factory=list)
    history: list[tuple[Path, Path]] = field(default_factory=list)  # (moved_to, original_inbox_path)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.inbox = self.data_dir / "inbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        for t in TARGETS:
            (self.data_dir / t).mkdir(parents=True, exist_ok=True)
        self.queue = list_images(self.inbox)

    # -- state ---------------------------------------------------------------
    @property
    def current(self) -> Path | None:
        return self.queue[0] if self.queue else None

    @property
    def remaining(self) -> int:
        return len(self.queue)

    def counts(self) -> dict[str, int]:
        return {t: len(list_images(self.data_dir / t)) for t in TARGETS}

    # -- actions -------------------------------------------------------------
    @staticmethod
    def _unique(dest: Path) -> Path:
        if not dest.exists():
            return dest
        stem, suf = dest.stem, dest.suffix
        i = 1
        while (cand := dest.with_name(f"{stem}_{i}{suf}")).exists():
            i += 1
        return cand

    def assign(self, target: str) -> Path:
        """Move the current photo to data/<target>/ and advance."""
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
        src = self.current
        if src is None:
            raise RuntimeError("inbox is empty")
        dest = self._unique(self.data_dir / target / src.name)
        shutil.move(str(src), str(dest))
        self.queue.pop(0)
        self.history.append((dest, src))
        return dest

    def undo(self) -> Path | None:
        """Move the last labelled photo back to the inbox and show it again."""
        if not self.history:
            return None
        moved_to, original = self.history.pop()
        back = self._unique(original) if original.exists() else original
        shutil.move(str(moved_to), str(back))
        self.queue.insert(0, back)
        return back


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
WINDOW = "hand-foul label"
CANVAS_W, CANVAS_H, PANEL_W = 1280, 800, 360

_TEXT_KO = {
    "title": "손 위치 라벨링",
    "remaining": "남은 사진",
    "rules": [
        "손이 빨간 테두리 안에 다 있으면 → normal",
        "손 일부라도 밖으로 나가면 → foul",
        "팔(손목 아래)은 보지 않는다",
    ],
    "keys": ["1  normal", "2  foul", "s  건너뛰기 (애매한 것)", "z  되돌리기", "q  종료"],
    "done": "inbox가 비었습니다. q 로 종료하세요.",
    "unreadable": "읽을 수 없는 파일 → skip 으로 이동",
}
_TEXT_EN = {
    "title": "Hand position labelling",
    "remaining": "remaining",
    "rules": [
        "Whole hand inside red border -> normal",
        "Any part of hand outside -> foul",
        "Ignore the arm (below the wrist)",
    ],
    "keys": ["1  normal", "2  foul", "s  skip (ambiguous)", "z  undo", "q  quit"],
    "done": "Inbox is empty. Press q to quit.",
    "unreadable": "Unreadable file -> moved to skip",
}


def _render(session: LabelSession, img: Image.Image | None, message: str) -> np.ndarray:
    txt = _TEXT_KO if korean_font_available() else _TEXT_EN
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), DARK)

    # -- photo area ----------------------------------------------------------
    area_w = CANVAS_W - PANEL_W
    if img is not None:
        shown = fit_within(img, area_w - 20, CANVAS_H - 20)
        x = (area_w - shown.width) // 2
        y = (CANVAS_H - shown.height) // 2
        canvas.paste(shown, (x, y))

    # -- side panel ----------------------------------------------------------
    draw = ImageDraw.Draw(canvas)
    px = area_w + 20
    draw.rectangle([area_w, 0, CANVAS_W, CANVAS_H], fill=(40, 42, 48))
    f_big, f_mid, f_small = get_font(26), get_font(20), get_font(17)

    y = 22
    draw.text((px, y), txt["title"], fill=WHITE, font=f_big)
    y += 48
    name = session.current.name if session.current else "-"
    if len(name) > 30:
        name = name[:14] + "…" + name[-13:]
    draw.text((px, y), name, fill=(190, 195, 205), font=f_small)
    y += 34
    draw.text((px, y), f"{txt['remaining']}: {session.remaining}", fill=WHITE, font=f_mid)
    y += 36
    c = session.counts()
    draw.text((px, y), f"normal {c['normal']}", fill=GREEN, font=f_mid)
    draw.text((px + 130, y), f"foul {c['foul']}", fill=RED, font=f_mid)
    draw.text((px + 230, y), f"skip {c['skip']}", fill=(170, 170, 170), font=f_mid)
    y += 50

    draw.line([px, y, CANVAS_W - 20, y], fill=(80, 84, 92), width=1)
    y += 16
    for line in txt["rules"]:
        draw.text((px, y), line, fill=(235, 235, 235), font=f_small)
        y += 28
    y += 14
    draw.line([px, y, CANVAS_W - 20, y], fill=(80, 84, 92), width=1)
    y += 16
    for line in txt["keys"]:
        draw.text((px, y), line, fill=(200, 205, 215), font=f_small)
        y += 26

    if message:
        draw.text((px, CANVAS_H - 60), message, fill=AMBER, font=f_small)
    if session.current is None:
        draw.text((px, CANVAS_H - 100), txt["done"], fill=AMBER, font=f_small)

    return pil_to_bgr(canvas)


def run_label_tool(data_dir: str | Path = "data") -> dict[str, int]:
    """Open the labelling window. Returns final per-folder counts."""
    import cv2  # imported lazily so headless use of the package never needs a GUI

    session = LabelSession(Path(data_dir))
    txt = _TEXT_KO if korean_font_available() else _TEXT_EN
    print(f"inbox: {session.inbox}  ({session.remaining} images)")
    if session.remaining == 0:
        print("inbox is empty - put photos into it and run again.")
        return session.counts()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    message = ""
    cached: tuple[Path | None, Image.Image | None] = (None, None)

    while True:
        cur = session.current
        if cur is not None and cached[0] != cur:
            try:
                cached = (cur, load_image(cur))
            except Exception:  # corrupt / not an image
                session.assign("skip")
                message = txt["unreadable"]
                continue
        img = cached[1] if cur is not None else None

        cv2.imshow(WINDOW, _render(session, img, message))
        key = cv2.waitKey(30) & 0xFF
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if key == 255:
            continue
        ch = chr(key).lower() if key < 128 else ""

        if ch == "q" or key == 27:
            break
        if ch in KEYMAP and cur is not None:
            dest = session.assign(KEYMAP[ch])
            message = f"{cur.name} → {dest.parent.name}"
        elif ch == "z":
            back = session.undo()
            message = f"undo: {back.name}" if back else "nothing to undo"

    cv2.destroyWindow(WINDOW)
    counts = session.counts()
    print("counts:", counts, "| remaining in inbox:", session.remaining)
    return counts
