"""Real-time foul detection from a camera: `hand-foul live`.

`--source` accepts a webcam index (0, 1, ...) or a stream URL, so a phone can
be used either as a virtual webcam (Camo, DroidCam, Windows "connected
camera") or as an IP camera (IP Webcam -> http://<phone-ip>:8080/video).

Each frame is classified; the class probabilities are smoothed over time so
the overlay does not flicker. While the smoothed foul probability is above
`threshold` the frame gets a thick red border and a WARNING banner.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .predict import Classifier

GREEN_BGR = (80, 160, 30)
RED_BGR = (40, 50, 210)
WHITE = (255, 255, 255)


def open_source(source: str | int):
    import cv2

    src: str | int = source
    if isinstance(source, str) and source.strip().lstrip("-").isdigit():
        src = int(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open camera/stream {source!r}. Try another index (--source 1) "
            "or a stream URL like http://<phone-ip>:8080/video"
        )
    return cap


def draw_overlay(frame: np.ndarray, foul_prob: float, warning: bool, fps: float | None = None, tick: float = 0.0) -> np.ndarray:
    """Green frame + label when normal; thick pulsing red frame + WARNING when foul."""
    import cv2

    h, w = frame.shape[:2]
    out = frame.copy()
    conf = foul_prob if warning else 1.0 - foul_prob
    label = f"FOUL  {conf * 100:.0f}%" if warning else f"NORMAL  {conf * 100:.0f}%"
    colour = RED_BGR if warning else GREEN_BGR

    if warning:
        pulse = 0.5 + 0.5 * np.sin(tick * 6.0)  # ~1 Hz pulse
        border = int(min(w, h) * (0.03 + 0.03 * pulse))
    else:
        border = max(4, int(min(w, h) * 0.01))
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), colour, border)

    scale = max(0.8, min(w, h) / 500)
    thick = max(2, int(scale * 2))
    if warning:
        text = "WARNING"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale * 2.2, thick + 1)
        cv2.rectangle(out, (0, border), (w, border + th + 30), colour, -1)
        cv2.putText(out, text, ((w - tw) // 2, border + th + 12), cv2.FONT_HERSHEY_DUPLEX, scale * 2.2, WHITE, thick + 1, cv2.LINE_AA)
        y = border + th + 30 + int(36 * scale)
    else:
        y = border + int(36 * scale)
    cv2.putText(out, label, (border + 12, y), cv2.FONT_HERSHEY_DUPLEX, scale, colour, thick, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} fps   q: quit  s: save", (border + 12, h - border - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, WHITE, 1, cv2.LINE_AA)
    return out


def run_live(
    source: str | int = 0,
    weights: str | Path = "weights/best.pt",
    threshold: float = 0.5,
    smoothing: float = 0.6,
    every: int = 1,
    width: int | None = None,
    mirror: bool = False,
    save_dir: str | Path = "live_captures",
    device: str | None = None,
    window: str = "hand-foul live",
    log=print,
) -> None:
    import cv2

    clf = Classifier.from_pretrained(weights, device=device)
    foul_idx = clf.classes.index("foul")
    cap = open_source(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    log(f"source: {source} | device: {clf.device} | threshold: {threshold} | press q to quit, s to save a frame")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    probs = None  # smoothed probability vector
    warning = False
    n = 0
    t_last, fps = time.time(), 0.0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log("stream ended / frame grab failed")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            if n % max(1, every) == 0:
                p = clf.predict_probs([frame])[0]
                probs = p if probs is None else smoothing * probs + (1 - smoothing) * p
            n += 1
            foul_prob = float(probs[foul_idx]) if probs is not None else 0.0
            # small hysteresis so the warning does not flicker around the threshold
            warning = foul_prob > threshold if not warning else foul_prob > threshold - 0.1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6)) if fps else 1.0 / max(now - t_last, 1e-6)
            t_last = now

            shown = draw_overlay(frame, foul_prob, warning, fps=fps, tick=now - t0)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path(save_dir)
                out.mkdir(parents=True, exist_ok=True)
                path = out / time.strftime(f"{'foul' if warning else 'normal'}_%Y%m%d-%H%M%S.jpg")
                cv2.imwrite(str(path), shown)
                log(f"saved {path}")
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
