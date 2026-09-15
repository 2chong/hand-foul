"""Real-time foul detection from a camera or video file: `hand-foul live` / `hand-foul video`.

`--source` accepts a webcam index (0, 1, ...), a video file path or a stream URL, so a phone can
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


def is_file_source(source: str | int) -> bool:
    return isinstance(source, str) and Path(source).is_file()


def open_source(source: str | int):
    """Open a webcam index, a video file or a stream URL."""
    import cv2

    src: str | int = source
    if isinstance(source, str) and source.strip().lstrip("-").isdigit():
        src = int(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open {source!r}. Use a webcam index (--source 1), a video file (clip.mp4) "
            "or a stream URL like http://<phone-ip>:8080/video"
        )
    return cap


def _fmt_t(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:04.1f}"


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
    record: str | Path | None = None,
    show_window: bool = True,
    realtime: bool | None = None,
    window: str = "hand-foul live",
    log=print,
) -> dict:
    """Run detection on a camera, stream or video file.

    record      write the overlaid frames to this video file (.mp4 / .avi)
    show_window False = headless (e.g. batch-process a file)
    realtime    pace a video file at its own fps (default: only when a window is shown)
    Returns a summary dict with frame counts and foul segments (seconds).
    """
    import cv2

    clf = Classifier.from_pretrained(weights, device=device)
    foul_idx = clf.classes.index("foul")
    cap = open_source(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    from_file = is_file_source(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not src_fps or src_fps != src_fps or src_fps > 240:  # missing / NaN / bogus
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if from_file else 0
    if realtime is None:
        realtime = show_window
    log(f"source: {source} | device: {clf.device} | threshold: {threshold}"
        + (f" | {total_frames} frames @ {src_fps:.1f} fps" if from_file else "")
        + (" | press q to quit, s to save a frame" if show_window else ""))

    writer = None
    if record:
        record = Path(record)
        record.parent.mkdir(parents=True, exist_ok=True)

    if show_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    probs = None  # smoothed probability vector
    warning = False
    n = n_foul = 0
    segments: list[list[float]] = []  # [start, end] in seconds
    t_last, fps = time.time(), 0.0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if not from_file:
                    log("stream ended / frame grab failed")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            t_frame = n / src_fps if from_file else time.time() - t0

            if n % max(1, every) == 0:
                p = clf.predict_probs([frame])[0]
                probs = p if probs is None else smoothing * probs + (1 - smoothing) * p
            n += 1
            foul_prob = float(probs[foul_idx]) if probs is not None else 0.0
            # small hysteresis so the warning does not flicker around the threshold
            warning = foul_prob > threshold if not warning else foul_prob > threshold - 0.1
            if warning:
                n_foul += 1
                if segments and segments[-1][1] >= t_frame - 1.5 / src_fps:
                    segments[-1][1] = t_frame
                else:
                    segments.append([t_frame, t_frame])

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6)) if fps else 1.0 / max(now - t_last, 1e-6)
            t_last = now

            shown = draw_overlay(frame, foul_prob, warning, fps=fps if show_window else None, tick=t_frame)
            if from_file:
                cv2.putText(shown, _fmt_t(t_frame), (shown.shape[1] - 110, shown.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

            if record:
                if writer is None:
                    codec = "MJPG" if record.suffix.lower() == ".avi" else "mp4v"
                    writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*codec), src_fps, (shown.shape[1], shown.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer for {record} (try an .avi name)")
                writer.write(shown)
                if not show_window and total_frames and n % 50 == 0:
                    log(f"  {n}/{total_frames} frames")

            if show_window:
                cv2.imshow(window, shown)
                delay = max(1, int(1000 / src_fps - (time.time() - now) * 1000)) if (from_file and realtime) else 1
                key = cv2.waitKey(delay) & 0xFF
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
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    summary = {
        "source": str(source),
        "frames": n,
        "foul_frames": n_foul,
        "fps": src_fps,
        "foul_segments": [(round(a, 2), round(b, 2)) for a, b in segments],
        "record": str(record) if record else None,
    }
    if n:
        log(f"{n} frames, foul in {n_foul} ({100 * n_foul / n:.0f}%)")
        if segments:
            log("foul segments: " + ", ".join(f"{_fmt_t(a)}-{_fmt_t(b)}" for a, b in segments))
        if record:
            log(f"annotated video -> {record}")
    return summary
