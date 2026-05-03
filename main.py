import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2

from capture import CameraCapture
from config import config
from detector import BoatDetector
from recorder import ClipRecorder, _draw_overlay

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy debug output from third-party libraries
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _check_terminal() -> None:
    import os
    term = os.environ.get("TERM_PROGRAM", "")
    if term != "iTerm.app":
        logger.warning(
            "Inline frame printing requires iTerm2 (detected: %s). "
            "Frames will be written but not visible. Download iTerm2 at https://iterm2.com",
            term or "unknown",
        )


def _next_test_run_dir() -> Path:
    base = Path("test_clips")
    base.mkdir(exist_ok=True)
    existing = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_"))
    n = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    return base / f"run_{n:03d}"


def run(
    input_path: Path | None = None,
    camera_url: str | None = None,
    background_frames: bool = False,
) -> None:
    _check_terminal()

    file_mode = input_path is not None
    source = input_path or camera_url or None

    if file_mode:
        config.output_dir = _next_test_run_dir()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraCapture(source=source)
    detector = BoatDetector()
    recorder = ClipRecorder(fps=camera.fps)

    def _shutdown(sig, frame):
        logger.info("Shutting down — flushing open clips")
        recorder.flush_all()
        camera.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if file_mode:
        logger.info(
            "Test mode — input=%s fps=%.1f device=%s classes=%s output=%s",
            input_path, camera.fps, config.device, config.target_classes, config.output_dir,
        )
    elif camera_url:
        logger.info(
            "Live mode (RTSP) — url=%s fps=%.1f device=%s classes=%s output=%s",
            camera_url, camera.fps, config.device, config.target_classes, config.output_dir,
        )
    else:
        logger.info(
            "Live mode — fps=%.1f device=%s classes=%s output=%s",
            camera.fps, config.device, config.target_classes, config.output_dir,
        )

    frame_count = 0
    run_start = time.monotonic()
    last_heartbeat = run_start
    last_snapshot = run_start
    known_track_ids: set[int] = set()
    last_detections: list = []

    while True:
        # Snapshot must come before read() so the current frame isn't in the pre-buffer
        pre_buffer = camera.pre_buffer_snapshot()
        ret, frame = camera.read()

        if not ret:
            if not file_mode:
                logger.error("Camera read failed — shutting down")
            else:
                logger.info("End of input file — flushing clips")
            break

        frame_count += 1

        if frame_count % config.inference_every_n_frames == 0:
            try:
                last_detections = detector.detect(frame)
            except Exception as exc:
                logger.warning("Detection failed on frame, skipping: %s", exc)
                last_detections = []
            detections = last_detections
        else:
            detections = []

        # Notify on new track IDs
        now_mono = time.monotonic()
        new_ids = {d.track_id for d in detections} - known_track_ids
        if new_ids:
            new_detections = [d for d in detections if d.track_id in new_ids]
            _notify_detection(frame, new_detections, frame_count)
            known_track_ids.update(new_ids)
            last_snapshot = now_mono  # reset background timer to avoid double-print
            last_heartbeat = now_mono  # reset heartbeat so it doesn't fire immediately after

        # Heartbeat every 10 seconds, only when no new tracks have fired recently
        elif now_mono - last_heartbeat >= 10.0:
            elapsed = now_mono - run_start
            video_seconds = frame_count / camera.fps
            if file_mode and camera.total_frames:
                progress = frame_count / camera.total_frames * 100
                logger.debug(
                    "frame=%d/%d (%.1f%%) | video=%.1fs | elapsed=%.1fs | detections=%d | active_tracks=%d",
                    frame_count, camera.total_frames, progress,
                    video_seconds, elapsed,
                    len(detections), recorder.active_track_count,
                )
            else:
                logger.debug(
                    "frame=%d | elapsed=%.1fs | detections=%d | active_tracks=%d",
                    frame_count, elapsed,
                    len(detections), recorder.active_track_count,
                )
            last_heartbeat = now_mono

        try:
            recorder.update(frame, detections, pre_buffer)
        except Exception as exc:
            logger.error("Recorder update failed: %s", exc)

        if background_frames and now_mono - last_snapshot >= config.snapshot_interval_seconds:
            logger.debug("Background frame — frame=%d active_tracks=%d", frame_count, recorder.active_track_count)
            _print_frame(frame, detections, frame_count)
            last_snapshot = now_mono

        if config.show_preview:
            preview = _draw_overlay(frame, detections) if detections else frame
            cv2.imshow("Boat Detector", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    recorder.flush_all()
    camera.release()
    cv2.destroyAllWindows()


def _notify_detection(frame: "cv2.Mat", detections: list, frame_count: int) -> None:
    """Called once per new track ID. Extend this function to add push notifications,
    webhooks, alerts, etc. alongside the terminal output."""
    for d in detections:
        logger.info(
            "--- BOAT DETECTED --- %s ID:%d conf:%.2f (frame %d)",
            d.class_name, d.track_id, d.confidence, frame_count,
        )
    _print_frame(frame, detections, frame_count)


def _print_frame(frame: "cv2.Mat", detections: list, frame_count: int) -> None:
    """Print a low-res annotated frame inline in the terminal (iTerm2 / compatible terminals)."""
    import base64
    import sys

    small = cv2.resize(frame, config.snapshot_resolution)

    if detections:
        sx = config.snapshot_resolution[0] / config.output_resolution[0]
        sy = config.snapshot_resolution[1] / config.output_resolution[1]
        scaled = [
            type(d)(
                track_id=d.track_id,
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_xyxy=(
                    d.bbox_xyxy[0] * sx,
                    d.bbox_xyxy[1] * sy,
                    d.bbox_xyxy[2] * sx,
                    d.bbox_xyxy[3] * sy,
                ),
            )
            for d in detections
        ]
        out = _draw_overlay(small, scaled)
    else:
        out = small

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, config.snapshot_quality])
    if not ok:
        logger.warning("Failed to encode snapshot frame")
        return

    b64 = base64.b64encode(buf).decode()
    w, h = config.snapshot_resolution
    # iTerm2 inline image protocol
    sys.stdout.write(f"\033]1337;File=inline=1;width={w}px;height={h}px;preserveAspectRatio=1:{b64}\a\n")
    sys.stdout.flush()
    logger.debug("frame=%d printed to terminal (%d detections)", frame_count, len(detections))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boat detector — live camera or test file")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="Path to a video file for test mode. Omit to use the live camera.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=None,
        metavar="RTSP_URL",
        help="RTSP URL of an IP camera, e.g. rtsp://admin:pass@192.168.1.100:554/h264Preview_01_main",
    )
    parser.add_argument(
        "--background-frames",
        action="store_true",
        default=False,
        help=f"Print a frame to the terminal every {config.snapshot_interval_seconds}s regardless of detections.",
    )
    args = parser.parse_args()

    if args.input is not None and not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    if args.input and args.camera:
        parser.error("--input and --camera are mutually exclusive")

    run(input_path=args.input, camera_url=args.camera, background_frames=args.background_frames)
