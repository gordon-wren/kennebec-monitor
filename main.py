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


def run(input_path: Path | None = None, background_frames: bool = False) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    file_mode = input_path is not None
    source = input_path if file_mode else None

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
            input_path,
            camera.fps,
            config.device,
            config.target_classes,
            config.output_dir,
        )
    else:
        logger.info(
            "Live mode — fps=%.1f device=%s classes=%s output=%s",
            camera.fps,
            config.device,
            config.target_classes,
            config.output_dir,
        )

    frame_count = 0
    last_heartbeat = time.monotonic()
    last_snapshot = time.monotonic()
    known_track_ids: set[int] = set()

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

        try:
            detections = detector.detect(frame)
        except Exception as exc:
            logger.warning("Detection failed on frame, skipping: %s", exc)
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
            logger.debug(
                "frame=%d | detections=%d | active_tracks=%d",
                frame_count,
                len(detections),
                recorder.active_track_count,
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
        "--background-frames",
        action="store_true",
        default=False,
        help=f"Print a frame to the terminal every {config.snapshot_interval_seconds}s regardless of detections.",
    )
    args = parser.parse_args()

    if args.input is not None and not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run(input_path=args.input, background_frames=args.background_frames)
