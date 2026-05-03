import json
import logging
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import config
from detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class _TrackState:
    track_id: int
    clip_path: Path
    metadata_path: Path
    writer: cv2.VideoWriter
    started_at: datetime
    last_seen_at: datetime
    fps: float
    frame_count: int = 0
    detections: list[Detection] = field(default_factory=list)
    last_bbox: Optional[tuple[float, float, float, float]] = None
    # Track the spread of centroid x-positions to detect real horizontal movement.
    # Jitter on a static pier is typically <50px; a boat crossing the frame is 200px+.
    cx_min: float = float("inf")
    cx_max: float = float("-inf")
    error: Optional[str] = None


class ClipRecorder:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self._active: dict[int, _TrackState] = {}
        # Maps a new BoT-SORT track ID to the canonical ID of an existing clip when
        # spatial proximity suggests it's the same physical object re-identified.
        self._id_remap: dict[int, int] = {}
        self._last_lost_log: dict[int, datetime] = {}

    @property
    def active_track_count(self) -> int:
        return len(self._active)

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        pre_buffer: list[np.ndarray],
    ) -> None:
        now = datetime.now(timezone.utc)

        # Resolve any remapped IDs before processing
        resolved = [
            Detection(
                track_id=self._id_remap.get(d.track_id, d.track_id),
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy,
            )
            for d in detections
        ]
        visible_ids = {d.track_id for d in resolved}

        # Render overlay once and reuse across all writers this frame
        out_frame = _draw_overlay(frame, resolved) if config.draw_overlay else frame

        # Open a new clip for any track ID we haven't seen before,
        # or remap to an existing clip if spatially close to a lost track.
        # Only consider tracks that are lost this frame (not in visible_ids) as merge
        # candidates — we must never merge a new boat into a clip for a different boat
        # that is still actively visible in the same frame.
        for det in resolved:
            if det.track_id not in self._active:
                canonical = self._find_nearby_lost_track(det.bbox_xyxy, frame.shape, visible_ids)
                if canonical is not None:
                    logger.info(
                        "Track %d merged into existing clip for track %d (proximity re-ID)",
                        det.track_id, canonical,
                    )
                    self._id_remap[det.track_id] = canonical
                    # Update the canonical state so it's treated as visible again
                    self._active[canonical].last_seen_at = now
                else:
                    self._start_track(det.track_id, now, pre_buffer, frame.shape)

        # Write the current frame to every open clip and check for timeouts
        for track_id in list(self._active.keys()):
            state = self._active[track_id]
            try:
                state.writer.write(out_frame)
                state.frame_count += 1
            except Exception as exc:
                state.error = str(exc)
                logger.error("Frame write failed for track %d: %s", track_id, exc)

            if track_id in visible_ids:
                det = next(d for d in resolved if d.track_id == track_id)
                state.last_seen_at = now
                state.last_bbox = det.bbox_xyxy
                state.detections.append(det)
                self._last_lost_log.pop(track_id, None)
                cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) / 2
                state.cx_min = min(state.cx_min, cx)
                state.cx_max = max(state.cx_max, cx)
            else:
                absent_for = (now - state.last_seen_at).total_seconds()
                if absent_for > config.track_loss_timeout_seconds:
                    self._finalize_track(track_id, now)
                else:
                    last_log = self._last_lost_log.get(track_id)
                    if last_log is None or (now - last_log).total_seconds() >= 2.0:
                        logger.debug(
                            "Track %d not visible for %.1fs — still waiting (timeout %.1fs)",
                            track_id,
                            absent_for,
                            config.track_loss_timeout_seconds,
                        )
                        self._last_lost_log[track_id] = now

    def flush_all(self) -> None:
        """Finalize every open clip — call on shutdown or camera loss."""
        now = datetime.now(timezone.utc)
        for track_id in list(self._active.keys()):
            self._finalize_track(track_id, now)

    # ------------------------------------------------------------------ private

    def _start_track(
        self,
        track_id: int,
        now: datetime,
        pre_buffer: list[np.ndarray],
        frame_shape: tuple,
    ) -> None:
        date_str = now.strftime("%Y-%m-%d")
        ts_str = now.strftime("%H%M%S")
        clip_dir = config.output_dir / date_str / f"track_{track_id}_{ts_str}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        clip_path = clip_dir / "clip.mp4"
        metadata_path = clip_dir / "metadata.json"

        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))

        if not writer.isOpened():
            error = f"VideoWriter failed to open at {clip_path}"
            logger.error(error)
            _write_error_metadata(metadata_path, track_id, now, error)
            return

        state = _TrackState(
            track_id=track_id,
            clip_path=clip_path,
            metadata_path=metadata_path,
            writer=writer,
            started_at=now,
            last_seen_at=now,
            fps=self.fps,
        )

        # Write buffered frames before the detection event (no overlay — no data for those frames)
        for buffered_frame in pre_buffer:
            writer.write(buffered_frame)
            state.frame_count += 1

        self._active[track_id] = state
        logger.info("Started clip for track %d → %s", track_id, clip_path)

    def _finalize_track(self, track_id: int, now: datetime) -> None:
        state = self._active.pop(track_id)
        self._last_lost_log.pop(track_id, None)
        # Remove any remaps that pointed to this canonical ID
        stale = [k for k, v in self._id_remap.items() if v == track_id]
        for k in stale:
            del self._id_remap[k]
        state.writer.release()

        x_range = state.cx_max - state.cx_min if state.cx_max != float("-inf") else 0.0
        threshold = config.min_track_displacement_px
        if threshold > 0 and x_range < threshold:
            logger.info(
                "Discarding track %d — static object (x-range %.1fpx < %.1fpx threshold) → deleting %s",
                track_id, x_range, threshold, state.clip_path.parent,
            )
            shutil.rmtree(state.clip_path.parent, ignore_errors=True)
            return

        _write_metadata(state, ended_at=now, x_range=x_range)
        logger.info(
            "Finalized track %d — %d frames, %.1fs, x-range %.1fpx → %s",
            track_id,
            state.frame_count,
            (now - state.started_at).total_seconds(),
            x_range,
            state.clip_path,
        )

    def _find_nearby_lost_track(
        self,
        bbox: tuple[float, float, float, float],
        frame_shape: tuple,
        visible_ids: set[int],
    ) -> Optional[int]:
        """Return the canonical track ID of any lost (not currently visible) active track
        whose last known position is within config.track_merge_proximity of bbox.
        Returns None if no match. visible_ids excludes currently-tracked objects so we
        never merge a genuinely new boat into a clip for a different boat still in frame."""
        h, w = frame_shape[:2]
        frame_diag = math.sqrt(w ** 2 + h ** 2)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        for track_id, state in self._active.items():
            if track_id in visible_ids:
                continue  # skip tracks still actively visible this frame
            if state.last_bbox is None:
                continue
            lx = (state.last_bbox[0] + state.last_bbox[2]) / 2
            ly = (state.last_bbox[1] + state.last_bbox[3]) / 2
            dist = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
            if dist / frame_diag <= config.track_merge_proximity:
                return track_id
        return None


# ------------------------------------------------------------------ helpers

def _write_metadata(state: _TrackState, ended_at: datetime, x_range: float = 0.0) -> None:
    confs = [d.confidence for d in state.detections]
    duration = (ended_at - state.started_at).total_seconds()

    meta = {
        "track_id": state.track_id,
        "started_at": state.started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration, 3),
        "frame_count": state.frame_count,
        "fps": state.fps,
        "resolution": list(config.output_resolution),
        "clip_path": state.clip_path.name,
        "detected_classes": list({d.class_name for d in state.detections}),
        "detection_count": len(state.detections),
        "confidence_min": round(min(confs), 4) if confs else None,
        "confidence_max": round(max(confs), 4) if confs else None,
        "confidence_mean": round(sum(confs) / len(confs), 4) if confs else None,
        "x_range_px": round(x_range, 1),
        "error": state.error,
    }
    with open(state.metadata_path, "w") as f:
        json.dump(meta, f, indent=2)


def _write_error_metadata(
    metadata_path: Path, track_id: int, now: datetime, error: str
) -> None:
    meta = {
        "track_id": track_id,
        "started_at": now.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "frame_count": 0,
        "error": error,
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)


def _draw_overlay(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{det.track_id} {det.class_name} {det.confidence:.2f}"
        cv2.putText(
            out, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return out
