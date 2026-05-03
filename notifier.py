"""Push notifications for new boat detections.

Providers
─────────
ntfy    — https://ntfy.sh  (free, no account, great iOS/Android app)
webhook — HTTP POST of a JSON payload to any URL

Both send a thumbnail of the detection frame alongside the alert.
Calls are fire-and-forget daemon threads — they never block the main loop.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import config

if TYPE_CHECKING:
    from detector import Detection

logger = logging.getLogger(__name__)

_THUMB_SIZE = (640, 360)
_JPEG_QUALITY = 80


def notify_detection(frame: np.ndarray, detections: list[Detection]) -> None:
    """Fire a push notification for one or more newly detected tracks. Non-blocking."""
    if not config.notify_enabled:
        return

    confs = [d.confidence for d in detections]
    if not confs or max(confs) < config.notify_min_confidence:
        logger.debug(
            "Notification suppressed — max confidence %.2f below threshold %.2f",
            max(confs) if confs else 0,
            config.notify_min_confidence,
        )
        return

    threading.Thread(target=_dispatch, args=(frame, list(detections)), daemon=True).start()


def _dispatch(frame: np.ndarray, detections: list[Detection]) -> None:
    thumb = _encode_thumbnail(frame, detections)
    provider = config.notify_provider.lower()
    try:
        if provider == "ntfy":
            _send_ntfy(detections, thumb)
        elif provider == "webhook":
            _send_webhook(detections, thumb)
        else:
            logger.warning("Unknown notify_provider '%s' — expected 'ntfy' or 'webhook'", provider)
    except Exception as exc:
        logger.warning("Notification failed (%s): %s", provider, exc)


# ── ntfy ──────────────────────────────────────────────────────────────────────

def _send_ntfy(detections: list[Detection], thumb: bytes | None) -> None:
    if not config.notify_ntfy_url:
        logger.warning("notify_provider=ntfy but notify_ntfy_url is not set")
        return

    ids = ", ".join(f"ID:{d.track_id}" for d in detections)
    confs = [d.confidence for d in detections]
    mean_conf = sum(confs) / len(confs) if confs else 0
    count = len(detections)
    message = f"{count} boat{'s' if count > 1 else ''} detected — {ids} — conf {mean_conf:.2f}"

    headers = {
        "Title": f"🚢 Boat detected — {config.camera_id}",
        "Message": message,
        "Tags": "boat,alert",
        "Priority": "default",
    }

    if thumb:
        # Attach the thumbnail as the request body so it appears inline in the notification.
        headers["Filename"] = "detection.jpg"
        headers["Content-Type"] = "image/jpeg"
        body = thumb
    else:
        headers["Content-Type"] = "text/plain"
        body = message.encode()

    req = urllib.request.Request(config.notify_ntfy_url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 201):
            logger.warning("ntfy returned HTTP %d", resp.status)
        else:
            logger.debug("ntfy notification sent (%d detection(s))", len(detections))


# ── Webhook ───────────────────────────────────────────────────────────────────

def _send_webhook(detections: list[Detection], thumb: bytes | None) -> None:
    if not config.notify_webhook_url:
        logger.warning("notify_provider=webhook but notify_webhook_url is not set")
        return

    import base64
    from datetime import datetime, timezone

    payload = {
        "camera_id": config.camera_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": [
            {
                "track_id": d.track_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
            }
            for d in detections
        ],
        "thumbnail_b64": base64.b64encode(thumb).decode() if thumb else None,
    }

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        config.notify_webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 201, 202, 204):
            logger.warning("Webhook returned HTTP %d", resp.status)
        else:
            logger.debug("Webhook notification sent (%d detection(s))", len(detections))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_thumbnail(frame: np.ndarray, detections: list[Detection]) -> bytes | None:
    try:
        small = cv2.resize(frame, _THUMB_SIZE)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return bytes(buf)
    except Exception as exc:
        logger.warning("Failed to encode notification thumbnail: %s", exc)
        return None
