"""Upload finalized clips to Cloudflare R2.

Credentials are read from environment variables:
    R2_ACCESS_KEY   — R2 API token key ID
    R2_SECRET_KEY   — R2 API token secret

Set config.r2_upload_enabled = True and populate config.r2_bucket /
config.r2_endpoint to activate uploads.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)


def clip_id(clip_dir: Path) -> str:
    """Derive a URL-safe clip ID from the clip directory.

    e.g. .../cam1/2026-05-02/track_6_012714 → cam1--2026-05-02--track_6_012714
    Segments are joined with '--'; reversed to '/' when building R2 keys.
    """
    # clip_dir.parts: [..., output_dir, camera_id, date, track_dirname]
    return f"{clip_dir.parent.parent.name}--{clip_dir.parent.name}--{clip_dir.name}"


def upload_clip(clip_dir: Path) -> None:
    """Fire-and-forget upload of clip.mp4, metadata.json, and thumb.jpg to R2.
    Runs in a daemon thread — does not block the detection loop.
    """
    if not config.r2_upload_enabled:
        return
    threading.Thread(target=_upload, args=(clip_dir,), daemon=True).start()


def _upload(clip_dir: Path) -> None:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        logger.error("boto3 not installed — R2 upload disabled. Run: pip install boto3")
        return

    cid = clip_id(clip_dir)
    r2_prefix = f"clips/{cid.replace('--', '/')}"

    try:
        client = boto3.client(
            "s3",
            endpoint_url=config.r2_endpoint,
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY", ""),
            aws_secret_access_key=os.environ.get("R2_SECRET_KEY", ""),
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )
        uploaded = []
        for filename in ("clip.mp4", "metadata.json", "thumb.jpg"):
            local = clip_dir / filename
            if not local.exists():
                continue
            key = f"{r2_prefix}/{filename}"
            client.upload_file(str(local), config.r2_bucket, key)
            uploaded.append(filename)

        logger.info("R2 upload complete for %s: %s", cid, ", ".join(uploaded))
    except Exception as exc:
        logger.error("R2 upload failed for %s: %s", cid, exc)
