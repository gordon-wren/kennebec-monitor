"""AIS vessel enrichment via AISstream.io WebSocket API.

Queries for nearby vessels when a new track is detected and writes
ais.json to the clip directory alongside clip.mp4 and metadata.json.

Free-tier AISstream.io accounts receive global AIS coverage. Sign up at
https://aisstream.io and generate an API key; set it in config.ais_api_key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import config

logger = logging.getLogger(__name__)

_WS_URL = "wss://stream.aisstream.io/v0/stream"


def enrich_clip(clip_dir: Path) -> None:
    """Start a background AIS query for a newly opened clip. Non-blocking."""
    if not config.ais_enabled:
        return
    if config.camera_latitude == 0.0 and config.camera_longitude == 0.0:
        logger.warning("ais_enabled=True but camera_latitude/camera_longitude are not set")
        return

    if config.ais_local_url:
        # Option B — kennebec-ais-catcher local relay (simple HTTP GET)
        threading.Thread(target=_run_local, args=(clip_dir,), daemon=True).start()
    else:
        # Option A — AISstream.io WebSocket
        if not config.ais_api_key:
            logger.warning("ais_enabled=True but neither ais_local_url nor ais_api_key is set")
            return
        threading.Thread(target=_run_remote, args=(clip_dir,), daemon=True).start()


def _run_remote(clip_dir: Path) -> None:
    try:
        asyncio.run(_query_remote(clip_dir))
    except Exception as exc:
        logger.warning("AIS remote query failed for %s: %s", clip_dir.name, exc)


def _run_local(clip_dir: Path) -> None:
    try:
        _query_local(clip_dir)
    except Exception as exc:
        logger.warning("AIS local query failed for %s: %s", clip_dir.name, exc)


def _query_local(clip_dir: Path) -> None:
    """Option B: single HTTP GET to kennebec-ais-catcher relay server."""
    import urllib.request

    lat = config.camera_latitude
    lon = config.camera_longitude
    # Convert bounding box half-width in degrees to a radius in km (1° ≈ 111 km)
    radius_km = config.ais_bounding_box_deg * 111.0

    url = f"{config.ais_local_url.rstrip('/')}?lat={lat}&lon={lon}&r={radius_km}"
    query_at = datetime.now(timezone.utc)

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            vessels = json.loads(resp.read())
    except Exception as exc:
        logger.warning("AIS local server unreachable (%s): %s", url, exc)
        vessels = []

    result: dict[str, Any] = {
        "query_at": query_at.isoformat(),
        "camera_lat": lat,
        "camera_lon": lon,
        "bounding_box_deg": config.ais_bounding_box_deg,
        "source": "local",
        "vessels": vessels,
    }
    ais_path = clip_dir / "ais.json"
    with open(ais_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "AIS (local): %d vessel(s) found near %s",
        len(vessels), clip_dir.name,
    )


async def _query_remote(clip_dir: Path) -> None:
    try:
        import websockets
    except ImportError:
        logger.error("websockets not installed — AIS disabled. Run: pip install 'websockets>=12.0'")
        return

    lat = config.camera_latitude
    lon = config.camera_longitude
    d = config.ais_bounding_box_deg
    bbox = [[lat - d, lon - d], [lat + d, lon + d]]

    subscribe_msg = json.dumps({
        "APIKey": config.ais_api_key,
        "BoundingBoxes": [bbox],
    })

    query_at = datetime.now(timezone.utc)
    vessels: dict[str, dict[str, Any]] = {}

    try:
        async with websockets.connect(_WS_URL, open_timeout=10) as ws:
            await ws.send(subscribe_msg)
            loop = asyncio.get_event_loop()
            deadline = loop.time() + config.ais_query_seconds
            while loop.time() < deadline:
                remaining = deadline - loop.time()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
                    _process(json.loads(raw), vessels)
                except asyncio.TimeoutError:
                    continue
    except Exception as exc:
        logger.warning("AIS WebSocket error: %s", exc)

    result: dict[str, Any] = {
        "query_at": query_at.isoformat(),
        "camera_lat": lat,
        "camera_lon": lon,
        "bounding_box_deg": d,
        "source": "aisstream",
        "vessels": list(vessels.values()),
    }
    ais_path = clip_dir / "ais.json"
    with open(ais_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "AIS: %d vessel(s) found near %s (%.1fs query window)",
        len(vessels), clip_dir.name, config.ais_query_seconds,
    )


def _process(msg: dict[str, Any], vessels: dict[str, dict[str, Any]]) -> None:
    msg_type = msg.get("MessageType", "")
    meta = msg.get("MetaData", {})
    mmsi = str(meta.get("MMSI", ""))
    if not mmsi:
        return

    payload = msg.get("Message", {})

    if msg_type == "PositionReport":
        report = payload.get("PositionReport", {})
        entry = vessels.setdefault(mmsi, {"mmsi": mmsi})
        entry.update({
            "name": meta.get("ShipName", entry.get("name", "")).strip(),
            "lat": meta.get("latitude"),
            "lon": meta.get("longitude"),
            "speed_knots": report.get("Sog"),
            "course": report.get("Cog"),
            "heading": report.get("TrueHeading"),
            "nav_status": report.get("NavigationalStatus"),
            "timestamp": meta.get("time_utc"),
        })

    elif msg_type == "ShipStaticData":
        static = payload.get("ShipStaticData", {})
        entry = vessels.setdefault(mmsi, {"mmsi": mmsi})
        entry.update({
            "name": static.get("Name", entry.get("name", "")).strip(),
            "ship_type": static.get("Type"),
            "call_sign": static.get("CallSign", "").strip(),
            "destination": static.get("Destination", "").strip(),
        })
        if "lat" not in entry:
            entry["lat"] = meta.get("latitude")
            entry["lon"] = meta.get("longitude")
            entry["timestamp"] = meta.get("time_utc")
