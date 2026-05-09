import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Camera — index 0 is usually the first connected UVC device
    camera_index: int = 0
    # Stable identifier for this camera — used in clip output paths and R2 keys so
    # clips from different cameras can be distinguished. Keep it short and URL-safe.
    camera_id: str = "cam1"

    # Output
    output_dir: Path = Path("clips")
    output_resolution: tuple[int, int] = (1920, 1080)
    # "mp4v" is safe on all platforms; swap to "avc1" on macOS for H.264 (smaller files)
    video_codec: str = "mp4v"

    # Detection
    # COCO class 8 = "boat" — set to None to track all detected object classes
    model_name: str = "yolo11s.pt"
    # Pass low-confidence detections through to BoT-SORT so its second-stage
    # matching can use them to keep existing tracks alive. The tracker's own
    # track_high_thresh / track_low_thresh in tracker.yaml control how they're used.
    confidence_threshold: float = 0.1

    # Run YOLO inference only every N frames. All frames are still written to clips
    # for smooth video output. Higher values improve performance but reduce detection
    # responsiveness — for slow vessels, 3–5 is a good starting point.
    # 1 = every frame (no skipping).
    inference_every_n_frames: int = 6
    target_classes: Optional[list[int]] = field(default_factory=lambda: [8])
    # "cpu" for Intel; "mps" for Apple Silicon (M-series)
    device: str = "cpu"

    # Recording
    pre_buffer_seconds: float = 20.0
    # How long a track can be absent before its clip is closed.
    # Too short → clips fragment when detection drops (waves, glare, occlusion).
    # Too long  → clips merge when two boats pass close together or one lingers off-frame.
    # For slow-moving vessels, set this to at least the longest expected detection gap.
    track_loss_timeout_seconds: float = 30.0

    # If a new track ID appears within this fraction of the frame diagonal from a
    # recently-lost track, it is treated as the same object and routed to the existing clip.
    # Handles cases where BoT-SORT drops and reassigns IDs despite a large track_buffer.
    # Range 0.0–1.0; 0.25 = 25% of frame diagonal (~480px on 1080p).
    track_merge_proximity: float = 0.25

    # Boats enter the frame from the left or right edge. A new track ID appearing within
    # this fraction of frame width from either edge is treated as a genuine new entry and
    # is never merged into an existing clip, regardless of proximity.
    # Range 0.0–0.5; 0.15 = leftmost/rightmost 15% of frame (~288px on 1920px wide).
    # Set to 0.0 to disable and fall back to proximity-only logic.
    edge_entry_zone: float = 0.15

    # Maximum bounding box area as a fraction of the total frame area. Detections larger
    # than this are discarded before reaching the tracker — catches piers, docks, and
    # large foreground structures that YOLO misclassifies as boats. A pier that fills
    # 25% of the frame will be filtered; a boat filling 5–10% will not.
    # Range 0.0–1.0; set to 1.0 to disable.
    max_detection_area_fraction: float = 0.10

    # Minimum horizontal x-range (pixels) a track's centroid must span over its lifetime
    # to be saved. Uses the spread of x-positions (max_cx - min_cx) rather than
    # displacement from start, making it robust against YOLO bbox jitter on large static
    # objects. A pier jitters <50px; a boat crossing the frame spans 200px+.
    # Set to 0.0 to disable.
    min_track_displacement_px: float = 100.0

    # Health watchdog — write a timestamp to this file every heartbeat cycle.
    # An external cron job can alert if the file goes stale (e.g. older than 5 minutes).
    # Leave empty to disable.
    # Example cron check (add to crontab -e):
    #   */5 * * * * find /tmp/boat-detector.watchdog -mmin +5 && echo "detector may be down"
    watchdog_file: str = ""

    # Logging — rotating file log for unattended production use.
    # Set log_file to an absolute path to enable (e.g. "/var/log/boat-detector/detector.log").
    # Leave empty to log to stdout only.
    log_file: str = ""
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB per file
    log_backup_count: int = 5              # keep 5 rotated files

    # Storage — delete clip directories older than this many days on startup.
    # Set to 0 to disable. Applies to config.output_dir only, not test_clips/.
    max_clip_age_days: int = 365

    # RTSP reconnect — seconds to wait between reconnect attempts (doubles each retry,
    # capped at max_reconnect_interval_seconds).
    reconnect_interval_seconds: float = 5.0
    max_reconnect_interval_seconds: float = 60.0

    # Notifications — fired once per new track ID (new boat detection).
    # Supported providers: "ntfy", "webhook" (set notify_provider accordingly).
    #
    # ntfy  — free push notifications, no account required.
    #   1. Pick a secret topic name (treat it like a password).
    #   2. Set notify_ntfy_url = "https://ntfy.sh/<your-topic>"
    #   3. Install the ntfy app on your phone and subscribe to the same topic.
    #
    # webhook — POST JSON payload to any URL (Zapier, Make.com, custom endpoint).
    #   Set notify_webhook_url to your endpoint.
    notify_enabled: bool = False
    notify_provider: str = "ntfy"        # "ntfy" | "webhook"
    notify_ntfy_url: str = ""            # https://ntfy.sh/<topic>
    notify_webhook_url: str = ""         # https://your-endpoint.example.com/hook
    notify_min_confidence: float = 0.35  # skip notification if all detections below this

    # AIS vessel enrichment — correlate detections with nearby vessels via AISstream.io.
    # Sign up at https://aisstream.io for a free API key (global AIS coverage).
    # Set camera_latitude / camera_longitude to the deployment position; the query
    # fetches all AIS-transmitting vessels within ±ais_bounding_box_deg of that point.
    ais_enabled: bool = False
    # Option A — AISstream.io cloud API (requires internet, free tier available)
    ais_api_key: str = field(default_factory=lambda: os.environ.get("AIS_API_KEY", ""))
    # Option B — kennebec-ais-catcher local relay server
    # Set to the server URL to use local RTL-SDR data instead of AISstream.io.
    # Example (same machine): "http://127.0.0.1:8080/vessels"
    # Example (LAN):          "http://192.168.1.50:8080/vessels"
    # When set, ais_api_key is ignored.
    ais_local_url: str = ""
    camera_latitude: float = 0.0       # Decimal degrees, e.g. 44.2374
    camera_longitude: float = 0.0     # Decimal degrees, e.g. -69.7626
    ais_bounding_box_deg: float = 0.05 # Half-width of query box in degrees (~5.5 km)
    ais_query_seconds: float = 30.0    # How long to listen per detection event (Option A only)

    # Cloudflare R2 upload — populate these and set R2_ACCESS_KEY / R2_SECRET_KEY
    # as environment variables, then flip r2_upload_enabled to True.
    r2_upload_enabled: bool = False
    r2_bucket: str = ""
    r2_endpoint: str = ""  # https://<account_id>.r2.cloudflarestorage.com

    # Development toggles — leave False for production
    draw_overlay: bool = False   # burn bounding boxes and track IDs into saved clips
    show_preview: bool = False   # display a live preview window while running

    # Debug frames — print a low-res annotated frame to the terminal every N seconds.
    # Rendered inline using the iTerm2 image protocol; set to 0 to disable.
    snapshot_interval_seconds: float = 10.0
    snapshot_resolution: tuple[int, int] = (640, 360)
    snapshot_quality: int = 60  # JPEG quality 0–100; lower = smaller terminal output


config = Config()
