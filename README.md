# Boat Detector

Records a clip for each tracked boat detected in a live or recorded video stream. Uses [YOLOv11](https://docs.ultralytics.com/) for detection and BoT-SORT for multi-object tracking. Designed for a fixed riverside camera — one clip per vessel, with a pre-event buffer and automatic upload to Cloudflare R2.

## Requirements

- Python 3.11+
- macOS on Intel or Apple Silicon (M-series)
- A PoE IP camera accessible over RTSP (e.g. Reolink RLC-810A), **or** a pre-recorded video file for test mode

## Installation

```bash
cd boat-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads `yolo11s.pt` (~22 MB) automatically.

## Camera setup

### Live mode — RTSP IP camera

Connect the camera to your local network via a PoE switch and note its IP address. The RTSP URL format for Reolink cameras is:

```
rtsp://<user>:<password>@<camera-ip>:554/h264Preview_01_main
```

Run:

```bash
python main.py --camera rtsp://admin:yourpass@192.168.1.100:554/h264Preview_01_main
```

### Test mode — video file

Feed a pre-recorded clip to tune settings without the camera present. Clips are written to a new numbered directory under `test_clips/` so runs don't overwrite each other.

```bash
python main.py --input /path/to/footage.mp4
```

Analyse the output against a known boat count:

```bash
python analyze_run.py test_clips/run_001 --expected 2
```

## Running

```bash
# Live camera (UVC device index 0)
python main.py

# Live camera (RTSP)
python main.py --camera rtsp://admin:pass@192.168.1.100:554/h264Preview_01_main

# Test file
python main.py --input footage.mp4

# Test file with periodic background frame prints
python main.py --input footage.mp4 --background-frames
```

Stop with `Ctrl+C`. Any open clips are flushed to disk before exit.

## Output

Each detected vessel gets its own subdirectory containing a video clip, a thumbnail, and metadata:

```
clips/
└── cam1/
    └── 2026-05-02/
        ├── summary.json    — daily activity summary for this camera
        └── track_6_012714/
            ├── clip.mp4        — full recording with pre-event buffer
            ├── thumb.jpg       — frame from the first detection
            ├── metadata.json
            └── ais.json        — nearby AIS vessels at detection time (if enabled)
```

### metadata.json

```json
{
  "camera_id": "cam1",
  "track_id": 6,
  "started_at": "2026-05-02T01:27:14.192345+00:00",
  "ended_at": "2026-05-02T01:29:18.723945+00:00",
  "duration_seconds": 124.532,
  "frame_count": 1895,
  "fps": 30.56,
  "resolution": [1920, 1080],
  "clip_path": "clip.mp4",
  "detected_classes": ["boat"],
  "detection_count": 80,
  "confidence_min": 0.3505,
  "confidence_max": 0.6430,
  "confidence_mean": 0.4329,
  "x_range_px": 545.4,
  "error": null
}
```

`x_range_px` is the horizontal spread of the vessel's centroid across its lifetime — used to filter static false positives (piers, docks). `error` is `null` on success.

## Configuration

All settings are in `config.py`.

### Detection

| Setting | Default | Description |
|---|---|---|
| `model_name` | `yolo11s.pt` | YOLO model variant — `n` is faster, `m`/`l` more accurate |
| `confidence_threshold` | `0.1` | Pre-filter before BoT-SORT. Keep low so the tracker's second-stage matcher sees weak detections |
| `target_classes` | `[8]` | COCO class 8 = boat. `None` detects all classes |
| `device` | `cpu` | `cpu` for Intel; `mps` for Apple Silicon |
| `inference_every_n_frames` | `6` | Run YOLO every N frames — all frames still written to clips. Higher values improve CPU performance at the cost of detection responsiveness |
| `max_detection_area_fraction` | `0.10` | Discard detections whose bounding box exceeds this fraction of the frame area. Filters foreground structures (piers, trees) misclassified as boats |

### Tracking

| Setting | Default | Description |
|---|---|---|
| `track_loss_timeout_seconds` | `30.0` | Seconds a vessel can be absent (occluded, low confidence) before its clip is closed |
| `track_merge_proximity` | `0.25` | Fraction of frame diagonal. A new BoT-SORT track ID appearing this close to a lost track is treated as the same vessel |
| `edge_entry_zone` | `0.15` | Fraction of frame width from each edge. New tracks appearing here are always treated as new vessel entries, never merged into an existing clip |
| `min_track_displacement_px` | `100.0` | Minimum horizontal centroid range (pixels) across a track's lifetime. Clips below this threshold are discarded as static objects |

### Recording

| Setting | Default | Description |
|---|---|---|
| `pre_buffer_seconds` | `20.0` | Seconds of footage prepended to each clip before the first detection |
| `output_dir` | `clips` | Root directory for saved clips |
| `output_resolution` | `(1920, 1080)` | Recording resolution |
| `video_codec` | `mp4v` | `avc1` gives smaller H.264 files on macOS |

### AIS vessel enrichment

When a new vessel is detected, boat-detector can query [AISstream.io](https://aisstream.io) for any AIS-transmitting vessels near the camera. Sign up for a free API key, then add it to `.env` (copy `.env.example` as a starting point):

```bash
AIS_API_KEY=your_key_here
```

 Results are written to `ais.json` in the clip directory and uploaded to R2 alongside the clip.

| Setting | Default | Description |
|---|---|---|
| `ais_enabled` | `False` | Set to `True` to enable AIS queries |
| `ais_api_key` | env `AIS_API_KEY` | AISstream.io API key — set in `.env`, not `config.py` |
| `camera_latitude` | `0.0` | Camera latitude in decimal degrees |
| `camera_longitude` | `0.0` | Camera longitude in decimal degrees |
| `ais_bounding_box_deg` | `0.05` | Half-width of the query bounding box in degrees (~5.5 km at mid-latitudes) |
| `ais_query_seconds` | `30.0` | How long to collect AIS messages per detection event |

#### ais.json format

```json
{
  "query_at": "2026-05-02T01:27:14+00:00",
  "camera_lat": 44.2374,
  "camera_lon": -69.7626,
  "bounding_box_deg": 0.05,
  "vessels": [
    {
      "mmsi": "338234567",
      "name": "KENNEBEC PILOT",
      "ship_type": 52,
      "call_sign": "WDG2345",
      "destination": "BATH",
      "lat": 44.239,
      "lon": -69.761,
      "speed_knots": 8.4,
      "course": 270.0,
      "heading": 268,
      "nav_status": 0,
      "timestamp": "2026-05-02T01:27:12Z"
    }
  ]
}
```

AIS only covers vessels that carry an AIS transponder — commercially required for vessels over 300 GT and passenger vessels. Recreational boats, kayaks, and small craft typically do not transmit AIS. If the clip's vessel does not appear in `ais.json`, it was either out of transponder range or not required to carry one.

> **Option B — RTL-SDR local receiver:** For offline or low-latency AIS decoding without a cloud API, see the companion project [`kennebec-ais-catcher`](../kennebec-ais-catcher). It runs an RTL-SDR USB dongle (~$25) through [AIS-catcher](https://github.com/jvde-github/AIS-catcher) to decode AIS on VHF 161.975 / 162.025 MHz, then serves a REST API on the local network. Set `ais_local_url = "http://127.0.0.1:8080/vessels"` in this project's `config.py` to use it instead of AISstream.io — no `ais_api_key` required. This works entirely offline with zero per-message cost, but requires a VHF antenna with line-of-sight to the river (~10–20 NM range for Class A transponders).

### Notifications

Push a notification once per new track ID (new vessel detection). Two providers are supported.

| Setting | Default | Description |
|---|---|---|
| `notify_enabled` | `False` | Set to `True` to enable notifications |
| `notify_provider` | `"ntfy"` | `"ntfy"` or `"webhook"` |
| `notify_ntfy_url` | `""` | ntfy topic URL, e.g. `https://ntfy.sh/<your-secret-topic>` |
| `notify_webhook_url` | `""` | Webhook endpoint URL (Zapier, Make.com, custom) |
| `notify_min_confidence` | `0.35` | Skip notification if all detections are below this confidence |

**ntfy** — free push notifications, no account required. Pick a secret topic name (treat it like a password), set `notify_ntfy_url = "https://ntfy.sh/<your-topic>"` in `config.py`, and subscribe to the same topic in the [ntfy mobile app](https://ntfy.sh).

**webhook** — set `notify_webhook_url` to any HTTP endpoint. The POST body is the clip's `metadata.json` payload.

### Upload

| Setting | Default | Description |
|---|---|---|
| `r2_upload_enabled` | `False` | Set to `True` to upload clips to Cloudflare R2 after finalization |
| `r2_bucket` | `""` | R2 bucket name |
| `r2_endpoint` | `""` | `https://<account_id>.r2.cloudflarestorage.com` |

R2 credentials are read from environment variables — do not put them in `config.py`. Add them to `.env`:

```bash
R2_ACCESS_KEY=your_key_id
R2_SECRET_KEY=your_secret
```

### Debug

| Setting | Default | Description |
|---|---|---|
| `draw_overlay` | `False` | Burn bounding boxes and track IDs into saved clips |
| `show_preview` | `False` | Display a live preview window (press `q` to quit) |
| `snapshot_interval_seconds` | `10.0` | Seconds between background terminal frame prints (requires `--background-frames`) |

## Apple Silicon

Change `device` to `mps` in `config.py` for GPU-accelerated inference on M-series Macs. You can also lower `inference_every_n_frames` to `2` or `3` for more responsive detection:

```python
device: str = "mps"
inference_every_n_frames: int = 3
```

## Troubleshooting

**No boats detected**
Lower `confidence_threshold` to `0.05` and check that `target_classes = [8]` matches your vessel type. Run in test mode with `--background-frames` to see what the model is detecting.

**Clips are fragmented (same vessel produces multiple clips)**
Increase `track_loss_timeout_seconds`. Also check `tracker.yaml` — `track_buffer` should be at least `timeout × fps` (e.g. 900 at 30fps for a 30s timeout).

**Static objects saved as clips**
Lower `max_detection_area_fraction` or raise `min_track_displacement_px`. The displacement filter is the stronger signal: a pier never moves, so its x-range stays near zero.

**Slow inference on Intel Mac**
Switch to the nano model (`yolo11n.pt`) or increase `inference_every_n_frames`. Live camera use on Intel is viable at `inference_every_n_frames = 6` with `yolo11n.pt`.

**RTSP stream drops or lags**
The stream uses `cv2.CAP_FFMPEG` with a buffer size of 1 to minimise latency. If drops are frequent, check network throughput between the camera and Mac — the main stream at 4K/8MP can be 8–16 Mbps.
