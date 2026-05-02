# Boat Detector

Records a clip for each tracked boat detected in a video stream. Uses [YOLOv11](https://docs.ultralytics.com/) for detection and BoT-SORT for per-object tracking. Optimised for Apple Silicon via the MPS backend.

## Requirements

- Python 3.11+
- A Sony a6700 or a7rV connected in USB Streaming mode, **or** an HDMI capture card (e.g. Elgato Cam Link 4K)
- macOS on Intel or Apple Silicon (M-series)

## Installation

```bash
# Clone / navigate to the project
cd boat-detector

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The first run will download `yolo11s.pt` (~22 MB) automatically.

## Camera setup

### Option A — USB Streaming (simplest)

1. On the camera: **Menu → Network → USB Streaming → Enable**
2. Connect the camera to your Mac via USB-C
3. The camera appears as a UVC webcam at device index `0`

### Option B — HDMI capture card (better image quality)

1. Run an HDMI cable from the camera to your capture card
2. Connect the capture card to your Mac via USB
3. The card appears as a UVC device — if index `0` is taken by another webcam, set `camera_index = 1` in `config.py`

## Running

### Live mode

```bash
python main.py
```

Stop with `Ctrl+C`. Any open clips are flushed to disk before exit.

### Test mode

Feed in a pre-recorded video file instead of a live camera. Useful for tuning settings without needing the camera present.

```bash
python main.py --input /path/to/footage.mp4
# or
python main.py -i footage.mp4
```

Processing runs as fast as the machine allows. Clips are written to the same output directory as live mode.

## Output

Clips are saved under `clips/` (configurable) with one subdirectory per tracked object:

```
clips/
└── 2024-06-01/
    ├── track_3_143022/
    │   ├── clip.mp4
    │   └── metadata.json
    └── track_7_143041/
        ├── clip.mp4
        └── metadata.json
```

Each clip includes a 10-second pre-buffer before the first detection, plus all frames until the boat has been absent for `track_loss_timeout_seconds`.

### metadata.json

```json
{
  "track_id": 3,
  "started_at": "2024-06-01T14:30:22.100000+00:00",
  "ended_at": "2024-06-01T14:31:05.800000+00:00",
  "duration_seconds": 43.7,
  "frame_count": 1311,
  "fps": 30.0,
  "resolution": [1920, 1080],
  "clip_path": "clip.mp4",
  "detected_classes": ["boat"],
  "detection_count": 1244,
  "confidence_min": 0.5021,
  "confidence_max": 0.9431,
  "confidence_mean": 0.7803,
  "error": null
}
```

`error` is `null` on success. If a clip fails to open or a frame write fails, the error message is recorded here.

## Configuration

All settings are in `config.py`.

| Setting | Default | Description |
|---|---|---|
| `camera_index` | `0` | UVC device index |
| `output_dir` | `clips` | Root directory for saved clips |
| `output_resolution` | `(1920, 1080)` | Frames are downsampled to this before recording |
| `video_codec` | `mp4v` | `avc1` gives smaller H.264 files on macOS |
| `model_name` | `yolo11s.pt` | YOLO model — `n` is faster, `m`/`l` are more accurate |
| `confidence_threshold` | `0.5` | Detections below this are ignored |
| `target_classes` | `[8]` | COCO class 8 = boat. Set to `None` to detect everything |
| `device` | `cpu` | `cpu` for Intel Macs; `mps` for Apple Silicon (M-series) |
| `pre_buffer_seconds` | `10.0` | Seconds of footage captured before the first detection |
| `track_loss_timeout_seconds` | `5.0` | **Likely needs tuning.** Seconds a boat can be absent before its clip is closed. Too short → fragmented clips. Too long → separate boats merged into one clip. |
| `draw_overlay` | `False` | Burn bounding boxes and track IDs into saved clips |
| `show_preview` | `False` | Show a live preview window (press `q` to quit) |

## Troubleshooting

**Camera not found**
Ensure the camera is in USB Streaming mode before connecting. Try `camera_index = 1` if another webcam occupies index `0`.

**Slow inference / dropped frames**
Switch to the nano model (`yolo11n.pt`) or lower `output_resolution`. On an Intel Mac, `yolo11n.pt` is recommended for live camera use. On Apple Silicon (`device = "mps"`), `yolo11s.pt` should comfortably handle 1080p30.

**No boats detected in test footage**
Lower `confidence_threshold` to `0.3` as a starting point. If the footage is aerial or contains unusual vessel types, consider fine-tuning the model on the [SeaDroneSee dataset](https://seadronessee.cs.uni-tuebingen.de/).

**Clips are fragmented (boat disappears and reappears)**
Increase `track_loss_timeout_seconds`. Start with `10` and adjust from there.
