from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Camera — index 0 is usually the first connected UVC device
    camera_index: int = 0

    # Output
    output_dir: Path = Path("clips")
    output_resolution: tuple[int, int] = (1920, 1080)
    # "mp4v" is safe on all platforms; swap to "avc1" on macOS for H.264 (smaller files)
    video_codec: str = "mp4v"

    # Detection
    # COCO class 8 = "boat" — set to None to track all detected object classes
    model_name: str = "yolo11s.pt"
    confidence_threshold: float = 0.35
    target_classes: Optional[list[int]] = field(default_factory=lambda: [8])
    # "cpu" for Intel; "mps" for Apple Silicon (M-series)
    device: str = "cpu"

    # Recording
    pre_buffer_seconds: float = 10.0
    # NOTE: 5 seconds is a starting point and will likely need tuning per deployment.
    # Too short → clips fragment when detection briefly drops (waves, glare, occlusion).
    # Too long  → clips merge when two boats pass close together or one lingers off-frame.
    track_loss_timeout_seconds: float = 5.0

    # Development toggles — leave False for production
    draw_overlay: bool = False   # burn bounding boxes and track IDs into saved clips
    show_preview: bool = False   # display a live preview window while running

    # Debug frames — print a low-res annotated frame to the terminal every N seconds.
    # Rendered inline using the iTerm2 image protocol; set to 0 to disable.
    snapshot_interval_seconds: float = 10.0
    snapshot_resolution: tuple[int, int] = (640, 360)
    snapshot_quality: int = 60  # JPEG quality 0–100; lower = smaller terminal output


config = Config()
