from collections import deque
from pathlib import Path

import cv2
import numpy as np

from config import config


class CameraCapture:
    def __init__(self, source: int | str | Path | None = None) -> None:
        """Open a camera or a video file.

        Args:
            source: Device index (int) for a live camera, or a file path (str/Path)
                    for test/file mode. Defaults to config.camera_index.
        """
        if source is None:
            source = config.camera_index

        self._source = source
        self.cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)

        if not self.cap.isOpened():
            if isinstance(source, int):
                raise RuntimeError(
                    f"Cannot open camera at index {source}. "
                    "Check that the Sony camera is connected in USB Streaming mode, "
                    "or that your capture card is recognized by the OS."
                )
            raise RuntimeError(f"Cannot open video file: {source}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Sony cameras over USB streaming sometimes report 0 — default to 30
        self.fps: float = raw_fps if raw_fps > 0 else 30.0

        pre_buffer_frames = int(self.fps * config.pre_buffer_seconds)
        self._buffer: deque[np.ndarray] = deque(maxlen=pre_buffer_frames)

    def pre_buffer_snapshot(self) -> list[np.ndarray]:
        """Return the current pre-buffer contents, oldest first.

        Call this BEFORE camera.read() so the snapshot excludes the frame
        about to be written — avoids writing the detection frame twice.
        """
        return list(self._buffer)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        frame = cv2.resize(frame, config.output_resolution)
        self._buffer.append(frame)
        return True, frame

    def release(self) -> None:
        self.cap.release()
