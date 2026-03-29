"""
Tier 1 – Data Loader
WebcamLoader: Singleton that owns the camera capture.

Uses ffmpeg piped to a background thread so the main loop always gets
the LATEST frame with no latency, and the FaceTime HD Camera is selected
by name (not by index) so the iPhone Continuity Camera is never used.

If ffmpeg is unavailable or the named camera is not found, falls back to
OpenCV VideoCapture with a numeric camera index.
"""

import os
import re
import subprocess
import threading
import time
import cv2
import numpy as np


def _av_index_by_name(name: str) -> str | None:
    """Return AVFoundation index for the named video device, else None."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    in_video_section = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video_section:
            continue
        m = re.search(r'\[(\d+)\]\s+(.+)', line)
        if m and name.lower() in m.group(2).lower():
            return m.group(1)
    return None


class WebcamLoader:
    _instance = None

    CAMERA_NAME  = os.getenv("GESTURE_CAMERA_NAME", "FaceTime HD Camera")
    CAMERA_INDEX = int(os.getenv("GESTURE_CAMERA_INDEX", "0"))
    WIDTH       = 1280
    HEIGHT      = 720
    FPS         = 30

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._latest_frame = None
            instance._lock         = threading.Lock()
            instance._running      = True
            instance._backend = None

            av_idx = _av_index_by_name(cls.CAMERA_NAME)
            if av_idx is not None:
                print(f"[WebcamLoader] '{cls.CAMERA_NAME}' → AVFoundation index {av_idx}")
                frame_bytes = cls.WIDTH * cls.HEIGHT * 3
                proc = subprocess.Popen([
                    "ffmpeg",
                    "-fflags", "nobuffer",          # disable input buffering
                    "-flags",  "low_delay",          # minimise decoder delay
                    "-f",      "avfoundation",
                    "-framerate", str(cls.FPS),
                    "-video_size", f"{cls.WIDTH}x{cls.HEIGHT}",
                    "-i",      av_idx,
                    "-vf",     f"scale={cls.WIDTH}:{cls.HEIGHT}",
                    "-pix_fmt","bgr24",
                    "-f",      "rawvideo",
                    "-"
                ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

                instance._proc = proc
                instance._backend = "ffmpeg"

                # Background thread: continuously reads frames from ffmpeg pipe
                # and keeps only the latest one → zero queue latency for caller
                def _reader():
                    if proc.stdout is None:
                        return
                    while instance._running:
                        raw = proc.stdout.read(frame_bytes)
                        if len(raw) < frame_bytes:
                            break
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                            (cls.HEIGHT, cls.WIDTH, 3)).copy()
                        with instance._lock:
                            instance._latest_frame = frame

                t = threading.Thread(target=_reader, daemon=True)
                t.start()
                instance._thread = t
            else:
                print(
                    "[WebcamLoader] ffmpeg camera selection unavailable. "
                    f"Falling back to OpenCV camera index {cls.CAMERA_INDEX}."
                )
                cap = cv2.VideoCapture(cls.CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cls.WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cls.HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, cls.FPS)
                if not cap.isOpened():
                    raise RuntimeError(
                        "Could not open camera. Check macOS Camera permission, "
                        "and/or set GESTURE_CAMERA_NAME or GESTURE_CAMERA_INDEX."
                    )
                instance._cap = cap
                instance._backend = "opencv"

            # Wait until first frame arrives
            print("[WebcamLoader] Waiting for first frame...")
            for _ in range(100):
                frame = instance.get_frame()
                if frame is not None:
                    with instance._lock:
                        instance._latest_frame = frame
                    break
                time.sleep(0.05)
            if instance._latest_frame is None:
                raise RuntimeError("Camera opened but no frames were received.")
            print("[WebcamLoader] Camera ready.")

            cls._instance = instance
        return cls._instance

    def get_frame(self):
        """Returns the latest captured frame (BGR numpy array)."""
        if self._backend == "opencv":
            ok, frame = self._cap.read()
            if not ok:
                return None
            return frame
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def release(self):
        self._running = False
        if getattr(self, "_backend", None) == "ffmpeg":
            self._proc.terminate()
        elif getattr(self, "_backend", None) == "opencv":
            self._cap.release()
        WebcamLoader._instance = None
