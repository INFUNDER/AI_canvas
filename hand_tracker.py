"""MediaPipe hand tracking + finger-state / gesture classification."""
import os
import subprocess
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MIN_MODEL_BYTES = 7_000_000

# Landmark indices
WRIST = 0
TIPS = (4, 8, 12, 16, 20)
PIPS = (3, 6, 10, 14, 18)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class Gesture:
    NONE = "none"    # no hand in view
    IDLE = "idle"    # hand visible, pen lifted
    DRAW = "draw"    # index finger only
    HOVER = "hover"  # index + middle: move / select


GESTURE_LABELS = {
    Gesture.NONE: "No hand",
    Gesture.IDLE: "Paused",
    Gesture.DRAW: "Drawing",
    Gesture.HOVER: "Selecting",
}


@dataclass
class HandState:
    points: np.ndarray       # (21, 2) pixel coordinates
    fingers: tuple           # (thumb, index, middle, ring, pinky) booleans
    gesture: str             # raw (unstabilised) gesture for this frame

    @property
    def index_tip(self):
        return self.points[8]


def ensure_model(path=MODEL_PATH):
    """Download the hand landmark model if it is missing or truncated."""
    if os.path.exists(path) and os.path.getsize(path) >= MIN_MODEL_BYTES:
        return path

    print("Downloading hand landmark model (~7.8 MB)...")
    tmp = path + ".part"
    try:
        def progress(blocks, block_size, total):
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                print(f"\r  {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, tmp, progress)
        print()
    except Exception as exc:  # e.g. missing SSL certs on macOS python.org builds
        print(f"\n  urllib failed ({exc}); retrying with curl...")
        subprocess.run(["curl", "-L", "--fail", "-o", tmp, MODEL_URL], check=True)

    if os.path.getsize(tmp) < MIN_MODEL_BYTES:
        os.remove(tmp)
        raise RuntimeError("Model download was incomplete. Check your connection and retry.")
    os.replace(tmp, path)
    return path


def fingers_up(points):
    """Rotation-invariant finger extension test.

    A finger counts as extended when its tip is clearly farther from the wrist
    than its PIP joint. The thumb uses distance from the pinky base instead,
    since it folds across the palm rather than toward the wrist.
    """
    def dist(a, b):
        return float(np.hypot(*(points[a] - points[b])))

    thumb = dist(4, 17) > dist(2, 17) * 1.2
    others = tuple(dist(tip, WRIST) > dist(pip, WRIST) * 1.1
                   for tip, pip in zip(TIPS[1:], PIPS[1:]))
    return (thumb,) + others


def classify(fingers):
    _, index, middle, ring, _ = fingers
    if index and not middle and not ring:
        return Gesture.DRAW
    if index and middle and not ring:
        return Gesture.HOVER
    return Gesture.IDLE


class GestureStabilizer:
    """Only switch gestures after the new one has held for `frames` frames.

    Prevents single-frame misclassifications from breaking strokes or
    leaving stray marks when moving between drawing and selecting.
    """

    def __init__(self, frames=3):
        self.frames = frames
        self.current = Gesture.NONE
        self._candidate = None
        self._count = 0

    def update(self, gesture):
        if gesture == self.current:
            self._candidate, self._count = None, 0
        elif gesture == self._candidate:
            self._count += 1
            if self._count >= self.frames:
                self.current, self._candidate, self._count = gesture, None, 0
        else:
            self._candidate, self._count = gesture, 1
            if self.frames <= 1:
                self.current = gesture
        return self.current

    def force(self, gesture):
        self.current, self._candidate, self._count = gesture, None, 0


class HandTracker:
    def __init__(self, confidence=0.6):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms):
        """Return a HandState for the most prominent hand, or None."""
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.hand_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        points = np.array([(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]],
                          dtype=np.float32)
        fingers = fingers_up(points)
        return HandState(points=points, fingers=fingers, gesture=classify(fingers))

    def close(self):
        self._detector.close()
