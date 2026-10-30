"""Drawing surface: a color layer + alpha mask with undo/redo history."""
import math

import cv2
import numpy as np


def render_shape(img, kind, p0, p1, color, thickness, line_type=cv2.LINE_AA):
    if kind == "line":
        cv2.line(img, p0, p1, color, thickness, line_type)
    elif kind == "rect":
        cv2.rectangle(img, p0, p1, color, thickness, line_type)
    elif kind == "circle":
        radius = int(round(math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        cv2.circle(img, p0, radius, color, thickness, line_type)


class Canvas:
    """Strokes live in a separate color layer plus an anti-aliased alpha mask,
    so any color (including black) can be drawn and edges blend smoothly onto
    whatever background is shown."""

    def __init__(self, width, height, history=25):
        self.width, self.height = width, height
        self.color = np.zeros((height, width, 3), np.uint8)
        self.alpha = np.zeros((height, width), np.uint8)
        self.history = history
        self._undo, self._redo = [], []

    # --- history ---------------------------------------------------------
    def _state(self):
        return self.color.copy(), self.alpha.copy()

    def checkpoint(self):
        """Save the current state; call before every change."""
        self._undo.append(self._state())
        if len(self._undo) > self.history:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._state())
        self.color, self.alpha = self._undo.pop()
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._state())
        self.color, self.alpha = self._redo.pop()
        return True

    @property
    def is_empty(self):
        return cv2.countNonZero(self.alpha) == 0

    def clear(self):
        if self.is_empty:
            return False
        self.checkpoint()
        self.color[:] = 0
        self.alpha[:] = 0
        return True

    # --- drawing ---------------------------------------------------------
    # The color layer is painted slightly wider and without AA so edge pixels
    # keep the pure stroke color; the AA alpha mask defines the visible shape.
    def stroke(self, p0, p1, color, thickness):
        cv2.line(self.color, p0, p1, color, thickness + 2, cv2.LINE_8)
        cv2.line(self.alpha, p0, p1, 255, thickness, cv2.LINE_AA)

    def erase(self, p0, p1, thickness):
        cv2.line(self.alpha, p0, p1, 0, thickness, cv2.LINE_AA)

    def shape(self, kind, p0, p1, color, thickness):
        render_shape(self.color, kind, p0, p1, color, thickness + 2, cv2.LINE_8)
        render_shape(self.alpha, kind, p0, p1, 255, thickness, cv2.LINE_AA)

    # --- output ----------------------------------------------------------
    def compose(self, background):
        """Alpha-blend the drawing over `background` (only within the drawn area)."""
        out = background.copy()
        x, y, w, h = cv2.boundingRect(self.alpha)
        if w == 0 or h == 0:
            return out
        a = self.alpha[y:y + h, x:x + w, None].astype(np.uint16)
        fg = self.color[y:y + h, x:x + w].astype(np.uint16)
        bg = out[y:y + h, x:x + w].astype(np.uint16)
        out[y:y + h, x:x + w] = ((fg * a + bg * (255 - a) + 127) // 255).astype(np.uint8)
        return out

    def to_bgra(self):
        """Drawing only, with a transparent background."""
        return np.dstack([self.color, self.alpha])
