"""Air Canvas: draw in the air with your index finger.

Gestures
  index finger up            draw with the current tool
  index + middle fingers up  move / select from the toolbars (hover to pick)
  anything else              lift the pen

Run `python main.py --help` for options; press H in the app for all controls.
"""
import argparse
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

import ui
from cameras import default_camera_index, list_cameras
from canvas import Canvas, render_shape
from filters import PointFilter
from hand_tracker import (GESTURE_LABELS, HAND_CONNECTIONS, Gesture,
                          GestureStabilizer, HandTracker)

WINDOW = "Air Canvas"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawings")

COLORS = [  # BGR
    ("Red", (60, 60, 235)),
    ("Orange", (40, 150, 255)),
    ("Yellow", (50, 225, 255)),
    ("Green", (90, 205, 70)),
    ("Cyan", (230, 210, 50)),
    ("Blue", (235, 120, 45)),
    ("Purple", (215, 80, 165)),
    ("White", (245, 245, 245)),
]
TOOLS = [
    ("brush", "Brush", ui.icon_brush),
    ("line", "Line", ui.icon_line),
    ("rect", "Rectangle", ui.icon_rect),
    ("circle", "Circle", ui.icon_circle),
]
TOOL_LABELS = {key: label for key, label, _ in TOOLS}
BACKGROUNDS =["Camera", "Dimmed", "Whiteboard", "Blackboard"]

MIN_SIZE, MAX_SIZE = 4, 60
ERASER_SCALE = 3
HAND_LOST_GRACE = 0.25   # seconds a lost hand may reappear without ending the stroke
HELP_ON_START = 6.0


class AirCanvasApp:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.canvas = Canvas(width, height)
        self.cursor_filter = PointFilter(min_cutoff=1.5, beta=0.01)
        self.stabilizer = GestureStabilizer(frames=3)
        self.toast = ui.Toast()

        self.color_idx = 5
        self.tool = "brush"
        self.erasing = False
        self.size = 14
        self.bg_idx = 0
        self.show_skeleton = False
        self.help_until = time.monotonic() + HELP_ON_START
        self.help_pinned = False

        self.gesture = Gesture.NONE
        self.hand = None
        self.cursor = None          # smoothed index-tip position (int tuple)
        self.last_seen = 0.0
        self.stroking = False
        self.prev_pt = None         # last freehand point
        self.shape_start = None
        self.fps = 0.0
        self._last_frame_t = None
        self._last_base = None

        self._build_ui()

    # --- layout -------------------------------------------------------------
    def _build_ui(self):
        w, h = self.w, self.h
        s = min(w / 1280, h / 720)
        self.top_h = int(100 * s)
        self.side_w = int(104 * s)

        r = int(28 * s)
        n = len(COLORS) + 1
        spacing = int(88 * s)
        x0, cy = w // 2 - spacing * (n - 1) // 2, self.top_h // 2
        self.palette = []
        for i, (name, bgr) in enumerate(COLORS):
            self.palette.append(ui.Button(
                (x0 + i * spacing, cy), r, tooltip=name, fill=bgr,
                on_select=lambda i=i: self.select_color(i),
                is_active=lambda i=i: not self.erasing and self.color_idx == i))
        self.palette.append(ui.Button(
            (x0 + (n - 1) * spacing, cy), r, tooltip="Eraser", icon=ui.icon_eraser,
            label_color=(40, 40, 40), fill=(235, 235, 235),
            on_select=self.select_eraser, is_active=lambda: self.erasing))

        lx = self.side_w // 2
        tr = int(30 * s)
        step = int(78 * s)
        ty = self.top_h + int(56 * s)
        self.tool_buttons = [
            ui.Button((lx, ty + i * step), tr, tooltip=label, icon=icon, tip_side="right",
                      on_select=lambda k=key: self.select_tool(k),
                      is_active=lambda k=key: not self.erasing and self.tool == k)
            for i, (key, label, icon) in enumerate(TOOLS)
        ]
        slider_top = ty + len(TOOLS) * step + int(30 * s)
        self.slider = ui.Slider(lx, slider_top, h - int(110 * s), MIN_SIZE, MAX_SIZE,
                                lambda: self.size, self.set_size)
        self.left_panel = ((int(10 * s), self.top_h + int(14 * s)),
                           (self.side_w - int(6 * s), h - int(60 * s)))

        rx = w - self.side_w // 2
        actions = [("UNDO", "Undo", self.undo, 0.45), ("REDO", "Redo", self.redo, 0.45),
                   ("SAVE", "Save PNG", self.save, 0.6), ("CLEAR", "Clear all", self.clear, 0.9)]
        self.action_buttons = [
            ui.Button((rx, ty + i * step), tr, label=label, tooltip=tip, dwell=dwell,
                      tip_side="left", on_select=fn,
                      fill=(55, 55, 150) if label == "CLEAR" else ui.BUTTON)
            for i, (label, tip, fn, dwell) in enumerate(actions)
        ]
        self.right_panel = ((w - self.side_w + int(6 * s), self.top_h + int(14 * s)),
                            (w - int(10 * s), ty + (len(actions) - 1) * step + int(56 * s)))
        self.buttons = self.palette + self.tool_buttons + self.action_buttons

    def in_ui(self, pt):
        x, y = pt
        if y < self.top_h:
            return True
        for (x1, y1), (x2, y2) in (self.left_panel, self.right_panel):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    # --- actions ------------------------------------------------------------
    @property
    def ink(self):
        return COLORS[self.color_idx][1]

    @property
    def stroke_width(self):
        return self.size * ERASER_SCALE if self.erasing else self.size

    def select_color(self, i):
        self.color_idx, self.erasing = i, False

    def select_eraser(self):
        self.erasing = True

    def select_tool(self, key):
        self.tool, self.erasing = key, False

    def set_size(self, value):
        self.size = int(min(MAX_SIZE, max(MIN_SIZE, value)))

    def undo(self):
        self._end_stroke()
        self._notify("Undo" if self.canvas.undo() else "Nothing to undo")

    def redo(self):
        self._end_stroke()
        self._notify("Redo" if self.canvas.redo() else "Nothing to redo")

    def clear(self):
        self._end_stroke()
        self._notify("Canvas cleared  (Z to undo)" if self.canvas.clear() else "Canvas is already empty")

    def save(self):
        if self._last_base is None or self.canvas.is_empty:
            self._notify("Nothing to save yet")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"air_canvas_{stamp}"
        cv2.imwrite(os.path.join(SAVE_DIR, name + ".png"), self.canvas.compose(self._last_base))
        cv2.imwrite(os.path.join(SAVE_DIR, name + "_transparent.png"), self.canvas.to_bgra())
        self._notify(f"Saved drawings/{name}.png")

    def cycle_background(self):
        self.bg_idx = (self.bg_idx + 1) % len(BACKGROUNDS)
        self._notify(f"Background: {BACKGROUNDS[self.bg_idx]}")

    def cycle_tool(self):
        keys = [t[0] for t in TOOLS]
        self.select_tool(keys[(keys.index(self.tool) + 1) % len(keys)])
        self._notify(f"Tool: {TOOL_LABELS[self.tool]}")

    def _notify(self, msg):
        self.toast.show(msg, time.monotonic())

    # --- per-frame logic -----------------------------------------------------
    def update(self, hand, now):
        """Advance the interaction state from this frame's hand detection."""
        if hand is not None:
            self.hand, self.last_seen = hand, now
            fx, fy = self.cursor_filter(hand.index_tip, now)
            self.cursor = (int(fx), int(fy))
            self.gesture = self.stabilizer.update(hand.gesture)
        elif now - self.last_seen > HAND_LOST_GRACE:
            self._end_stroke()
            self.hand, self.cursor = None, None
            self.cursor_filter.reset()
            self.stabilizer.force(Gesture.NONE)
            self.gesture = Gesture.NONE
        else:
            return  # brief dropout: hold state, don't extend the stroke

        hovering = self.gesture == Gesture.HOVER
        pointer = self.cursor if hovering else None
        for b in self.buttons:
            b.update(pointer, now)
        self.slider.update(pointer)

        if self.gesture == Gesture.DRAW:
            self._continue_stroke(self.cursor)
        else:
            self._end_stroke()

    def _continue_stroke(self, pt):
        if not self.stroking:
            if self.in_ui(pt):
                return  # don't start strokes on the toolbars
            self.stroking = True
            self.help_until = 0  # first stroke dismisses the startup help
            self.prev_pt = pt
            self.shape_start = pt
            if self.erasing or self.tool == "brush":
                self.canvas.checkpoint()
            return

        if self.erasing or self.tool == "brush":
            if self.in_ui(pt):
                self.prev_pt = None
                return
            if self.prev_pt is not None:
                if self.erasing:
                    self.canvas.erase(self.prev_pt, pt, self.stroke_width)
                else:
                    self.canvas.stroke(self.prev_pt, pt, self.ink, self.stroke_width)
            self.prev_pt = pt

    def _end_stroke(self):
        if not self.stroking:
            return
        self.stroking = False
        if not self.erasing and self.tool != "brush" and self.cursor and self.shape_start:
            if math.dist(self.shape_start, self.cursor) > 6:
                self.canvas.checkpoint()
                self.canvas.shape(self.tool, self.shape_start, self.cursor, self.ink, self.size)
        self.prev_pt = self.shape_start = None

    # --- rendering -----------------------------------------------------------
    def _background(self, frame):
        mode = BACKGROUNDS[self.bg_idx]
        if mode == "Camera":
            return frame
        if mode == "Dimmed":
            return cv2.convertScaleAbs(frame, alpha=0.35)
        shade = 245 if mode == "Whiteboard" else 24
        return np.full_like(frame, shade)

    def render(self, frame, now):
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else self.fps * 0.9 + 0.1 / dt
        self._last_frame_t = now

        base = self._background(frame)
        self._last_base = base
        out = self.canvas.compose(base)

        board = BACKGROUNDS[self.bg_idx] in ("Whiteboard", "Blackboard")
        if self.hand is not None and (self.show_skeleton or board):
            self._draw_skeleton(out, board)
        if self.stroking and not self.erasing and self.tool != "brush" and self.cursor:
            render_shape(out, self.tool, self.shape_start, self.cursor, self.ink, self.size)

        self._draw_ui(out, now)
        self._draw_cursor(out)

        if self.help_pinned or now < self.help_until:
            ui.draw_help(out)
        self.toast.draw(out, now, (self.w // 2, self.h - 110))
        return out

    def _draw_skeleton(self, img, board):
        pts = self.hand.points.astype(int)
        color = (150, 150, 150) if board else (220, 220, 220)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, ui.AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, ui.ACCENT, -1, ui.AA)

    def _draw_cursor(self, img):
        if self.cursor is None:
            return
        c = self.cursor
        if self.gesture == Gesture.HOVER:
            cv2.circle(img, c, 11, ui.WHITE, -1, ui.AA)
            cv2.circle(img, c, 15, (40, 40, 40), 2, ui.AA)
            if self.hand is not None:
                mid = tuple(self.hand.points[12].astype(int))
                cv2.line(img, c, mid, ui.WHITE, 2, ui.AA)
                cv2.circle(img, mid, 7, (200, 200, 200), -1, ui.AA)
        elif self.gesture == Gesture.DRAW:
            radius = max(3, self.stroke_width // 2)
            if self.erasing:
                cv2.circle(img, c, radius, ui.WHITE, 2, ui.AA)
                cv2.circle(img, c, radius, (40, 40, 40), 1, ui.AA)
            else:
                cv2.circle(img, c, radius + 2, ui.WHITE, 2, ui.AA)
                cv2.circle(img, c, max(2, radius - 1), self.ink, -1, ui.AA)
        else:
            cv2.circle(img, c, 8, (200, 200, 200), 2, ui.AA)

    def _draw_ui(self, img, now):
        ui.glass_panel(img, (-30, -30), (self.w + 30, self.top_h), radius=30)
        ui.glass_panel(img, *self.left_panel)
        ui.glass_panel(img, *self.right_panel)
        for b in self.buttons:
            b.draw(img, now)
        self.slider.draw(img, (235, 235, 235) if self.erasing else self.ink)

        tool = "Eraser" if self.erasing else TOOL_LABELS[self.tool]
        status = (f"{GESTURE_LABELS[self.gesture]}   |   {tool}  {self.stroke_width}px   |   "
                  f"{BACKGROUNDS[self.bg_idx]}   |   {self.fps:4.1f} fps   |   H for help")
        ui.pill(img, status, (self.w // 2, self.h - 34), 0.48)

    # --- keyboard -------------------------------------------------------------
    def handle_key(self, key):
        """Return False when the app should quit."""
        if key in (ord("q"), 27):
            return False
        if key == -1 or key == 255:
            return True
        if self.help_until > time.monotonic():
            self.help_until = 0  # any key dismisses the startup help

        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch == "h":
            self.help_pinned = not self.help_pinned
        elif ch == "z" or ch == "u":
            self.undo()
        elif ch == "y" or ch == "r":
            self.redo()
        elif ch == "c":
            self.clear()
        elif ch == "s":
            self.save()
        elif ch == "b":
            self.cycle_background()
        elif ch == "t":
            self.cycle_tool()
        elif ch == "e":
            self.select_eraser()
        elif ch == "k":
            self.show_skeleton = not self.show_skeleton
        elif ch in ("[", "-"):
            self.set_size(self.size - 2)
        elif ch in ("]", "=", "+"):
            self.set_size(self.size + 2)
        elif ch.isdigit() and 1 <= int(ch) <= len(COLORS):
            self.select_color(int(ch) - 1)
        return True


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame
    if not cap.isOpened():
        return None, None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None
    return cap, frame


def main():
    parser = argparse.ArgumentParser(description="Draw in the air with hand tracking.")
    parser.add_argument("--camera", type=int, default=None,
                        help="webcam index (default: the built-in camera)")
    parser.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true", help="don't flip the camera image")
    args = parser.parse_args()

    if args.list_cameras:
        found = list_cameras()
        if not found:
            print("Camera names unavailable (macOS only; needs pyobjc-framework-AVFoundation).")
        for index, name, builtin in found:
            print(f"  {index}: {name}{'  (built-in, default)' if builtin else ''}")
        return

    camera = default_camera_index() if args.camera is None else args.camera
    names = {i: n for i, n, _ in list_cameras()}
    print(f"Using camera {camera}: {names.get(camera, 'unknown')}  (change with --camera N)")

    tracker = HandTracker()
    cap, frame = open_camera(camera, args.width, args.height)
    if cap is None:
        print("Error: could not open the webcam. On macOS, allow camera access for your "
              "terminal in System Settings > Privacy & Security > Camera.")
        return

    h, w = frame.shape[:2]
    app = AirCanvasApp(w, h)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w, h)
    fullscreen = False
    t0 = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            app.update(tracker.process(frame, (now - t0) * 1000), now)
            cv2.imshow(WINDOW, app.render(frame, now))

            key = cv2.waitKey(1)
            if key != -1 and chr(key & 0xFF).lower() == "f":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif not app.handle_key(key & 0xFF if key != -1 else -1):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the mouse
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
