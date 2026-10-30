"""On-screen widgets: glass panels, dwell-to-select buttons, slider, toasts, help."""
import math

import cv2

FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

WHITE = (255, 255, 255)
MUTED = (175, 175, 180)
PANEL = (30, 28, 26)
BUTTON = (62, 58, 55)
ACCENT = (255, 190, 70)  # BGR: warm sky blue


# --- primitives ------------------------------------------------------------
def text_size(text, scale, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0]


def put_text(img, text, org, scale, color=WHITE, thickness=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color, thickness, AA)


def put_text_centered(img, text, center, scale, color=WHITE, thickness=1):
    tw, th = text_size(text, scale, thickness)
    put_text(img, text, (center[0] - tw / 2, center[1] + th / 2), scale, color, thickness)


def rounded_rect(img, p1, p2, color, radius):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    if r:
        for c in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, c, r, color, -1, AA)


def glass_panel(img, p1, p2, color=PANEL, alpha=0.72, radius=18):
    """Translucent rounded panel, blended only inside its bounding box."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(w, int(p2[0])), min(h, int(p2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    rounded_rect(overlay, (p1[0] - x1, p1[1] - y1), (p2[0] - x1, p2[1] - y1), color, radius)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def pill(img, text, center, scale=0.5, color=WHITE, bg=PANEL, alpha=0.8, pad=(14, 9)):
    tw, th = text_size(text, scale)
    cx, cy = center
    glass_panel(img, (cx - tw / 2 - pad[0], cy - th / 2 - pad[1]),
                (cx + tw / 2 + pad[0], cy + th / 2 + pad[1]), bg, alpha, radius=th)
    put_text_centered(img, text, (cx, cy), scale, color)


# --- icons (drawn inside a button of radius r) ------------------------------
def icon_brush(img, c, r, color):
    k = r * 0.45
    pts = [(c[0] - k + i * k / 6, c[1] + math.sin(i / 12 * 2 * math.pi) * k * 0.5)
           for i in range(13)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, 3, AA)


def icon_line(img, c, r, color):
    k = int(r * 0.42)
    cv2.line(img, (c[0] - k, c[1] + k), (c[0] + k, c[1] - k), color, 3, AA)


def icon_rect(img, c, r, color):
    k = int(r * 0.42)
    cv2.rectangle(img, (c[0] - k, c[1] - int(k * 0.75)), (c[0] + k, c[1] + int(k * 0.75)), color, 2, AA)


def icon_circle(img, c, r, color):
    cv2.circle(img, c, int(r * 0.42), color, 2, AA)


def icon_eraser(img, c, r, color):
    k = r * 0.38
    box = cv2.boxPoints(((c[0], c[1]), (k * 2.2, k * 1.2), -35)).astype(int)
    cv2.fillConvexPoly(img, box, (210, 200, 245), AA)
    cv2.polylines(img, [box], True, color, 2, AA)


# --- widgets -----------------------------------------------------------------
class Button:
    """Circular button activated by hovering over it for `dwell` seconds."""

    def __init__(self, center, radius, *, label="", tooltip="", fill=BUTTON,
                 label_color=WHITE, icon=None, dwell=0.35, on_select=None,
                 is_active=None, tip_side="below"):
        self.center = (int(center[0]), int(center[1]))
        self.radius = int(radius)
        self.label, self.tooltip = label, tooltip
        self.fill, self.label_color, self.icon = fill, label_color, icon
        self.dwell = dwell
        self.on_select, self.is_active = on_select, is_active
        self.tip_side = tip_side
        self.hovered = False
        self.progress = 0.0
        self._hover_since = None
        self._latched = False  # must leave the button before it can fire again
        self._flash_until = 0.0

    def hit(self, pt):
        if pt is None:
            return False
        return math.hypot(pt[0] - self.center[0], pt[1] - self.center[1]) <= self.radius * 1.3

    def update(self, pt, now):
        if not self.hit(pt):
            self.hovered, self.progress = False, 0.0
            self._hover_since, self._latched = None, False
            return False
        self.hovered = True
        if self._latched:
            return False
        if self._hover_since is None:
            self._hover_since = now
        self.progress = min(1.0, (now - self._hover_since) / self.dwell) if self.dwell > 0 else 1.0
        if self.progress >= 1.0:
            self._latched = True
            self._flash_until = now + 0.3
            if self.on_select:
                self.on_select()
            return True
        return False

    def draw(self, img, now):
        cx, cy = self.center
        r = int(self.radius * (1.12 if self.hovered else 1.0))
        active = bool(self.is_active and self.is_active())

        cv2.circle(img, (cx, cy + 3), r, (12, 12, 12), -1, AA)  # drop shadow
        cv2.circle(img, (cx, cy), r, self.fill, -1, AA)
        if self.icon:
            self.icon(img, (cx, cy), r, self.label_color)
        elif self.label:
            scale = 0.42 * self.radius / 32
            put_text_centered(img, self.label, (cx, cy), scale, self.label_color)

        if active:
            cv2.circle(img, (cx, cy), r + 6, WHITE, 2, AA)
        if self.hovered and 0 < self.progress < 1:
            cv2.ellipse(img, (cx, cy), (r + 6, r + 6), -90, 0, 360 * self.progress, ACCENT, 4, AA)
        if now < self._flash_until:
            grow = int((1 - (self._flash_until - now) / 0.3) * 10)
            cv2.circle(img, (cx, cy), r + 6 + grow, ACCENT, 2, AA)

        if self.hovered and self.tooltip:
            off = r + 26
            pos = {"below": (cx, cy + off), "right": (cx + off + 30, cy),
                   "left": (cx - off - 30, cy)}[self.tip_side]
            pill(img, self.tooltip, pos, 0.45)


class Slider:
    """Vertical slider; value follows the cursor while hovered (no dwell)."""

    def __init__(self, x, y1, y2, vmin, vmax, get, set_, label="Size"):
        self.x, self.y1, self.y2 = int(x), int(y1), int(y2)
        self.vmin, self.vmax = vmin, vmax
        self.get, self.set = get, set_
        self.label = label
        self.hovered = False

    def hit(self, pt):
        return (pt is not None and abs(pt[0] - self.x) <= 36
                and self.y1 - 14 <= pt[1] <= self.y2 + 14)

    def update(self, pt):
        self.hovered = self.hit(pt)
        if self.hovered:
            frac = (pt[1] - self.y1) / max(1, self.y2 - self.y1)
            frac = min(1.0, max(0.0, frac))
            self.set(int(round(self.vmax - frac * (self.vmax - self.vmin))))

    def draw(self, img, color):
        value = self.get()
        frac = (value - self.vmin) / (self.vmax - self.vmin)
        ky = int(self.y2 - frac * (self.y2 - self.y1))
        cv2.line(img, (self.x, self.y1), (self.x, self.y2), (80, 76, 72), 6, AA)
        cv2.line(img, (self.x, ky), (self.x, self.y2), ACCENT, 6, AA)
        knob = max(6, min(22, value // 2))
        cv2.circle(img, (self.x, ky), knob + 3, WHITE if self.hovered else (40, 40, 40), -1, AA)
        cv2.circle(img, (self.x, ky), knob, color, -1, AA)
        put_text_centered(img, self.label, (self.x, self.y1 - 26), 0.42, MUTED)
        put_text_centered(img, f"{value}px", (self.x, self.y2 + 26), 0.42, WHITE)


class Toast:
    def __init__(self):
        self.message, self.until = "", 0.0

    def show(self, message, now, duration=2.2):
        self.message, self.until = message, now + duration

    def draw(self, img, now, center):
        if now >= self.until:
            return
        fade = min(1.0, (self.until - now) / 0.4)
        pill(img, self.message, center, 0.6, bg=(50, 46, 42), alpha=0.85 * fade, pad=(20, 12))


HELP_LINES = [
    ("GESTURES", None),
    ("Index finger up", "Draw with the current tool"),
    ("Index + middle up", "Move without drawing / select"),
    ("Hover on a button", "Hold until the ring fills to pick it"),
    ("Fist or open palm", "Lift the pen"),
    ("KEYBOARD", None),
    ("Z / Y", "Undo / redo"),
    ("C", "Clear canvas (undoable)"),
    ("S", "Save PNG to drawings/"),
    ("B", "Cycle background"),
    ("[ / ]", "Brush size"),
    ("1-8  E  T", "Colors, eraser, cycle tool"),
    ("K / F", "Hand skeleton / fullscreen"),
    ("H / Q", "Toggle help / quit"),
]


def draw_help(img):
    h, w = img.shape[:2]
    pw, ph = 620, 40 + 34 * len(HELP_LINES)
    x1, y1 = (w - pw) // 2, (h - ph) // 2
    glass_panel(img, (x1, y1), (x1 + pw, y1 + ph), (22, 20, 18), 0.88, 22)
    y = y1 + 40
    for left, right in HELP_LINES:
        if right is None:
            put_text(img, left, (x1 + 32, y), 0.5, ACCENT)
        else:
            put_text(img, left, (x1 + 48, y), 0.52, WHITE)
            put_text(img, right, (x1 + 270, y), 0.52, MUTED)
        y += 34
