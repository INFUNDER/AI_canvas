# Air Canvas

Draw in the air with your index finger using a webcam, MediaPipe hand tracking and OpenCV.

**Try it in your browser:** https://infunder.github.io/AI_canvas/

The web version in [`web/`](web) runs entirely in the browser (MediaPipe Tasks for JavaScript), so no video ever leaves your device. It has the same tools and gestures as the desktop app, and also supports drawing with a mouse or touch and choosing a camera from the status bar.

## Run the desktop app (Python)

```bash
pip install -r requirements.txt
python main.py            # --camera 1, --width 1920 --height 1080, --no-mirror
```

The hand model (`hand_landmarker.task`) downloads automatically on first run.

On macOS the app picks the built-in FaceTime camera by name, so it won't use an iPhone that's connected through Continuity Camera. Run `python main.py --list-cameras` to see every camera, and `--camera N` to choose one.

## Gestures

| Gesture | Action |
| --- | --- |
| Index finger up | Draw with the current tool |
| Index + middle up | Move without drawing; hover a button until its ring fills to pick it |
| Fist / open palm | Lift the pen |

## Features

- **Tools:** freehand brush, straight line, rectangle, circle (live preview while drawing), eraser
- **8 colors** and a **size slider** (hover over it to set 4–60 px)
- **Undo / redo** (25 steps). Clear can also be undone.
- **Save** writes two files to `drawings/`: the composite image, and a transparent PNG of just the drawing
- **Backgrounds:** camera, dimmed camera, whiteboard, blackboard (the hand skeleton is shown on boards)
- **Smooth tracking:** a One Euro filter on the fingertip, VIDEO-mode tracking, a 3-frame gesture debounce and a short grace period when the hand drops out, so strokes don't break or leave stray marks
- Anti-aliased strokes with a proper alpha layer, so any color blends cleanly on any background
- Dwell-to-select buttons, so sweeping across the toolbar doesn't trigger anything by accident

## Keyboard

`Z`/`Y` undo/redo · `C` clear · `S` save · `B` background · `[` `]` size · `1`–`8` colors · `E` eraser · `T` cycle tool · `K` hand skeleton · `F` fullscreen · `H` help · `Q`/`Esc` quit

## Project layout

| File | Purpose |
| --- | --- |
| `main.py` | App loop, interaction state, layout and rendering |
| `hand_tracker.py` | Model download, MediaPipe wrapper, finger and gesture detection |
| `canvas.py` | Color and alpha layers, shapes, undo/redo, compositing |
| `filters.py` | One Euro filter for cursor smoothing |
| `ui.py` | Panels, buttons, slider, toasts, help overlay |
| `cameras.py` | Picks the built-in Mac camera over an iPhone (Continuity Camera) |
| `web/` | Browser version, published with GitHub Pages |

To run the web version locally, serve the folder with `python -m http.server -d web`, then open http://localhost:8000. Browsers only allow camera access on `localhost` or over HTTPS.
