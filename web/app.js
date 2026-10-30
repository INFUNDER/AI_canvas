// Air Canvas (browser version): MediaPipe hand tracking + canvas drawing.
import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/" +
  "hand_landmarker/float16/1/hand_landmarker.task";

const COLORS = [
  ["Red", "#eb3c3c"], ["Orange", "#ff9628"], ["Yellow", "#ffe132"], ["Green", "#46cd5a"],
  ["Cyan", "#32d2e6"], ["Blue", "#2d78eb"], ["Purple", "#a550d7"], ["White", "#f5f5f5"],
];
const TOOL_LABELS = { brush: "Brush", line: "Line", rect: "Rectangle", circle: "Circle" };
const BACKGROUNDS = ["Camera", "Dimmed", "Whiteboard", "Blackboard"];
const MIN_SIZE = 4, MAX_SIZE = 60, ERASER_SCALE = 3;
const HISTORY = 20;
const HAND_LOST_GRACE = 0.25;  // seconds a lost hand may reappear without ending the stroke
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const Gesture = { NONE: "No hand", IDLE: "Paused", DRAW: "Drawing", HOVER: "Selecting" };

const $ = (sel) => document.querySelector(sel);
const stage = $("#stage"), video = $("#video");
const bg = $("#bg").getContext("2d");
const ink = $("#ink").getContext("2d", { willReadFrequently: true });
const fx = $("#fx").getContext("2d");
const sizeInput = $("#size");

let W = 1280, H = 720;

// --- smoothing -------------------------------------------------------------
// One Euro filter (Casiez et al., 2012): strong smoothing when slow, little lag when fast.
class OneEuro {
  constructor(minCutoff = 1.5, beta = 0.01, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.reset();
  }
  reset() { this.x = this.dx = this.t = null; }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(x, t) {
    if (this.t === null) { this.x = x; this.dx = 0; this.t = t; return x; }
    const dt = Math.max(t - this.t, 1e-6);
    this.t = t;
    const a = OneEuro.alpha(this.dCutoff, dt);
    this.dx = a * ((x - this.x) / dt) + (1 - a) * this.dx;
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    const b = OneEuro.alpha(cutoff, dt);
    this.x = b * x + (1 - b) * this.x;
    return this.x;
  }
}

// Only switch gestures after the new one holds for a few frames.
class GestureStabilizer {
  constructor(frames = 3) { this.frames = frames; this.force(Gesture.NONE); }
  force(g) { this.current = g; this.candidate = null; this.count = 0; }
  update(g) {
    if (g === this.current) { this.candidate = null; this.count = 0; }
    else if (g === this.candidate) { if (++this.count >= this.frames) this.force(g); }
    else { this.candidate = g; this.count = 1; }
    return this.current;
  }
}

// --- hand geometry -----------------------------------------------------------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

// Rotation-invariant: a finger is up when its tip is clearly farther from the wrist than its PIP joint.
function fingersUp(p) {
  const thumb = dist(p[4], p[17]) > dist(p[2], p[17]) * 1.2;
  const others = [[8, 6], [12, 10], [16, 14], [20, 18]].map(([tip, pip]) =>
    dist(p[tip], p[0]) > dist(p[pip], p[0]) * 1.1);
  return [thumb, ...others];
}

function classify([, index, middle, ring]) {
  if (index && !middle && !ring) return Gesture.DRAW;
  if (index && middle && !ring) return Gesture.HOVER;
  return Gesture.IDLE;
}

// --- state -------------------------------------------------------------------
const state = {
  colorIdx: 5, tool: "brush", erasing: false, size: 14, bgIdx: 0, skeleton: false,
  gesture: Gesture.NONE, hand: null, cursor: null, lastSeen: -Infinity,
};
const filterX = new OneEuro(), filterY = new OneEuro();
const stabilizer = new GestureStabilizer(3);
const stroke = { active: false, source: null, pts: [], start: null, end: null };
const undoStack = [], redoStack = [];

const inkColor = () => COLORS[state.colorIdx][1];
const strokeWidth = () => state.erasing ? state.size * ERASER_SCALE : state.size;
const freehand = () => state.erasing || state.tool === "brush";

// --- coordinates -------------------------------------------------------------
function toClient(p) {
  const r = stage.getBoundingClientRect();
  return { x: r.left + (p.x / W) * r.width, y: r.top + (p.y / H) * r.height };
}
function toCanvas(clientX, clientY) {
  const r = stage.getBoundingClientRect();
  return { x: ((clientX - r.left) / r.width) * W, y: ((clientY - r.top) / r.height) * H };
}
const uiRects = () => [...document.querySelectorAll(".bar")].map((el) => el.getBoundingClientRect());
function inUI(p) {
  const c = toClient(p);
  return uiRects().some((r) => c.x >= r.left && c.x <= r.right && c.y >= r.top && c.y <= r.bottom);
}

// --- history -----------------------------------------------------------------
function checkpoint() {
  undoStack.push(ink.getImageData(0, 0, W, H));
  if (undoStack.length > HISTORY) undoStack.shift();
  redoStack.length = 0;
}
function restore(from, to) {
  if (!from.length) return false;
  to.push(ink.getImageData(0, 0, W, H));
  ink.putImageData(from.pop(), 0, 0);
  return true;
}
function isEmpty() {
  const d = ink.getImageData(0, 0, W, H).data;
  for (let i = 3; i < d.length; i += 4) if (d[i]) return false;
  return true;
}

// --- drawing -----------------------------------------------------------------
function styleFor(ctx, width, color) {
  ctx.lineWidth = width;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.strokeStyle = ctx.fillStyle = color;
}

function drawShape(ctx, kind, a, b) {
  ctx.beginPath();
  if (kind === "line") { ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); }
  else if (kind === "rect") ctx.rect(a.x, a.y, b.x - a.x, b.y - a.y);
  else if (kind === "circle") ctx.arc(a.x, a.y, dist(a, b), 0, Math.PI * 2);
  ctx.stroke();
}

// Freehand segments are drawn as quadratic curves through midpoints for smooth lines.
function drawFreehand(final = false) {
  const pts = stroke.pts, n = pts.length;
  if (!n) return;
  ink.save();
  styleFor(ink, strokeWidth(), inkColor());
  if (state.erasing) ink.globalCompositeOperation = "destination-out";
  ink.beginPath();
  if (final) {
    if (n < 3) { ink.restore(); return; }
    const m = mid(pts[n - 2], pts[n - 1]);
    ink.moveTo(m.x, m.y); ink.lineTo(pts[n - 1].x, pts[n - 1].y); ink.stroke();
  } else if (n === 1) {
    ink.arc(pts[0].x, pts[0].y, strokeWidth() / 2, 0, Math.PI * 2); ink.fill();
  } else if (n === 2) {
    ink.moveTo(pts[0].x, pts[0].y); ink.lineTo(pts[1].x, pts[1].y); ink.stroke();
  } else {
    const [a, b, c] = pts.slice(-3), m1 = mid(a, b), m2 = mid(b, c);
    ink.moveTo(m1.x, m1.y); ink.quadraticCurveTo(b.x, b.y, m2.x, m2.y); ink.stroke();
  }
  ink.restore();
}

function beginStroke(p, source) {
  if (stroke.active) return;
  if (source === "hand" && inUI(p)) return;  // don't start strokes on the toolbars
  hideIntroHelp();
  Object.assign(stroke, { active: true, source, pts: [], start: p, end: p });
  if (freehand()) { checkpoint(); stroke.pts.push(p); drawFreehand(); }
}

function extendStroke(p) {
  if (!stroke.active) return;
  stroke.end = p;
  if (!freehand()) return;
  if (stroke.source === "hand" && inUI(p)) { drawFreehand(true); stroke.pts = []; return; }
  const last = stroke.pts[stroke.pts.length - 1];
  if (last && dist(last, p) < 1) return;
  stroke.pts.push(p);
  drawFreehand();
}

function endStroke() {
  if (!stroke.active) return;
  stroke.active = false;
  if (freehand()) drawFreehand(true);
  else if (dist(stroke.start, stroke.end) > 6) {
    checkpoint();
    styleFor(ink, state.size, inkColor());
    drawShape(ink, state.tool, stroke.start, stroke.end);
  }
  stroke.pts = [];
}

// --- actions -----------------------------------------------------------------
let toastTimer;
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2200);
}

const actions = {
  undo() { endStroke(); toast(restore(undoStack, redoStack) ? "Undo" : "Nothing to undo"); },
  redo() { endStroke(); toast(restore(redoStack, undoStack) ? "Redo" : "Nothing to redo"); },
  clear() {
    endStroke();
    if (isEmpty()) return toast("Canvas is already empty");
    checkpoint();
    ink.clearRect(0, 0, W, H);
    toast("Canvas cleared  (Z to undo)");
  },
  save() {
    if (isEmpty()) return toast("Nothing to save yet");
    const out = Object.assign(document.createElement("canvas"), { width: W, height: H });
    const ctx = out.getContext("2d");
    ctx.drawImage(bg.canvas, 0, 0);
    ctx.drawImage(ink.canvas, 0, 0);
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
    out.toBlob((blob) => {
      const a = Object.assign(document.createElement("a"),
        { href: URL.createObjectURL(blob), download: `air_canvas_${stamp}.png` });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      toast("Saved PNG");
    });
  },
};

function selectColor(i) { state.colorIdx = i; state.erasing = false; refreshUI(); }
function selectEraser() { state.erasing = true; refreshUI(); }
function selectTool(t) { state.tool = t; state.erasing = false; refreshUI(); }
function setSize(v) {
  state.size = Math.round(Math.min(MAX_SIZE, Math.max(MIN_SIZE, v)));
  sizeInput.value = state.size;
  refreshUI();
}
function cycleBackground() {
  state.bgIdx = (state.bgIdx + 1) % BACKGROUNDS.length;
  toast(`Background: ${BACKGROUNDS[state.bgIdx]}`);
  refreshUI();
}
function cycleTool() {
  const keys = Object.keys(TOOL_LABELS);
  selectTool(keys[(keys.indexOf(state.tool) + 1) % keys.length]);
  toast(`Tool: ${TOOL_LABELS[state.tool]}`);
}

// --- UI ----------------------------------------------------------------------
function buildPalette() {
  const bar = $("#palette");
  COLORS.forEach(([name, hex], i) => {
    const b = Object.assign(document.createElement("button"),
      { className: "btn swatch", ariaLabel: name });
    Object.assign(b.dataset, { tip: name, dwell: 350, color: i });
    b.style.setProperty("--c", hex);
    b.onclick = () => selectColor(i);
    bar.append(b);
  });
  const eraser = Object.assign(document.createElement("button"),
    { className: "btn eraser", ariaLabel: "Eraser", onclick: selectEraser });
  Object.assign(eraser.dataset, { tip: "Eraser", dwell: 350, eraser: "" });
  eraser.innerHTML = '<svg viewBox="0 0 24 24"><path d="m14 4 6 6-9 9H6l-3-3z" fill="#f0c8d2"/><path d="M8 19h12"/></svg>';
  bar.append(eraser);
}

function refreshUI() {
  document.querySelectorAll("[data-color]").forEach((b) =>
    b.classList.toggle("active", !state.erasing && +b.dataset.color === state.colorIdx));
  document.querySelector("[data-eraser]").classList.toggle("active", state.erasing);
  document.querySelectorAll("[data-tool]").forEach((b) =>
    b.classList.toggle("active", !state.erasing && b.dataset.tool === state.tool));
  $("#size-out").textContent = `${state.size}px`;
  $("#st-tool").textContent = `${state.erasing ? "Eraser" : TOOL_LABELS[state.tool]} ${strokeWidth()}px`;
  $("#st-bg").textContent = BACKGROUNDS[state.bgIdx];
}

// Dwell-to-select: hover a button with two fingers until its ring fills.
const dwell = { el: null, since: 0, latched: false };
function updateDwell(clientPt, now) {
  let best = null, bestD = Infinity;
  if (clientPt) {
    for (const el of document.querySelectorAll("[data-dwell]")) {
      const r = el.getBoundingClientRect();
      const d = Math.hypot(clientPt.x - (r.left + r.width / 2), clientPt.y - (r.top + r.height / 2));
      if (d < r.width * 0.65 && d < bestD) { best = el; bestD = d; }
    }
  }
  if (best !== dwell.el) {
    dwell.el?.classList.remove("hover");
    dwell.el?.style.setProperty("--p", 0);
    Object.assign(dwell, { el: best, since: now, latched: false });
    best?.classList.add("hover");
  }
  if (!dwell.el || dwell.latched) return;
  const p = Math.min(1, (now - dwell.since) / (+dwell.el.dataset.dwell / 1000));
  dwell.el.style.setProperty("--p", p);
  if (p >= 1) {
    dwell.latched = true;  // must leave the button before it can fire again
    dwell.el.style.setProperty("--p", 0);
    dwell.el.click();
    dwell.el.classList.remove("flash");
    void dwell.el.offsetWidth;
    dwell.el.classList.add("flash");
  }
}

function updateSliderHover(clientPt) {
  const r = sizeInput.getBoundingClientRect();
  const hit = clientPt && Math.abs(clientPt.x - (r.left + r.width / 2)) < r.width * 1.5 &&
    clientPt.y >= r.top - 10 && clientPt.y <= r.bottom + 10;
  sizeInput.classList.toggle("hover", !!hit);
  if (hit) {
    const frac = Math.min(1, Math.max(0, (clientPt.y - r.top) / r.height));
    setSize(MAX_SIZE - frac * (MAX_SIZE - MIN_SIZE));
  }
}

// --- per-frame hand logic --------------------------------------------------------
function onHand(landmarks, t) {
  if (landmarks) {
    const pts = landmarks.map((lm) => ({ x: (1 - lm.x) * W, y: lm.y * H }));  // mirrored
    const fingers = fingersUp(pts);
    state.hand = pts;
    state.lastSeen = t;
    state.cursor = { x: filterX.filter(pts[8].x, t), y: filterY.filter(pts[8].y, t) };
    state.gesture = stabilizer.update(classify(fingers));
  } else if (t - state.lastSeen > HAND_LOST_GRACE) {
    if (stroke.source === "hand") endStroke();
    state.hand = state.cursor = null;
    filterX.reset(); filterY.reset();
    stabilizer.force(Gesture.NONE);
    state.gesture = Gesture.NONE;
  } else {
    return;  // brief dropout: hold state, don't extend the stroke
  }

  const pointer = state.gesture === Gesture.HOVER ? toClient(state.cursor) : null;
  updateDwell(pointer, t);
  updateSliderHover(pointer);

  if (state.gesture === Gesture.DRAW) {
    if (!stroke.active) beginStroke(state.cursor, "hand");
    else if (stroke.source === "hand") extendStroke(state.cursor);
  } else if (stroke.source === "hand") {
    endStroke();
  }
}

// --- rendering ---------------------------------------------------------------
function drawBackground() {
  const mode = BACKGROUNDS[state.bgIdx];
  if (mode === "Whiteboard" || mode === "Blackboard") {
    bg.fillStyle = mode === "Whiteboard" ? "#f5f5f5" : "#181818";
    bg.fillRect(0, 0, W, H);
    return;
  }
  bg.save();
  bg.translate(W, 0);
  bg.scale(-1, 1);
  bg.drawImage(video, 0, 0, W, H);
  bg.restore();
  if (mode === "Dimmed") { bg.fillStyle = "rgba(0,0,0,0.65)"; bg.fillRect(0, 0, W, H); }
}

function renderFx() {
  fx.clearRect(0, 0, W, H);
  const board = state.bgIdx >= 2;

  if (state.hand && (state.skeleton || board)) {
    fx.lineWidth = 2;
    fx.strokeStyle = board ? "#969696" : "#dcdcdc";
    fx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      fx.moveTo(state.hand[a].x, state.hand[a].y);
      fx.lineTo(state.hand[b].x, state.hand[b].y);
    }
    fx.stroke();
    fx.fillStyle = "#46beff";
    for (const p of state.hand) { fx.beginPath(); fx.arc(p.x, p.y, 4, 0, Math.PI * 2); fx.fill(); }
  }

  if (stroke.active && !freehand()) {
    styleFor(fx, state.size, inkColor());
    drawShape(fx, state.tool, stroke.start, stroke.end);
  }

  const c = state.cursor;
  if (!c) return;
  const circle = (r, color, fill, width = 2) => {
    fx.beginPath(); fx.arc(c.x, c.y, r, 0, Math.PI * 2);
    if (fill) { fx.fillStyle = color; fx.fill(); } else { fx.lineWidth = width; fx.strokeStyle = color; fx.stroke(); }
  };
  if (state.gesture === Gesture.HOVER) {
    const m = state.hand[12];
    fx.lineWidth = 2; fx.strokeStyle = "#fff";
    fx.beginPath(); fx.moveTo(c.x, c.y); fx.lineTo(m.x, m.y); fx.stroke();
    fx.fillStyle = "#c8c8c8"; fx.beginPath(); fx.arc(m.x, m.y, 7, 0, Math.PI * 2); fx.fill();
    circle(11, "#fff", true);
    circle(15, "#282828", false);
  } else if (state.gesture === Gesture.DRAW) {
    const r = Math.max(3, strokeWidth() / 2);
    if (state.erasing) { circle(r, "#fff", false); circle(r, "#282828", false, 1); }
    else { circle(r + 2, "#fff", false); circle(Math.max(2, r - 1), inkColor(), true); }
  } else {
    circle(8, "#c8c8c8", false);
  }
}

let fpsAvg = 0, lastDetect = 0, lastStatus = 0;
function updateStatus(now) {
  if (now - lastStatus < 0.25) return;
  lastStatus = now;
  $("#st-gesture").textContent = state.gesture;
  $("#st-fps").textContent = `${fpsAvg.toFixed(0)} fps`;
}

// --- camera ------------------------------------------------------------------
let landmarker = null, stream = null, running = false;
const isPhone = (label) => /iphone|continuity|desk view/i.test(label);
const isBuiltin = (label) => /facetime|built-in|integrated/i.test(label);

// Resizing a canvas wipes it, so the drawing is copied over and rescaled.
function resizeCanvases(w, h) {
  if (ink.canvas.width === w && ink.canvas.height === h) return;
  const old = Object.assign(document.createElement("canvas"),
    { width: ink.canvas.width, height: ink.canvas.height });
  old.getContext("2d").drawImage(ink.canvas, 0, 0);
  W = w; H = h;
  for (const ctx of [bg, ink, fx]) Object.assign(ctx.canvas, { width: W, height: H });
  ink.drawImage(old, 0, 0, W, H);
  undoStack.length = redoStack.length = 0;
  fitStage();
}

function fitStage() {
  const ar = W / H;
  const width = Math.min(innerWidth, innerHeight * ar);
  stage.style.width = `${width}px`;
  stage.style.aspectRatio = `${W} / ${H}`;
}

async function startCamera(deviceId) {
  stream?.getTracks().forEach((t) => t.stop());
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: deviceId ? { deviceId: { exact: deviceId }, ...size } : size,
  });
  video.srcObject = stream;
  await video.play();

  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const track = stream.getVideoTracks()[0];
  // Prefer the laptop's built-in camera over an iPhone connected via Continuity Camera.
  if (!deviceId && isPhone(track.label)) {
    const builtin = cams.find((d) => isBuiltin(d.label) && !isPhone(d.label));
    if (builtin) return startCamera(builtin.deviceId);
  }

  const select = $("#camera-select");
  const current = track.getSettings().deviceId;
  select.replaceChildren(...cams.map((d, i) =>
    new Option(d.label || `Camera ${i + 1}`, d.deviceId, false, d.deviceId === current)));
  try { localStorage.setItem("airCanvasCamera", current); } catch {}

  resizeCanvases(video.videoWidth || 1280, video.videoHeight || 720);
}

function loop(ms) {
  requestAnimationFrame(loop);
  if (!running) return;
  const now = ms / 1000;
  if (video.readyState >= 2) {
    if (state.bgIdx < 2) drawBackground();
    if (video.currentTime !== loop.lastVideoTime) {
      loop.lastVideoTime = video.currentTime;
      const res = landmarker.detectForVideo(video, ms);
      onHand(res.landmarks?.[0], now);
      if (lastDetect) fpsAvg = fpsAvg ? fpsAvg * 0.9 + 0.1 / (now - lastDetect) : 1 / (now - lastDetect);
      lastDetect = now;
    }
  }
  if (state.bgIdx >= 2) drawBackground();
  renderFx();
  updateStatus(now);
}

// --- input -------------------------------------------------------------------
function toggleHelp(force) { $("#help").classList.toggle("hidden", force === undefined ? undefined : !force); }
function hideIntroHelp() { if (!toggleHelp.pinned) toggleHelp(false); }

function bindInput() {
  const canvasFx = fx.canvas;
  canvasFx.addEventListener("pointerdown", (e) => {
    canvasFx.setPointerCapture(e.pointerId);
    beginStroke(toCanvas(e.clientX, e.clientY), "pointer");
  });
  canvasFx.addEventListener("pointermove", (e) => {
    if (stroke.source === "pointer") extendStroke(toCanvas(e.clientX, e.clientY));
  });
  for (const type of ["pointerup", "pointercancel"]) {
    canvasFx.addEventListener(type, () => { if (stroke.source === "pointer") endStroke(); });
  }

  document.querySelectorAll("[data-tool]").forEach((b) => (b.onclick = () => selectTool(b.dataset.tool)));
  document.querySelectorAll("[data-action]").forEach((b) => (b.onclick = () => actions[b.dataset.action]()));
  sizeInput.oninput = () => setSize(+sizeInput.value);
  $("#st-bg").onclick = cycleBackground;
  $("#help-btn").onclick = () => { toggleHelp.pinned = true; toggleHelp(true); };
  $("#help").onclick = () => { toggleHelp.pinned = false; toggleHelp(false); };
  $("#camera-select").onchange = (e) => startCamera(e.target.value).catch(cameraError);
  addEventListener("resize", fitStage);

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === "SELECT") return;
    const k = e.key.toLowerCase();
    const handlers = {
      h: () => { toggleHelp.pinned = $("#help").classList.contains("hidden"); toggleHelp(toggleHelp.pinned); },
      z: actions.undo, u: actions.undo, y: actions.redo, r: actions.redo,
      c: actions.clear, s: actions.save, b: cycleBackground, t: cycleTool, e: selectEraser,
      k: () => { state.skeleton = !state.skeleton; },
      f: () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(),
      "[": () => setSize(state.size - 2), "-": () => setSize(state.size - 2),
      "]": () => setSize(state.size + 2), "=": () => setSize(state.size + 2),
    };
    if (handlers[k]) handlers[k]();
    else if (/^[1-8]$/.test(k)) selectColor(+k - 1);
    else return;
    e.preventDefault();
  });
}

function cameraError(err) {
  console.error(err);
  $("#intro").classList.remove("hidden");
  $("#intro-status").textContent = err.name === "NotAllowedError"
    ? "Camera access was blocked. Allow it in your browser's site settings and try again."
    : `Could not start the camera (${err.message}).`;
  $("#start").disabled = false;
  $("#start").textContent = "Try again";
}

async function loadLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  try { return await HandLandmarker.createFromOptions(fileset, options("GPU")); }
  catch { return await HandLandmarker.createFromOptions(fileset, options("CPU")); }
}

async function init() {
  buildPalette();
  bindInput();
  refreshUI();
  resizeCanvases(W, H);
  fitStage();
  requestAnimationFrame(loop);

  const start = $("#start");
  try {
    landmarker = await loadLandmarker();
    start.disabled = false;
    start.textContent = "Start camera";
  } catch (err) {
    console.error(err);
    start.textContent = "Could not load the hand model";
    $("#intro-status").textContent = "Check your connection and reload the page.";
    return;
  }

  start.onclick = async () => {
    start.disabled = true;
    start.textContent = "Starting camera…";
    let saved = null;
    try { saved = localStorage.getItem("airCanvasCamera"); } catch {}
    try {
      await startCamera(saved).catch(() => startCamera());
      $("#intro").classList.add("hidden");
      toggleHelp(true);
      running = true;
    } catch (err) {
      cameraError(err);
    }
  };
}

init();
