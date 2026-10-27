/* SensorLine Live: the forms that start a run, and the screen that draws the history the server
   keeps; the page never asks the server to compute a window again.
   Stream coordinates: p = sensor (0 at the top), t = model sample. */
'use strict';

const SPAN_WINDOWS = 5; // windows of signal shown in the strip
const LIVE_FPS = 30; // redraw cap
const POLL_MS = 250;
const EVENT_ROWS = 50;
const FEED_TYPES = new Set(['counted', 'track_rejected', 'merged', 'miss', 'track_confirmed']);

const $ = (id) => document.getElementById(id);
const state = {
  view: null,
  meta: null,
  mode: null,
  version: -1,
  windows: [],
  columns: new Map(),
  trackLog: new Map(),
  events: [],
  cursor: -1,
  cursorF: 0,
  playing: false,
  follow: true,
  speed: 1,
  status: null,
  detail: null,
  detailK: -1,
  dirty: true,
  fps: 0,
  frames: 0,
  fpsAt: performance.now(),
  lastTick: 0,
  entering: false,
};

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function trackColor(id) {
  return `hsl(${(id * 137.508) % 360} 70% 55%)`;
}
const RAMP = (() => {
  // Perceptual dark-to-bright ramp (viridis-like), 256 RGB entries.
  const stops = [
    [13, 8, 135],
    [84, 2, 163],
    [139, 10, 165],
    [185, 50, 137],
    [219, 92, 104],
    [244, 136, 73],
    [254, 188, 43],
    [240, 249, 33]
  ];
  const ramp = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255 * (stops.length - 1),
      a = Math.floor(x),
      b = Math.min(a + 1, stops.length - 1),
      f = x - a;
    for (let c = 0; c < 3; c++) ramp[i * 3 + c] = stops[a][c] + (stops[b][c] - stops[a][c]) * f;
  }
  ramp[0] = 12;
  ramp[1] = 12;
  ramp[2] = 20;
  return ramp;
})();

async function getJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}
async function postJSON(url, value) {
  const response = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value)
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function decode(base64) {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

function reset() {
  Object.assign(state, {
    windows: [], columns: new Map(), trackLog: new Map(), events: [], detail: null, detailK: -1, dirty: true
  });
}

function addWindow(window) {
  state.windows.push(window);
  const sensors = state.meta.sensors;
  if (window.columns) {
    const bytes = decode(window.columns);
    for (let i = 0; i * sensors < bytes.length; i++)
      state.columns.set(window.columns_start + i, bytes.subarray(i * sensors, (i + 1) * sensors));
  }
  for (const row of window.tracks) {
    const id = row[0];
    if (!state.trackLog.has(id)) state.trackLog.set(id, []);
    const log = state.trackLog.get(id);
    if (log.length && log[log.length - 1].k === window.k) log[log.length - 1].row = row;
    else log.push({ k: window.k, row });
  }
  for (const event of window.events) {
    state.events.push({ k: window.k, t: event[0], type: event[1], id: event[2], payload: event[3] });
  }
}

function showView(name) {
  if (state.view === name) return;
  state.view = name;
  $('start').hidden = name !== 'start';
  $('screen').hidden = name !== 'screen';
  $('new-run').hidden = name !== 'screen';
  if (name === 'start') {
    $('title').textContent = '';
    setRunState('idle', 'Choose a recording and the weights');
    loadForm();
  }
  state.dirty = true;
}

async function enterScreen(status) {
  if (state.entering) return;
  state.entering = true;
  try {
    const meta = await getJSON('/api/meta');
    const sameRecording = state.view === 'screen' && state.meta && state.meta.recording === meta.recording
      && meta.mode === 'replay';
    state.meta = meta;
    state.mode = meta.mode;
    state.version = status.version;
    reset();
    if (!sameRecording) state.cursor = -1;
    $('title').textContent = `${meta.recording} · ${meta.model} · W ${meta.width}, S ${meta.stride} · ` +
      `${meta.match_mode}`;
    $('tau').value = meta.score_thr;
    $('nms').value = meta.nms_iou;
    $('tau-out').textContent = Number(meta.score_thr).toFixed(2);
    $('nms-out').textContent = Number(meta.nms_iou).toFixed(2);
    if (meta.mode === 'live') {
      state.follow = true;
      $('follow').checked = true;
      $('follow-label').hidden = false;
      state.playing = !status.paused;
      $('speed').value = String(status.speed);
      state.speed = status.speed;
      $('apply').disabled = true;
      $('apply-note').textContent = 'Changing thresholds for the whole recording works in a replay.';
    } else {
      state.follow = false;
      $('follow').checked = false;
      $('follow-label').hidden = true;
      state.playing = false;
      $('apply').disabled = false;
      if (!sameRecording) $('apply-note').textContent = '';
    }
    $('play').textContent = state.playing ? 'Pause' : 'Play';
    drawBookmarks();
    showView('screen');
  } finally {
    state.entering = false;
  }
}

async function poll() {
  try {
    const status = await getJSON('/api/status');
    state.status = status;
    if (!status.active) {
      showView('start');
    } else {
      if (state.view !== 'screen' || status.version !== state.version) await enterScreen(status);
      while (state.windows.length < status.produced) {
        const page = await getJSON(`/api/windows?from=${state.windows.length}`);
        if (page.version !== state.version || !page.windows.length) break;
        page.windows.forEach(addWindow);
      }
      if (state.cursor < 0 && state.windows.length) {
        // #k=123 in the address opens that window.
        const asked = Number((location.hash.match(/k=(\d+)/) || [])[1]);
        setCursor(Number.isFinite(asked) ? asked : 0);
      }
      if (state.mode === 'live' && state.follow && state.windows.length) setCursor(state.windows.length - 1);
      state.dirty = true;
    }
  } catch (error) {
    setRunState('failed', `No connection: ${error.message}`);
  }
  setTimeout(poll, POLL_MS);
}

function fillList(id, items) {
  $(id).innerHTML = items.map((item) => `<option value="${item.replace(/"/g, '&quot;')}"></option>`).join('');
}

async function loadForm() {
  try {
    const form = await getJSON('/api/form');
    const fill = (id, value) => {
      if (!$(id).value) $(id).value = value;
    };
    fill('f-csv', form.csv);
    fill('f-weights', form.weights);
    fill('f-run', form.run);
    $('f-rate').value = $('f-rate').value || form.rate;
    $('f-speed').value = String(form.speed);
    $('f-input').value = form.input_kind;
    fillList('recent-csv', form.recent.csv);
    fillList('recent-weights', form.recent.weights);
    fillList('recent-runs', form.recent.runs);
  } catch (error) {
    $('start-error').textContent = error.message;
  }
}

async function submitForm(event, url, body, buttonId, errorId) {
  event.preventDefault();
  const button = $(buttonId),
    label = button.textContent;
  $(errorId).textContent = '';
  button.disabled = true;
  button.textContent = 'Starting…';
  try {
    const status = await postJSON(url, body());
    state.status = status;
    await enterScreen(status);
  } catch (error) {
    $(errorId).textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

/* The file chooser: the server lists folders, the page never sees file contents. */
const chooser = { kind: 'file', ext: '', input: null, isRun: false };

async function openChooser(button) {
  chooser.input = $(button.dataset.target);
  chooser.kind = button.dataset.kind;
  chooser.ext = button.dataset.ext || '';
  $('chooser-title').textContent = button.dataset.title;
  $('chooser-pick').hidden = chooser.kind !== 'run';
  $('chooser-note').textContent = '';
  $('chooser').showModal();
  await browse(chooser.input.value);
}

async function browse(path) {
  try {
    const listing = await getJSON(`/api/browse?path=${encodeURIComponent(path)}&kind=${chooser.kind}&ext=` +
      `${encodeURIComponent(chooser.ext)}`);
    chooser.isRun = listing.is_run;
    $('chooser-path').value = listing.path;
    $('chooser-up').disabled = !listing.parent;
    $('chooser-up').dataset.path = listing.parent || '';
    $('chooser-drives').innerHTML = listing.drives.map((d) => `<option>${d}</option>`).join('');
    const drive = listing.drives.find((d) => listing.path.toLowerCase().startsWith(d.toLowerCase()));
    if (drive) $('chooser-drives').value = drive;
    $('chooser-list').innerHTML = listing.entries.map((entry, i) =>
        `<li tabindex="0" class="${entry.dir ? 'dir' : 'file'}" data-i="${i}">` +
        `<span>${entry.dir ? '▸ ' : ''}${entry.name.replace(/</g, '&lt;')}</span>` +
        `<span class="size">${entry.dir ? '' : formatSize(entry.size)}</span></li>`).join('') ||
      '<li class="empty">Nothing to choose in this folder</li>';
    chooser.entries = listing.entries;
    $('chooser-pick').disabled = chooser.kind === 'run' && !listing.is_run;
    $('chooser-note').textContent = chooser.kind === 'run'
      ? (listing.is_run ? 'This folder is a run.' : 'Open folders until you are in a run folder.')
      : '';
  } catch (error) {
    $('chooser-note').textContent = error.message;
  }
}

function formatSize(bytes) {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`;
  if (bytes >= 1 << 10) return `${Math.round(bytes / (1 << 10))} kB`;
  return `${bytes} B`;
}

function chooseEntry(item) {
  if (!item || item.dataset.i === undefined) return;
  const entry = chooser.entries[Number(item.dataset.i)];
  if (entry.dir) {
    browse(entry.path);
    return;
  }
  chooser.input.value = entry.path;
  $('chooser').close();
}

function setCursor(k) {
  if (!state.windows.length) return;
  const clamped = Math.max(0, Math.min(state.windows.length - 1, k));
  if (clamped !== state.cursor) {
    state.cursor = clamped;
    state.dirty = true;
  }
  state.cursorF = state.cursor;
}

function leaveLive() {
  state.follow = false;
  $('follow').checked = false;
}

function tick() {
  const now = performance.now();
  const elapsed = (now - state.lastTick) / 1000;
  if (state.view !== 'screen' || !state.meta) return;
  if (elapsed >= 1 / LIVE_FPS) {
    state.lastTick = now;
    // A replay plays by moving the cursor. In a live run Play and Pause control the run itself, and the
    // cursor either follows the newest window or stays where the user put it.
    if (state.playing && state.mode === 'replay' && state.windows.length) {
      const perWindow = state.meta.stride / (state.meta.model_rate_hz || 20);
      const step = state.speed ? elapsed * state.speed / perWindow : 4;
      state.cursorF = Math.min(state.cursorF + step, state.windows.length - 1);
      const next = Math.floor(state.cursorF);
      if (next !== state.cursor) {
        state.cursor = next;
        state.dirty = true;
      }
      if (state.cursor === state.windows.length - 1 && state.status && state.status.finished) togglePlay(false);
    }
    if (state.dirty && state.cursor >= 0) {
      draw();
      state.dirty = false;
      state.frames++;
    }
    if (now - state.fpsAt >= 1000) {
      state.fps = state.frames * 1000 / (now - state.fpsAt);
      state.frames = 0;
      state.fpsAt = now;
      if (state.cursor >= 0) drawStatus();
    }
  }
}

async function togglePlay(play = !state.playing) {
  state.playing = play;
  $('play').textContent = play ? 'Pause' : 'Play';
  if (state.mode === 'live') {
    try {
      await postJSON('/api/control', { action: play ? 'play' : 'pause' });
    } catch (error) {
      console.warn(error);
    }
  }
}

function trackAt(id, k) {
  const log = state.trackLog.get(id);
  let low = 0,
    high = log.length - 1,
    found = null;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (log[middle].k <= k) {
      found = log[middle];
      low = middle + 1;
    } else high = middle - 1;
  }
  return found;
}

function tracksAt(k, from, to) {
  const shown = [];
  for (const id of state.trackLog.keys()) {
    const entry = trackAt(id, k);
    if (!entry) continue;
    const row = entry.row;
    if (row[5] <= from || row[3] >= to) continue;
    shown.push({
      id: row[0], state: row[1], p1: row[2], t1: row[3], p2: row[4], t2: row[5],
      hits: row[6], misses: row[7], score: row[8], cuts: row[9], merged: row[10]
    });
  }
  return shown;
}

function fitCanvas(canvas, cssHeight) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(200, Math.floor(canvas.clientWidth * ratio));
  const height = Math.floor((cssHeight || canvas.clientHeight) * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function draw() {
  if (!state.meta || state.cursor < 0) return;
  drawStrip();
  drawWindow();
  drawCounter();
  drawTracks();
  drawEvents();
  drawStatus();
}

function drawStrip() {
  const canvas = $('strip');
  const ratio = fitCanvas(canvas, 300);
  const context = canvas.getContext('2d');
  const { sensors, width } = state.meta;
  const current = state.windows[state.cursor];
  const to = current.t0 + width,
    span = SPAN_WINDOWS * width,
    from = to - span;
  const band = 22 * ratio,
    plot = canvas.height - band - 12 * ratio;
  const x = (t) => (t - from) / span * canvas.width;
  const y = (p) => band + p / sensors * plot;
  context.clearRect(0, 0, canvas.width, canvas.height);

  // Heatmap: one pixel per sample and sensor, scaled without smoothing.
  const image = new ImageData(span, sensors);
  for (let column = 0; column < span; column++) {
    const values = state.columns.get(from + column);
    for (let p = 0; p < sensors; p++) {
      const v = values ? values[p] : 0,
        offset = (p * span + column) * 4;
      image.data[offset] = values ? RAMP[v * 3] : 0;
      image.data[offset + 1] = values ? RAMP[v * 3 + 1] : 0;
      image.data[offset + 2] = values ? RAMP[v * 3 + 2] : 0;
      image.data[offset + 3] = values ? 255 : 40;
    }
  }
  const buffer = document.createElement('canvas');
  buffer.width = span;
  buffer.height = sensors;
  buffer.getContext('2d').putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(buffer, 0, band, canvas.width, plot);

  // Windows k-1 (dashed) and k (solid), and their overlap.
  const previous = state.windows[state.cursor - 1];
  if (previous && previous.segment === current.segment) {
    context.fillStyle = css('--accent-wash');
    context.fillRect(x(current.t0), band, x(previous.t0 + width) - x(current.t0), plot);
    context.setLineDash([6 * ratio, 4 * ratio]);
    context.strokeStyle = 'rgba(255,255,255,0.75)';
    context.lineWidth = 1.5 * ratio;
    context.strokeRect(x(previous.t0), band + 1, x(previous.t0 + width) - x(previous.t0), plot - 2);
    context.setLineDash([]);
  }
  context.strokeStyle = 'rgba(255,255,255,0.95)';
  context.lineWidth = 2 * ratio;
  context.strokeRect(x(current.t0), band + 1, x(to) - x(current.t0) - 1, plot - 2);
  if (current.skip) {
    context.fillStyle = 'rgba(255,255,255,0.85)';
    context.font = `${12 * ratio}px system-ui`;
    context.fillText(`${current.skip} window: not given to the model`, x(current.t0) + 6 * ratio, band + 16 * ratio);
  }

  // Tracks in stream coordinates.
  context.font = `${11 * ratio}px system-ui`;
  for (const track of tracksAt(state.cursor, from, to)) {
    if (track.state === 'merged') continue;
    const left = x(track.t1),
      top = y(track.p1),
      w = x(track.t2) - left,
      h = y(track.p2) - top;
    context.setLineDash(track.state === 'tentative' ? [4 * ratio, 3 * ratio] : []);
    context.lineWidth = (track.state === 'rejected' ? 1 : 2) * ratio;
    context.strokeStyle = track.state === 'closed' ? 'rgba(150,150,150,0.75)'
      : track.state === 'rejected' ? css('--critical') : trackColor(track.id);
    if (track.state === 'closed') context.setLineDash([2 * ratio, 2 * ratio]);
    context.strokeRect(left, top, w, h);
    context.setLineDash([]);
    if (track.state !== 'rejected') {
      context.fillStyle = context.strokeStyle;
      context.fillText(`#${track.id} ${track.score.toFixed(2)}·${track.hits}`, left, top - 3 * ratio);
    }
  }

  // Cut detections of the current window: hatched at the cut edge.
  context.strokeStyle = 'rgba(255,255,255,0.9)';
  context.lineWidth = 1 * ratio;
  for (const [p1, t1, p2, t2, , cutLeft, cutRight] of current.detections) {
    for (const [cut, edge, direction] of [
        [cutLeft, t1, 1],
        [cutRight, t2, -1]
      ]) {
      if (!cut) continue;
      for (let p = p1; p < p2; p += 0.8) {
        context.beginPath();
        context.moveTo(x(edge), y(p));
        context.lineTo(x(edge) + direction * 6 * ratio, y(Math.min(p + 0.8, p2)));
        context.stroke();
      }
    }
  }

  const seconds = (t) => (t / (state.meta.model_rate_hz || 20)).toFixed(1);
  $('strip-from').textContent = `t ${from} (${seconds(from)} s)`;
  $('strip-to').textContent = `t ${to} (${seconds(to)} s)`;
}

async function loadDetail(k) {
  if (state.detailK === k) return;
  state.detailK = k;
  try {
    const detail = await getJSON(`/api/window?index=${k}`);
    if (state.detailK === k) {
      state.detail = detail;
      drawWindow();
    }
  } catch (error) {
    console.warn(error);
  }
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])),
    h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h,
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/* NMS and τ as in the detector, over the window's candidates (for the sliders). */
function postprocess(candidates, tau, nmsIou, valid) {
  const order = candidates.map((c, i) => i).sort((a, b) => candidates[b][4] - candidates[a][4] || a - b);
  const kept = [];
  for (const i of order) {
    if (kept.length >= state.meta.max_detections) break;
    if (kept.every((j) => iou(candidates[i], candidates[j]) <= nmsIou)) kept.push(i);
  }
  return {
    threshold: candidates.filter((c) => c[4] >= tau),
    nms: kept.map((i) => candidates[i]).filter((c) => c[4] >= tau && c[1] < valid),
  };
}

function drawWindow() {
  const current = state.windows[state.cursor];
  if (!current) return;
  $('window-k').textContent = `${current.k} · t0 ${current.t0}`;
  loadDetail(current.k);
  const canvas = $('frame');
  const ratio = fitCanvas(canvas, canvas.clientWidth);
  const context = canvas.getContext('2d');
  const { sensors, width } = state.meta;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const detail = state.detail && state.detail.k === current.k ? state.detail : null;
  if (!detail) return;
  // Same orientation as the training frames: sensors across, time down.
  if (detail.frame) {
    const bytes = decode(detail.frame),
      image = new ImageData(sensors, width);
    for (let p = 0; p < sensors; p++)
      for (let t = 0; t < width; t++) {
        const v = bytes[p * width + t],
          offset = (t * sensors + p) * 4;
        image.data[offset] = RAMP[v * 3];
        image.data[offset + 1] = RAMP[v * 3 + 1];
        image.data[offset + 2] = RAMP[v * 3 + 2];
        image.data[offset + 3] = 255;
      }
    const buffer = document.createElement('canvas');
    buffer.width = sensors;
    buffer.height = width;
    buffer.getContext('2d').putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(buffer, 0, 0, canvas.width, canvas.height);
  }
  const tau = Number($('tau').value),
    nmsIou = Number($('nms').value);
  const stage = document.querySelector('input[name="stage"]:checked').value;
  const result = postprocess(detail.candidates, tau, nmsIou, current.valid);
  const boxes = stage === 'all' ? detail.candidates : stage === 'threshold' ? result.threshold : result.nms;
  const px = canvas.width / sensors,
    py = canvas.height / width;
  for (const [p1, t1, p2, t2, score] of boxes) {
    context.strokeStyle = stage === 'nms' ? 'rgba(255,255,255,0.95)' : `rgba(255,255,255,${0.15 + 0.8 * score})`;
    context.lineWidth = (stage === 'nms' ? 2 : 1) * ratio;
    context.strokeRect(p1 * px, t1 * py, (p2 - p1) * px, (t2 - t1) * py);
    if (stage === 'nms') {
      context.fillStyle = 'rgba(255,255,255,0.95)';
      context.font = `${11 * ratio}px system-ui`;
      context.fillText(score.toFixed(2), p1 * px, t1 * py - 2 * ratio);
    }
  }
  const changed = tau !== state.meta.score_thr || nmsIou !== state.meta.nms_iou;
  $('frame-counts').textContent = `${detail.candidates.length} candidates · ${result.threshold.length}` +
    ` with score ≥ τ · ${result.nms.length} after NMS` +
    (changed ? ` (run used τ ${state.meta.score_thr}, NMS ${state.meta.nms_iou}: ${current.detections.length})` : '');
  $('window-note').textContent = current.skip ? `${current.skip}` +
    `: not given to the model` : current.peak == null ? '' : `peak ${Math.round(current.peak)} ADC`;
}

function drawCounter() {
  const current = state.windows[state.cursor];
  $('count').textContent = current.count;
  const canvas = $('chart');
  const ratio = fitCanvas(canvas, 190);
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  const windows = state.windows,
    last = windows[windows.length - 1];
  const tMax = Math.max(last.t0 + state.meta.width, 1);
  const total = Math.max(windows[windows.length - 1].count, 1);
  const pad = 28 * ratio,
    x = (t) => pad + t / tMax * (canvas.width - pad - 6 * ratio),
    y = (n) => canvas.height - pad + -n / total * (canvas.height - pad - 8 * ratio);
  context.strokeStyle = css('--grid');
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(pad, y(0));
  context.lineTo(canvas.width, y(0));
  context.moveTo(pad, y(total));
  context.lineTo(canvas.width, y(total));
  context.stroke();
  context.fillStyle = css('--muted');
  context.font = `${10 * ratio}px system-ui`;
  context.fillText(String(total), 2 * ratio, y(total) + 4 * ratio);
  context.fillText('0', 2 * ratio, y(0));
  context.fillText(`${(tMax / (state.meta.model_rate_hz || 20)).toFixed(0)}` +
    ` s`, canvas.width - 40 * ratio, canvas.height - 8 * ratio);
  // Counted over time (window by window).
  context.strokeStyle = css('--series-1');
  context.lineWidth = 2 * ratio;
  context.beginPath();
  windows.forEach((w, i) => {
    const px = x(w.t0 + state.meta.width),
      py = y(w.count);
    if (i) context.lineTo(px, py);
    else context.moveTo(px, py);
  });
  context.stroke();
  context.strokeStyle = css('--ink-2');
  context.lineWidth = 1 * ratio;
  context.beginPath();
  context.moveTo(x(current.t0 + state.meta.width), 4 * ratio);
  context.lineTo(x(current.t0 + state.meta.width), canvas.height - pad);
  context.stroke();
}

function drawTracks() {
  const current = state.windows[state.cursor];
  const rows = current.tracks.filter((r) => r[1] === 'tentative' || r[1] === 'confirmed').sort((a, b) => a[0] - b[0]);
  $('tracks-count').textContent = `${rows.length} active`;
  $('tracks').tBodies[0].innerHTML = rows.map((r) =>
    `<tr><td><span class="swatch" style="background:${trackColor(r[0])}"></span>${r[0]}</td><td>${r[1]}</td>` +
    `<td>${r[6]}</td><td>${r[7]}</td>` +
    `<td>${r[8].toFixed(2)}</td><td>${r[3].toFixed(1)}–${r[5].toFixed(1)}</td><td>${r[9]}</td></tr>`).join('');
}

function eventText(event) {
  if (event.type === 'counted') return `#${event.id} → ${event.payload.count}`;
  if (event.type === 'merged') return `#${event.id} → #${event.payload.into}`;
  if (event.type === 'track_rejected') return `#${event.id} score ${event.payload.score.toFixed(2)}`;
  if (event.type === 'miss') return `#${event.id} misses ${event.payload.misses}`;
  return `#${event.id}`;
}

function drawEvents() {
  const k = state.cursor;
  const shown = [];
  for (let i = state.events.length - 1; i >= 0 && shown.length < EVENT_ROWS; i--) {
    const event = state.events[i];
    if (event.k <= k && FEED_TYPES.has(event.type)) shown.push(event);
  }
  $('events').innerHTML = shown.map((e) =>
    `<li tabindex="0" data-k="${e.k}"><span>t ${e.t.toFixed(1)}</span>` +
    `<span class="type-${e.type}">${e.type.replace('track_', '')}</span><span>${eventText(e)}</span></li>`).join('');
}

function drawStatus() {
  if (!state.meta || state.view !== 'screen') return;
  const current = state.windows[state.cursor];
  const status = state.status || {};
  const rate = state.meta.model_rate_hz || 20;
  const t = current ? current.t0 + state.meta.width : 0;
  const clock = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
  const parts = [
    `window <b>${current ? current.k : '—'}</b> / ${state.windows.length}${status.finished ? '' : '…'}`,
    `t <b>${t}</b> · ${clock(t / rate)}`,
    `infer <b>${current ? current.infer_ms.toFixed(2) : '—'}</b> ms`,
    `window <b>${current ? current.window_ms.toFixed(2) : '—'}</b> ms`,
    `render <b>${state.playing ? state.fps.toFixed(0) : '—'}</b> fps`,
    `RTF <b>${status.rtf == null ? '—' : status.rtf.toFixed(4)}</b>`,
    `${state.meta.recording_rate_hz ? state.meta.recording_rate_hz + ' Hz' : ''}${state.meta.pool > 1 ? ` ÷ ` +
    `${state.meta.pool} (pooled)` : ''} → ${rate} Hz model`,
  ];
  if (state.mode === 'replay' && state.meta.replay_matches_log != null)
    parts.push(state.meta.replay_matches_log ? 'replay = logged events ✓' : 'replay differs from the log!');
  $('statusbar').innerHTML = parts.map((p) => `<span>${p}</span>`).join('');
  // The Play and Pause of a live run control the run; once it has ended there is nothing to control.
  $('play').disabled = state.mode === 'live' && !!status.finished;
  if (status.failure) setRunState('failed', status.failure);
  else if (state.mode === 'live') setRunState(status.finished ? 'idle' : status.paused ? 'paused' : 'running',
    status.finished ? 'Live run finished' : status.paused ? 'Paused' : 'Running live');
  else setRunState('idle', 'Replay of saved logs');
}

function setRunState(kind, text) {
  $('run-state').dataset.state = kind;
  $('run-state-text').textContent = text;
}

function bookmarkKey() {
  return `sensorline-live-bookmarks:${state.meta ? state.meta.recording : ''}`;
}

function loadBookmarks() {
  try {
    return JSON.parse(localStorage.getItem(bookmarkKey()) || '[]');
  } catch (error) {
    return [];
  }
}

function saveBookmarks(list) {
  try {
    localStorage.setItem(bookmarkKey(), JSON.stringify(list));
  } catch (error) { /* storage unavailable */ }
}

function drawBookmarks() {
  $('bookmarks').innerHTML = loadBookmarks().map((k) => `<li><button type="button" data-k="${k}">k ${k}</button>` +
    `</li>`).join('');
}

function download(name, url) {
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
}

let recorder = null;

function toggleRecord() {
  if (recorder) {
    recorder.stop();
    return;
  }
  const stream = $('strip').captureStream(LIVE_FPS);
  const chunks = [];
  recorder = new MediaRecorder(stream, {
    mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm'
  });
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.onstop = () => {
    download(`${state.meta.recording}_k${state.cursor}` +
      `.webm`, URL.createObjectURL(new Blob(chunks, { type: 'video/webm' })));
    recorder = null;
    $('record').textContent = 'Record video';
  };
  recorder.start();
  $('record').textContent = 'Stop recording';
}

function wire() {
  // The forms
  $('start-form').addEventListener('submit', (event) => submitForm(event, '/api/start', () => ({
    csv: $('f-csv').value.trim(),
    weights: $('f-weights').value.trim(),
    rate: $('f-rate').value.trim(),
    speed: Number($('f-speed').value),
    input_kind: $('f-input').value,
  }), 'start-button', 'start-error'));
  $('open-form').addEventListener('submit', (event) => submitForm(event, '/api/open', () => ({
    run: $('f-run').value.trim(),
  }), 'open-button', 'open-error'));
  document.querySelectorAll('button[data-target]')
    .forEach((button) => button.addEventListener('click', () => openChooser(button)));
  $('chooser-list').addEventListener('click', (event) => chooseEntry(event.target.closest('li')));
  $('chooser-list').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') chooseEntry(event.target.closest('li'));
  });
  $('chooser-up').addEventListener('click', () => browse($('chooser-up').dataset.path));
  $('chooser-drives').addEventListener('change', () => browse($('chooser-drives').value));
  $('chooser-path').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') browse($('chooser-path').value);
  });
  $('chooser-close').addEventListener('click', () => $('chooser').close());
  $('chooser-pick').addEventListener('click', () => {
    chooser.input.value = $('chooser-path').value;
    $('chooser').close();
  });
  $('new-run').addEventListener('click', async () => {
    try {
      await postJSON('/api/reset', {});
    } catch (error) {
      console.warn(error);
    }
    state.version = -1;
    state.cursor = -1;
    state.meta = null;
    showView('start');
  });

  // The screen
  $('play').addEventListener('click', () => togglePlay());
  $('back').addEventListener('click', () => {
    leaveLive();
    setCursor(state.cursor - 1);
  });
  $('forward').addEventListener('click', () => setCursor(state.cursor + 1));
  $('speed').addEventListener('change', async () => {
    state.speed = Number($('speed').value);
    if (state.mode === 'live') try {
      await postJSON('/api/control', { action: 'speed', value: state.speed });
    } catch (error) {
      console.warn(error);
    }
  });
  $('follow').addEventListener('change', () => {
    state.follow = $('follow').checked;
    if (state.follow) setCursor(state.windows.length - 1);
  });
  $('events').addEventListener('click', (event) => {
    const item = event.target.closest('li');
    if (item) {
      leaveLive();
      setCursor(Number(item.dataset.k));
    }
  });
  $('events').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') event.target.click();
  });
  $('bookmark').addEventListener('click', () => {
    const list = loadBookmarks();
    if (!list.includes(state.cursor)) list.push(state.cursor);
    saveBookmarks(list.sort((a, b) => a - b));
    drawBookmarks();
  });
  $('bookmarks').addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (button) {
      leaveLive();
      setCursor(Number(button.dataset.k));
    }
  });
  $('png-strip').addEventListener('click', () => download(`${state.meta.recording}_strip_k${state.cursor}` +
    `.png`, $('strip').toDataURL('image/png')));
  $('png-window').addEventListener('click', () => download(`${state.meta.recording}_window_k${state.cursor}` +
    `.png`, $('frame').toDataURL('image/png')));
  $('record').addEventListener('click', toggleRecord);
  for (const id of ['tau', 'nms']) $(id).addEventListener('input', () => {
    $(`${id}-out`).textContent = Number($(id).value).toFixed(2);
    drawWindow();
  });
  document.querySelectorAll('input[name="stage"]').forEach((input) => input.addEventListener('change', drawWindow));
  $('apply').addEventListener('click', async () => {
    $('apply-note').textContent = 'Recomputing from the saved candidates…';
    try {
      await postJSON('/api/recompute', { score_thr: Number($('tau').value), nms_iou: Number($('nms').value) });
      $('apply-note').textContent =
        `Recomputed with τ ${$('tau').value}, NMS ${$('nms').value}; logged run unchanged.`;
    } catch (error) {
      $('apply-note').textContent = error.message;
    }
  });
  $('theme').addEventListener('click', () => {
    const dark = getComputedStyle(document.documentElement).colorScheme === 'dark';
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    try {
      localStorage.setItem('sensorline-theme', document.documentElement.dataset.theme);
    } catch (error) { /* ignore */ }
    state.dirty = true;
  });
  document.addEventListener('keydown', (event) => {
    if (state.view !== 'screen' || event.target.matches('input, select, textarea') || $('chooser').open) return;
    if (event.key === ' ') {
      event.preventDefault();
      togglePlay();
    }
    if (event.key === 'ArrowLeft') {
      leaveLive();
      setCursor(state.cursor - 1);
    }
    if (event.key === 'ArrowRight') setCursor(state.cursor + 1);
  });
  window.addEventListener('resize', () => {
    state.dirty = true;
  });
}

wire();
// A timer rather than requestAnimationFrame: it also draws in a background tab.
setInterval(tick, 1000 / LIVE_FPS / 2);
poll();
