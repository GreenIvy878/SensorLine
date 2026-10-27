/* Inference section: run a saved checkpoint on a dataset and read the result. */
'use strict';

const inferenceUI = {
  checkpoints: null,
  selected: null,
  filter: '',
  source: 'val',
  folder: '',
  labeled: true,
  device: 'auto',
  state: null,
  message: '',
  resultKey: '',
  frameFilter: 'all',
  cell: null,
  page: 0,
  frame: null,
  showBelow: false,
  timer: null,
};
const FRAMES_PER_PAGE = 12;
const SOURCE_TEXT = { val: 'validation set', train: 'training set', folder: 'folder' };
/* Overlay colors sit on grayscale images, not on the page, so they are fixed. */
const BOX_COLORS = {
  found: '#6cff9a', missed: '#ff6b9d', tp: '#80c8ff', fp: '#ffb571', prediction: '#80c8ff', below: '#dcafff'
};

function checkpointLabel(row) {
  const parts = [row.name];
  if (row.f1 != null) parts.push('F1 ' + percentText(row.f1));
  if (row.timestamp) parts.push(new Date(row.timestamp).toLocaleDateString());
  return parts.join(' · ');
}

function inferenceFormHTML() {
  const ui = inferenceUI,
    running = ui.state?.status === 'running';
  const needle = ui.filter.toLowerCase();
  const rows = ui.checkpoints.filter(row => row.name.toLowerCase().includes(needle) || row.id === ui.selected);
  const sources = [
    ['val', 'Validation set'],
    ['train', 'Training set'],
    ['folder', 'Folder']
  ];
  if (!ui.checkpoints.length)
    return empty('No trained checkpoint in the runs folder yet. Train one with train.py first.');
  let html = `<section class="card"><div class="form-grid">
      <label class="field"><span>Find a run<small>${rows.length} of ${ui.checkpoints.length}</small></span>` +
        `<input id="inference-filter" value="${esc(ui.filter)}" placeholder="part of the name"></label>
      <label class="field wide"><span>Checkpoint</span>` +
        `<select id="inference-checkpoint">${rows.map(row => `<option value="${esc(row.id)}" ` +
        `${row.id === ui.selected ? 'selected' : ''}>${esc(checkpointLabel(row))}</option>`).join('')}</select></label>
      <label class="field"><span>Device</span><select id="inference-device">` +
        `${['auto', 'cuda', 'cpu'].map(name => `<option ${name === ui.device ? 'selected' : ''}>${name}` +
        `</option>`).join('')}</select></label>
    </div>
    <div class="actions"><span class="message">Run on</span>` +
      `<div class="segmented" role="group" aria-label="Data">` +
      `${sources.map(([key, label]) => `<button type="button" data-source="${key}" aria-pressed="${ui.source === key}` +
      `">${label}</button>`).join('')}</div>
      <span class="message">` +
        `${ui.source === 'folder'
          ? 'Any folder with an images/ directory; add labels/ to get quality metrics. ' +
            'Pixels are scaled as the data.yaml of the dataset says; without one, as in training.'
          : `The ${SOURCE_TEXT[ui.source]} this run was trained with.`}</span></div>`;
  if (ui.source === 'folder') html += `<div class="form-grid"><label class="field wide"><span>Folder path</span>` +
    `<input id="inference-folder" value="${esc(ui.folder)}" placeholder="C:\\data\\new_recordings"></label>
      <label class="field checkbox"><input type="checkbox" id="inference-labeled" ${ui.labeled ? 'checked' : ''}` +
        `> The folder has labels/</label></div>`;
  html += `<div class="actions" style="margin-bottom:0">` +
    `<button id="inference-run" class="primary" ${running ? 'disabled' : ''}>Run inference</button>` +
    `<span class="message${ui.message ? ' error' : ''}" role="status">${esc(ui.message)}</span></div>`;
  if (running) html += `<div style="margin-top:14px"><div class="progress-meta">` +
    `<span>Processed <strong>${ui.state.done}</strong> of ${ui.state.total || '…'} frames</span>` +
    `</div>${progressBar(ui.state.done, ui.state.total, 'Frames')}</div>`;
  return html + '</section>';
}

function renderInferenceForm() {
  const ui = inferenceUI,
    node = $('inference-form');
  if (!node) return;
  const focused = document.activeElement?.id,
    caret = document.activeElement?.selectionStart;
  node.innerHTML = inferenceFormHTML();
  if (!$('inference-run')) return;
  if (focused && $(focused)) {
    $(focused).focus();
    if (caret != null && $(focused).setSelectionRange) $(focused).setSelectionRange(caret, caret);
  }
  $('inference-filter').oninput = () => {
    ui.filter = $('inference-filter').value;
    renderInferenceForm();
    ui.selected = $('inference-checkpoint').value || ui.selected;
  };
  $('inference-checkpoint').onchange = () => {
    ui.selected = $('inference-checkpoint').value;
  };
  $('inference-device').onchange = () => {
    ui.device = $('inference-device').value;
  };
  node.querySelectorAll('[data-source]').forEach(button => button.onclick = () => {
    ui.source = button.dataset.source;
    ui.message = '';
    renderInferenceForm();
  });
  if ($('inference-folder')) $('inference-folder').oninput = () => {
    ui.folder = $('inference-folder').value;
  };
  if ($('inference-labeled')) $('inference-labeled').onchange = () => {
    ui.labeled = $('inference-labeled').checked;
  };
  $('inference-run').onclick = startInference;
}
async function startInference() {
  const ui = inferenceUI;
  ui.message = '';
  try {
    ui.state = (await postJSON('/api/inference/start', {
      checkpoint: $('inference-checkpoint').value,
      source: ui.source,
      folder: ui.folder,
      labeled: ui.labeled,
      device: ui.device
    })).inference;
    // The previous result belongs to another run; do not leave it under the new progress bar.
    ui.resultKey = '';
    ui.frame = null;
    if ($('inference-result')) $('inference-result').innerHTML = '';
  } catch (error) {
    ui.message = error.message;
  }
  renderInferenceForm();
  scheduleInferencePoll();
}

function scheduleInferencePoll() {
  clearTimeout(inferenceUI.timer);
  if (inferenceUI.state?.status !== 'running') return;
  inferenceUI.timer = setTimeout(async () => {
    try {
      inferenceUI.state = (await request('/api/inference/status')).inference;
    } catch (error) { /* keep the last state */ }
    if (section === 'Inference') {
      renderInferenceForm();
      renderInferenceResult();
    }
    scheduleInferencePoll();
  }, 400);
}

/* Frames by labeled count (rows) and detected count (columns). Exact counts lie on
   the diagonal in the first hue; every other cell is a miscount in the second. */
function matrixCap(frames) {
  // A few very crowded frames would stretch the grid with empty rows: counts above
  // the level that covers 97% of the frames share one last "N+" row and column.
  const largest = frames.map(frame => Math.max(frame.true_count, frame.predicted_count)).sort((a, b) => a - b);
  const typical = largest[Math.floor((largest.length - 1) * 0.97)],
    highest = largest[largest.length - 1];
  return highest > typical + 1 ? typical + 1 : null;
}

function countMatrix(frames) {
  const counts = new Map(),
    cap = matrixCap(frames),
    bucket = value => cap != null && value >= cap ? cap : value;
  let lowest = Infinity,
    highest = 0,
    peak = 0;
  for (const frame of frames) {
    const labeled = bucket(frame.true_count),
      detected = bucket(frame.predicted_count);
    const key = labeled + ':' + detected,
      value = (counts.get(key) || 0) + 1;
    counts.set(key, value);
    lowest = Math.min(lowest, labeled, detected);
    highest = Math.max(highest, labeled, detected);
    peak = Math.max(peak, value);
  }
  const name = value => value === cap ? `${cap}+` : String(value);
  const size = highest - lowest + 1,
    cell = Math.max(14, Math.min(34, Math.floor(440 / size))),
    left = 46,
    top = 30;
  const base = cssValue('--surface-2'),
    exact = cssValue('--series-1'),
    wrong = cssValue('--series-2'),
    ink = cssValue('--ink');
  const luminance = color => {
    const [r, g, b] = color.match(/\d+/g).map(Number);
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  };
  // Drawn at its own size: a small matrix must not be blown up to the card width.
  let svg = `<svg class="chart count-matrix" viewBox="0 0 ${left + size * cell + 8} ${top + size * cell + 30}` +
    `" style="max-width:${left + size * cell + 8}px" role="img"` +
    ` aria-label="Frames by labeled and detected object count">`;
  for (let row = 0; row < size; row++)
    for (let column = 0; column < size; column++) {
      const labeled = lowest + row,
        detected = lowest + column,
        value = counts.get(labeled + ':' + detected) || 0;
      const x = left + column * cell,
        y = top + row * cell;
      // Empty cells are a faint placeholder; a cell with one frame is already clearly tinted.
      const fill = value ? mixColor(base, labeled === detected ? exact : wrong, 0.45 + 0.55 * value / peak) : base;
      const selected = inferenceUI.cell && inferenceUI.cell[0] === labeled && inferenceUI.cell[1] === detected;
      const tip = `${value} frame${value === 1 ? '' : 's'}: ${name(labeled)} labeled, ${name(detected)} detected`;
      svg += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" style="fill:${fill}"` +
        `${value ? ` data-tip="${tip}" data-cell="${labeled}:${detected}" tabindex="0"` : ' opacity=".45"'}` +
        `${selected ? ' class="selected"' : ''}/>`;
      if (value && cell >= 20) svg += `<text x="${x + cell / 2}" y="${y + cell / 2 + 4}` +
        `" text-anchor="middle" style="fill:${luminance(fill) > 0.6 ? ink : '#fff'};pointer-events:none">${value}` +
        `</text>`;
    }
  const every = size > 16 ? 2 : 1;
  for (let index = 0; index < size; index += every) {
    svg += `<text x="${left + index * cell + cell / 2}" y="${top - 8}" text-anchor="middle">${name(lowest + index)}` +
      `</text><text x="${left - 8}" y="${top + index * cell + cell / 2 + 4}" text-anchor="end">` +
      `${name(lowest + index)}</text>`;
  }
  svg += `<text class="axis-title" x="${left + size * cell / 2}" y="${top + size * cell + 22}` +
    `" text-anchor="middle">Detected objects</text><text class="axis-title" transform="translate(12 ` +
    `${top + size * cell / 2}) rotate(-90)" text-anchor="middle">Labeled objects</text></svg>`;
  return svg + `<div class="chart-legend"><span><i style="background:var(--series-1)"></i>Exact count</span><span>` +
    `<i style="background:var(--series-2)"></i>Miscount</span></div>` +
    `<p class="legend">Rows are the labeled count of a frame, columns the detected count; a stronger cell holds more ` +
    `frames. Select a cell to list its frames below.</p>`;
}

function frameFilters(labeled) {
  return labeled ?
    [
      ['all', 'All frames', () => true],
      ['errors', 'With errors', frame => frame.fp + frame.fn > 0],
      ['missed', 'Missed objects', frame => frame.fn > 0],
      ['extra', 'False alarms', frame => frame.fp > 0]
    ] :
    [
      ['all', 'All frames', () => true],
      ['empty', 'No objects detected', frame => frame.predicted_count === 0],
      ['busy', 'Most objects first', () => true]
    ];
}

function visibleFrames(state) {
  const ui = inferenceUI,
    filters = frameFilters(state.labeled);
  const chosen = filters.find(item => item[0] === ui.frameFilter) || filters[0];
  let frames = state.frames.filter(chosen[2]);
  if (ui.cell) {
    // The last row and column of the matrix collect every count from the cap upwards.
    const cap = matrixCap(state.frames),
      bucket = value => cap != null && value >= cap ? cap : value;
    frames = frames.filter(frame =>
      bucket(frame.true_count) === ui.cell[0] && bucket(frame.predicted_count) === ui.cell[1]);
  }
  if (ui.frameFilter === 'busy') frames = [...frames].sort((a, b) => b.predicted_count - a.predicted_count);
  else if (ui.frameFilter !== 'all') frames = [...frames].sort((a, b) => (b.fp + b.fn) - (a.fp + a.fn));
  return frames;
}

function paintInferenceFrame(canvas, sample, showBelow, withScores) {
  const values = sample.matrix.flat();
  paintMatrix(canvas, {
    matrix: sample.matrix, min: Math.min(...values), max: Math.max(...values), boxes_image_xyxy: []
  }, false);
  const context = canvas.getContext('2d');
  const draw = (box, color, dashed, score) => {
    const x = box[0] * canvas.width,
      y = box[1] * canvas.height;
    context.strokeStyle = color;
    context.setLineDash(dashed ? [5, 3] : []);
    context.strokeRect(x, y, (box[2] - box[0]) * canvas.width, (box[3] - box[1]) * canvas.height);
    if (withScores && score != null) {
      context.fillStyle = color;
      context.font = '12px system-ui';
      context.fillText(score.toFixed(2), x, Math.max(12, y - 3));
    }
  };
  context.lineWidth = 2;
  for (const target of sample.targets) draw(target.box, target.detected ? BOX_COLORS.found : BOX_COLORS.missed, false);
  for (const detection of sample.detections)
    draw(detection.box, BOX_COLORS[detection.classification], sample.labeled, detection.score);
  if (showBelow)
    for (const detection of sample.below_threshold) draw(detection.box, BOX_COLORS.below, true, detection.score);
  context.setLineDash([]);
}

function boxLegend(labeled) {
  const key = (color, text, dashed = false) => `<span>` +
    `<i class="${dashed ? 'dashed' : ''}" style="border-color:${color}"></i>${text}</span>`;
  return `<div class="box-legend">${labeled
    ? key(BOX_COLORS.found, 'labeled object, found') + key(BOX_COLORS.missed, 'labeled object, missed') +
      key(BOX_COLORS.tp, 'correct detection', true) + key(BOX_COLORS.fp, 'false alarm', true)
    : key(BOX_COLORS.prediction, 'detected object')}${key(BOX_COLORS.below, 'below the confidence threshold', true)}` +
      `</div>`;
}
const frameURL = id => '/api/inference/sample?id=' + encodeURIComponent(id);

async function fillFrameGallery(frames) {
  const container = $('inference-gallery'),
    key = inferenceUI.resultKey + inferenceUI.page + inferenceUI.frameFilter + String(inferenceUI.cell);
  if (!container) return;
  container.dataset.key = key;
  for (const frame of frames) {
    let sample;
    try {
      sample = await request(frameURL(frame.id));
    } catch (error) {
      continue;
    }
    if (!container.isConnected || container.dataset.key !== key) return;
    const item = document.createElement('div'),
      canvas = document.createElement('canvas'),
      button = document.createElement('button');
    canvas.className = 'thumbnail';
    paintInferenceFrame(canvas, sample, false, false);
    button.className = 'thumb-button';
    button.textContent = sample.labeled ? `${frame.id} · ${frame.predicted_count} of ${frame.true_count}` +
      `${frame.fp ? ` · ${frame.fp} false` : ''}${frame.fn ? ` · ${frame.fn} missed` : ''}` : `${frame.id} · ` +
      `${frame.predicted_count} object${frame.predicted_count === 1 ? '' : 's'}`;
    button.onclick = () => {
      inferenceUI.frame = frame.id;
      showInferenceFrame();
    };
    item.append(canvas, button);
    container.append(item);
  }
}
async function showInferenceFrame() {
  const ui = inferenceUI,
    node = $('inference-frame');
  if (!node || !ui.frame) return;
  let sample;
  try {
    sample = await request(frameURL(ui.frame));
  } catch (error) {
    node.innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  const frame = ui.state.frames.find(row => row.id === ui.frame);
  const facts = sample.labeled ?
    [
      ['Labeled objects', frame.true_count],
      ['Detected objects', frame.predicted_count],
      ['Correct detections', frame.tp],
      ['False alarms', frame.fp],
      ['Missed objects', frame.fn]
    ] :
    [
      ['Detected objects', frame.predicted_count]
    ];
  facts.push(['Candidates below the threshold', sample.below_threshold.length]);
  const scores = sample.detections.map(detection => detection.score.toFixed(2)).join(', ') || 'none';
  node.innerHTML = card(ui.frame, `<div class="frame-detail"><div><canvas id="inference-canvas" class="heatmap">` +
    `</canvas>${boxLegend(sample.labeled)}</div>
    <div><ul class="frame-facts">${facts.map(([label, value]) => `<li>${esc(label)}<strong>${esc(value)}</strong>` +
      `</li>`).join('')}</ul>
    <p class="legend" style="margin-top:12px">Confidence of the detections: ${esc(scores)}.</p>
    <label><input type="checkbox" id="inference-below" ${ui.showBelow ? 'checked' : ''}` +
      `> Show candidates below the threshold</label>
    <p class="legend" style="margin-top:12px">X → ${sample.sensor_axis === 'x' ? 'sensors' : 'time'}; Y ↓ ` +
      `${sample.sensor_axis === 'x' ? 'time' : 'sensors'}.</p></div></div>`);
  paintInferenceFrame($('inference-canvas'), sample, ui.showBelow, true);
  $('inference-below').onchange = () => {
    ui.showBelow = $('inference-below').checked;
    showInferenceFrame();
  };
  node.scrollIntoView({ block: 'nearest' });
}

function renderFrames(state) {
  const ui = inferenceUI,
    frames = visibleFrames(state),
    pages = Math.max(1, Math.ceil(frames.length / FRAMES_PER_PAGE));
  ui.page = Math.min(ui.page, pages - 1);
  const chips = frameFilters(state.labeled).map(([key, label]) => `<button type="button" data-filter="${key}` +
    `" aria-pressed="${ui.frameFilter === key}">${label}</button>`).join('') +
    (ui.cell ? `<button type="button" id="inference-cell" aria-pressed="true" title="Remove this filter">` +
      `${ui.cell[0]} labeled, ${ui.cell[1]} detected ×</button>` : '');
  $('inference-frames').innerHTML = card('Frames', `<div class="controls"><div class="chips">${chips}</div>` +
    `<span class="toolbar-spacer"></span>
      <button id="inference-prev" ${ui.page === 0 ? 'disabled' : ''}>Previous</button>` +
        `<span class="muted">${frames.length} frames · page ${ui.page + 1} of ${pages}</span>` +
        `<button id="inference-next" ${ui.page + 1 >= pages ? 'disabled' : ''}>Next</button></div>
    ${frames.length ? '<div id="inference-gallery" class="gallery"></div>' : empty('No frame matches this filter.')}` +
      `${boxLegend(state.labeled)}`) + '<div id="inference-frame"></div>';
  $('inference-frames').querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
    ui.frameFilter = button.dataset.filter;
    ui.page = 0;
    renderFrames(state);
  });
  if ($('inference-cell')) $('inference-cell').onclick = () => {
    ui.cell = null;
    ui.page = 0;
    renderInferenceResult(true);
  };
  $('inference-prev').onclick = () => {
    ui.page--;
    renderFrames(state);
  };
  $('inference-next').onclick = () => {
    ui.page++;
    renderFrames(state);
  };
  fillFrameGallery(frames.slice(ui.page * FRAMES_PER_PAGE, (ui.page + 1) * FRAMES_PER_PAGE));
  showInferenceFrame();
}

function labeledResultHTML(state) {
  const metrics = state.metrics,
    detected = metrics.tp + metrics.fp,
    exact = Math.round(metrics.exact_count_accuracy * metrics.samples);
  const bias = Math.abs(metrics.count_bias) < 0.005 ? 'no systematic over- or undercount' :
    `${metrics.count_bias > 0 ? 'over' : 'under'}counts by ${decimalText(Math.abs(metrics.count_bias), 2)} per frame`;
  const tiles = tile('Objects counted', String(detected), `${metrics.objects} labeled`) +
    tile('Frames with the exact count', percentText(metrics.exact_count_accuracy), `${exact} of ${metrics.samples}` +
      ` frames`) +
    tile('Count error per frame (MAE)', decimalText(metrics.count_mae), bias) +
    tile('F1', percentText(metrics.f1), `precision ${percentText(metrics.precision)} · recall ` +
      `${percentText(metrics.recall)}`);
  const saved = state.evaluation.confidence;
  const sweep = plot({
    id: 'sweep',
    rows: state.sweep,
    x: 'confidence',
    xDomain: [0, 1],
    xTicks: [0, .2, .4, .6, .8, 1],
    xLabel: 'Confidence threshold',
    percent: true,
    label: 'Precision, recall and F1 by confidence threshold',
    series: QUALITY_SERIES,
    title: row => `Confidence ≥ ${row.confidence}`,
    marker: { x: saved, label: `saved ${saved}` }
  });
  // Two short cards share the left column; the taller matrix takes the right one.
  return `<div class="metrics">${tiles}</div><div class="grid"><div class="stack">
    ${card('Found, missed and false alarms', tray(metrics.tp, metrics.fp, metrics.fn))}
    ${card('Effect of the confidence threshold, %', sweep, state.request.source === 'val'
      ? 'Computed on the validation set that also selected the checkpoint, so the best threshold here is optimistic.'
      : 'The same detections, accepted at other thresholds. The model and the NMS threshold are unchanged.')}
    </div>${card('Count agreement by frame', countMatrix(state.frames))}</div>`;
}
/* How many frames hold 0, 1, 2… detected objects: one thin column per count. */
function countColumns(counts) {
  const highest = Math.max(...counts),
    frames = Array.from({ length: highest + 1 }, (_, value) => counts.filter(count => count === value).length);
  const width = 620,
    height = 230,
    left = 44,
    right = 12,
    top = 14,
    bottom = 42;
  const scale = niceScale(0, Math.max(...frames)),
    slot = (width - left - right) / frames.length,
    bar = Math.min(24, slot * 0.7);
  const py = value => height - bottom - value / scale.max * (height - top - bottom);
  let svg = `<svg class="chart" viewBox="0 0 ${width} ${height}` +
    `" role="img" aria-label="Frames by number of detected objects">`;
  for (const tick of scale.ticks.filter(Number.isInteger)) svg += `<line class="grid-line" x1="${left}" x2="` +
    `${width - right}" y1="${py(tick)}" y2="${py(tick)}"/>` +
    `<text x="${left - 8}" y="${py(tick) + 4}" text-anchor="end">${tick}</text>`;
  const every = Math.ceil(frames.length / 16);
  frames.forEach((value, count) => {
    const x = left + slot * count + slot / 2;
    // Rounded at the data end only; the hit area is the whole slot, not just the column.
    const radius = Math.min(4, (py(0) - py(value)) / 2, bar / 2);
    if (value) svg += `<path d="M${x - bar / 2},${py(0)}V${py(value) + radius}q0,${-radius} ${radius},${-radius}h` +
      `${bar - 2 * radius}q${radius},0 ${radius},${radius}V${py(0)}z" style="fill:var(--series-1)"/>`;
    svg += `<rect x="${x - slot / 2}" y="${top}" width="${slot}" height="${height - top - bottom}` +
      `" fill="transparent" data-tip="${value} frame${value === 1 ? '' : 's'} with ${count} detected object` +
      `${count === 1 ? '' : 's'}"/>`;
    if (count % every === 0) svg += `<text x="${x}" y="${height - bottom + 18}" text-anchor="middle">${count}</text>`;
  });
  svg += `<line class="axis-line" x1="${left}" x2="${width - right}" y1="${py(0)}" y2="${py(0)}"/>` +
    `<text class="axis-title" x="${(left + width - right) / 2}" y="${height - 6}` +
    `" text-anchor="middle">Detected objects in a frame</text></svg>`;
  return svg;
}

function unlabeledResultHTML(state) {
  const counts = state.frames.map(frame => frame.predicted_count),
    total = counts.reduce((sum, value) => sum + value, 0);
  return `<div class="metrics">${tile('Objects counted', String(total))}${tile('Frames', String(counts.length))}` +
    `${tile('Objects per frame', decimalText(total / counts.length, 2))}` +
    `${tile('Most in one frame', String(Math.max(...counts)))}</div>
    <div class="grid">${card('Frames by object count', countColumns(counts),
      'There are no labels for this data, so quality cannot be measured here.')}</div>`;
}

/* A folder may store pixels differently from the training data; say which rule was applied. */
function signalText(preprocessing) {
  const pixel = preprocessing.offset ? `(pixel − ${preprocessing.offset})` : 'pixel';
  return `signal = ${pixel} ÷ ${preprocessing.scale}` +
    `, ` + (preprocessing.source === 'checkpoint' ? 'as in training (nothing describes this folder)' : `as ` +
    `${preprocessing.source.split(/[\\/]/).pop()} describes this data`);
}

function renderInferenceResult(force = false) {
  const ui = inferenceUI,
    state = ui.state,
    node = $('inference-result');
  if (!node) return;
  if (!state || state.status === 'running') {
    if (!state) node.innerHTML = '';
    return;
  }
  if (state.status === 'failed') {
    node.innerHTML = `<section class="card"><h3>Inference failed</h3><p class="error">${esc(state.error)}</p>` +
      `</section>`;
    ui.resultKey = '';
    return;
  }
  if (!force && ui.resultKey === state.id && node.dataset.key === state.id) return;
  if (ui.resultKey !== state.id) {
    ui.frameFilter = 'all';
    ui.cell = null;
    ui.page = 0;
    ui.frame = null;
  }
  ui.resultKey = state.id;
  node.dataset.key = state.id;
  const checkpoint = ui.checkpoints?.find(row => row.id === state.request.checkpoint_id);
  const thresholds = state.evaluation;
  node.innerHTML = `<p class="legend"><strong>${esc(checkpoint ? checkpoint.name : 'Checkpoint')}</strong> (epoch ` +
    `${esc(state.checkpoint_epoch)}) on the ${esc(SOURCE_TEXT[state.request.source])} · ${state.frames.length}` +
    ` frames · confidence ≥ ${thresholds.confidence} · NMS IoU ${thresholds.nms_iou}` +
    `${state.labeled ? ` · a detection is correct at IoU ≥ ${thresholds.match_iou}` : ''} · ${esc(state.device)}` +
    `${state.request.source === 'folder' ? ` · ${esc(signalText(state.preprocessing))}` : ''}</p>` +
    (state.labeled ? labeledResultHTML(state) : unlabeledResultHTML(state)) +
    `<div id="inference-frames"></div><p class="legend">Predictions and metrics are saved in <span class="path">` +
    `${esc(state.output_directory)}</span> · data: <span class="path">${esc(state.data_root)}</span></p>`;
  node.querySelectorAll('[data-cell]').forEach(rect => {
    const select = () => {
      ui.cell = rect.dataset.cell.split(':').map(Number);
      ui.frameFilter = 'all';
      ui.page = 0;
      renderInferenceResult(true);
    };
    rect.onclick = select;
    rect.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        select();
      }
    };
  });
  renderFrames(state);
}

async function renderInference() {
  const ui = inferenceUI;
  $('content').innerHTML = '<h2>Inference</h2><p class="section-hint">Run a trained checkpoint on its own data or ' +
    'on a new folder, and see what it counted, found and missed.</p>' +
    '<div id="inference-form"><p>Loading…</p></div><div id="inference-result"></div>';
  try {
    if (!ui.checkpoints) ui.checkpoints = await request('/api/inference/checkpoints');
    if (!ui.state) ui.state = (await request('/api/inference/status')).inference;
  } catch (error) {
    if ($('inference-form')) $('inference-form').innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  if (section !== 'Inference') return;
  if (!ui.checkpoints.some(row => row.id === ui.selected)) ui.selected = ui.checkpoints[0]?.id || null;
  renderInferenceForm();
  renderInferenceResult(true);
  scheduleInferencePoll();
}
