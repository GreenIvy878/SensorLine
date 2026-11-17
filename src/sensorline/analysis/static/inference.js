/* Inference section: run a saved checkpoint on its validation, test or training data, or on a folder, and read the
   result. The setup stays on the left; the result, in three tabs, on the right. */
'use strict';

const inferenceUI = {
  checkpoints: null,
  selected: null,
  filter: '',
  source: 'val',
  folder: '',
  labeled: true,
  device: 'auto',
  borderIgnore: '',
  state: null,
  message: '',
  resultKey: '',
  tab: 'summary',
  frameFilter: 'all',
  page: 0,
  frame: null,
  showBelow: false,
  timer: null,
};
const FRAMES_PER_PAGE = 5; // one row of five
const FILTER_FROM = 8; // a text filter over the runs is offered when there are more than this many
const SOURCE_TEXT = { val: 'validation set', test: 'test set', train: 'training set', folder: 'folder' };
/* Overlay colors sit on grayscale images, not on the page, so they are fixed. */
const BOX_COLORS = {
  found: '#6cff9a', missed: '#ff6b9d', tp: '#80c8ff', fp: '#ffb571', prediction: '#80c8ff', below: '#dcafff',
  ignored: '#9aa0a6'
};

function checkpointLabel(row) {
  const parts = [row.name];
  if (row.count_mae != null) parts.push('MAE ' + decimalText(row.count_mae, 2));
  if (row.timestamp) parts.push(new Date(row.timestamp).toLocaleDateString());
  return parts.join(' · ');
}

/* A folder is long and its end is what tells it apart: …\main_flow\test. The whole path stays in the title. */
function shortPath(path) {
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  return parts.length > 2 ? '…\\' + parts.slice(-2).join('\\') : String(path);
}

const selectedCheckpoint = () => inferenceUI.checkpoints?.find(row => row.id === inferenceUI.selected);

/* The data a run can be evaluated on, with what each one means for the numbers. */
function sourceChoices(row) {
  const splits = row?.splits || {};
  const own = key => {
    const split = splits[key];
    if (!split) return { path: '', missing: key === 'test' ? 'No test split in this run or in a model YAML' : 'Not recorded' };
    if (!split.found) return { path: split.path, missing: 'Folder not found' };
    return split;
  };
  const choice = (key, title, detail) => {
    const split = own(key);
    return { key, title, detail: split.missing || detail(split), path: split.path, disabled: Boolean(split.missing) };
  };
  return [
    choice('val', 'Validation', () => ''),
    choice('test', 'Test', split => split.origin === 'run' ? '' : `from ${split.origin}`),
    choice('train', 'Training', () => ''),
    { key: 'folder', title: 'Another folder', path: '', disabled: false, detail: '' }
  ];
}

function sourceListHTML(row) {
  const ui = inferenceUI;
  return `<div class="choices" role="group" aria-label="Data to run on">` +
    sourceChoices(row).map(item => `<button type="button" class="choice" data-source="${item.key}" ` +
      `aria-pressed="${ui.source === item.key}" ${item.disabled ? 'disabled' : ''}>` +
      `<b>${esc(item.title)}</b>${item.detail ? `<span>${esc(item.detail)}</span>` : ''}` +
      `${item.path ? `<span class="path" title="${esc(item.path)}">${esc(shortPath(item.path))}</span>` : ''}` +
      `</button>`).join('') + '</div>';
}

function inferenceFormHTML() {
  const ui = inferenceUI,
    running = ui.state?.status === 'running',
    needle = ui.filter.toLowerCase();
  if (!ui.checkpoints.length)
    return empty('No trained checkpoint in the runs folder yet.');
  const rows = ui.checkpoints.filter(row => row.name.toLowerCase().includes(needle) || row.id === ui.selected),
    row = selectedCheckpoint();
  let html = `<section class="card setup">` +
    `<div class="form-section">` +
    `${ui.checkpoints.length > FILTER_FROM ? `<input id="inference-filter" class="compact" value="${esc(ui.filter)}" ` +
      `placeholder="Find a run: ${rows.length} of ${ui.checkpoints.length}" aria-label="Find a run by name">` : ''}` +
    `<select id="inference-checkpoint" aria-label="Checkpoint">${rows.map(item => `<option value="${esc(item.id)}" ` +
    `${item.id === ui.selected ? 'selected' : ''}>${esc(checkpointLabel(item))}</option>`).join('')}</select>` +
    `</div>` +
    `<div class="form-section">${sourceListHTML(row)}`;
  if (ui.source === 'folder') html += `<div class="form-grid single" style="margin:12px 0 0">` +
    `<label class="field"><span class="name">Folder path</span>` +
    `<input id="inference-folder" value="${esc(ui.folder)}" placeholder="C:\\data\\new_recordings"></label>` +
    `<label class="field checkbox"><input type="checkbox" id="inference-labeled" ${ui.labeled ? 'checked' : ''}> ` +
    `The folder has labels/</label></div>`;
  html += `</div><details class="more"><summary>More options</summary><div class="form-grid single">` +
    `<label class="field"><span class="name">Device</span><select id="inference-device">` +
    `${['auto', 'cuda', 'cpu'].map(name => `<option ${name === ui.device ? 'selected' : ''}>${name}</option>`).join('')}` +
    `</select></label>` +
    `<label class="field"><span class="name">Ignore cut objects<small>samples</small></span>` +
    `<input id="inference-border" type="number" min="0" step="0.5" value="${esc(ui.borderIgnore)}" ` +
    `placeholder="as trained" title="Objects that a time edge of the frame cuts to fewer samples than this are ` +
    `counted neither as found nor as missed. Empty keeps the setting of the checkpoint."></label>` +
    `</div></details>` +
    `<div class="actions"><button id="inference-run" class="primary wide-button" ` +
    `${running ? 'disabled' : ''}>${running ? 'Running…' : 'Run inference'}</button></div>` +
    `<p class="message${ui.message ? ' error' : ''}" role="status" ` +
    `style="margin:8px 0 0">${esc(ui.message)}</p>`;
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
  const open = node.querySelector('details.more')?.open;
  // A run without a test split cannot be run on one: fall back instead of leaving a dead choice selected.
  const row = selectedCheckpoint();
  if (row && !sourceChoices(row).some(item => item.key === ui.source && !item.disabled)) ui.source = 'val';
  node.innerHTML = inferenceFormHTML();
  if (!$('inference-run')) return;
  if (open) node.querySelector('details.more').open = true;
  if (focused && $(focused)) {
    $(focused).focus();
    if (caret != null && $(focused).setSelectionRange) $(focused).setSelectionRange(caret, caret);
  }
  if ($('inference-filter')) $('inference-filter').oninput = () => {
    ui.filter = $('inference-filter').value;
    renderInferenceForm();
    ui.selected = $('inference-checkpoint').value || ui.selected;
  };
  $('inference-checkpoint').onchange = () => {
    ui.selected = $('inference-checkpoint').value;
    ui.message = '';
    renderInferenceForm();
  };
  $('inference-device').onchange = () => {
    ui.device = $('inference-device').value;
  };
  $('inference-border').oninput = () => {
    ui.borderIgnore = $('inference-border').value;
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
      device: ui.device,
      border_ignore: ui.borderIgnore === '' ? null : Number(ui.borderIgnore)
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
  for (const target of sample.targets)
    draw(target.box, target.detected ? BOX_COLORS.found : target.ignored ? BOX_COLORS.ignored : BOX_COLORS.missed, false);
  for (const detection of sample.detections)
    draw(detection.box, BOX_COLORS[detection.classification], sample.labeled, detection.score);
  if (showBelow)
    for (const detection of sample.below_threshold) draw(detection.box, BOX_COLORS.below, true, detection.score);
  context.setLineDash([]);
}

function boxLegend(labeled, ignoring = false) {
  const key = (color, text, dashed = false) => `<span>` +
    `<i class="${dashed ? 'dashed' : ''}" style="border-color:${color}"></i>${text}</span>`;
  return `<div class="box-legend">${labeled
    ? key(BOX_COLORS.found, 'labeled object, found') + key(BOX_COLORS.missed, 'labeled object, missed') +
      key(BOX_COLORS.tp, 'correct detection', true) + key(BOX_COLORS.fp, 'false alarm', true) +
      (ignoring ? key(BOX_COLORS.ignored, 'cut by the frame edge, not counted') : '')
    : key(BOX_COLORS.prediction, 'detected object')}${key(BOX_COLORS.below, 'below the confidence threshold', true)}` +
      `</div>`;
}
const frameURL = id => '/api/inference/sample?id=' + encodeURIComponent(id);

/* The line under a thumbnail: the frame, then what the model did with it. */
function frameCaption(sample, frame) {
  if (!sample.labeled) return `<b>${esc(frame.id)}</b><span>${frame.predicted_count} object` +
    `${frame.predicted_count === 1 ? '' : 's'} detected</span>`;
  const notes = [`${frame.predicted_count} detected of ${frame.true_count} labeled`];
  if (frame.fp) notes.push(`${frame.fp} false alarm${frame.fp === 1 ? '' : 's'}`);
  if (frame.fn) notes.push(`${frame.fn} missed`);
  return `<b>${esc(frame.id)}</b><span class="${frame.fp + frame.fn ? 'bad' : ''}">${esc(notes.join(' · '))}</span>`;
}

async function fillFrameGallery(frames) {
  const container = $('inference-gallery'),
    key = inferenceUI.resultKey + inferenceUI.page + inferenceUI.frameFilter ;
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
    button.innerHTML = frameCaption(sample, frame);
    button.setAttribute('aria-pressed', String(inferenceUI.frame === frame.id));
    button.onclick = () => {
      inferenceUI.frame = frame.id;
      container.querySelectorAll('.thumb-button').forEach(node => node.setAttribute('aria-pressed', 'false'));
      button.setAttribute('aria-pressed', 'true');
      showInferenceFrame();
    };
    item.append(canvas, button);
    container.append(item);
  }
}
async function showInferenceFrame() {
  const ui = inferenceUI,
    node = $('inference-frame');
  if (!node) return;
  if (!ui.frame) {
    node.innerHTML = '';
    return;
  }
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
    `</canvas>${boxLegend(sample.labeled, sample.targets.some(target => target.ignored))}</div>
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

/* The Frames tab: filters, one page of thumbnails, and the frame that was opened. */
function renderFrames(state) {
  const ui = inferenceUI,
    frames = visibleFrames(state),
    pages = Math.max(1, Math.ceil(frames.length / FRAMES_PER_PAGE));
  ui.page = Math.min(ui.page, pages - 1);
  const chips = frameFilters(state.labeled).map(([key, label]) => `<button type="button" data-filter="${key}` +
    `" aria-pressed="${ui.frameFilter === key}">${label}</button>`).join('');
  $('inference-panel').innerHTML = card('Frames', `<div class="controls"><div class="chips">${chips}</div>` +
    `<span class="spacer"></span>
      <button id="inference-prev" ${ui.page === 0 ? 'disabled' : ''}>Previous</button>` +
        `<span class="muted">${frames.length} frames · page ${ui.page + 1} of ${pages}</span>` +
        `<button id="inference-next" ${ui.page + 1 >= pages ? 'disabled' : ''}>Next</button></div>
    ${frames.length ? '<div id="inference-gallery" class="gallery"></div>' : empty('No frame matches this filter.')}` +
      `${boxLegend(state.labeled, state.evaluation.border_ignore > 0)}`) + '<div id="inference-frame"></div>';
  $('inference-panel').querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
    ui.frameFilter = button.dataset.filter;
    ui.page = 0;
    renderFrames(state);
  });
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

/* The Summary tab of a labeled result: the four numbers, then how the objects and the frames came out. */
function labeledSummaryHTML(state) {
  const metrics = state.metrics,
    detected = metrics.tp + metrics.fp,
    exact = Math.round(metrics.exact_count_accuracy * metrics.samples);
  const tiles = tile('Objects counted', String(detected), `${metrics.objects} labeled`) +
    tile('Exact-count frames', percentText(metrics.exact_count_accuracy), `${exact} of ${metrics.samples} frames`) +
    tile('Count MAE', decimalText(metrics.count_mae), 'mean error per frame') +
    tile('F1', percentText(metrics.f1), `precision ${percentText(metrics.precision)} · recall ` +
      `${percentText(metrics.recall)}`);
  return `<div class="metrics">${tiles}</div>` + card('Found, missed and false alarms',
    `<div class="fill">${tray(metrics.tp, metrics.fp, metrics.fn)}</div>`);
}

/* The Threshold tab: what the confidence threshold does to precision, recall and F1, with the numbers behind it. */
function thresholdHTML(state) {
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
    series: curveSeries.quality.map(series => ({ ...series, key: series.key.replace('val_', '') })),
    title: row => `Confidence ≥ ${row.confidence}`,
    marker: { x: saved, label: `saved ${saved}` }
  });
  const note = state.request.source === 'val' ?
    'Computed on the validation set that also chose the checkpoint, so the best threshold here is optimistic.' :
    'The same detections, accepted at other thresholds. The model and the NMS threshold are unchanged.';
  const rows = table([
    ['Confidence ≥', 'confidence'],
    ['Precision', 'precision', percentText],
    ['Recall', 'recall', percentText],
    ['F1', 'f1', percentText],
    ['Found', 'tp'],
    ['False alarms', 'fp'],
    ['Missed', 'fn'],
    ['Count MAE', 'count_mae', value => decimalText(value)]
  ], state.sweep, row => row.confidence === saved, true);
  return card('Effect of the confidence threshold', sweep +
    `<details class="plain"><summary>Table of the same numbers</summary>${rows}</details>`, note);
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

function unlabeledSummaryHTML(state) {
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

function resultHeadHTML(state) {
  const ui = inferenceUI,
    checkpoint = ui.checkpoints?.find(row => row.id === state.request.checkpoint_id),
    thresholds = state.evaluation;
  const chips = [
    `epoch ${state.checkpoint_epoch}`, `${state.frames.length} frames`, `confidence ≥ ${thresholds.confidence}`,
    `NMS IoU ${thresholds.nms_iou}`, ...(state.labeled ? [`correct at IoU ≥ ${thresholds.match_iou}`] : []),
    ...(thresholds.border_ignore > 0 ? [`cut objects under ${thresholds.border_ignore} samples not counted`] : []),
    state.device
  ];
  const where = [
    ['Predictions and metrics', state.output_directory, true],
    ['Data', state.data_root, true],
    ...(state.request.source === 'folder' ? [['Pixels', signalText(state.preprocessing), false]] : [])
  ];
  return `<section class="card result-head"><div class="run-head"><h3>${esc(checkpoint ? checkpoint.name : 'Checkpoint')}` +
    ` on the ${esc(SOURCE_TEXT[state.request.source])}</h3>` +
    `<span class="spacer"></span>${checkpoint ? `<button id="inference-open-run">Open the run</button>` : ''}</div>` +
    `<ul class="facts">${chips.map(text => `<li class="chip">${esc(text)}</li>`).join('')}</ul>` +
    `<details class="plain"><summary>Where the files are</summary><dl class="kv" style="margin-top:10px">` +
    `${where.map(([label, value, mono]) => `<dt>${esc(label)}</dt><dd${mono ? ' class="mono"' : ''}>` +
      `${esc(value)}</dd>`).join('')}</dl></details></section>`;
}

function resultTabs(state) {
  const tabs = [['summary', 'Summary']];
  if (state.labeled) tabs.push(['threshold', 'Confidence threshold']);
  tabs.push(['frames', 'Frames', state.frames.length]);
  return tabs;
}

/* The body of the open tab; switching tabs redraws only this. */
function renderResultPanel() {
  const ui = inferenceUI,
    state = ui.state,
    panel = $('inference-panel');
  if (!panel || !state || state.status !== 'completed') return;
  if (!resultTabs(state).some(([key]) => key === ui.tab)) ui.tab = 'summary';
  document.querySelectorAll('[data-result-tab]').forEach(button =>
    button.setAttribute('aria-selected', String(button.dataset.resultTab === ui.tab)));
  if (ui.tab === 'frames') {
    renderFrames(state);
    return;
  }
  panel.innerHTML = ui.tab === 'threshold' ? thresholdHTML(state) :
    state.labeled ? labeledSummaryHTML(state) : unlabeledSummaryHTML(state);
}

function renderInferenceResult(force = false) {
  const ui = inferenceUI,
    state = ui.state,
    node = $('inference-result');
  if (!node) return;
  if ($('inference-head') && (!state || state.status !== 'completed')) $('inference-head').innerHTML = '';
  if (!state || state.status === 'running') {
    node.innerHTML = state ? '' : `<div class="empty-state"><h3>No result yet</h3>` +
      `<p>Choose a checkpoint and the data, then run inference.</p></div>`;
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
    ui.tab = 'summary';
    ui.frameFilter = 'all';
    ui.page = 0;
    ui.frame = null;
  }
  ui.resultKey = state.id;
  node.dataset.key = state.id;
  $('inference-head').innerHTML = resultHeadHTML(state);
  node.innerHTML = tabStrip(resultTabs(state), ui.tab, 'result-tab', 'Parts of the result') +
    '<div id="inference-panel"></div>';
  node.querySelectorAll('[data-result-tab]').forEach(button => button.onclick = () => {
    ui.tab = button.dataset.resultTab;
    renderResultPanel();
  });
  if ($('inference-open-run')) $('inference-open-run').onclick = () => openRun(state.request.checkpoint_id);
  renderResultPanel();
}

async function renderInference() {
  const ui = inferenceUI;
  $('content').innerHTML = pageHeader('Inference', 'Run a trained checkpoint on its validation, test or training ' +
    'data, or on a new folder, and see what it counted, found and missed.') +
    '<div id="inference-head"></div>' +
    '<div class="split"><div id="inference-form"><p>Loading…</p></div><div id="inference-result"></div></div>';
  try {
    if (!ui.checkpoints) ui.checkpoints = await request('/api/inference/checkpoints');
    if (!ui.state) ui.state = (await request('/api/inference/status')).inference;
  } catch (error) {
    if ($('inference-form')) $('inference-form').innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  if (section !== 'Inference') return;
  // The first visit shows the run that is on the screen of the server already: its checkpoint and its data.
  if (!ui.synced && ui.state?.request) {
    ui.selected = ui.state.request.checkpoint_id;
    ui.source = ui.state.request.source;
    ui.folder = ui.state.request.folder || '';
  }
  ui.synced = true;
  if (!ui.checkpoints.some(row => row.id === ui.selected)) ui.selected = ui.checkpoints[0]?.id || null;
  renderInferenceForm();
  renderInferenceResult(true);
  scheduleInferencePoll();
}

/* Other sections send you here with a run and the data to run it on. */
function openInference(experimentId, source) {
  inferenceUI.selected = experimentId;
  if (source) inferenceUI.source = source;
  inferenceUI.checkpoints = null;
  inferenceUI.message = '';
  setSection('Inference');
}

renderers.Inference = renderInference;
