/* Model section: size, cost and geometry of one detector at once; its speed on request. */
'use strict';

const modelUI = {
  options: null,
  source: 'config',
  name: null,
  time: 40,
  summary: null,
  summaryKey: '',
  loading: false,
  message: '',
  device: 'auto',
  batch: null,
  bench: null,
  benchMessage: '',
  timer: null,
};

function countText(value) {
  if (!num(value)) return '–';
  for (const [size, unit] of [
      [1e9, 'G'],
      [1e6, 'M'],
      [1e3, 'k']
    ])
    if (Math.abs(value) >= size) return (value / size).toPrecision(3) + ' ' + unit;
  return String(Math.round(value));
}

function bytesText(value) {
  if (!num(value)) return '–';
  for (const [size, unit] of [
      [2 ** 30, 'GiB'],
      [2 ** 20, 'MiB'],
      [2 ** 10, 'KiB']
    ])
    if (value >= size) return (value / size).toPrecision(3) + ' ' + unit;
  return value + ' B';
}
const msText = value => num(value) ? (value < 10 ? value.toFixed(2) : value.toPrecision(3)) + ' ms' : '–';
const rateText = value => num(value) ? countText(value) + '/s' : '–';
const pairText = pair => pair
  ? pair.map(value => Number.isInteger(value) ? value : Number(value.toPrecision(3))).join(' × ')
  : 'global';

function shareCell(value) {
  if (!num(value)) return '–';
  return `<span class="share"><i style="width:${(value * 100).toFixed(1)}%"></i></span> ${percentText(value)}`;
}
const modelGroup = (title, tiles) => `<div class="form-section"><h4>${esc(title)}</h4>` +
  `<div class="metrics">${tiles.join('')}</div></div>`;

function modelChoices() {
  const ui = modelUI;
  return ui.source === 'config' ?
    ui.options.configs.map(row => ({ value: row.name, label: `${row.name} · ${row.architecture}` })) :
    ui.options.checkpoints.map(row => ({ value: row.id, label: checkpointLabel(row) }));
}

function chooseModel() {
  const ui = modelUI,
    rows = modelChoices();
  if (!rows.some(row => row.value === ui.name)) {
    const preferred = ui.source === 'config' ? rows.find(row => row.value === ui.options.default_config) : null;
    ui.name = (preferred || rows[0])?.value || null;
  }
}

function modelFormHTML() {
  const ui = modelUI,
    rows = modelChoices();
  const sources = [
    ['config', 'Model YAML'],
    ['checkpoint', 'Trained checkpoint']
  ];
  const picker = rows.length ?
    `<select id="model-name">${rows.map(row => `<option value="${esc(row.value)}" ` +
    `${row.value === ui.name ? 'selected' : ''}>${esc(row.label)}</option>`).join('')}</select>` :
    `<select id="model-name" disabled>` +
    `<option>${ui.source === 'config'
      ? 'No valid YAML next to --config'
      : 'No trained checkpoint in the runs folder'}</option></select>`;
  return `<section class="card"><div class="actions"><div class="segmented" role="group" aria-label="Model source">` +
    `${sources.map(([key, label]) => `<button type="button" data-model-source="${key}" aria-pressed="` +
    `${ui.source === key}">${label}</button>`).join('')}</div>
      <span class="message">` +
        `${ui.source === 'config'
          ? 'The architecture of a YAML with freshly initialised weights. Size and cost do not depend on the weights.'
          : 'A trained run: the same numbers, and a speed measurement with its real detections.'}` +
        `</span></div>
    <div class="form-grid" style="margin-bottom:0">
      <label class="field wide"><span>Model</span>${picker}</label>
      <label class="field"><span>Signal length T<small>time steps</small></span>` +
        `<input id="model-time" type="number" min="1" max="4096" step="1" value="${esc(ui.time)}"></label>
    </div>${ui.message ? `<p class="error">${esc(ui.message)}</p>` : ''}</section>`;
}

function summaryHTML(m) {
  const frame = `${m.input_shape[2]} × ${m.input_shape[3]}`;
  const field = m.receptive_field_sensor_time;
  const grid = m.output_grid_sensor_time.map(pairText).join(', ');
  const stride = m.output_stride_sensor_time.map(pairText).join(', ');
  let html = `<p class="legend">${esc(m.label)} · input B × C × sensor × time = ` +
    `${esc(m.input_shape.join(' × '))}</p>`;
  html += modelGroup('Size', [
    tile('Parameters', countText(m.parameters), `${m.parameters.toLocaleString('en')} weights`),
    tile('Weights, fp32', bytesText(m.weight_bytes), `fp16: ${bytesText(m.weight_bytes / 2)} · buffers ` +
      `${bytesText(m.buffer_bytes)}`),
    tile('Layers with weights', String(m.weighted_layers), 'convolutions and linear layers'),
    tile('Activations per frame', bytesText(m.activation_bytes),
      'all layer outputs, fp32; training keeps most of them'),
  ]);
  html += modelGroup(`Compute for one ${frame} frame`, [
    tile('MACs', countText(m.macs), 'multiply-accumulates of weighted layers'),
    tile('FLOPs', countText(m.flops), '2 × MACs'),
    tile('MACs per time step', countText(m.macs_per_time_step), 'cost grows linearly with T'),
    tile('MACs per parameter', countText(m.macs / m.parameters), 'how often each weight is reused'),
    tile('Elementwise operations', countText(m.elementwise_operations), 'norms, activations, pooling'),
  ]);
  html += modelGroup('Geometry', [
    tile('Receptive field, sensor × time', pairText(field), field ? `largest head context; frame ` +
      `${frame}` : 'a global block (SE/ECA/GRN) sees the whole frame'),
    tile('Output grid, sensor × time', grid, `stride ${stride}`),
    tile('Box candidates per frame', countText(m.predictions_per_frame), 'before the confidence filter and NMS'),
  ]);
  html += card('Where parameters and compute go', table([
    ['Part', 'name'],
    ['Parameters', 'parameters'],
    ['Share', 'fraction', shareCell],
    ['MACs', 'macs', countText],
    ['Share of MACs', 'mac_fraction', shareCell]
  ], m.parameter_distribution));
  html += card('Layers', table([
    ['Block', 'name'],
    ['Type', 'type'],
    ['Output B,C,S,T', 'shape', (value, row) => esc((value || row.shapes?.[0] || []).join(' × '))],
    ['Kernel S,T', 'kernel_sensor_time', pairText],
    ['Stride S,T', 'block_stride_sensor_time', pairText],
    ['Dilation S,T', 'dilation_sensor_time', pairText],
    ['RF S,T', 'receptive_field_sensor_time', value => value ? pairText(value) : '–'],
    ['Parameters', 'parameters'],
    ['MACs', 'macs', countText],
    ['Share of MACs', 'mac_fraction', shareCell]
  ], m.layers), m.rf_limitation);
  const coverage = m.dataset_coverage.map(row => ({
    ...row,
    axis: row.axis === 'sensor_width' ? 'sensors (width)' : 'time (length)',
    ratio: num(row.receptive_field) && num(row.p95) ? row.receptive_field / row.p95 : null
  }));
  html += card('Receptive field against the objects of the validation set',
    m.extents_error ? empty(m.extents_error) : table([
      ['Axis', 'axis'],
      ['Head', 'head_level'],
      ['Receptive field', 'receptive_field'],
      ['Median object', 'median'],
      ['p90', 'p90'],
      ['p95', 'p95'],
      ['RF / p95', 'ratio', value => num(value) ? value.toFixed(2) + '×' : '–']
    ], coverage),
    `Object sizes in cells of ${m.data_val}. A ratio below 1 means the largest objects do not fit in what one output ` +
    `cell sees.`);
  return html;
}
async function loadModelSummary() {
  const ui = modelUI;
  const key = `${ui.source}|${ui.name}|${ui.time}`;
  if (!ui.name || key === ui.summaryKey) return;
  ui.summaryKey = key;
  ui.loading = true;
  ui.message = '';
  renderModelSummary();
  try {
    const summary = await request(`/api/model/summary?source=${encodeURIComponent(ui.source)}&name=` +
      `${encodeURIComponent(ui.name)}&time=${encodeURIComponent(ui.time)}`);
    if (ui.summaryKey !== key) return;
    ui.summary = summary;
    if (ui.batch == null) ui.batch = summary.batch_size;
  } catch (error) {
    if (ui.summaryKey !== key) return;
    ui.summary = null;
    ui.message = error.message;
    renderModelForm();
  }
  ui.loading = false;
  if (section === 'Model') {
    renderModelSummary();
    renderBenchmark();
  }
}

function renderModelSummary() {
  const node = $('model-summary');
  if (!node) return;
  const ui = modelUI;
  node.innerHTML = ui.loading
    ? '<p class="legend">Building the model…</p>'
    : ui.summary ? summaryHTML(ui.summary) : '';
}

function benchmarkResultHTML(state) {
  const r = state.result,
    batch = state.batch_size;
  let html = `<p class="legend">Measured: ${esc(r.label)} · ${esc(r.device)} · ${r.frames} validation frames of ` +
    `${esc(r.frame_shape_sensor_time.join(' × '))} from ${esc(r.data_val)} · PyTorch ${esc(r.torch)}</p>`;
  const warnings = [];
  if (!r.trained) warnings.push('Untrained weights produce almost no box candidates, so decoding and NMS ' +
    'are faster than with a trained model.');
  if (warnings.length) html += warnings.map(text => `<p class="error">${esc(text)}</p>`).join('');
  html += modelGroup('Latency, one frame at a time', [
    tile('Model forward', msText(r.latency_forward.median_ms), `median; p90 ${msText(r.latency_forward.p90_ms)}` +
      `; transfer included`),
    tile('Full pipeline', msText(r.latency_pipeline.median_ms), `forward + decoding + NMS; p90 ` +
      `${msText(r.latency_pipeline.p90_ms)}`),
    tile('Decoding and NMS', msText(r.latency_postprocess.median_ms), 'median, on the CPU'),
  ]);
  html += modelGroup(`Throughput, batches of ${batch}`, [
    tile('Model forward', rateText(r.throughput_forward_fps), 'frames per second'),
    tile('Full pipeline', rateText(r.throughput_pipeline_fps), 'frames per second'),
    tile('Signal per second', rateText(r.throughput_pipeline_fps * r.time_steps_per_frame),
      'time steps through the full pipeline'),
    tile('Decoding and NMS', percentText(r.postprocess_share), 'share of the pipeline time'),
    tile('Peak memory', r.peak_inference_bytes == null ? 'CPU' : bytesText(r.peak_inference_bytes),
      'GPU memory above the loaded weights'),
  ]);
  const epoch = num(r.train_frames) ? Math.ceil(r.train_frames / batch) * r.step.median_ms / 1000 : null;
  html += modelGroup(`Training, batches of ${batch}`, [
    tile('Training step', msText(r.step.median_ms), `forward + loss + backward; p90 ${msText(r.step.p90_ms)}`),
    tile('Peak memory', r.peak_training_bytes == null ? 'CPU' : bytesText(r.peak_training_bytes),
      'activations and gradients of one step'),
    tile('Epoch, steps only', num(epoch) ? (epoch < 60 ? epoch.toFixed(1) + ' s' : durationText(epoch)) : '–',
      num(r.train_frames)
        ? `${r.train_frames} training frames; no data loading, validation or optimizer`
        : 'training data not readable'),
  ]);
  html += modelGroup('Output', [
    tile('Candidates per frame', num(r.candidates_per_frame) ? r.candidates_per_frame.toFixed(1) : '–',
      'after NMS, above ap_min_confidence'),
    tile('Detections per frame', num(r.detections_per_frame) ? r.detections_per_frame.toFixed(2) : '–',
      `confidence ≥ ${r.evaluation.confidence}`),
  ]);
  return html;
}

function benchmarkHTML() {
  const ui = modelUI,
    state = ui.bench,
    running = state?.status === 'running';
  const devices = ['auto', 'cuda', 'cpu'];
  let html = `<section class="card"><h3>Measured speed</h3>` +
    `<p class="legend">Runs the model on the validation frames of its own data. Nothing is measured until you start ` +
    `it; it takes a few seconds on a GPU and longer on a CPU. Frames are read from disk before timing.</p>
    <div class="form-grid">
      <label class="field"><span>Device<small>${esc(ui.options.device)}</small></span>` +
        `<select id="model-device">${devices.map(name => `<option ${name === ui.device ? 'selected' : ''} ` +
        `${name === 'cuda' && !ui.options.cuda ? 'disabled' : ''}>${name}</option>`).join('')}</select></label>
      <label class="field"><span>Batch size</span><input id="model-batch" type="number" min="1" max="1024" step="1" ` +
        `value="${esc(ui.batch ?? '')}"></label>
    </div>
    <div class="actions" style="margin-bottom:0"><button id="model-measure" class="primary" ` +
      `${running || !ui.name ? 'disabled' : ''}>Measure on the validation set</button>` +
      `<span class="message${ui.benchMessage ? ' error' : ''}" role="status">${esc(ui.benchMessage)}</span></div>`;
  if (running) html += `<div style="margin-top:14px"><div class="progress-meta"><span>${esc(state.stage)}</span>` +
    `<span>${state.total ? `${state.done} of ${state.total}` : ''}</span>` +
    `</div>${progressBar(state.done, state.total, state.stage)}</div>`;
  if (state?.status === 'failed') html += `<p class="error">${esc(state.error)}</p>`;
  if (state?.status === 'completed') html += benchmarkResultHTML(state);
  return html + '</section>';
}

function renderBenchmark() {
  const ui = modelUI,
    node = $('model-benchmark');
  if (!node || !ui.options) return;
  node.innerHTML = benchmarkHTML();
  $('model-device').onchange = () => {
    ui.device = $('model-device').value;
  };
  $('model-batch').oninput = () => {
    ui.batch = $('model-batch').value.trim() === '' ? null : Number($('model-batch').value);
  };
  $('model-measure').onclick = startBenchmark;
}
async function startBenchmark() {
  const ui = modelUI;
  ui.benchMessage = '';
  try {
    ui.bench = (await postJSON('/api/model/benchmark', {
      source: ui.source, name: ui.name, device: ui.device, batch_size: ui.batch
    })).benchmark;
  } catch (error) {
    ui.benchMessage = error.message;
  }
  renderBenchmark();
  scheduleBenchmarkPoll();
}

function scheduleBenchmarkPoll() {
  clearTimeout(modelUI.timer);
  if (modelUI.bench?.status !== 'running') return;
  modelUI.timer = setTimeout(async () => {
    try {
      modelUI.bench = (await request('/api/model/benchmark')).benchmark;
    } catch (error) { /* keep the last state */ }
    if (section === 'Model') renderBenchmark();
    scheduleBenchmarkPoll();
  }, 400);
}

function renderModelForm() {
  const ui = modelUI,
    node = $('model-form');
  if (!node) return;
  node.innerHTML = modelFormHTML();
  node.querySelectorAll('[data-model-source]').forEach(button => button.onclick = () => {
    ui.source = button.dataset.modelSource;
    chooseModel();
    renderModelForm();
    loadModelSummary();
  });
  if (!$('model-name').disabled) $('model-name').onchange = () => {
    ui.name = $('model-name').value;
    loadModelSummary();
  };
  $('model-time').onchange = () => {
    const value = Number($('model-time').value);
    if (!Number.isInteger(value) || value < 1 || value > 4096) {
      ui.message = 'Signal length must be a whole number from 1 to 4096.';
      renderModelForm();
      return;
    }
    ui.time = value;
    ui.message = '';
    loadModelSummary();
  };
}
async function renderModel() {
  const ui = modelUI;
  $('content').innerHTML = `<h2>Model</h2><p class="section-hint">Size, cost and geometry of one detector, computed ` +
    `from its architecture at once. Speed and memory are measured on real frames when you ask for it.</p>
    <div id="model-form"><p class="legend">Loading models…</p></div><div id="model-summary"></div>` +
      `<div id="model-benchmark"></div>`;
  try {
    const [options, bench] = await Promise.all([request('/api/model/options'), request('/api/model/benchmark')]);
    ui.options = options;
    ui.bench = bench.benchmark;
  } catch (error) {
    $('model-form').innerHTML = `<p class="error">${esc(error.message)}</p>`;
    return;
  }
  if (section !== 'Model') return;
  if (ui.source === 'checkpoint' && !ui.options.checkpoints.length) ui.source = 'config';
  chooseModel();
  // Coming back to the section shows the last summary at once instead of reloading it.
  ui.summaryKey = ui.summary && ui.summaryKey === `${ui.source}|${ui.name}|${ui.time}` ? ui.summaryKey : '';
  renderModelForm();
  renderModelSummary();
  renderBenchmark();
  loadModelSummary();
  scheduleBenchmarkPoll();
}
