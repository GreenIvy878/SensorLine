/* Experiments section: the finished runs side by side, and one run opened into its results, curves, architecture
   and details. Reads cached JSON only. */
'use strict';
const experimentUI = {
  rows: null,
  detail: null,
  tab: 'results',
  analysis: null,
  model: null,
  token: 0,
  filter: '',
  status: 'all',
  sort: 'count_mae',
  descending: false
};
const primaryMetrics = ['count_mae', 'map50', 'map50_95', 'precision', 'recall', 'f1', 'exact_count_accuracy'];
const jsonText = value => esc(JSON.stringify(value, null, 2));
const experimentURL = (path, id) => `${path}?id=${encodeURIComponent(id)}`;
/* Rates read as percentages, everything else as plain numbers; counts stay whole. */
const RATE_METRICS = new Set(['precision', 'recall', 'f1', 'map50', 'map75', 'map50_95', 'exact_count_accuracy']);
const WHOLE_METRICS = new Set(['samples', 'objects', 'tp', 'fp', 'fn', 'ignored_objects']);
const metricText = (key, value) => value == null ? '–' : RATE_METRICS.has(key) ? percentText(value) :
  WHOLE_METRICS.has(key) ? String(value) : decimalText(value, 3);
const dateText = timestamp => timestamp ? new Date(timestamp).toLocaleString('en-GB', {
  day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'
}) : '–';

async function experimentAction(action) {
  try {
    await action();
  } catch (error) {
    if (section === 'Experiments' && $('experiment-message')) $('experiment-message').textContent = error.message;
  }
}

/* Other sections send you here to read one run: the list is refreshed first, so a new run is in it. */
async function openRun(id) {
  try {
    experimentUI.rows = await request('/api/experiments');
    setSection('Experiments');
    await openExperiment(id);
  } catch (error) {
    if ($('experiment-message')) $('experiment-message').textContent = error.message;
  }
}

async function renderExperiments() {
  const token = ++experimentUI.token;
  const head = pageHeader('Experiments', 'Every training run found in the runs folder. Open one for its curves, ' +
    'architecture and files, or run it on its test data.');
  if (!experimentUI.rows) {
    $('content').innerHTML = head + '<p id="experiment-message" role="status">Loading runs…</p>';
    try {
      experimentUI.rows = await request('/api/experiments');
    } catch (error) {
      if (token === experimentUI.token) $('experiment-message').textContent = error.message;
      return;
    }
    if (token !== experimentUI.token || section !== 'Experiments') return;
  }
  $('content').innerHTML = '<div id="experiment-body"></div>';
  if (experimentUI.detail) renderExperimentDetail();
  else renderExperimentList();
}

/* Columns of the list: [key, title, how a value reads, where it comes from]. Metrics are those of the best epoch. */
const runColumns = [
  ['f1', 'F1', row => metricText('f1', row.best_metrics?.f1)],
  ['count_mae', 'Count MAE', row => metricText('count_mae', row.best_metrics?.count_mae)],
  ['test_f1', 'Test F1', row => metricText('f1', row.best_test_metrics?.f1), true],
  ['test_count_mae', 'Test MAE', row => metricText('count_mae', row.best_test_metrics?.count_mae), true],
  ['parameters', 'Parameters', row => num(row.parameters) ? countText(row.parameters) : '–'],
  ['training_seconds', 'Time', row => num(row.training_seconds) ? durationText(row.training_seconds) : '–']
];
const BETTER_WHEN_LOWER = new Set(['count_mae', 'test_count_mae', 'training_seconds', 'parameters']);

function runSortValue(row, key) {
  if (key === 'test_f1') return row.best_test_metrics?.f1;
  if (key === 'test_count_mae') return row.best_test_metrics?.count_mae;
  return row.best_metrics?.[key] ?? row[key];
}

function renderExperimentList() {
  const ui = experimentUI,
    runs = ui.rows.filter(row => row.id),
    withTest = runs.some(row => row.best_test_metrics);
  const counts = status => status === 'all' ? runs.length : runs.filter(row => row.status === status).length;
  const rows = runs.filter(row => (ui.status === 'all' || row.status === ui.status) &&
    `${row.name} ${row.architecture} ${row.id}`.toLowerCase().includes(ui.filter.toLowerCase()));
  rows.sort((a, b) => {
    const av = runSortValue(a, ui.sort),
      bv = runSortValue(b, ui.sort);
    if (av == null) return bv == null ? 0 : 1;
    if (bv == null) return -1;
    const comparison = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv));
    return ui.descending ? -comparison : comparison;
  });
  const columns = runColumns.filter(column => !column[3] || withTest);
  const arrow = key => ui.sort === key ? `<span aria-hidden="true">${ui.descending ? '▼' : '▲'}</span>` : '';
  const sortable = (key, title) => `<th class="num"><button class="sort" data-sort="${key}">${esc(title)}${arrow(key)}` +
    `</button></th>`;
  const statusChips = ['all', 'completed', 'running', 'failed'].filter(status => status === 'all' || counts(status))
    .map(status => `<button type="button" data-status="${status}" aria-pressed="${ui.status === status}">` +
      `${status === 'all' ? 'All' : STATUS_TEXT[status] || status} ${counts(status)}</button>`).join('');
  let html = pageHeader('Experiments', 'Every training run found in the runs folder. Open one for its curves, ' +
    'architecture and files, or run it on its test data.') +
    `<div class="controls"><input id="experiment-filter" class="compact" type="search" value="${esc(ui.filter)}" ` +
    `placeholder="Find a run" aria-label="Find a run by name or architecture"><div class="chips">${statusChips}</div>` +
    `<span class="spacer"></span><button id="experiment-refresh">Refresh</button></div>`;
  html += `<p id="experiment-message" class="message error" role="status"></p>`;
  html += rows.length ? `<div class="scroll"><table class="runs"><thead><tr>` +
    `<th><button class="sort" data-sort="timestamp">Run${arrow('timestamp')}</button></th><th>Architecture</th>` +
    `${columns.map(([key, title]) => sortable(key, title)).join('')}<th>Status</th><th></th></tr></thead><tbody>` +
    rows.map(row => `<tr><td class="run-name"><button class="experiment-open" data-id="${esc(row.id)}">${esc(row.name)}` +
      `</button><small>${esc(dateText(row.timestamp))}</small></td>` +
      `<td class="arch" title="${esc(row.architecture)}">${esc(row.architecture)}</td>` +
      `${columns.map(([, , read]) => `<td class="num">${esc(read(row))}</td>`).join('')}` +
      `<td>${pill(row.status, STATUS_TEXT[row.status] || row.status)}</td>` +
      `<td class="row-actions"><button class="quiet" data-infer="${esc(row.id)}"` +
      `${row.status === 'running' ? ' disabled' : ''}>Inference</button></td></tr>`).join('') +
    `</tbody></table></div>` : empty(runs.length ? 'No run matches the filter.' :
      'No training run in the runs folder yet.');
  html += `<details class="plain"><summary>How runs are compared</summary><p>Metrics are those of the best epoch on ` +
    `the validation split. An ordinary run keeps the epoch with the lowest count MAE, then the highest AP50, then ` +
    `the lowest loss; searches and saved-parameter runs use their recorded objective. Compare the data and the ` +
    `settings of two runs before reading a ranking. Training time is wall time for training and validation, not ` +
    `inference latency. ${withTest ? 'Test columns come from the test split monitored during training: it never ' +
      'chose a checkpoint.' : ''}</p></details>`;
  const scored = runs.filter(row => row.status === 'completed' && num(row.parameters) && num(runSortValue(row, 'f1')));
  const metric = primaryMetrics.includes(ui.sort) ? ui.sort : 'f1';
  // Runs of one size would stack in a single column: the plot says nothing until sizes differ.
  if (new Set(scored.map(row => row.parameters)).size > 1) html += card('Quality against model size', chart([{
    name: metricName(metric),
    points: scored.filter(row => num(row.best_metrics?.[metric]))
      .map(row => ({ x: row.parameters, y: row.best_metrics[metric], label: row.name }))
  }], 'Parameters', metricName(metric), true, false, { yPercent: RATE_METRICS.has(metric) }));
  const invalid = ui.rows.filter(row => !row.id);
  if (invalid.length) html += card('Unreadable artifacts', table([
    ['Directory', 'directory'],
    ['Issue', 'error']
  ], invalid));
  $('experiment-body').innerHTML = html;
  $('experiment-filter').oninput = () => {
    ui.filter = $('experiment-filter').value;
    renderExperimentList();
    const input = $('experiment-filter');
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  };
  document.querySelectorAll('[data-status]').forEach(node => node.onclick = () => {
    ui.status = node.dataset.status;
    renderExperimentList();
  });
  document.querySelectorAll('[data-sort]').forEach(node => node.onclick = () => {
    const key = node.dataset.sort;
    // A first click puts the best value (or the newest run) on top; a second reverses it.
    ui.descending = ui.sort === key ? !ui.descending : key === 'timestamp' || !BETTER_WHEN_LOWER.has(key);
    ui.sort = key;
    renderExperimentList();
  });
  $('experiment-refresh').onclick = () => experimentAction(async () => {
    ui.rows = await request('/api/experiments');
    renderExperimentList();
  });
  document.querySelectorAll('.experiment-open').forEach(node =>
    node.onclick = () => experimentAction(() => openExperiment(node.dataset.id)));
  document.querySelectorAll('[data-infer]').forEach(node => node.onclick = () => openInference(node.dataset.infer));
}

async function openExperiment(id) {
  const ui = experimentUI;
  ui.detail = await request(experimentURL('/api/experiment', id));
  ui.tab = 'results';
  ui.model = ui.analysis = null;
  // Training does not write the model summary; the first opening of a finished run prepares it.
  let failure = '';
  if (!ui.detail.analysis_ready && ui.detail.status === 'completed') {
    try {
      await request(experimentURL('/api/experiment-prepare', id), { method: 'POST' });
      ui.detail.analysis_ready = true;
    } catch (error) {
      failure = error.message;
    }
  }
  if (ui.detail.analysis_ready) await loadExperimentAnalysis();
  renderExperimentDetail();
  if (failure && $('experiment-message')) $('experiment-message').textContent = failure;
}
async function loadExperimentAnalysis() {
  const id = experimentUI.detail.id;
  const [analysis, model] = await Promise.all([
    request(experimentURL('/api/experiment-analysis', id)), request(experimentURL('/api/experiment-model', id))
  ]);
  if (experimentUI.detail.id !== id) return;
  Object.assign(experimentUI, { analysis, model });
}

/* Validation of the best epoch and of the last one, and the test split when the run monitored one. */
function metricsTable(d) {
  const best = d.best_metrics || {},
    final = d.final_metrics || {},
    test = d.best_test_metrics || null;
  const keys = [...new Set([...Object.keys(best), ...Object.keys(final), ...Object.keys(test || {})])];
  const columns = [['Metric', 'name'], ['Validation, best epoch', 'best'], ['Validation, last epoch', 'final']];
  if (test) columns.push(['Test, best epoch', 'test']);
  return table(columns, keys.map(key => ({
    name: metricName(key), best: metricText(key, best[key]), final: metricText(key, final[key]),
    test: metricText(key, test?.[key])
  })), () => false, true);
}

function resultsPanel(d) {
  const best = d.best_metrics || {},
    test = d.best_test_metrics;
  const tiles = tile('F1', percentText(best.f1), `precision ${percentText(best.precision)} · recall ` +
    `${percentText(best.recall)}`) +
    tile('Count MAE', decimalText(best.count_mae), `${percentText(best.exact_count_accuracy)} of frames exact`) +
    tile('mAP@50', percentText(best.map50), `mAP@50–95 ${percentText(best.map50_95)}`) +
    (test ? tile('Test F1', percentText(test.f1), `count MAE ${decimalText(test.count_mae)} · ` +
      `${test.samples} frames`) : '');
  const selection = d.checkpoint_selection ?
    `${metricName(d.checkpoint_selection.metric)} (${d.checkpoint_selection.direction})` :
    'count MAE, then AP50, then loss';
  return `<div class="metrics">${tiles}</div>` +
    card('All metrics', metricsTable(d));
}

function curvesPanel(d) {
  const job = { history: d.history, epochs: Math.max(d.history.length, 2), best_epoch: d.best_epoch };
  if (!d.history.length) return empty('This run recorded no epochs.');
  return '<div class="grid three">' +
    card('Loss', curve('exp-loss', job, 'loss', 'Training and validation loss by epoch', { clip: true })) +
    card('Detection quality, %', curve('exp-quality', job, 'quality', 'Precision, recall and F1 by epoch',
      { percent: true, clip: true })) +
    card('Count error per frame (MAE)', curve('exp-count', job, 'count', 'Count MAE by epoch', { clip: true })) +
    '</div>' + `<details class="plain"><summary>Epoch table</summary>${historyTable(d.history)}</details>`;
}

function modelLayerTable(model) {
  return table([
    ['Block', 'name'],
    ['Type', 'type'],
    ['Output B×C×S×T', 'shape', (v, row) => esc((v || row.shapes || []).join(' × '))],
    ['Kernel S×T', 'kernel_sensor_time', v => esc(pairText(v))],
    ['Stride S×T', 'block_stride_sensor_time', v => esc(pairText(v))],
    ['Dilation S×T', 'dilation_sensor_time', v => esc(pairText(v))],
    ['Receptive field S×T', 'receptive_field_sensor_time', v => v ? esc(pairText(v)) : '–'],
    ['Parameters', 'parameters']
  ], model.layers);
}

function architecturePanel(d) {
  const m = experimentUI.model;
  const link = `<p class="legend">Size, compute, speed on real frames and the receptive field against the objects of ` +
    `the validation set: <button class="experiment-open" id="experiment-model-link">open this run in the Model ` +
    `section</button>.</p>`;
  if (!m) return link + (d.status === 'completed' ?
    `<div class="empty-state"><h3>The layer summary has not been prepared</h3><p>It is built once from the ` +
    `checkpoint.</p><button id="experiment-prepare" class="primary">Prepare the summary</button></div>` :
    empty('The layer summary is built when the run has finished.'));
  return link + card('Layers and receptive field', `<p class="legend">Input B×C×S×T: ${esc(m.input_shape.join(' × '))}. ` +
    `${esc(m.axes)}</p>${modelLayerTable(m)}`, m.rf_limitation) +
    card('Parameters by major block', table([
      ['Block', 'name'],
      ['Parameters', 'parameters'],
      ['Share', 'fraction', value => percentText(value)]
    ], m.parameter_distribution)) +
    card('Receptive field against the objects', table([
      ['Axis', 'axis', value => value === 'sensor_width' ? 'sensors (width)' : 'time (length)'],
      ['Head', 'head_level'],
      ['Receptive field', 'receptive_field'],
      ['Median object', 'median'],
      ['p90', 'p90'],
      ['p95', 'p95']
    ], m.dataset_coverage));
}

function detailsPanel(d) {
  const facts = [
    ['Run folder', d.directory, true],
    ['Checkpoint', d.checkpoint, true],
    ['Seed', d.seed ?? '–', false],
    ['Architecture', d.architecture, false]
  ];
  return card('Run', `<dl class="kv">${facts.map(([label, value, mono]) => `<dt>${esc(label)}</dt>` +
    `<dd${mono ? ' class="mono"' : ''}>${esc(value)}</dd>`).join('')}</dl>`) +
    (d.error ? `<p class="error">${esc(d.error)}</p>` : '') +
    `<details class="plain"><summary>Configuration used</summary><pre>${jsonText(d.config)}</pre></details>` +
    `<details class="plain"><summary>Layers as built</summary><pre>${esc(d.model_info.structure || d.architecture)}` +
    `</pre></details>` +
    `<details class="plain"><summary>Datasets of the run</summary><pre>${jsonText(d.dataset)}</pre></details>` +
    `<details class="plain"><summary>Provenance</summary><pre>${jsonText(d.provenance)}</pre></details>` +
    (experimentUI.analysis ? `<p><a class="button" href="${experimentURL('/api/experiment-analysis', d.id)}">` +
      `Export experiment_analysis.json</a></p>` : '');
}

function renderExperimentDetail() {
  const ui = experimentUI,
    d = ui.detail;
  const facts = [
    pill(d.status, STATUS_TEXT[d.status] || d.status), dateText(d.timestamp),
    num(d.parameters) ? `${countText(d.parameters)} parameters` : null,
    d.best_epoch ? `best epoch ${d.best_epoch}` : null,
    num(d.training_seconds) ? `trained in ${durationText(d.training_seconds)}` : null
  ].filter(Boolean);
  const done = d.status === 'completed';
  const actions = `<button id="experiment-infer" class="primary" ${done ? '' : 'disabled'}>Run inference</button>` +
    `${d.best_test_metrics ? '<button id="experiment-infer-test">Run on the test set</button>' : ''}` +
    `<button id="experiment-model" ${done ? '' : 'disabled'}>Model</button>`;
  $('experiment-body').innerHTML = `<p class="crumb"><button class="experiment-open" id="experiment-back">` +
    `← All experiments</button></p>` + pageHeader(d.name, d.architecture, actions) +
    `<ul class="facts">${facts.map(item => `<li>${item.startsWith('<') ? item : `<span class="chip">${esc(item)}</span>`}` +
      `</li>`).join('')}</ul><p id="experiment-message" class="message error" role="status"></p>` +
    tabStrip([['results', 'Results'], ['curves', 'Curves'], ['architecture', 'Architecture'], ['details', 'Details']],
      ui.tab, 'tab', 'Parts of the run') + '<div id="experiment-panel"></div>';
  $('experiment-back').onclick = () => {
    ui.detail = null;
    renderExperimentList();
  };
  $('experiment-infer').onclick = () => openInference(d.id);
  if ($('experiment-infer-test')) $('experiment-infer-test').onclick = () => openInference(d.id, 'test');
  $('experiment-model').onclick = () => openModel(d.id);
  document.querySelectorAll('[data-tab]').forEach(node => node.onclick = () => {
    ui.tab = node.dataset.tab;
    renderExperimentDetail();
  });
  const panel = $('experiment-panel');
  panel.innerHTML = { results: resultsPanel, curves: curvesPanel, architecture: architecturePanel,
    details: detailsPanel }[ui.tab](d);
  if ($('experiment-model-link')) $('experiment-model-link').onclick = () => openModel(d.id);
  if ($('experiment-prepare')) $('experiment-prepare').onclick = () => experimentAction(async () => {
    $('experiment-prepare').disabled = true;
    $('experiment-message').textContent = 'Preparing the model summary…';
    try {
      await request(experimentURL('/api/experiment-prepare', d.id), { method: 'POST' });
      await loadExperimentAnalysis();
      $('experiment-message').textContent = '';
      renderExperimentDetail();
    } finally {
      if ($('experiment-prepare')) $('experiment-prepare').disabled = false;
    }
  });
}

renderers.Experiments = renderExperiments;
