/* Experiment context in the existing dashboard. Filters read cached JSON only. */
'use strict';
const experimentUI = {
  rows: null,
  detail: null,
  tab: 'Overview',
  analysis: null,
  model: null,
  token: 0,
  filter: '',
  status: 'all',
  sort: 'count_mae',
  descending: false
};
const primaryMetrics = ['count_mae', 'map50', 'map50_95', 'precision', 'recall', 'exact_count_accuracy'];
const jsonText = value => esc(JSON.stringify(value, null, 2));
const experimentURL = (path, id) => `${path}?id=${encodeURIComponent(id)}`;

async function experimentAction(action) {
  try {
    await action();
  } catch (error) {
    if (section === 'Experiments') $('experiment-message').textContent = error.message;
  }
}
async function renderExperiments() {
  const token = ++experimentUI.token;
  if (!experimentUI.rows) {
    $('content').innerHTML = '<h2>Experiments</h2><p id="experiment-message">Loading runs…</p>';
    try {
      experimentUI.rows = await request('/api/experiments');
    } catch (error) {
      if (token === experimentUI.token) $('experiment-message').textContent = error.message;
      return;
    }
    if (token !== experimentUI.token || section !== 'Experiments') return;
  }
  $('content').innerHTML = '<h2>Experiments</h2><p id="experiment-message" role="status"></p>' +
    '<div id="experiment-body"></div>';
  if (experimentUI.detail) renderExperimentDetail();
  else renderExperimentList();
}

function renderExperimentList() {
  const ui = experimentUI;
  let rows = ui.rows.filter(row => row.id && (ui.status === 'all' || row.status === ui.status) &&
    `${row.name} ${row.architecture} ${row.id}`.toLowerCase().includes(ui.filter.toLowerCase()));
  const sortValue = row => row.best_metrics?.[ui.sort] ?? row[ui.sort];
  rows.sort((a, b) => {
    const av = sortValue(a),
      bv = sortValue(b);
    if (av == null) return bv == null ? 0 : 1;
    if (bv == null) return -1;
    const comparison = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv));
    return ui.descending ? -comparison : comparison;
  });
  const sortKeys = [...primaryMetrics, 'parameters', 'best_epoch', 'training_seconds', 'name', 'timestamp', 'status'];
  let html = `<div class="controls"><label>Filter <input id="experiment-filter" value="${esc(ui.filter)}"></label>
    <label>Status <select id="experiment-status">${['all', 'completed', 'running', 'failed'].map(s => `<option ` +
      `${s === ui.status ? 'selected' : ''}>${s}</option>`).join('')}</select></label>
    <label>Rank / sort <select id="experiment-sort">${sortKeys.map(s => `<option ${s === ui.sort ? 'selected' : ''}>` +
      `${s}</option>`).join('')}</select></label>
    <label><input type="checkbox" id="experiment-descending" ${ui.descending ? 'checked' : ''}> Descending</label>
    <button id="experiment-refresh">Refresh runs</button></div>`;
  const view = rows.map(row => ({ ...row, best: row.best_metrics?.[ui.sort], final: row.final_metrics?.[ui.sort] }));
  html += table([
    ['Experiment', 'name', (name, row) => `<button class="experiment-open" data-id="${esc(row.id)}">${esc(name)}` +
      `</button><br><small>${esc(row.id)}</small>`],
    ['Architecture', 'architecture'],
    ['Parameters', 'parameters'],
    [`Best ${ui.sort}`, 'best'],
    [`Final ${ui.sort}`, 'final'],
    ['Best epoch', 'best_epoch'],
    ['Training s', 'training_seconds'],
    ['Timestamp', 'timestamp'],
    ['Status', 'status']
  ], view);
  html += `<p>Ordinary training selects the best checkpoint by validation count MAE, then AP50, then loss. Search ` +
    `and saved-parameter runs use their recorded objective; see Overview. Leaderboard uses recorded best-checkpoint ` +
    `metrics. Compare provenance before interpreting rankings. Timing is train + validation wall time, not inference ` +
    `latency.</p>`;
  if (primaryMetrics.includes(ui.sort)) html += card('Metric vs model complexity', chart([{
    name: ui.sort,
    points: rows.filter(r => r.status === 'completed')
      .map(r => ({ x: r.parameters, y: r.best_metrics?.[ui.sort], label: r.name }))
  }], 'Parameters', ui.sort, true, false, { yPercent: ui.sort !== 'count_mae' }));
  const invalid = ui.rows.filter(row => !row.id);
  if (invalid.length) html += card('Unreadable artifacts', table([
    ['Directory', 'directory'],
    ['Issue', 'error']
  ], invalid));
  $('experiment-body').innerHTML = html;
  $('experiment-filter').onchange = () => {
    ui.filter = $('experiment-filter').value;
    renderExperimentList();
  };
  $('experiment-status').onchange = () => {
    ui.status = $('experiment-status').value;
    renderExperimentList();
  };
  $('experiment-sort').onchange = () => {
    ui.sort = $('experiment-sort').value;
    ui.descending = ui.sort !== 'count_mae' && primaryMetrics.includes(ui.sort);
    renderExperimentList();
  };
  $('experiment-descending').onchange = () => {
    ui.descending = $('experiment-descending').checked;
    renderExperimentList();
  };
  $('experiment-refresh').onclick = () => experimentAction(async () => {
    ui.rows = await request('/api/experiments');
    renderExperimentList();
  });
  document.querySelectorAll('.experiment-open').forEach(node =>
    node.onclick = () => experimentAction(() => openExperiment(node.dataset.id)));
}
async function openExperiment(id) {
  const ui = experimentUI;
  ui.detail = await request(experimentURL('/api/experiment', id));
  ui.tab = 'Overview';
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
  if (failure) $('experiment-message').textContent = failure;
}
async function loadExperimentAnalysis() {
  const id = experimentUI.detail.id;
  const [analysis, model] = await Promise.all([
    request(experimentURL('/api/experiment-analysis', id)), request(experimentURL('/api/experiment-model', id))
  ]);
  if (experimentUI.detail.id !== id) return;
  Object.assign(experimentUI, { analysis, model });
}

function metricsTable(best, final) {
  const keys = [...new Set([...Object.keys(best), ...Object.keys(final)])];
  return table([
    ['Metric', 'metric'],
    ['Best checkpoint', 'best'],
    ['Last epoch', 'final']
  ], keys.map(metric => ({ metric, best: best[metric], final: final[metric] })));
}

function renderExperimentDetail() {
  const ui = experimentUI,
    d = ui.detail;
  $('experiment-body').innerHTML = `<div class="controls"><button id="experiment-back">← Experiments</button>` +
    `<strong>${esc(d.name)}</strong><span>${esc(d.status)} · ${esc(d.timestamp)}</span></div>
    <div class="controls">${['Overview', 'Training', 'Model'].map(tab => `<button data-tab="${tab}` +
      `" class="experiment-tab ${ui.tab === tab ? 'active' : ''}">${tab}</button>`).join('')}
    ${d.status === 'completed' && !ui.analysis
      ? '<button id="experiment-prepare">Prepare analysis once</button>'
      : ''}</div><div id="experiment-panel"></div>`;
  $('experiment-back').onclick = () => {
    ui.detail = null;
    renderExperimentList();
  };
  document.querySelectorAll('.experiment-tab').forEach(node => node.onclick = () => {
    ui.tab = node.dataset.tab;
    renderExperimentDetail();
  });
  if ($('experiment-prepare')) $('experiment-prepare').onclick = () => experimentAction(async () => {
    $('experiment-prepare').disabled = true;
    $('experiment-message').textContent = 'Preparing the model summary…';
    try {
      await request(experimentURL('/api/experiment-prepare', d.id), { method: 'POST' });
      await loadExperimentAnalysis();
      $('experiment-message').textContent = 'Analysis ready';
      renderExperimentDetail();
    } finally {
      if ($('experiment-prepare')) $('experiment-prepare').disabled = false;
    }
  });
  const panel = $('experiment-panel');
  if (ui.tab === 'Overview') {
    panel.innerHTML = `<div class="metrics">` +
      `${[
        ['Parameters', d.parameters], ['Best epoch', d.best_epoch], ['Seed', d.seed],
        ['Training s', d.training_seconds]
      ].map(([name, value]) => `<div class="metric"><strong>${fmt(value)}</strong><span>${name}</span></div>`)
        .join('')}</div>` +
      (d.error ? `<p class="error">${esc(d.error)}</p>` : '') +
      `<p>Checkpoint selection: ` +
      `${d.checkpoint_selection
        ? esc(d.checkpoint_selection.metric + ' (' + d.checkpoint_selection.direction + ')')
        : 'count MAE → AP50 → loss'}.</p>` +
      card('Results', metricsTable(d.best_metrics, d.final_metrics)) +
      `<details><summary>Actual configuration</summary><pre>${jsonText(d.config)}</pre></details><details>` +
      `<summary>Architecture and dataset provenance</summary>` +
      `<pre>${esc(d.model_info.structure || d.architecture)}</pre><pre>${jsonText(d.dataset)}</pre>` +
      `<pre>${jsonText(d.provenance)}</pre></details><p>Checkpoint: ${esc(d.directory + '/' + d.checkpoint)}</p>` +
      (ui.analysis ? `<a href="${experimentURL('/api/experiment-analysis', d.id)}` +
        `">Export experiment_analysis.json</a>` : '');
  } else if (ui.tab === 'Training') {
    const series = keys => keys.map(key => ({
      name: key, points: d.history.map(row => ({ x: row.epoch, y: row[key] }))
    }));
    panel.innerHTML = '<div class="grid">' +
      card('Loss', chart(series(['train_loss_total', 'val_loss_total']), 'Epoch', 'Loss')) +
      card('Count error', chart(series(['val_count_mae']), 'Epoch', 'Count MAE')) +
      card('Localization and detection', chart(series(['val_map50', 'val_precision', 'val_recall']), 'Epoch', 'Metric',
        false, false, { yPercent: true })) + '</div>' +
      `<details><summary>History including learning rate / epoch timing when recorded</summary>` +
      `<pre>${jsonText(d.history)}</pre></details>`;
  } else renderModelPanel(panel);
}

function modelLayerTable(model) {
  return table([
    ['Block', 'name'],
    ['Type', 'type'],
    ['B,C,sensor,time', 'shape', (v, row) => esc(JSON.stringify(v || row.shapes))],
    ['Kernel S,T', 'kernel_sensor_time', v => esc(JSON.stringify(v))],
    ['Stride S,T', 'block_stride_sensor_time', v => esc(JSON.stringify(v))],
    ['Dilation S,T', 'dilation_sensor_time', v => esc(JSON.stringify(v))],
    ['RF S,T', 'receptive_field_sensor_time', v => v ? esc(v.join(' × ')) : 'Unavailable'],
    ['Parameters', 'parameters']
  ], model.layers);
}

function renderModelPanel(panel) {
  const m = experimentUI.model;
  if (!m) {
    panel.innerHTML = empty('Prepare analysis to inspect the instantiated model.');
    return;
  }
  panel.innerHTML = card('Resolution and receptive field', `<p>Input B,C,sensor,time: ` +
    `${esc(m.input_shape.join(' × '))}. ${esc(m.axes)}</p>` + modelLayerTable(m) + `<p>${esc(m.rf_limitation)}</p>`) +
    card('Parameters by major block', table([
      ['Block', 'name'],
      ['Parameters', 'parameters'],
      ['Fraction', 'fraction']
    ], m.parameter_distribution)) +
    card('Dataset extent vs head RF', table([
      ['Axis', 'axis'],
      ['Head', 'head_level'],
      ['RF', 'receptive_field'],
      ['Median', 'median'],
      ['p90', 'p90'],
      ['p95', 'p95']
    ], m.dataset_coverage));
}
