/* Rendering only: tab changes reuse the saved report; matrices load on demand. */
'use strict';
const $ = id => document.getElementById(id);
const dataPages = ['Overview', 'Samples', 'Sensors', 'Objects', 'Foreground / Background', 'Train vs Val',
  'Noise'];
let report = null,
  section = 'Overview',
  scope = 'train',
  selectedSample = null,
  sampleRequest = 0;
let showBoxes = true;
const colors = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)'];
/* The steps of the work, in the order they are taken. Each owns one or more pages (`pages`). A section file adds
   its step with `steps.splice(...)` and its page drawer with `renderers[page] = ...`; the page needs no other change. */
const steps = [
  { key: 'Data', label: 'Data', hint: 'Look at a dataset', pages: dataPages },
  { key: 'Inference', label: 'Inference', hint: 'Run a model on validation, test or new data', pages: ['Inference'] },
  { key: 'Experiments', label: 'Experiments', hint: 'Compare and open finished runs', pages: ['Experiments'] },
  { key: 'Model', label: 'Model', hint: 'Size, cost and speed of an architecture', pages: ['Model'] }
];
const stepOf = page => steps.find(step => step.pages.includes(page));
const renderers = {};
let lastDataPage = 'Overview';
const sectionHints = {
  Overview: 'Size and composition of the selected population, and the overall signal level.',
  Samples: 'One recording at a time: the stored matrix, its boxes and its profiles.',
  Sensors: 'Level, spread and noise of each sensor.',
  Objects: 'Sizes, positions and overlaps of the annotated objects.',
  'Foreground / Background': 'Whether raw signal magnitude separates boxes from background.',
  'Train vs Val': 'Differences between the training and validation splits.',
  Noise: 'Noise estimates per sensor and per sample.'
};
const esc = value => String(value ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = value => value == null ? 'Unavailable'
  : typeof value === 'number'
    ? Number.isInteger(value) ? String(value) : value.toLocaleString('en', { maximumFractionDigits: 3 })
    : esc(value);
const num = value => typeof value === 'number' && Number.isFinite(value);
/* A note is not printed: it is the tooltip of the title. */
const card = (title, body, note = '') => `<section class="card">` +
  `<h3${note ? ` title="${esc(note)}"` : ''}>${esc(title)}</h3>${body}</section>`;
/* Small figures under a chart: the label above, the value below. */
const statLine = (items, bare = false) => `<dl class="stats${bare ? ' bare' : ''}">${items.map(([label, value]) => `<div><dt>${esc(label)}</dt>` +
  `<dd>${esc(value)}</dd></div>`).join('')}</dl>`;
/* The top of every page: its name, one line on what it is for, and the actions that belong to it. */
const pageHeader = (title, lead = '', actions = '') => `<header class="page-head"><div><h2>${esc(title)}</h2>` +
  `${lead ? `<p class="lead">${esc(lead)}</p>` : ''}</div>${actions ? `<div class="page-actions">${actions}</div>` : ''}` +
  `</header>`;
/* Pages of one section; items are [key, label, count?]. The caller wires the clicks on [data-<attribute>]. */
const tabStrip = (items, current, attribute, label) => `<div class="tabs" role="tablist" aria-label="${esc(label)}">` +
  `${items.map(([key, text, count]) => `<button type="button" role="tab" data-${attribute}="${esc(key)}" ` +
  `aria-selected="${key === current}">${esc(text)}${count != null ? `<span class="count">${esc(count)}</span>` : ''}` +
  `</button>`).join('')}</div>`;
/* A metric key of the runs as a person reads it. */
const METRIC_NAMES = {
  f1: 'F1', precision: 'Precision', recall: 'Recall', map50: 'mAP@50', map75: 'mAP@75', map50_95: 'mAP@50–95',
  count_mae: 'Count MAE', count_rmse: 'Count RMSE', count_bias: 'Count bias', exact_count_accuracy: 'Exact-count frames',
  loss_total: 'Loss', loss_bbox: 'Box loss', loss_objectness: 'Objectness loss', samples: 'Frames', objects: 'Objects',
  tp: 'Found', fp: 'False alarms', fn: 'Missed', ignored_objects: 'Objects not counted (cut by the edge)'
};
const metricName = key => METRIC_NAMES[key] || String(key).replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());

/* `flat` shows every row; otherwise a long table scrolls inside its box with the header in view. */
function table(columns, rows, flagged = () => false, flat = false) {
  // Numbers align right so digits line up; text and links stay left.
  const numeric = columns.map(column => {
    const values = rows.map(row => row[column[1]]).filter(value => value != null);
    return values.length > 0 && values.every(value => typeof value === 'number'
      || (column[2] && typeof value === 'object' && !Array.isArray(value)));
  });
  const cell = (tag, index, content) => `<${tag}${numeric[index] ? ' class="num"' : ''}>${content}</${tag}>`;
  const head = columns.map((column, index) => cell('th', index, esc(column[0]))).join('');
  const body = rows.length ?
    rows.map(row => `<tr class="${flagged(row) ? 'flag' : ''}">` +
      `${columns.map((column, index) =>
        cell('td', index, column[2] ? column[2](row[column[1]], row) : fmt(row[column[1]]))).join('')}` +
      `</tr>`).join('') :
    `<tr><td class="no-rows" colspan="${columns.length}">Nothing to show</td></tr>`;
  return `<div class="scroll${flat ? ' flat' : ''}"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function empty(reason) {
  return `<p class="empty">${esc(reason)}</p>`;
}

function integerStep(span) {
  const target = Math.max(1, span / 6);
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const multiple = [1, 2, 5, 10].find(value => value * magnitude >= target);
  return multiple * magnitude;
}

function integerAxis(values, label, percent = false, includeZero = false, magnify = true) {
  let minimum = Math.min(...values),
    maximum = Math.max(...values);
  if (includeZero) {
    minimum = Math.min(0, minimum);
    maximum = Math.max(0, maximum);
  }
  let scale = percent ? 100 : 1;
  // Preserve detail for small continuous values without fractional tick labels.
  const span = maximum - minimum || Math.abs(maximum) || 1;
  if (magnify && !percent && !values.every(Number.isInteger) && span < 1) {
    scale = 10 ** Math.min(12, Math.max(0, Math.ceil(Math.log10(1 / span))));
  }
  const step = integerStep((maximum - minimum) * scale);
  let lower = Math.floor(minimum * scale / step) * step;
  let upper = Math.ceil(maximum * scale / step) * step;
  if (lower === upper) {
    if (includeZero && lower === 0) upper += step;
    else {
      lower -= step;
      upper += step;
    }
  }
  const ticks = [];
  const intervals = Math.round((upper - lower) / step);
  for (let index = 0; index <= intervals; index++) ticks.push(lower + index * step);
  const unitLabel = percent ? label + ' (%)' : scale === 1 ? label : label + ' (×' + scale + ')';
  return { minimum: lower, maximum: upper, ticks, scale, label: unitLabel };
}

function chart(series, xLabel, yLabel, scatter = false, bars = false, options = {}) {
  const points = series.flatMap(s => s.points).filter(p => num(p.x) && num(p.y));
  if (!points.length) return empty('No valid observations for this plot.');
  const width = 620,
    height = 285,
    left = 76,
    right = 30,
    top = 18,
    bottom = 58;
  const xValues = points.flatMap(point => {
    if (!bars) return [point.x];
    // Histograms use their actual edges; categorical bars have unit spacing.
    return [num(point.x0) ? point.x0 : point.x - .4,
      num(point.x1) ? point.x1 : point.x + .4
    ];
  });
  const continuousX = !bars || points.some(p => num(p.x0) && num(p.x1));
  const xAxis = integerAxis(xValues, xLabel, options.xPercent, false, continuousX);
  const yAxis = integerAxis(points.map(p => p.y), yLabel, options.yPercent, bars);
  const px = x => left + (x * xAxis.scale - xAxis.minimum) /
    (xAxis.maximum - xAxis.minimum) * (width - left - right);
  const py = y => height - bottom - (y * yAxis.scale - yAxis.minimum) /
    (yAxis.maximum - yAxis.minimum) * (height - top - bottom);
  let html = `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="` +
    `${esc(yAxis.label + ' versus ' + xAxis.label)}">`;
  for (const tick of yAxis.ticks) {
    const y = py(tick / yAxis.scale);
    html += `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" class="grid-line"/>` +
      `<text class="axis-tick" data-axis="y" x="${left - 8}" y="${y + 4}" text-anchor="end">${tick}</text>`;
  }
  for (const tick of xAxis.ticks) {
    html += `<text class="axis-tick" data-axis="x" x="${px(tick / xAxis.scale)}" y="${height - bottom + 20}` +
      `" text-anchor="middle">${tick}</text>`;
  }
  html += `<line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}` +
    `" class="axis-line"/>`;
  html += `<text class="axis-title" x="${(left + width - right) / 2}" y="${height - 8}" text-anchor="middle">` +
    `${esc(xAxis.label)}</text><text class="axis-title" transform="translate(14 ${height / 2}` +
    `) rotate(-90)" text-anchor="middle">${esc(yAxis.label)}</text>`;
  series.forEach((s, index) => {
    const color = s.color || colors[index % colors.length];
    const valid = s.points.filter(p => num(p.x) && num(p.y));
    if (!scatter && !bars) {
      // Null entries break curves rather than interpolating unavailable lags.
      let path = '';
      let connected = false;
      s.points.forEach(p => {
        if (num(p.x) && num(p.y)) {
          path += `${connected ? 'L' : 'M'}${px(p.x)},${py(p.y)} `;
          connected = true;
        } else connected = false;
      });
      html += `<path class="series-line" d="${path}" style="stroke:${color}"/>`;
    }
    valid.forEach(p => {
      let coordinates = `${xAxis.label}: ${fmt(p.x * xAxis.scale)}; ${yAxis.label}: ${fmt(p.y * yAxis.scale)}`;
      if (bars && num(p.x0) && num(p.x1)) {
        coordinates = `${xAxis.label}: ${fmt(p.x0 * xAxis.scale)} … ${fmt(p.x1 * xAxis.scale)}; ${yAxis.label}: ` +
          `${fmt(p.y * yAxis.scale)}`;
      }
      const tooltip = `data-tip="${esc((p.label || s.name) + ': ' + coordinates)}"`;
      if (bars) {
        const start = num(p.x0) ? p.x0 : p.x - .4;
        const end = num(p.x1) ? p.x1 : p.x + .4;
        const barWidth = px(end) - px(start);
        const barTop = Math.min(py(p.y), py(0));
        const gap = barWidth > 5 ? 1 : 0; // surface gap between neighbouring bars
        html += `<rect data-bin-start="${start}" data-bin-end="${end}" x="${px(start) + gap}" y="${barTop}" width="` +
          `${Math.max(1, barWidth - 2 * gap)}" height="${Math.abs(py(0) - py(p.y))}" rx="${Math.min(2, barWidth / 3)}` +
          `" style="fill:${color}" ${tooltip}/>`;
      } else html += `<circle${!scatter && valid.length > 60 ? ' class="quiet"' : ''} cx="${px(p.x)}" cy="${py(p.y)}` +
        `" r="${scatter ? 3.5 : 2.5}" style="fill:${p.color || color}" opacity="${scatter ? .75 : 1}" ${tooltip}/>`;
    });
  });
  html += '</svg>';
  // One series is named by its card title; several get a legend with swatches.
  const legend = options.legend || (series.length > 1
    ? series.map((s, i) => ({ name: s.name, color: s.color || colors[i % colors.length] }))
    : []);
  return html + (legend.length ? `<div class="chart-legend">${legend.map(item => `<span>` +
    `<i style="background:${item.color}"></i>${esc(item.name)}</span>`).join('')}</div>` : '');
}

function line(values, name = 'Correlation', x = 'Lag / distance', y = 'Correlation') {
  const yPercent = y === 'Correlation';
  return chart([{ name, points: values.map((value, index) => ({ x: index, y: value })) }],
    x, y, false, false, { yPercent });
}

function histogram(distribution, title, units = 'Stored pixel value', xPercent = false) {
  if (!distribution || !distribution.count) return card(title, empty('No values in this region/population.'));
  const { edges, counts } = distribution.histogram;
  const total = counts.reduce((a, b) => a + b, 0);
  const points = counts.map((count, i) => ({
    x: (edges[i] + edges[i + 1]) / 2,
    x0: edges[i],
    x1: edges[i + 1],
    y: total ? count / total : null
  }));
  return card(title,
    chart([{ name: 'Sample frequency', points }], units, 'Samples in bin', false, true, { xPercent, yPercent: true }) +
    statLine([['Values', fmt(distribution.count)], ['Mean', fmt(distribution.mean)],
      ['Median', fmt(distribution.median)], ['Std. deviation', fmt(distribution.std)]]));
}

function setSection(name) {
  section = name;
  if (dataPages.includes(name)) lastDataPage = name;
  document.body.dataset.view = dataPages.includes(name) ? 'dataset' : name.toLowerCase();
  const current = stepOf(name)?.key;
  document.querySelectorAll('#nav button').forEach(button => {
    const here = button.dataset.step === current;
    button.classList.toggle('active', here);
    if (here) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  document.querySelectorAll('#data-tabs button').forEach(button =>
    button.setAttribute('aria-selected', String(button.dataset.section === name)));
  // The address keeps the open section, so reload and links return to it.
  const hash = '#' + encodeURIComponent(name);
  if (location.hash !== hash) history.replaceState(null, '', hash);
  render();
}

function sectionFromAddress() {
  // A suffix after '/' belongs to the section itself, for example '#Training/optuna' (what follows belongs to that section).
  const name = decodeURIComponent(location.hash.slice(1)).split('/')[0];
  if (name === 'Data') return lastDataPage;
  return steps.some(step => step.pages.includes(name)) ? name : 'Overview';
}
function buildNav() {
  steps.forEach(step => {
    const button = document.createElement('button');
    button.className = 'step';
    button.dataset.step = step.key;
    button.innerHTML = `<b>${esc(step.label)}</b><small>${esc(step.hint)}</small>`;
    // The Data step opens the page that was open last.
    button.onclick = () => setSection(step.key === 'Data' ? lastDataPage : step.pages[0]);
    $('nav').append(button);
  });
}
dataPages.forEach(name => {
  const button = document.createElement('button');
  button.type = 'button';
  button.setAttribute('role', 'tab');
  button.textContent = name;
  button.title = sectionHints[name];
  button.dataset.section = name;
  button.onclick = () => setSection(name);
  $('data-tabs').append(button);
});
window.addEventListener('hashchange', () => {
  const name = sectionFromAddress();
  if (name !== section) setSection(name);
});
$('scope').onchange = () => {
  scope = $('scope').value;
  selectedSample = null;
  render();
};
async function request(url, options) {
  let response;
  // fetch rejects only when no answer arrived at all; the browser's own text for that is "Failed to fetch".
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw Error('The dashboard server is not answering. Check the console where dashboard.py was started: ' +
      'if it has stopped, start it again and reload this page.');
  }
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function run(refresh) {
  try {
    await request(refresh ? '/api/refresh' : '/api/run', { method: 'POST' });
    await poll();
  } catch (error) {
    $('status').textContent = error.message;
  }
}
$('run').onclick = () => run(false);
$('refresh').onclick = () => run(true);
async function poll() {
  try {
    const status = await request('/api/status');
    $('status').textContent = status.message + (status.error ? ` · ${status.error}` : '') + (report ? ` · ` +
      `${new Date(report.generated_at).toLocaleString()} · values in stored pixel units` : '');
    $('run').disabled = $('refresh').disabled = status.running;
    $('status').parentElement.dataset.state =
      status.running ? 'running' : status.error ? 'error' : status.available ? 'ready' : 'idle';
    if (status.available && (!report || report.source_signature !== status.signature
        || report.generated_at !== status.generated_at)) {
      report = await request('/api/analysis');
      render();
    }
    $('offline').hidden = true;
  } catch (error) {
    $('status').textContent = `Connection error: ${error.message}`;
    $('status').parentElement.dataset.state = 'error';
    $('offline').textContent = 'The dashboard server is not answering. Check the console where dashboard.py was ' +
      'started: if it has stopped, start it again and reload this page.';
    $('offline').hidden = false;
  }
}

function openSample(split, id) {
  selectedSample = { split, id };
  setSection('Samples');
}

function bindInspect() {
  document.querySelectorAll('[data-sample]').forEach(button => {
    button.onclick = () => openSample(button.dataset.split, button.dataset.sample);
  });
}

function inspectButton(row, label) {
  return `<button data-sample="${esc(row.id)}" data-split="${esc(row.split)}">${esc(label || row.id)}</button>`;
}
async function gallery(rows) {
  const container = $('sample-gallery');
  if (!container) return;
  const token = ++sampleRequest;
  for (const row of rows) {
    try {
      const sample = await request(`/api/sample?split=${row.split}&id=${encodeURIComponent(row.id)}`);
      if (token !== sampleRequest || !container.isConnected) return;
      const node = document.createElement('div');
      const canvas = document.createElement('canvas');
      canvas.className = 'thumbnail';
      paintMatrix(canvas, sample, true);
      node.append(canvas);
      const button = document.createElement('button');
      button.className = 'thumb-button';
      button.textContent = `${row.id} · ${row.object_count} object${row.object_count === 1 ? '' : 's'}`;
      button.onclick = () => openSample(row.split, row.id);
      node.append(button);
      container.append(node);
    } catch (error) {
      if (container.isConnected) {
        const p = document.createElement('p');
        p.textContent = error.message;
        container.append(p);
      }
    }
  }
}

function paintMatrix(canvas, sample, boxes) {
  const matrix = sample.matrix,
    h = matrix.length,
    w = matrix[0].length;
  const factor = Math.max(1, Math.floor(480 / Math.max(h, w)));
  canvas.width = w * factor;
  canvas.height = h * factor;
  const context = canvas.getContext('2d'),
    range = sample.max - sample.min;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const c = range ? Math.round((matrix[y][x] - sample.min) / range * 255) : 128;
      context.fillStyle = `rgb(${c},${c},${c})`;
      context.fillRect(x * factor, y * factor, factor, factor);
    }
  if (boxes) {
    context.strokeStyle = '#6cff9a';
    context.lineWidth = 2;
    sample.boxes_image_xyxy.forEach(b => context.strokeRect(b[0] * canvas.width, b[1] * canvas.height,
      (b[2] - b[0]) * canvas.width, (b[3] - b[1]) * canvas.height));
  }
}

function renderOverview(result) {
  const o = result.overview,
    s = o.signal;
  const cards = [
    ['Valid samples', o.samples],
    ['Objects', o.objects],
    ['Empty samples', o.empty_samples],
    ['Train / Val', `${report.splits.train.overview.samples} / ${report.splits.val.overview.samples}`]
  ];
  let html = `<div class="metrics">${cards.map(([label, value]) => `<div class="metric">` +
    `<strong>${fmt(value)}</strong><span>${esc(label)}</span></div>`).join('')}</div>`;
  html += `<div class="grid">${histogram(s, 'Signal values')}` +
    `${histogram(result.distributions.object_count, 'Objects per sample', 'Objects')}`;
  const label = { std: 'Std. deviation', dynamic_range: 'Dynamic range' };
  html += card('Signal summary', statLine(['min', 'max', 'mean', 'median', 'std', 'dynamic_range']
      .map(name => [label[name] || metricName(name), fmt(s[name])]), true) +
    `<h4 style="margin:22px 0 10px">Percentiles</h4>` + statLine(Object.entries(s.percentiles)
      .filter(([name]) => ['1', '5', '25', '50', '75', '95', '99'].includes(name))
      .map(([name, value]) => [`${name}th`, fmt(value)]), true));
  const duplicates = report.duplicates;
  html += card('Data coverage', table([
    ['Split', 'split'],
    ['Frames', 'valid'],
    ['Empty', 'empty'],
    ['Objects', 'objects']
  ], ['train', 'val'].map(split => ({
    split: split === 'val' ? 'validation' : split,
    valid: report.splits[split].overview.samples,
    empty: report.splits[split].overview.empty_samples,
    objects: report.splits[split].overview.objects
  })), () => false, true) + statLine([['Input issues', report.issues.length],
    ['Repeated frames in train', duplicates.within_split_counts.train],
    ['Repeated in validation', duplicates.within_split_counts.val],
    ['Identical in both', duplicates.cross_split.length]]));
  html += '</div>' + card('Random sample gallery', '<div class="gallery" id="sample-gallery"></div>');
  if (report.issues.length || duplicates.cross_split.length) html += card('Input issues and identical frames',
    (report.issues.length ? table([['Split', 'split'], ['Input', 'id'], ['Issue', 'reason']], report.issues,
      () => false, true) : '') + (duplicates.cross_split.length ? `<details class="plain"><summary>Identical frames, ` +
    `by name</summary><pre>${esc(JSON.stringify(duplicates.cross_split, null, 2))}</pre></details>` : ''));
  return html;
}

function renderSamples(result) {
  const rows = result.samples;
  if (!rows.length) return empty('No valid samples in this population.');
  let index = selectedSample
    ? rows.findIndex(row => row.id === selectedSample.id && row.split === selectedSample.split)
    : 0;
  // A link may open a sample outside the current filter; keep it inspectable.
  if (index < 0) {
    const target = report.splits[selectedSample.split];
    if (scope !== selectedSample.split && target.samples.some(row => row.id === selectedSample.id)) {
      scope = selectedSample.split;
      $('scope').value = scope;
      return renderSamples(target);
    }
    index = 0;
  }
  selectedSample = { split: rows[index].split, id: rows[index].id };
  return `<div class="controls"><button id="sample-prev">← Previous</button>` +
    `<select id="sample-select" aria-label="Sample">${rows.map((row, i) => `<option value="${i}" ` +
    `${i === index ? 'selected' : ''}>${esc(row.split + ' / ' + row.id)}</option>`).join('')}</select>` +
    `<button id="sample-next">Next →</button><label><input id="boxes" type="checkbox" ${showBoxes ? 'checked' : ''}` +
    `> YOLO boxes</label></div><div id="sample-details">Loading matrix…</div>`;
}
async function loadSample() {
  const token = ++sampleRequest,
    selected = selectedSample;
  const target = $('sample-details');
  if (!target || !selected) return;
  try {
    const sample = await request(`/api/sample?split=${selected.split}&id=${encodeURIComponent(selected.id)}`);
    if (token !== sampleRequest || !target.isConnected) return;
    target.innerHTML = `<div class="grid three">` +
      `${card('Original matrix', `<canvas id="matrix" class="heatmap" aria-label="Original sensor matrix"></canvas>`)}
      ${card('Profiles across sensors', chart([
        { name: 'Mean', points: sample.profiles.sensor_mean.map((v, i) => ({ x: i, y: v })) },
        { name: 'Max', points: sample.profiles.sensor_max.map((v, i) => ({ x: i, y: v })) }
      ], 'Sensor index', 'Stored pixel value'))}
      ${card('Profiles over time', chart([
        { name: 'Mean', points: sample.profiles.time_mean.map((v, i) => ({ x: i, y: v })) },
        { name: 'Max', points: sample.profiles.time_max.map((v, i) => ({ x: i, y: v })) }
      ], 'Time step', 'Stored pixel value'))}</div>`;
    const canvas = $('matrix');
    paintMatrix(canvas, sample, showBoxes);
    $('boxes').onchange = () => {
      showBoxes = $('boxes').checked;
      paintMatrix(canvas, sample, showBoxes);
    };
  } catch (error) {
    if (target.isConnected) target.textContent = error.message;
  }
}

function sensorTable(result) {
  const flagged = new Set(result.sensor_diagnostics.map(row => row.sensor));
  return table([
    ['Sensor', 'sensor'],
    ['Mean', 'mean'],
    ['Median', 'median'],
    ['STD', 'std'],
    ['Min', 'min'],
    ['Max', 'max'],
    ['Range', 'dynamic_range'],
    ['Noise', 'noise']
  ], result.sensors, row => flagged.has(row.sensor));
}

function renderSensors(result) {
  return `<div class="grid">${card('Per-sensor level', chart(['mean', 'median'].map(name => ({
    name, points: result.sensors.map(row => ({ x: row.sensor, y: row[name] }))
  })), 'Sensor index', 'Stored pixel value'))}
  ${card('Per-sensor variation', chart(['std', 'dynamic_range', 'noise'].map(name => ({
    name, points: result.sensors.map(row => ({ x: row.sensor, y: row[name] }))
  })), 'Sensor index', 'Stored pixel scale'))}</div>` +
    card('Sensor statistics', sensorTable(result),
      'Noise = Gaussian-equivalent MAD of second temporal differences; object edges also contribute.') +
    card('Suspicious sensor diagnostics', table([
      ['Sensor', 'sensor'],
      ['Feature', 'feature'],
      ['Value', 'value'],
      ['Peer median', 'peer_median'],
      ['Robust score', 'robust_score']
    ], result.sensor_diagnostics),
    'Flags use |robust score| >3.5 against sensor peers; they do not establish hardware failure.');
}

const OBJECT_PLOTS = {
  sensor_width: 'Object width, in sensors', temporal_length: 'Object length, in time steps',
  sensor_center: 'Object position along the sensors', time_center: 'Object position along time',
  boundary_distance: 'Distance of an object to the frame edge', pair_iou: 'Overlap of neighbouring objects (IoU)',
  nearest_center_distance: 'Distance to the nearest object'
};

function renderObjects(result) {
  const o = result.objects;
  const rows = result.samples;
  const sparse = [...rows].sort((a, b) => a.object_count - b.object_count).slice(0, 2);
  const dense = [...rows].sort((a, b) => b.object_count - a.object_count).slice(0, 2);
  const overlaps = [...rows].sort((a, b) => b.max_iou - a.max_iou).filter(row => row.max_iou > 0).slice(0, 2);
  return `<div class="metrics">${tile('Empty frames', percentText(result.overview.empty_fraction))}` +
    `${tile('Frames with several objects', percentText(result.overview.multiple_object_fraction))}` +
    `${tile('Overlapping pairs', `${o.overlapping_pairs} of ${o.pair_count}`)}` +
    `${tile('Frames with overlapping boxes', String(o.samples_with_overlap))}</div><div class="grid">` +
    card('Object width and length', chart([{
      name: 'Annotated objects',
      points: o.scatter.map(row => ({ x: row.sensor_width, y: row.temporal_length, label: row.split + ' / ' + row.id }))
    }], 'Sensor width (cells)', 'Temporal length (steps)', true),
      o.scatter.length < o.scatter_population ? `Shows ${o.scatter.length} of ${o.scatter_population} objects.` : '') +
    ['sensor_width', 'temporal_length', 'sensor_center', 'time_center', 'boundary_distance', 'pair_iou',
      'nearest_center_distance'
    ].map(name => histogram(result.distributions[name], OBJECT_PLOTS[name],
      name === 'pair_iou' ? 'Dimensionless' : 'Cells / steps', name === 'pair_iou')).join('') +
    '</div>' + card('Object extent percentiles', table([
      ['Axis', 'axis'],
      ['p50', '50'],
      ['p75', '75'],
      ['p90', '90'],
      ['p95', '95'],
      ['p99', '99']
    ], ['sensor_width', 'temporal_length'].map(axis => ({ axis, ...result.distributions[axis]?.percentiles })))) +
    card('Representative samples', `<p>Sparse: ${sparse.map(row => inspectButton(row)).join(' ')}</p>` +
      `<p>Dense: ${dense.map(row => inspectButton(row)).join(' ')}</p>` +
      `<p>Overlapping: ${overlaps.length ? overlaps.map(row => inspectButton(row)).join(' ') : 'None measured'}` +
      `</p>`,
    'Nearest distance uses Euclidean bbox centers in cell coordinates: one sensor cell and one time step ' +
    'are treated as one unit.');
}

function renderForeground(result) {
  const f = result.foreground_background,
    separation = f.separation,
    h = separation.histograms;
  const plot = h
    ? chart(['foreground', 'background'].map(name => ({
      name, points: h[name].map((value, i) => ({ x: (h.edges[i] + h.edges[i + 1]) / 2, y: value }))
    })), 'Stored pixel value', 'Samples in bin', false, false, { yPercent: true })
    : empty('Both foreground and background values are required.');
  return `<div class="grid">${card('Foreground vs background', plot)}${histogram(f.foreground, 'Foreground values')}` +
    `${histogram(f.background, 'Background values')}` +
    `${card('Separation diagnostics',
      table([['Metric', 'name'], ['Value', 'value']], ['mean_difference', 'median_difference',
        'standardized_effect_size', 'histogram_overlap', 'wasserstein', 'ks', 'magnitude_auc'
      ].map(name => ({ name: { ks: 'KS statistic', magnitude_auc: 'Magnitude AUC' }[name] || metricName(name), value: separation[name] })), () => false, true),
      'Mean differences are foreground − background. Effect size divides that difference by pooled STD. ' +
      'Magnitude AUC treats brighter pixels as foreground (below 0.5 means darker foreground). ' +
      'Histogram overlap depends on bins; both distributions use bounded pixel samples.')}` +
    `</div>` +
''
}

function renderDrift() {
  const rows = report.drift.rows;
  return `<div class="grid">` +
    card('Sensor mean distributions by split', chart(['train', 'val'].map(name => ({
      name, points: report.splits[name].sensors.map(row => ({ x: row.sensor, y: row.mean }))
    })), 'Sensor index', 'Stored pixel mean')) +
    card('Sensor STD by split', chart(['train', 'val'].map(name => ({
      name, points: report.splits[name].sensors.map(row => ({ x: row.sensor, y: row.std }))
    })), 'Sensor index', 'Stored pixel STD')) + '</div>' +
    card('Train vs validation distances', table([
      ['Feature', 'feature'],
      ['Grain', 'grain'],
      ['Train mean', 'train'],
      ['Val mean', 'val'],
      ['Wasserstein', 'wasserstein'],
      ['KS', 'ks'],
      ['Normalized mean delta', 'normalized_mean_difference']
    ], rows, row => row.large_difference),
    'Mean delta = val − train, divided by sqrt((train variance + val variance)/2). ' +
    'Null means unavailable or a nonzero delta with zero pooled variance.') +
''
}

function renderNoise(result) {
  const ranked = [...result.sensors].sort((a, b) => (b.noise ?? -1) - (a.noise ?? -1));
  return `<div class="grid">` +
    card('Noise across physical sensors',
      line(result.sensors.map(row => row.noise), 'Second-difference MAD', 'Sensor index', 'Stored pixel scale')) +
    histogram(result.distributions.noise, 'Per-sample noise scale', 'Stored pixel scale') + '</div>' +
    card('Sensor noise ranking', table([
      ['Sensor', 'sensor'],
      ['Noise', 'noise'],
      ['STD', 'std'],
      ['Range', 'dynamic_range']
    ], ranked));
}

function render() {
  ++sampleRequest;
  if (renderers[section]) {
    renderers[section]();
    return;
  }
  if (!report) {
    $('content').innerHTML = `<p class="empty">No analysis result yet. It appears here when the current run finishes; use <strong>Run ` +
      `analysis</strong> above to start one.</p>`;
    return;
  }
  const result = report.splits[scope];
  const dataRenderers = {
    Overview: renderOverview,
    Samples: renderSamples,
    Sensors: renderSensors,
    Objects: renderObjects,
    'Foreground / Background': renderForeground,
    'Train vs Val': renderDrift,
    Noise: renderNoise
  };
  $('content').innerHTML = dataRenderers[section](result);
  bindInspect();
  if (section === 'Overview') {
    // Reproducible gallery permutation; no expensive analysis here.
    let seed = report.provenance.seed;
    const randomized = result.samples.map(row => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return { row, key: seed };
    }).sort((a, b) => a.key - b.key);
    gallery(randomized.slice(0, 8).map(item => item.row));
  }
  if (section === 'Samples' && $('sample-select')) {
    const rows = report.splits[scope].samples;
    const select = $('sample-select');
    const change = () => {
      selectedSample = { id: rows[Number(select.value)].id, split: rows[Number(select.value)].split };
      loadSample();
      $('sample-prev').disabled = select.selectedIndex === 0;
      $('sample-next').disabled = select.selectedIndex === rows.length - 1;
    };
    select.onchange = change;
    $('sample-prev').onclick = () => {
      select.selectedIndex = Math.max(0, select.selectedIndex - 1);
      change();
    };
    $('sample-next').onclick = () => {
      select.selectedIndex = Math.min(rows.length - 1, select.selectedIndex + 1);
      change();
    };
    change();
  }
}
$('theme').onclick = () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('sensorline-theme', next);
  } catch (error) {}
  render(); // colours computed in script (correlation matrix) follow the theme
};
document.addEventListener('mousemove', event => {
  const tip = $('tooltip'),
    mark = event.target.closest?.('[data-tip]');
  if (!mark) {
    tip.hidden = true;
    return;
  }
  tip.textContent = mark.dataset.tip;
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - box.width - 8, event.clientX + 14) + 'px';
  tip.style.top = Math.max(8, event.clientY - box.height - 12) + 'px';
});
/* Where "auto" runs: the GPU's name, or the CPU. */
async function showDevice() {
  try {
    const options = await request('/api/model/options');
    $('device-chip').textContent = options.device;
    $('device-chip').title = options.cuda ? 'The GPU used when a device is set to auto' :
      'No CUDA device found: everything runs on the CPU';
    $('device-chip').hidden = false;
  } catch (error) { /* the connection banner reports a lost server */ }
}
// Deferred scripts have all run by now, so every section renderer exists.
document.addEventListener('DOMContentLoaded', () => {
  buildNav();
  showDevice();
  setSection(sectionFromAddress());
  poll();
  setInterval(poll, 2000);
});
