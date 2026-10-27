/* Pieces shared by the workspace sections (Inference, Model):
   number formats, line charts with a crosshair tooltip, outcome bars, tiles. */
'use strict';

const percentText = value => value == null ? '–' : (value * 100).toFixed(1) + '%';
const decimalText = (value, digits = 3) => value == null ? '–' : Number(value).toFixed(digits);

function compactText(value) {
  if (typeof value !== 'number') return value == null ? '–' : String(value);
  return String(Number(value.toPrecision(4)));
}

function durationText(seconds) {
  if (seconds == null) return '';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600),
    minutes = Math.floor(total % 3600 / 60),
    rest = total % 60;
  const pad = value => String(value).padStart(2, '0');
  return hours ? `${hours}:${pad(minutes)}:${pad(rest)}` : `${minutes}:${pad(rest)}`;
}
const postJSON = (url, body) => request(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {})
});

const plotSpecs = new Map();

function niceScale(minimum, maximum, count = 5) {
  // A single value still needs a readable axis around it.
  if (minimum === maximum) {
    const pad = Math.abs(minimum) * 0.05 || 0.5;
    minimum -= pad;
    maximum += pad;
  }
  const raw = (maximum - minimum) / count,
    magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find(multiple => multiple * magnitude >= raw) * magnitude;
  const lower = Math.floor(minimum / step + 1e-9) * step,
    upper = Math.ceil(maximum / step - 1e-9) * step;
  const ticks = [];
  for (let value = lower; value <= upper + step / 2; value += step) ticks.push(Number(value.toPrecision(12)));
  return { min: lower, max: upper, ticks };
}

function integerTicks(minimum, maximum) {
  const step = integerStep(maximum - minimum),
    ticks = [];
  for (let value = Math.ceil(minimum / step) * step; value <= maximum; value += step) ticks.push(value);
  if (ticks[0] !== minimum && minimum === 1) ticks.unshift(1);
  return ticks;
}
/* The first epochs are often far from the rest: loss far above, quality far below.
   Keep the scale on the bulk of the values and say how many points left it,
   instead of flattening every curve into one line. */
function upperLimit(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const typical = sorted[Math.floor((sorted.length - 1) * 0.9)];
  const maximum = sorted[sorted.length - 1];
  return sorted.length >= 8 && maximum > typical * 2 ? typical * 1.25 : maximum;
}

function lowerLimit(values) {
  // Quality climbs within a few epochs and then stays on a plateau: the median is on it.
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)],
    maximum = sorted[sorted.length - 1];
  const limit = median - 6 * Math.max(maximum - median, 0.005);
  return sorted.length >= 8 && sorted[0] < limit ? limit : sorted[0];
}

/* One line/step/dot chart on a shared x. spec: {id, rows, x, xDomain, xLabel, xTicks,
   series:[{name, key, color, line, step, dots}], percent, clip, marker, title(row)} */
function plot(spec) {
  const rows = spec.rows.filter(row => num(row[spec.x]));
  const values = rows.flatMap(row => spec.series.map(series => row[series.key])).filter(num);
  if (!values.length) return empty(spec.emptyText || 'Nothing to plot yet.');
  const width = 620,
    height = 258,
    left = 52,
    right = 18,
    top = 24,
    bottom = 42;
  // Bounded percentages can only stray downwards, unbounded measures upwards.
  const limit = spec.clip && !spec.percent ? upperLimit(values) : Math.max(...values);
  const floor = spec.clip && spec.percent
    ? lowerLimit(values)
    : spec.percent ? Math.min(...values) : Math.min(0, ...values);
  const clipped = values.filter(value => value > limit || value < floor).length;
  const scale = niceScale(Math.max(0, floor), limit);
  if (spec.percent) scale.ticks = scale.ticks.filter(tick => tick <= 1);
  const [xMin, xMax] = spec.xDomain || [Math.min(...rows.map(r => r[spec.x])), Math.max(...rows.map(r => r[spec.x]))];
  const px = x => left + (xMax === xMin ? .5 : (x - xMin) / (xMax - xMin)) * (width - left - right);
  const py = y => height - bottom - (y - scale.min) / (scale.max - scale.min) * (height - top - bottom);
  const yText = value => spec.percent ? String(Number((value * 100).toPrecision(6))) : compactText(value);
  const clip = `clip-${spec.id}`;
  let svg = `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(spec.label)}` +
    `" data-plot="${esc(spec.id)}">`;
  // The clip keeps off-scale curves inside the plot but leaves room for dots on its edges.
  svg += `<defs><clipPath id="${clip}"><rect x="${left - 8}" y="${top - 8}" width="${width - left - right + 16}` +
    `" height="${height - top - bottom + 9}"/></clipPath></defs>`;
  for (const tick of scale.ticks) svg += `<line class="grid-line" x1="${left}" x2="${width - right}" y1="${py(tick)}` +
    `" y2="${py(tick)}"/><text x="${left - 8}" y="${py(tick) + 4}" text-anchor="end">${yText(tick)}</text>`;
  for (const tick of spec.xTicks || integerTicks(xMin, xMax)) svg += `<text x="${px(tick)}" y="` +
    `${height - bottom + 18}" text-anchor="middle">${esc(spec.xTickText ? spec.xTickText(tick) : tick)}</text>`;
  svg += `<line class="axis-line" x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}"/>`;
  svg += `<text class="axis-title" x="${(left + width - right) / 2}" y="${height - 6}" text-anchor="middle">` +
    `${esc(spec.xLabel)}</text>`;
  if (spec.marker && num(spec.marker.x)) {
    const x = px(spec.marker.x),
      anchor = x > width - 90 ? 'end' : 'start';
    // The label sits in the margin above the plot, where no curve can run through it.
    svg += `<line class="marker-line" x1="${x}" x2="${x}" y1="${top - 8}" y2="${height - bottom}"/>` +
      `<text class="marker-text" x="${x + (anchor === 'end' ? -5 : 5)}" y="${top - 12}" text-anchor="${anchor}">` +
      `${esc(spec.marker.label)}</text>`;
  }
  svg += `<g clip-path="url(#${clip})">`;
  for (const series of spec.series) {
    const points = rows.filter(row => num(row[series.key]));
    if (series.line !== false) {
      let path = '',
        previous = null;
      for (const row of points) {
        const x = px(row[spec.x]),
          y = py(row[series.key]);
        path += previous == null ? `M${x},${y}` : series.step ? `H${x}V${y}` : `L${x},${y}`;
        previous = y;
      }
      if (series.step && points.length) path += `H${px(xMax)}`;
      svg += `<path class="series-line" d="${path}" style="stroke:${series.color}"/>`;
    }
    if (series.dots)
      for (const row of points) svg += `<circle class="dot" cx="${px(row[spec.x])}" cy="${py(row[series.key])}` +
        `" r="5" style="fill:${series.color}"/>`;
  }
  svg += '</g><g class="crosshair" hidden><line/></g></svg>';
  plotSpecs.set(spec.id, { ...spec, rows, px, py, top, bottom: height - bottom, width, limit, floor: scale.min });
  const legend = spec.series.length > 1 ?
    `<div class="chart-legend">${spec.series.map(series => `<span>` +
    `<i class="${series.line === false ? '' : 'line'}" style="background:${series.color}">` +
    `</i>${esc(series.name)}</span>`).join('')}</div>` : '';
  const note = clipped ? `<p class="legend">${clipped} early value${clipped > 1 ? 's are' : ' is'}` +
    ` outside the scale; hover or open the table to read ${clipped > 1 ? 'them' : 'it'}.</p>` : '';
  return `<div class="plot">${svg}</div>${legend}${note}`;
}

/* Crosshair and one tooltip for every series at the nearest x. */
function chartTip() {
  let tip = $('chart-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'chart-tip';
    tip.className = 'chart-tip';
    tip.hidden = true;
    document.body.append(tip);
  }
  return tip;
}
document.addEventListener('pointermove', event => {
  const svg = event.target.closest?.('svg[data-plot]'),
    tip = $('chart-tip');
  if (!svg || !plotSpecs.has(svg.dataset.plot)) {
    if (tip) tip.hidden = true;
    document.querySelectorAll('.crosshair').forEach(node => {
      node.setAttribute('hidden', '');
    });
    return;
  }
  const spec = plotSpecs.get(svg.dataset.plot),
    box = svg.getBoundingClientRect();
  const x = (event.clientX - box.left) / box.width * spec.width;
  const row = spec.rows.reduce((best, candidate) =>
    Math.abs(spec.px(candidate[spec.x]) - x) < Math.abs(spec.px(best[spec.x]) - x) ? candidate : best);
  const cross = svg.querySelector('.crosshair'),
    cx = spec.px(row[spec.x]);
  cross.removeAttribute('hidden');
  cross.innerHTML = `<line x1="${cx}" x2="${cx}" y1="${spec.top}" y2="${spec.bottom}` +
    `"/>` + spec.series.filter(series => num(row[series.key])).map(series =>
    `<circle cx="${cx}" cy="${spec.py(Math.max(spec.floor, Math.min(row[series.key], spec.limit)))}` +
    `" r="4.5" style="fill:${series.color}"/>`).join('');
  const target = chartTip();
  target.replaceChildren();
  const title = document.createElement('span');
  title.className = 'tip-title';
  title.textContent = spec.title(row);
  target.append(title);
  for (const series of spec.series) {
    if (!num(row[series.key])) continue;
    const line = document.createElement('div');
    line.className = 'tip-row';
    const key = document.createElement('i');
    key.style.background = series.color;
    const value = document.createElement('strong');
    value.textContent = spec.percent ? percentText(row[series.key]) : compactText(row[series.key]);
    const name = document.createElement('span');
    name.textContent = series.name;
    line.append(key, value, name);
    target.append(line);
  }
  target.hidden = false;
  const size = target.getBoundingClientRect();
  target.style.left = Math.min(window.innerWidth - size.width - 8, event.clientX + 16) + 'px';
  target.style.top = Math.max(8, event.clientY - size.height - 12) + 'px';
});

/* Detection outcomes as one bar of all objects, split into TP, FP and FN. */
function tray(tp, fp, fn, noun = 'objects') {
  const total = tp + fn + fp;
  if (!total) return empty('No objects and no detections.');
  const parts = [
    { kind: 'tp', name: 'Found (TP)', count: tp },
    { kind: 'fp', name: 'False alarm (FP)', count: fp },
    { kind: 'fn', name: 'Missed (FN)', count: fn },
  ];
  // The width follows the count; the CSS minimum keeps a few errors visible.
  const segments = parts.filter(part => part.count)
    .map(part => `<i class="${part.kind}" style="flex-grow:${part.count}" data-tip="${part.name}: ${part.count} of ` +
      `${total} (${percentText(part.count / total)})"></i>`).join('');
  const bar = `<div class="outcome-bar" role="img" aria-label="${tp} found, ${fp} false alarms, ${fn} missed">` +
    `${segments}</div>`;
  const legend = `<div class="chart-legend outcome-legend">${parts.map(part => `<span><i class="${part.kind}">` +
    `</i>${part.name} ${part.count}</span>`).join('')}</div>`;
  const recall = tp + fn ? percentText(tp / (tp + fn)) : '–',
    precision = tp + fp ? percentText(tp / (tp + fp)) : '–';
  return `${bar}${legend}<p class="tray-caption"><strong>${tp} of ${tp + fn}</strong> labeled ${esc(noun)}` +
    ` found (recall ${recall}). <strong>${tp} of ${tp + fp}</strong> detections are correct (precision ${precision}` +
    `).</p>`;
}

function progressBar(done, total, label, thin = false) {
  const share = total ? Math.min(1, done / total) : 0;
  return `<div class="progress${thin ? ' thin' : ''}" role="progressbar" aria-label="${esc(label)}` +
    `" aria-valuemin="0" aria-valuemax="${total || 0}" aria-valuenow="${Math.round(done * 100) / 100}">` +
    `<span style="width:${(share * 100).toFixed(1)}%"></span></div>`;
}
const tile = (label, value, note = '') => `<div class="metric"><strong>${esc(value)}</strong>` +
  `<span>${esc(label)}</span>${note ? `<small>${esc(note)}</small>` : ''}</div>`;

/* Precision, recall and F1 of a threshold sweep, in the order and colors of the charts. */
const QUALITY_SERIES = [
  { name: 'Precision', key: 'precision', color: 'var(--series-1)' },
  { name: 'Recall', key: 'recall', color: 'var(--series-2)' },
  { name: 'F1', key: 'f1', color: 'var(--series-3)' }
];
