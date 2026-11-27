const statuses = {
  incorrect: 'Incorrect output',
  unavailable: 'Unavailable',
  regression: 'Regression',
  improvement: 'Improvement',
  inconclusive: 'Inconclusive',
  'within-threshold': 'Within cutoff',
};

const shapeFormatter = new Intl.NumberFormat('en-US');
const axisFormatter = new Intl.NumberFormat('en-US', {maximumSignificantDigits: 3});
const dateFormatter = new Intl.DateTimeFormat('en-US', {dateStyle: 'medium', timeZone: 'UTC'});
const shaPattern = /^[0-9a-f]{40,64}$/i;
const caseIdPattern = /^[a-z0-9][a-z0-9-]{0,31}$/;
const publications = {softmax: 'report.json', rmsnorm: 'rmsnorm.json'};
const requestedPublication = new URLSearchParams(window.location?.search ?? '').get('benchmark');
const publication = Object.hasOwn(publications, requestedPublication) ? requestedPublication : 'softmax';
const publicationPath = publications[publication];
const byId = (id) => document.getElementById(id);
const number = (value) => typeof value === 'number' && Number.isFinite(value);
const isSha = (value) => typeof value === 'string' && shaPattern.test(value);
const hasStatus = (value) => Object.hasOwn(statuses, value);
const gpuName = (card) => card.gpu_name || card.requested_gpu;
const namedCase = (item) => typeof item.case_id === 'string' && caseIdPattern.test(item.case_id) &&
  typeof item.label === 'string' && item.label.length > 0 && item.label.length <= 80;
const shapeKey = (item) => namedCase(item) ? item.case_id : `${item.rows}x${item.columns}`;
const shapeLabel = (item) => namedCase(item) ? item.label : `${shapeFormatter.format(item.rows)} × ${shapeFormatter.format(item.columns)}`;
const percent = (value) => `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
const latency = (value) => `${(value * 1000).toFixed(1)} µs`;
const dollars = (value) => `$${value.toFixed(4)}`;

let report;
let selectedShape;
let selectedView = 'change';
let objectUrl;

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function badge(status, label = statuses[status]) {
  return element('span', `status-badge ${status}`, label);
}

function shortGpuName(card) {
  return gpuName(card).replace(/^NVIDIA (?:GeForce )?/, '').replace(/^A100-SXM4-\d+GB$/, 'A100 SXM4');
}

function hasTimings(item) {
  return item && item.status !== 'incorrect' && item.status !== 'unavailable';
}

function validCase(item) {
  if (!item || !hasStatus(item.status)) return false;
  const hasDimensions = Number.isInteger(item.rows) && item.rows > 0 && Number.isInteger(item.columns) && item.columns > 0;
  if (!namedCase(item) && !hasDimensions) return false;
  if (!hasTimings(item)) return true;
  return number(item.base_ms) && item.base_ms > 0 && number(item.head_ms) && item.head_ms > 0 &&
    number(item.base_ms * 1000) && number(item.head_ms * 1000) &&
    number(item.change_pct) && Array.isArray(item.interval_pct) && item.interval_pct.length === 2 &&
    item.interval_pct.every(number) && item.interval_pct[0] <= item.interval_pct[1] &&
    Number.isInteger(item.paired_samples) && item.paired_samples > 0;
}

function validate(data) {
  if (!data || data.schema_version !== 1 || !hasStatus(data.status) ||
    !number(data.margin_pct) || data.margin_pct < 0 ||
    !number(data.active_gpu_seconds) || data.active_gpu_seconds < 0 ||
    !isSha(data.base?.sha) || !isSha(data.head?.sha) ||
    !Array.isArray(data.cards) || data.cards.length < 1 || data.cards.length > 4) {
    throw new Error('Choose a WarpDiff version 1 report.');
  }
  for (const card of data.cards) {
    if (!card || typeof gpuName(card) !== 'string' || !gpuName(card).trim() || !hasStatus(card.status) ||
      (card.cases != null && (!Array.isArray(card.cases) || card.cases.length > 50 || !card.cases.every(validCase)))) {
      throw new Error('This report contains an invalid GPU result.');
    }
  }
  return data;
}

function roundedLimit(maximum) {
  if (maximum <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(maximum));
  const step = [1, 2, 2.5, 4, 5, 8, 10].find((value) => value * power >= maximum);
  const limit = step * power;
  return number(limit) && limit > 0 ? limit : maximum;
}

function setPlotCaption(timeView) {
  const legend = element('span', 'plot-legend');
  const entries = timeView
    ? [['original', 'Original'], ['patched', 'Patched']]
    : [['cutoff', `Within ±${report.margin_pct}%`], ['interval', '95% confidence interval']];
  for (const [color, label] of entries) {
    const item = element('span', 'legend-item');
    item.append(element('span', `legend-swatch ${color}`), label);
    legend.append(item);
  }
  byId('chart-title').replaceChildren(
    element('span', '', timeView ? 'Median time (µs)' : 'Change in latency (%)'),
    legend,
  );
}

function renderAxis(limit, timeView) {
  const ticks = element('div', 'axis-ticks');
  for (const fraction of [0, .25, .5, .75, 1]) {
    const value = timeView ? fraction * limit : (fraction * 2 - 1) * limit;
    const label = axisFormatter.format(value).replace('-', '−');
    const tick = element('span', 'axis-tick', !timeView && value > 0 ? `+${label}` : label);
    tick.style.setProperty('--at', `${fraction * 100}%`);
    ticks.append(tick);
  }
  byId('chart-axis').replaceChildren(ticks);
}

function changeTrack(item, limit) {
  const track = element('div', 'graph-track');
  const place = (value) => (value / limit + 1) * 50;
  const band = element('span', 'tolerance-band');
  band.style.setProperty('--start', `${place(-report.margin_pct)}%`);
  band.style.setProperty('--size', `${report.margin_pct / limit * 100}%`);
  const [low, high] = item.interval_pct.map(place);
  const interval = element('span', 'confidence-range');
  interval.style.setProperty('--start', `${low}%`);
  interval.style.setProperty('--size', `${high - low}%`);
  const median = element('span', 'median-dot');
  median.style.setProperty('--at', `${place(item.change_pct)}%`);
  track.append(band, element('span', 'zero-line'), interval, median);
  track.setAttribute('aria-hidden', 'true');
  return track;
}

function timeTrack(item, limit) {
  const track = element('div', 'graph-track time-track');
  for (const [name, elapsed] of [['original', item.base_ms], ['patched', item.head_ms]]) {
    const bar = element('span', `time-bar ${name}`);
    bar.style.setProperty('--size', `${elapsed * 1000 / limit * 100}%`);
    track.append(bar);
  }
  track.setAttribute('aria-hidden', 'true');
  return track;
}

function gpuRow(card, item, limit, timeView) {
  const status = item?.status || 'unavailable';
  const row = element('article', `gpu-card ${status}`);
  const identity = element('div', 'gpu-identity');
  const name = element('h4', '', shortGpuName(card));
  name.title = gpuName(card);
  identity.append(name);
  const specs = [];
  if (number(card.vram_mib)) specs.push(`${(card.vram_mib / 1024).toFixed(1)} GiB`);
  if (typeof card.compute_capability === 'string') specs.push(`SM ${card.compute_capability}`);
  if (specs.length) identity.append(element('p', 'gpu-meta', specs.join(' · ')));

  const measurement = element('div', 'gpu-measurement');
  const verdict = element('div', 'gpu-verdict');
  if (hasTimings(item)) {
    const note = timeView
      ? `Original ${latency(item.base_ms)} · Patched ${latency(item.head_ms)}`
      : `95% interval ${item.interval_pct.map(percent).join(' to ')} · ${item.paired_samples} pairs`;
    measurement.append(
      timeView ? timeTrack(item, limit) : changeTrack(item, limit),
      element('p', 'measurement-note', note),
    );
    verdict.append(element('span', 'delta', percent(item.change_pct)));
  } else {
    const note = status === 'incorrect' ? 'An output differed from PyTorch.' : 'No timing for this GPU.';
    measurement.append(element('p', 'missing', note));
  }
  verdict.append(badge(status));
  row.append(identity, measurement, verdict);
  return row;
}

function updateComparison() {
  for (const button of byId('shape-controls').children) {
    button.setAttribute('aria-pressed', String(button.dataset.shape === selectedShape));
  }
  for (const button of byId('view-controls').children) {
    button.setAttribute('aria-pressed', String(button.dataset.view === selectedView));
  }
  for (const row of byId('results-table').children) {
    row.classList.toggle('selected-case', row.dataset.shape === selectedShape);
  }

  const pairs = report.cards.map((card) => [card, card.cases?.find((item) => shapeKey(item) === selectedShape)]);
  const timed = pairs.map(([, item]) => item).filter(hasTimings);
  const timeView = selectedView === 'time';
  const values = timeView
    ? timed.flatMap((item) => [item.base_ms * 1000, item.head_ms * 1000])
    : [report.margin_pct, ...timed.flatMap((item) => [item.change_pct, ...item.interval_pct].map(Math.abs))];
  const maximum = Math.max(0, ...values);
  const padded = maximum * 1.12;
  const limit = roundedLimit(number(padded) ? padded : maximum);
  const regressions = timed.filter((item) => item.status === 'regression').length;
  const improvements = timed.filter((item) => item.status === 'improvement').length;
  const summary = byId('selection-summary');
  if (!timed.length) {
    summary.textContent = selectedShape ? 'No timings for this input.' : 'This report has no recorded timings.';
  } else if (regressions) {
    const count = timed.length === 1 ? '1 GPU' : `${regressions} of ${timed.length} measured GPUs`;
    summary.replaceChildren(element('strong', '', count), ' showed a regression on this input.');
  } else if (timed.some((item) => item.status === 'inconclusive')) {
    summary.textContent = 'At least one GPU had inconclusive timings for this input.';
  } else if (improvements) {
    const count = timed.length === 1 ? '1 GPU' : `${improvements} of ${timed.length} measured GPUs`;
    summary.replaceChildren(element('strong', '', count), ' showed an improvement on this input.');
  } else {
    summary.textContent = 'All measured GPUs stayed within the cutoff.';
  }
  byId('chart-axis').hidden = !timed.length;
  byId('view-controls').hidden = !timed.length;
  if (timed.length) {
    setPlotCaption(timeView);
    renderAxis(limit, timeView);
  } else {
    byId('chart-title').textContent = 'GPU results';
  }
  byId('gpu-grid').replaceChildren(...pairs.map(([card, item]) => gpuRow(card, item, limit, timeView)));
}

function cell(className, value, label) {
  const td = element('td', className, value);
  td.setAttribute('role', 'cell');
  if (label) td.dataset.label = label;
  return td;
}

function showTable() {
  const rows = [];
  for (const card of report.cards) {
    for (const item of card.cases || []) {
      const row = element('tr');
      row.setAttribute('role', 'row');
      row.dataset.shape = shapeKey(item);
      const timed = hasTimings(item);
      const status = cell('result-cell');
      status.append(badge(item.status));
      row.append(
        cell('device-name', shortGpuName(card)),
        cell('measurement case-name', shapeLabel(item)),
        cell('measurement numeric', timed ? latency(item.base_ms) : '—', 'Original'),
        cell('measurement numeric', timed ? latency(item.head_ms) : '—', 'Patched'),
        cell(`measurement numeric table-change ${item.status}`, timed ? percent(item.change_pct) : '—', 'Change'),
        status,
      );
      rows.push(row);
    }
  }
  byId('results-table').replaceChildren(...rows);
  byId('results').hidden = !rows.length;
  byId('results-link').hidden = !rows.length;
}

function detail(label, value) {
  const group = element('div');
  const content = element('dd');
  content.append(value);
  group.append(element('dt', '', label), content);
  return group;
}

function showStats() {
  const costs = report.cards.map((card) => card.charges?.total_usd);
  const total = costs.every(number) ? dollars(costs.reduce((sum, value) => sum + value, 0)) : 'Pending';
  const rentals = report.cards.filter((card) => card.cleanup === 'verified' || card.cleanup === 'pending');
  const removed = rentals.filter((card) => card.cleanup === 'verified').length;
  const label = report.billing_status === 'settled' ? 'Vast charges' : 'Vast charges at report time';
  const seconds = report.active_gpu_seconds;
  const activeTime = seconds >= 60 ? `${(seconds / 60).toFixed(2)} min` : `${seconds.toFixed(3)} s`;
  byId('stats').replaceChildren(
    detail('Active GPU time', activeTime),
    detail(label, total),
    detail('Rentals removed', rentals.length ? `${removed} of ${rentals.length}` : '—'),
  );
}

function githubLink(href, text) {
  const link = element('a', '', text);
  link.href = href;
  return link;
}

function commitLabel(repoUrl, shas) {
  const label = document.createDocumentFragment();
  shas.forEach((sha, index) => {
    if (index) label.append(' / ');
    label.append(repoUrl ? githubLink(`${repoUrl}/commit/${sha}`, sha.slice(0, 8)) : sha.slice(0, 8));
  });
  return label;
}

function showEnvironment() {
  const entries = [];
  for (const card of report.cards) {
    const versions = [['CUDA', card.cuda], ['PyTorch', card.torch], ['Triton', card.triton]]
      .filter(([, version]) => typeof version === 'string')
      .map(([name, version]) => `${name} ${version}`);
    const costs = card.charges;
    if (!versions.length && !number(costs?.total_usd)) continue;
    const entry = element('section');
    entry.append(element('h4', '', shortGpuName(card)));
    if (versions.length) entry.append(element('p', '', versions.join(' · ')));
    if (number(costs?.total_usd)) {
      const amounts = [['GPU', costs.compute_usd], ['Disk', costs.storage_usd], ['Transfer', costs.transfer_usd]]
        .filter(([, value]) => number(value))
        .map(([name, value]) => `${name} ${dollars(value)}`);
      entry.append(element('p', '', amounts.length ? amounts.join(' · ') : dollars(costs.total_usd)));
    }
    entries.push(entry);
  }
  byId('environment-list').replaceChildren(...entries);
  byId('environment').hidden = !entries.length;
  byId('environment').open = false;
}

function showProvenance(repoUrl, sources) {
  const baselines = new Set([report.base.sha]);
  for (const source of sources) {
    if (isSha(source?.base?.sha)) baselines.add(source.base.sha);
  }
  const fields = [
    detail(baselines.size > 1 ? 'Original commits' : 'Original commit', commitLabel(repoUrl, [...baselines])),
    detail('Patched commit', commitLabel(repoUrl, [report.head.sha])),
  ];
  if (typeof report.kernel === 'string') fields.push(detail('Kernel', report.kernel));
  if (repoUrl && typeof report.benchmark?.id === 'string') {
    fields.push(detail('Config', githubLink(`${repoUrl}/blob/${report.base.sha}/warpdiff.toml`, 'warpdiff.toml')));
  }
  byId('revisions').replaceChildren(...fields);

  const links = new Map();
  for (const source of sources) {
    const card = report.cards.find((item) => item.source_run_id && item.source_run_id === source?.run_id);
    links.set(source?.workflow_url, card ? `${shortGpuName(card)} run` : 'Workflow run');
  }
  if (!links.has(report.workflow_url)) links.set(report.workflow_url, 'Workflow run');
  const safeLinks = [...links].filter(([url]) => typeof url === 'string' &&
    /^https:\/\/github\.com\/[a-z0-9-]+\/[a-z0-9_.-]+\/actions\/runs\/\d+$/i.test(url) &&
    (!repoUrl || url.toLowerCase().startsWith(`${repoUrl.toLowerCase()}/`)));
  byId('run-links').replaceChildren(...safeLinks.map(([url, label]) => githubLink(url, `${label} ↗`)));
  byId('run-links').hidden = !safeLinks.length;
}

function setNotice(message, error = false) {
  const notice = byId('load-status');
  notice.hidden = !message;
  notice.textContent = message;
  notice.classList.toggle('error', error);
}

function render(data, source, file) {
  report = validate(data);
  const tabs = byId('benchmark-tabs');
  const currentBenchmark = file ? null : (report.benchmark?.id || publication);
  for (const tab of tabs.children) {
    if (tab.dataset.benchmark === currentBenchmark) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  }
  tabs.hidden = false;
  const sources = Array.isArray(report.sources) ? report.sources : [];
  const labels = [source];
  if (sources.length > 1) labels.push(`${sources.length} runs`);
  const finished = typeof report.finished_at === 'string' ? Date.parse(report.finished_at) : NaN;
  if (number(finished)) labels.push(dateFormatter.format(finished));
  byId('run-source').textContent = labels.join(' · ');
  const isSoftmax = typeof report.kernel === 'string' && report.kernel.split('/').at(-1) === 'softmax.py';
  byId('report-title').textContent = typeof report.benchmark?.name === 'string'
    ? report.benchmark.name : (isSoftmax ? 'Softmax benchmark' : 'GPU benchmark');
  byId('shape-label').textContent = report.benchmark ? 'Test case' : 'Input shape';
  const result = byId('run-result');
  result.className = `status-badge ${report.status}`;
  result.textContent = report.status === 'regression' ? 'Regression found' : statuses[report.status];

  const repository = report.repository;
  const repoUrl = typeof repository === 'string' && /^[a-z0-9-]+\/[a-z0-9_.-]+$/i.test(repository)
    ? `https://github.com/${repository}` : '';
  const pr = byId('pr-link');
  pr.hidden = !(repoUrl && Number.isInteger(report.pull_request) && report.pull_request > 0);
  if (pr.hidden) pr.removeAttribute('href');
  else {
    const checkPrefix = `${repoUrl}/runs/`;
    const check = report.check_url;
    pr.href = typeof check === 'string' && check.startsWith(checkPrefix) && /^\d+$/.test(check.slice(checkPrefix.length))
      ? check : `${repoUrl}/pull/${report.pull_request}/checks`;
  }

  const cases = report.cards.flatMap((card) => card.cases || []);
  const shapes = new Map(cases.map((item) => [shapeKey(item), item]));
  byId('shape-controls').replaceChildren(...Array.from(shapes, ([key, item]) => {
    const button = element('button', '', shapeLabel(item));
    button.type = 'button';
    button.dataset.shape = key;
    button.addEventListener('click', () => {
      selectedShape = key;
      updateComparison();
    });
    return button;
  }));
  byId('shape-picker').hidden = !shapes.size;
  const lead = cases.find((item) => item.status === 'regression') || cases[0];
  selectedShape = lead ? shapeKey(lead) : null;
  showTable();
  updateComparison();
  showStats();
  showProvenance(repoUrl, sources);
  showEnvironment();

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = file ? URL.createObjectURL(file) : null;
  const download = byId('evidence-link');
  download.href = objectUrl || publicationPath;
  download.download = file?.name || 'warpdiff-report.json';
  download.hidden = false;
  setNotice('');
  byId('report').hidden = false;
  byId('benchmark-link').hidden = false;
}

for (const button of byId('view-controls').children) {
  button.addEventListener('click', () => {
    selectedView = button.dataset.view;
    updateComparison();
  });
}

const fileInput = byId('report-input');
byId('open-report').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const [file] = fileInput.files;
  fileInput.value = '';
  if (!file) return;
  try {
    if (file.size > 2_000_000) throw new Error('Choose a WarpDiff report under 2 MB.');
    render(JSON.parse(await file.text()), 'Imported report', file);
  } catch (error) {
    setNotice(error instanceof SyntaxError ? 'The selected file is not valid JSON.' : error.message, true);
  }
});

fetch(publicationPath, {cache: 'no-cache'}).then((response) => {
  if (response.status === 404) return null;
  if (!response.ok) throw new Error('Report request failed');
  return response.json();
}).then((data) => {
  if (report) return;
  if (data) render(data, data.provenance === 'vast' ? 'Vast.ai benchmark' : 'Published report');
  else setNotice('Import a WarpDiff JSON report to see its GPU timings.');
}).catch(() => {
  if (!report) setNotice('Couldn’t load the benchmark. You can still import a WarpDiff JSON report.', true);
});
