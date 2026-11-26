const statuses = {
  incorrect: 'Incorrect output', unavailable: 'Unavailable', regression: 'Regression',
  improvement: 'Improvement', inconclusive: 'Inconclusive', 'within-threshold': 'Within threshold',
};

const shapeFormatter = new Intl.NumberFormat('en-US');
const byId = (id) => document.getElementById(id);
const number = (value) => typeof value === 'number' && Number.isFinite(value);
const hasStatus = (value) => Object.hasOwn(statuses, value);
const gpuName = (card) => card.gpu_name || card.requested_gpu;
const shapeKey = ({rows, columns}) => `${rows}x${columns}`;
const shapeLabel = ({rows, columns}) => `${shapeFormatter.format(rows)} × ${shapeFormatter.format(columns)}`;
const percent = (value) => `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
const latency = (value) => `${(value * 1000).toFixed(1)} µs`;

let report;
let objectUrl;

function element(tag, className, content) {
  const item = document.createElement(tag);
  item.className = className;
  if (content !== undefined) item.textContent = content;
  return item;
}

function pill(status) {
  return element('span', `pill ${status}`, statuses[status]);
}

function validCase(item) {
  if (!item || !Number.isInteger(item.rows) || item.rows < 1 || !Number.isInteger(item.columns) || item.columns < 1 || !hasStatus(item.status)) {
    return false;
  }
  return item.status === 'incorrect' || (
    number(item.base_ms) && item.base_ms > 0 && number(item.head_ms) && item.head_ms > 0 &&
    number(item.change_pct) && Array.isArray(item.interval_pct) && item.interval_pct.length === 2 &&
    item.interval_pct.every(number) && Number.isInteger(item.paired_samples)
  );
}

function validate(data) {
  if (!data || data.schema_version !== 1 || !hasStatus(data.status) || !number(data.margin_pct) ||
    !number(data.active_gpu_seconds) || !/^[0-9a-f]{40,64}$/i.test(data.base?.sha) || !/^[0-9a-f]{40,64}$/i.test(data.head?.sha) ||
    !Array.isArray(data.cards) || data.cards.length < 1 || data.cards.length > 4) {
    throw new Error('Select a WarpDiff version 1 report.');
  }
  for (const card of data.cards) {
    if (!card || typeof gpuName(card) !== 'string' || !hasStatus(card.status) || (card.cases && (
      !Array.isArray(card.cases) || card.cases.length > 50 || !card.cases.every(validCase)
    ))) {
      throw new Error('The report has an invalid GPU measurement.');
    }
  }
  return data;
}

function latencyRow(name, elapsed, scale, candidate) {
  const row = element('div', 'latency');
  const track = element('span', 'track');
  const bar = element('span', candidate ? 'bar bar-candidate' : 'bar');
  bar.style.width = `${elapsed * 100 / scale}%`;
  track.append(bar);
  track.setAttribute('aria-hidden', 'true');
  row.append(element('span', '', name), track, element('span', 'latency-value', latency(elapsed)));
  return row;
}

function cardFor(card, item, scale) {
  const status = item?.status || 'unavailable';
  const panel = element('article', `gpu-card ${status}`);
  const heading = element('div', 'gpu-top');
  const hardware = element('div', '');
  hardware.append(element('h3', '', gpuName(card)));
  const details = [];
  if (number(card.vram_mib)) details.push(`${Math.round(card.vram_mib / 1024)} GB VRAM`);
  if (typeof card.compute_capability === 'string') details.push(`SM ${card.compute_capability}`);
  if (details.length) hardware.append(element('p', 'gpu-meta', details.join(' · ')));
  heading.append(hardware, pill(status));
  panel.append(heading);
  if (item && number(item.change_pct)) {
    panel.append(element('p', 'delta', percent(item.change_pct)),
      element('p', 'confidence', `95% interval ${item.interval_pct.map(percent).join(' to ')} · ${item.paired_samples} pairs`),
      latencyRow('Baseline', item.base_ms, scale, false), latencyRow('Candidate', item.head_ms, scale, true));
  } else {
    panel.append(element('p', 'missing', item?.status === 'incorrect' ? 'One or both revisions disagreed with PyTorch.' : 'No timing for this GPU.'));
  }
  return panel;
}

function showShape(key) {
  for (const button of byId('shape-controls').children) {
    button.setAttribute('aria-pressed', String(button.dataset.shape === key));
  }
  const pairs = report.cards.map((card) => [card, card.cases?.find((item) => shapeKey(item) === key)]);
  const scale = Math.max(...pairs.flatMap(([, item]) => [item?.base_ms || 0, item?.head_ms || 0]));
  byId('gpu-grid').replaceChildren(...pairs.map(([card, item]) => cardFor(card, item, scale)));
}

function showTable() {
  const rows = [];
  for (const card of report.cards) {
    for (const item of card.cases || []) {
      const row = element('tr', '');
      const timed = item.status !== 'incorrect';
      row.append(element('td', '', gpuName(card)), element('td', 'measurement', shapeLabel(item)),
        element('td', 'measurement', timed ? latency(item.base_ms) : '—'),
        element('td', 'measurement', timed ? latency(item.head_ms) : '—'),
        element('td', `measurement change ${item.status}`, timed ? percent(item.change_pct) : '—'));
      const status = element('td', '');
      status.append(pill(item.status));
      row.append(status);
      rows.push(row);
    }
  }
  byId('results-table').replaceChildren(...rows);
  document.querySelector('.all-shapes').hidden = rows.length === 0;
}

function showStats() {
  const values = report.cards.map((card) => card.charges?.total_usd);
  const receipt = values.every(number) ? `$${values.reduce((sum, value) => sum + value, 0).toFixed(4)}` : 'Pending';
  const chargeLabel = report.billing_status === 'settled' ? 'Vast charges' : 'Vast charges at report time';
  const cleaned = report.cards.filter((card) => card.cleanup === 'verified').length;
  const stats = [
    [`${(report.active_gpu_seconds / 60).toFixed(2)} min`, 'CUDA-timed work'],
    [receipt, chargeLabel],
    [`${cleaned} / ${report.cards.length}`, 'Rentals removed'],
  ];
  byId('stats').replaceChildren(...stats.map(([value, label]) => {
    const stat = element('div', 'stat');
    stat.append(element('span', 'stat-value', value), element('span', 'stat-label', label));
    return stat;
  }));
}

function setNotice(message, error = false) {
  const notice = byId('load-status');
  notice.hidden = !message;
  notice.textContent = message;
  notice.classList.toggle('error', error);
}

function render(data, source, file) {
  report = validate(data);
  setNotice('');
  byId('report').hidden = false;
  const sources = Array.isArray(report.sources) ? report.sources : [];
  byId('run-source').textContent = sources.length > 1 ? `${source} · ${sources.length} RUNS` : source;
  const filename = typeof report.kernel === 'string' ? report.kernel.split('/').at(-1) : null;
  byId('report-title').textContent = filename === 'softmax.py' ? 'Softmax benchmark' : 'GPU benchmark';
  const bases = new Set([report.base.sha]);
  for (const source of sources) {
    if (/^[0-9a-f]{40,64}$/i.test(source?.base?.sha)) bases.add(source.base.sha);
  }
  const baseline = Array.from(bases, (sha) => sha.slice(0, 8)).join(' / ');
  byId('revisions').replaceChildren(bases.size > 1 ? 'Baselines ' : 'Baseline ', element('code', '', baseline),
    ' → candidate ', element('code', '', report.head.sha.slice(0, 8)));
  const status = byId('run-result');
  status.className = `pill ${report.status}`;
  status.textContent = report.status === 'regression' ? 'Regression detected' : statuses[report.status];
  byId('threshold').textContent = `Tolerance ±${report.margin_pct}%`;
  const pr = byId('pr-link');
  const repository = report.repository;
  pr.hidden = !(/^[a-z0-9_.-]+\/[a-z0-9_.-]+$/i.test(repository) && Number.isInteger(report.pull_request) && report.pull_request > 0);
  if (!pr.hidden) pr.href = `https://github.com/${repository}/pull/${report.pull_request}`;

  const allCases = report.cards.flatMap((card) => card.cases || []);
  const shapes = new Map(allCases.map((item) => [shapeKey(item), item]));
  const controls = byId('shape-controls');
  controls.replaceChildren(...Array.from(shapes, ([key, item]) => {
    const button = element('button', '', shapeLabel(item));
    button.type = 'button';
    button.dataset.shape = key;
    button.addEventListener('click', () => showShape(key));
    return button;
  }));
  const lead = allCases.find((item) => item.status === 'regression') || shapes.values().next().value;
  if (lead) showShape(shapeKey(lead));
  else showShape(null);
  showTable();
  showStats();

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = file ? URL.createObjectURL(file) : null;
  const download = byId('evidence-link');
  download.href = objectUrl || 'report.json';
  download.download = file?.name || 'warpdiff-report.json';
  download.hidden = false;
}

const fileInput = byId('report-input');
byId('open-report').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const [file] = fileInput.files;
  fileInput.value = '';
  if (!file) return;
  try {
    if (file.size > 2_000_000) throw new Error('Choose a WarpDiff report under 2 MB.');
    render(JSON.parse(await file.text()), 'IMPORTED REPORT', file);
  } catch (error) {
    setNotice(error instanceof SyntaxError ? 'The selected file is not valid JSON.' : error.message, true);
  }
});

fetch('report.json', {cache: 'no-cache'}).then((response) => {
  if (response.status === 404) return null;
  if (!response.ok) throw new Error('Report request failed');
  return response.json();
}).then((data) => {
  if (report) return;
  if (data) render(data, data.provenance === 'vast' ? 'VAST.AI HARDWARE RUN' : 'PUBLISHED REPORT');
  else setNotice('Open a WarpDiff JSON report to compare the measured GPUs.');
}).catch(() => {
  if (!report) setNotice('The published benchmark could not be loaded. Open a local WarpDiff JSON report to continue.', true);
});
