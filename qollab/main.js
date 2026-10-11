import { QuantumCircuit, transpile } from 'qiskit';
import { JobStatus } from 'qiskit.providers.jobstatus';
import { DATA } from './webapp/data.js';
import { CATALOG } from './webapp/catalog.js';
import { DOC_HTML } from './webapp/docs.js';
import { FIGURE } from './webapp/figure.js';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const COLORS = { IonQ: 'var(--ionq)', IBM: 'var(--ibm)', Rigetti: 'var(--rigetti)' };
const state = { runQ: '9', resQ: '9', metric: 'better', session: [] };
const RAW = `${DATA.repo.replace('https://github.com/', 'https://raw.githubusercontent.com/')}/main/`;
const BLOB = `${DATA.repo}/blob/main/`;
const hasBackend = typeof backend !== 'undefined';
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function setSeg(attr, value) {
  $$(`[${attr}]`).forEach((b) => b.setAttribute('aria-checked', String(b.getAttribute(attr) === value)));
}

function showTab(name) {
  $$('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  $$('.tab').forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
}
$$('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
$('#wim').addEventListener('click', (event) => {
  const link = event.target.closest('a.term');
  if (!link) return;
  event.preventDefault(); showTab('docs');
  $('#glossary').open = true;
  const target = $(link.getAttribute('href'));
  if (target) { target.focus(); target.scrollIntoView({ block: 'center' }); }
});

function renderSpec() {
  const c = DATA.circuits[state.runQ];
  const rows = [
    ['Qubits / board regions', c.qubits], ['ZZ interactions', c.zz], ['Trotter steps', c.steps],
    ['γ, t', `${c.gamma}, ${c.t}`], ['Copper-rich regions at the start', c.start_copper.join(', ')],
    ['Better-map rate: random', pct(c.floor)], ['Better-map rate: ideal', pct(c.ideal)],
  ];
  $('#circuit-spec').innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  $('#circuit-note').textContent = `Choose a simulator or device in Qollab. This circuit represents ${c.qubits} board regions and suggests a layout each time it is measured. Run it to see how many suggestions fit better than the starting layout.`;
}

$$('[data-q]').forEach((b) => b.addEventListener('click', () => {
  state.runQ = b.dataset.q; state.resQ = state.runQ; setSeg('data-q', state.runQ); setSeg('data-rq', state.resQ); $('#run-detail').hidden = true; renderSpec(); renderChart();
}));


function scaleHTML(e) {
  const max = Math.max(e.ideal, e.p_better + e.p_err, 0.01) * 1.15;
  const at = (v) => `${(Math.max(0, v) / max * 100).toFixed(1)}%`;
  return `<div class="scale" aria-hidden="true">
      <div class="scale-track"></div>
      <div class="scale-err" style="left:${at(e.p_better - e.p_err)};width:calc(${at(e.p_better + e.p_err)} - ${at(e.p_better - e.p_err)})"></div>
      <div class="scale-mark ref" style="left:${at(e.floor)}"><span>random guessing<br>${pct(e.floor)}</span></div>
      <div class="scale-mark ref" style="left:${at(e.ideal)}"><span>ideal circuit<br>${pct(e.ideal)}</span></div>
      <div class="scale-dot" style="left:${at(e.p_better)}"></div>
    </div>`;
}

function explainHTML(e, title) {
  const n = e.circuit === '9' ? 9 : 12;
  const rows = e.top.map((t) => `<tr><td><code>${t.bits}</code></td><td class="num">${t.count}</td><td><span class="chip ${t.better ? 'yes' : 'no'}">${t.better ? 'better' : 'not better'}</span></td><td>${t.copper.length ? t.copper.join(', ') : 'none'}</td><td class="num">${pct(t.ideal, 1)}</td></tr>`).join('');
  const f3 = (v) => v.toFixed(3);
  const f5 = (v) => v.toFixed(5);
  return `<div class="answer">
      <div class="answer-num"><span class="big">${pct(e.p_better)}</span><span class="muted small">${e.n_better} of ${e.shots} samples proposed a better-fitting layout</span></div>
      <p class="sentence">For the same starting layout, random proposals would fit better <b>${pct(e.floor)}</b> of the time. The ideal circuit gives <b>${pct(e.ideal)}</b>. Here, “better” means a closer fit to the simulated bending. It does not necessarily mean the true copper layout has been found.</p>
      ${scaleHTML(e)}
    </div>

    <details class="learn-more"><summary>Metrics and calculations</summary>
    <ol class="derive">
      <li><div class="d-label">Better-map rate</div>
        <div class="d-body"><div class="d-def">shots that proposed a better-fitting map ÷ all shots</div>
        <div class="d-calc">${e.n_better} ÷ ${e.shots} = <b>${f3(e.p_better)}</b> <span class="muted">± ${f3(e.p_err)}</span></div></div></li>
      <li><div class="d-label">Share of ideal gain</div>
        <div class="d-body"><div class="d-def">(this run − random guessing) ÷ (ideal − random guessing)</div>
        <div class="d-calc">(${f3(e.p_better)} − ${f3(e.floor)}) ÷ (${f3(e.ideal)} − ${f3(e.floor)}) = <b>${e.kept.toFixed(0)}%</b> <span class="muted">± ${e.kept_err.toFixed(0)} percentage points</span></div></div></li>
      <li><div class="d-label">XEB score</div>
        <div class="d-body"><div class="d-def">(average ideal probability of what was measured − 1/(2^${n})) ÷ (Σ p² − 1/(2^${n}))</div>
        <div class="d-calc">(${f5(e.mean_p)} − ${f5(e.uniform_p)}) ÷ (${f5(e.sum_p2)} − ${f5(e.uniform_p)}) = <b>${f3(e.xeb)}</b> <span class="muted">± ${f3(e.xeb_err)}</span></div>
        <div class="d-why muted small">XEB summarizes agreement with the ideal circuit in one number. Its expected value is 1 for ideal samples and 0 for uniformly random samples. Different distributions can share the same score, and a finite set of samples can give a value below 0 or above 1. The ± value describes sampling uncertainty within this run. It does not include changes between hardware runs.</div></div></li>
    </ol>

    </details>
    <details class="learn-more"><summary>Measured bitstrings and counts</summary>
    <p class="muted small">Across ${e.shots} shots, this run produced ${e.distinct} different bitstrings. The ${e.top.length} most frequent are shown below. Count positions from the right, starting at 0: bit i represents region i, and 1 means copper-rich. “Better” means a closer fit to the warpage than the starting layout, not necessarily the true layout. The last column gives each outcome’s probability in the ideal ${n}-qubit circuit.</p>
    <div class="table-wrap"><table class="outcomes"><thead><tr><th>Bitstring</th><th class="num">Count</th><th>Fit</th><th>Copper-rich regions</th><th class="num">Ideal prob.</th></tr></thead><tbody>${rows}</tbody></table></div>

    </details>
    ${e.log ? `<details class="log"><summary>Console log <span>summary calculated from these counts</span></summary><pre>${esc(e.log)}</pre></details>` : ''}`;
}

function backendName() {
  try {
    const n = typeof backend.name === 'function' ? backend.name() : backend.name;
    return n ? String(n) : 'selected backend';
  } catch (e) { return 'selected backend'; }
}

function explain(q, counts, top = 8) {
  const c = DATA.circuits[q], n = c.qubits, d = 2 ** n;
  const clean = {};
  for (const [k, v] of Object.entries(counts)) {
    let b = String(k).replace(/\s/g, '');
    if (/^0x/i.test(b)) b = BigInt(b).toString(2);
    else if (!/^[01]+$/.test(b) || b.length < n) b = Number(b).toString(2);
    b = b.padStart(n, '0').slice(-n);
    clean[b] = (clean[b] || 0) + Math.round(Number(v));
  }
  let shots = 0, nBetter = 0, sumP = 0, sumP2obs = 0;
  for (const [b, k] of Object.entries(clean)) {
    const i = parseInt(b, 2), p = c.ideal_p[i];
    shots += k; nBetter += c.better[i] * k; sumP += p * k; sumP2obs += p * p * k;
  }
  const pb = nBetter / shots, err = Math.sqrt(pb * (1 - pb) / shots);
  const meanP = sumP / shots, sdP = Math.sqrt(Math.max(0, sumP2obs / shots - meanP * meanP));
  const den = c.sum_p2 - 1 / d;
  const rows = Object.entries(clean).sort((x, y) => y[1] - x[1]).slice(0, top).map(([b, k]) => {
    const i = parseInt(b, 2);
    const copper = [];
    for (let r = 0; r < n; r++) if (b[n - 1 - r] === '1') copper.push(r);
    return { bits: b, count: k, copper, better: !!c.better[i], ideal: c.ideal_p[i] };
  });
  return {
    circuit: q, shots, distinct: Object.keys(clean).length, n_better: nBetter, p_better: pb, p_err: err,
    ideal: c.ideal, floor: c.floor, kept: 100 * (pb - c.floor) / (c.ideal - c.floor), kept_err: 100 * err / (c.ideal - c.floor),
    mean_p: meanP, uniform_p: 1 / d, sum_p2: c.sum_p2, xeb: (meanP - 1 / d) / den, xeb_err: sdP / Math.sqrt(shots) / den, top: rows,
  };
}

function logText(e) {
  const n = Number(e.circuit), f = (v, k) => v.toFixed(k);
  const lines = [
    `circuit: ${e.circuit} qubits | backend: ${e.backend || 'recorded file'} | job: ${e.job || '-'} | shots: ${e.shots}`,
    `distinct bitstrings: ${e.distinct}`,
    'most frequent outcomes:',
    `  bitstring${' '.repeat(Math.max(1, n - 7))}  count  better  copper-rich regions`,
    ...e.top.map((t) => `  ${t.bits}  ${String(t.count).padStart(5)}  ${(t.better ? 'yes' : 'no').padEnd(6)}  [${t.copper.join(', ')}]`),
    `P(better map) = ${e.n_better} / ${e.shots} = ${f(e.p_better, 3)} +/- ${f(e.p_err, 3)}`,
    `   random guessing ${f(e.floor, 3)}, ideal circuit ${f(e.ideal, 3)}`,
    `signal kept   = (${f(e.p_better, 3)} - ${f(e.floor, 3)}) / (${f(e.ideal, 3)} - ${f(e.floor, 3)}) = ${f(e.kept, 0)}% +/- ${f(e.kept_err, 0)}`,
    `XEB           = (${f(e.mean_p, 5)} - ${f(e.uniform_p, 5)}) / (${f(e.sum_p2, 5)} - ${f(e.uniform_p, 5)}) = ${f(e.xeb, 3)} +/- ${f(e.xeb_err, 3)}`,
  ];
  return lines.join('\n');
}

function dispose(v) { try { if (v && typeof v.destroy === 'function') v.destroy(); } catch (_) { /* proxy already released */ } }

function buildCircuit(q) {
  const c = DATA.circuits[q], n = c.qubits;
  const qc = QuantumCircuit(n, n);
  for (const [g, theta, qs] of c.gates) {
    if (theta === null) qc[g](...qs);
    else qc[g](theta, ...qs);
  }
  for (let i = 0; i < n; i++) qc.measure(i, i);
  return qc;
}

function countsToObject(raw) {
  if (raw && typeof raw.toJs === 'function') raw = raw.toJs();
  if (raw instanceof Map) raw = Object.fromEntries(raw);
  if (!raw || typeof raw !== 'object') throw Error('The backend did not return measurement counts in the expected format.');
  return raw;
}

function statusName(s) {
  if (s === JobStatus.DONE) return 'DONE';
  if (s === JobStatus.ERROR) return 'ERROR';
  if (s === JobStatus.CANCELLED) return 'CANCELLED';
  const t = String(s).toUpperCase();
  for (const k of ['CANCELLED', 'CANCELED', 'INITIALIZING', 'VALIDATING', 'QUEUED', 'RUNNING', 'DONE', 'ERROR', 'FAILED']) if (t.includes(k)) return k;
  return t;
}

let wake = null, busy = false, ionqUsed = false;
function requestAction(a) { if (wake) { const r = wake; wake = null; r(a); } }
function nextAction() { return new Promise((r) => { wake = r; }); }

async function execute() {
  const out = $('#run-out'), btn = $('#run-btn');
  const q = state.runQ, name = backendName(), ionq = /ionq/i.test(name);
  if (ionq && ionqUsed) {
    out.innerHTML = '<p class="err">This demo allows one IonQ job per Qollab Run.</p><p class="muted small">To submit another job, press Stop in Qollab, then Run again.</p>';
    return;
  }
  let shots = Math.max(50, Math.min(5000, parseInt($('#shots').value, 10) || 1000));
  if (ionq) shots = Math.min(shots, 300);
  busy = true; btn.disabled = true;
  let circuit = null, compiled = null, job = null;
  const status = (t) => { out.innerHTML = `<p class="muted">${esc(t)}</p>`; };
  try {
    status('Building the circuit and compiling it for the selected backend.');
    circuit = buildCircuit(q);
    compiled = transpile.callKwargs(circuit, backend, { optimization_level: 1 });
    status(`Submitting one job of ${shots} shots to ${name}.`);
    if (ionq) ionqUsed = true;
    job = await backend.run(compiled, { shots });
    let jid = '';
    try { jid = String(await job.job_id()); } catch (_) { jid = ''; }
    for (;;) {
      const st = statusName(await job.status());
      status(`Job ${jid || ''} status: ${st.toLowerCase()}. IonQ jobs can queue; this page keeps checking.`);
      if (st === 'DONE') break;
      if (['ERROR', 'FAILED', 'CANCELLED', 'CANCELED'].includes(st)) throw Error(`Job ended with status ${st.toLowerCase()}.`);
      await setTimeout(2000);
    }
    const result = await job.result();
    const counts = countsToObject(result.get_counts());
    dispose(result);
    const r = { ...explain(q, counts), backend: name, job: jid };
    r.log = logText(r);
    console.log(r.log);
    state.session.push(r);
    out.innerHTML = `<p class="muted small">${esc(name)} · ${q} qubits · ${r.shots} shots${jid ? ` · job ${esc(jid)}` : ''}. Added to the Results tab as a new run, kept separate from the recorded runs.</p>${explainHTML(r, 'This run')}`;
    renderChart();
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    console.log(`Run failed:\n${msg}`);
    out.innerHTML = `<p class="err">The run did not complete. Full error:</p><pre class="errlog">${esc(msg)}</pre><p class="muted small">You can still explore the previously recorded experiments in Results.</p>`;
  } finally {
    dispose(compiled); dispose(circuit); dispose(job);
    busy = false; btn.disabled = false;
  }
}

$('#run-btn').addEventListener('click', () => { if (!busy) requestAction('run'); });

const METRIC = {
  better: {
    key: 'p_better', err: 'p_err', min: 0, max: 0.45, ticks: [0, 0.1, 0.2, 0.3, 0.4], fmt: (v) => pct(v, 0), fmtv: (v) => pct(v),
    note: 'How often did the circuit suggest a layout that fits better than the starting one? The grey lines show the expected rates for random proposals and the ideal circuit.',
  },
  xeb: {
    key: 'xeb', err: 'xeb_err', min: -0.1, max: 1.3, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2), fmtv: (v) => v.toFixed(2),
    note: 'XEB compares the measured samples with the ideal circuit’s output. Ideal samples have an expected score of 1; uniformly random samples, 0. This score checks the circuit, not the accuracy of the physical model.',
  },
};
const KINDS = [['hardware', 'Real quantum hardware'], ['noise model', 'Noise-model simulation'], ['ideal', 'Ideal simulation']];

function marker(kind, x, y, color) {
  if (kind === 'hardware') return `<circle cx="${x}" cy="${y}" r="5.5" fill="${color}"/>`;
  if (kind === 'noise model') return `<circle cx="${x}" cy="${y}" r="5" fill="#fff" stroke="${color}" stroke-width="2"/>`;
  return `<rect x="${x - 5}" y="${y - 5}" width="10" height="10" transform="rotate(45 ${x} ${y})" fill="${color}"/>`;
}

function renderChart() {
  const m = state.metric === 'better' && state.resQ === '12'
    ? { ...METRIC.better, max: 0.25, ticks: [0, 0.05, 0.1, 0.15, 0.2] } : METRIC[state.metric];
  const c = DATA.circuits[state.resQ];
  $('#metric-note').textContent = m.note;
  const runs = DATA[`runs${state.resQ}`];
  const session = state.session.filter((r) => r.circuit === state.resQ).map((r) => ({
    label: `This session: ${r.backend}`, vendor: 'session', kind: 'session', shots: r.shots,
    xeb: r.xeb, xeb_err: r.xeb_err, p_better: r.p_better, p_err: r.p_err,
  }));
  const groups = KINDS.map(([k, t]) => [t, runs.filter((r) => r.kind === k)]);
  if (session.length) groups.unshift(['New runs (this session)', session]);
  const W = 1000, L = 250, R = 800, rowH = 26;
  const x = (v) => L + (Math.max(m.min, Math.min(m.max, v)) - m.min) / (m.max - m.min) * (R - L);
  let y = 30, body = '';
  groups.forEach(([title, rs]) => {
    if (!rs.length) return;
    y += 22;
    body += `<text x="0" y="${y}" font-size="12" fill="var(--muted)" font-weight="600" letter-spacing="0.04em">${title.toUpperCase()}</text>`;
    y += 2;
    rs.forEach((r, k) => {
      y += k === 0 ? rowH - 4 : rowH;
      const v = r[m.key], e = r[m.err];
      const color = r.vendor === 'session' ? 'var(--ink)' : (COLORS[r.vendor] || 'var(--other)');
      const idx = runs.indexOf(r);
      body += `<g class="row" data-i="${idx}">
        <rect class="hit" x="0" y="${y - rowH / 2 - 1}" width="${W}" height="${rowH}" fill="transparent"/>
        <text x="0" y="${y + 4}" font-size="13.5" fill="var(--ink)">${esc(r.label)}</text>
        <line x1="${x(v - e)}" x2="${x(v + e)}" y1="${y}" y2="${y}" stroke="${color}" stroke-width="2" opacity="0.55"/>
        ${marker(r.kind === 'session' ? 'hardware' : r.kind, x(v), y, color)}
        <text x="${R + 16}" y="${y + 4}" font-size="12.5" fill="var(--muted)" font-family="IBM Plex Mono, monospace">${m.fmtv(v)} ± ${m.fmtv(e)}  n=${r.shots}</text>
      </g>`;
    });
    y += 6;
  });
  const H = y + 34;
  let axis = m.ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="26" y2="${H - 22}" stroke="var(--line)"/>
    <text x="${x(t)}" y="${H - 6}" font-size="12" text-anchor="middle" fill="var(--muted)">${m.fmt(t)}</text>`).join('');
  if (state.metric === 'better') {
    axis += [[c.floor, 'random guessing', 'end', -6], [c.ideal, 'ideal circuit', 'start', 6]].map(([v, t, anchor, dx]) =>
      `<line x1="${x(v)}" x2="${x(v)}" y1="26" y2="${H - 22}" stroke="#9aa0a8" stroke-width="1.5" stroke-dasharray="4 4"/>
       <text x="${x(v) + dx}" y="16" font-size="12" text-anchor="${anchor}" fill="var(--muted)">${t} ${pct(v)}</text>`).join('');
  }
  $('#chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Results by backend">${axis}${body}</svg>
    <p class="muted small">● hardware  ○ noise model  ◆ ideal. Colour: <span style="color:var(--ionq)">IonQ</span>, <span style="color:var(--ibm)">IBM</span>, <span style="color:var(--rigetti)">Rigetti</span>. Bars: ±1 standard error. Each point is a separate run. Shots from different runs are not combined.</p>`;
  $$('#chart .row').forEach((g) => g.addEventListener('click', () => {
    const i = parseInt(g.dataset.i, 10);
    if (i >= 0) showDetail(runs[i]);
  }));
}

function showDetail(r) {
  const d = $('#run-detail');
  d.hidden = false;
  const e = { ...explain(state.resQ, r.counts), backend: r.label, job: r.job };
  e.log = logText(e);
  d.innerHTML = `<div class="detail-head"><b>${esc(r.label)}</b><span class="muted">${esc(r.kind)} · ${esc(r.route)} · ${r.shots} shots</span></div>
    <dl><dt>Job ID</dt><dd>${r.job ? esc(r.job) : '<span class="muted">not recorded in the counts file</span>'}</dd>
    <dt>Raw counts</dt><dd><button class="linkbtn" data-file="${esc(r.file)}">${esc(r.file)}</button></dd></dl>${explainHTML(e, r.label)}`;
  d.querySelector('[data-file]').addEventListener('click', (ev) => openFile(ev.currentTarget.dataset.file));
  d.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

$$('[data-rq]').forEach((b) => b.addEventListener('click', () => {
  state.resQ = b.dataset.rq; state.runQ = state.resQ; setSeg('data-rq', state.resQ); setSeg('data-q', state.runQ); $('#run-detail').hidden = true; renderSpec(); renderChart();
}));
$$('[data-m]').forEach((b) => b.addEventListener('click', () => {
  state.metric = b.dataset.m; setSeg('data-m', state.metric); renderChart();
}));

function renderBaselines() {
  const res = DATA.baselines.results;
  const names = Object.keys(res['0.01']).sort((a, b) => res['0.03'][a][3] - res['0.03'][b][3]);
  const rows = names.map((n) => `<tr><td>${esc(n)}</td><td class="num">${res['0.01'][n][3].toFixed(3)}</td><td class="num">${res['0.03'][n][3].toFixed(3)}</td></tr>`).join('');
  $('#baselines').innerHTML = `<table><thead><tr><th>Sampler</th><th class="num">1% noise</th><th class="num">3% noise</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderTsai() {
  const t = DATA.tsai, W = 520, H = 320, L = 46, R = 505, T = 14, B = 284;
  const x = (v) => L + (v - 20) / 240 * (R - L);
  const y = (v) => T + (40 - v) / 110 * (B - T);
  const line = (k) => t.map((p) => `${x(p.T).toFixed(1)},${y(p[k]).toFixed(1)}`).join(' ');
  const ticksY = [40, 20, 0, -20, -40, -60].map((v) => `<line x1="${L}" x2="${R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 8}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">${v}</text>`).join('');
  const ticksX = [20, 60, 100, 140, 180, 220, 260].map((v) => `<text x="${x(v)}" y="${B + 16}" font-size="11" text-anchor="middle" fill="var(--muted)">${v}</text>`).join('');
  const dots = t.map((p) => `<circle cx="${x(p.T)}" cy="${y(p.moire)}" r="3.5" fill="#c0392b"/>`).join('');
  $('#tsai-chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Corner warpage vs temperature: model vs moiré">
    ${ticksY}${ticksX}
    <polyline fill="none" stroke="#7d5ba6" stroke-width="2" stroke-dasharray="6 4" points="${line('fem')}"/>
    <polyline fill="none" stroke="var(--ionq)" stroke-width="2.5" points="${line('ours')}"/>
    ${dots}
    <text x="${L + 8}" y="${T + 14}" font-size="12" fill="#c0392b">● moiré, heating (paper)</text>
    <text x="${L + 8}" y="${T + 30}" font-size="12" fill="#7d5ba6">- - 3D FEM (paper)</text>
    <text x="${L + 8}" y="${T + 46}" font-size="12" fill="var(--ionq)">▬ this project's FEA</text>
    <text x="${(L + R) / 2}" y="${H - 2}" font-size="11" text-anchor="middle" fill="var(--muted)">Temperature (°C); corner warpage in µm</text>
  </svg>`;
}

function renderCatalog(filter = '') {
  const f = filter.trim().toLowerCase();
  $('#catalog').innerHTML = CATALOG.map((cat) => {
    const files = cat.files.filter(([p, d]) => !f || p.toLowerCase().includes(f) || d.toLowerCase().includes(f));
    if (!files.length) return '';
    const items = files.map(([p, d]) => `<li><button class="linkbtn" data-file="${esc(p)}"><code>${esc(p)}</code></button><span>${esc(d)}</span></li>`).join('');
    return `<details ${f ? 'open' : ''}><summary>${esc(cat.title)}<span>${files.length}</span></summary><ul class="files">${items}</ul></details>`;
  }).join('');
  $$('#catalog [data-file]').forEach((b) => b.addEventListener('click', () => openFile(b.dataset.file)));
}
$('#file-search').addEventListener('input', (e) => renderCatalog(e.target.value));

const TEXT = /\.(py|txt|json|csv|qasm|js|html|css|cfg|toml|ini|mac)$|LICENSE$/i;
async function openFile(path) {
  $('#viewer-title').textContent = path;
  $('#viewer-links').innerHTML = `<a href="${BLOB}${esc(path)}" target="_blank" rel="noopener">View on GitHub</a><a href="${RAW}${esc(path)}" target="_blank" rel="noopener" download>Download</a>`;
  $('#viewer').hidden = false;
  if (!TEXT.test(path)) { $('#viewer-body').textContent = 'This file cannot be previewed here. Use the links above.'; return; }
  $('#viewer-body').textContent = 'Loading from GitHub.';
  try {
    const res = await fetch(RAW + path);
    if (!res.ok) throw Error(String(res.status));
    let text = await res.text();
    if (path.endsWith('.json') && text.length < 400000) {
      try { text = JSON.stringify(JSON.parse(text), null, 1); } catch (_) { /* keep raw */ }
    }
    $('#viewer-body').textContent = text.length > 200000 ? `${text.slice(0, 200000)}\n\n[truncated]` : text;
  } catch (e) {
    $('#viewer-body').textContent = 'Could not load the preview. Use the links above to open the file on GitHub.';
  }
}
$('#viewer-close').addEventListener('click', () => { $('#viewer').hidden = true; });

$('#tsai-fig').src = FIGURE;
$('#doc-body').innerHTML = DOC_HTML;
$$('.repo-link').forEach((a) => { a.href = a.dataset.path ? BLOB + a.dataset.path : DATA.repo; });

for (const q of ['9', '12']) {
  const c = DATA.circuits[q];
  c.floor = c.better.reduce((sum, value) => sum + Number(value), 0) / c.better.length;
  c.ideal = c.ideal_p.reduce((sum, p, i) => sum + p * Number(c.better[i]), 0);
  for (const run of DATA[`runs${q}`]) {
    const scored = explain(q, run.counts);
    Object.assign(run, { p_better: scored.p_better, p_err: scored.p_err, xeb: scored.xeb,
      xeb_err: scored.xeb_err, kept: scored.kept, kept_err: scored.kept_err });
  }
}
renderSpec();
renderChart();
renderBaselines();
renderTsai();
renderCatalog();

$('#backend-name').textContent = hasBackend ? `Backend: ${backendName()}` : 'Select a backend and press Run in Qollab to enable this demo.';
$('#run-btn').disabled = !hasBackend;
// Keep the Qollab run alive so the Run button can submit jobs while the page is open.
while (hasBackend) {
  const action = await nextAction();
  if (action === 'run') await execute();
}
