import { loadPrimeKG, createBrain } from './brain.js';
import { createTelemetry } from './telemetry.js';

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

let kg;
let brain;
let telemetry;
let source = null;
let paths = [];
let summary = null;
let expectSummary = true;
let selectedPath = null;
let pinned = -1;
let hovered = -1;
let hoverTimer = 0;
let connectFrom = -1;

// ---------- boot ----------

$('#boot-scale').innerHTML = Array.from({ length: 11 }, (_, k) => `<span>${k * 10}%</span>`).join('');
function bootProgress(p) {
  const cur = Math.floor(p * 10 + 1e-9);
  $('#boot-scale')
    .querySelectorAll('span')
    .forEach((s, k) => (s.className = k < cur ? 'on' : k === cur ? 'cur' : ''));
}
bootProgress(0);

async function boot() {
  try {
    kg = await loadPrimeKG(bootProgress);
  } catch (e) {
    $('#boot-msg').innerHTML = esc(e.message).replace(/`([^`]+)`/g, '<code>$1</code>');
    $('#boot').classList.add('error');
    return;
  }
  brain = createBrain($('#scene'), kg, { onHover, onClick: (i) => (i >= 0 ? inspect(i) : deselect()) });
  brain.setInsets(insets(), { place: true });
  telemetry = createTelemetry(kg, () => brain.time);
  if (import.meta.env.DEV) window.verdia = { brain, kg, telemetry };
  $('#meta-nodes').textContent = `• ${kg.N.toLocaleString()}`;
  renderLegend();
  telemetry.idle();
  $('#boot').classList.add('done');
  setTimeout(() => $('#boot').remove(), 1200);

  fetch('/api/health')
    .then((r) => r.json())
    .then(({ llm, primekg }) => {
      if (!primekg) return status('Server', 'PrimeKG not loaded · run npm run primekg and restart', true);
      status('Agent LLM', llm.ready ? `${llm.provider}/${llm.model}` : 'no key · add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env', !llm.ready || llm.provider === 'mock');
    })
    .catch(() => status('Agent server', 'offline', true));

  const params = new URLSearchParams(location.search);
  const pair = params.get('connect')?.split(',').map(Number);
  if (pair?.length === 2 && pair.every((i) => Number.isInteger(i) && i >= 0 && i < kg.N)) connect(pair[0], pair[1]);
  else if (params.get('q')) {
    $('#nct').value = params.get('q');
    submit(params.get('q'));
  } else captionTimer = setTimeout(() => caption(`Every point is one of PrimeKG's ${kg.N.toLocaleString()} entities`), 900);
}

/** How much of the stage canvas the side panels cover, in pixels. */
function insets() {
  const stage = $('#scene').getBoundingClientRect();
  const left = $('.panel.left').getBoundingClientRect();
  const right = $('.panel.right').getBoundingClientRect();
  return {
    left: left.width ? Math.max(0, left.right - stage.left) : 0,
    right: right.width ? Math.max(0, stage.right - right.left) : 0,
    top: 0,
    bottom: 0,
  };
}

function status(label, value, bad) {
  const el = $('#llm-status');
  el.innerHTML = `${esc(label)}: <b>${esc(value)}</b>`;
  el.classList.toggle('bad', !!bad);
}

function tickClock() {
  const d = new Date();
  const date = d.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' });
  const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  $('#clock').textContent = `• ${date} ${time}`;
}
tickClock();
setInterval(tickClock, 1000);

let captionTimer = 0;
/** Type a bracketed caption over the stage, then fade it out. */
function caption(text, hold = 4000) {
  const el = $('#caption');
  clearTimeout(captionTimer);
  if (!text) return el.classList.add('out');
  const stage = $('#scene').getBoundingClientRect();
  const side = insets();
  el.style.left = `${stage.left + side.left + (stage.width - side.left - side.right) / 2}px`;
  el.style.top = `${stage.top + stage.height / 2}px`;
  el.classList.remove('out');
  let k = 0;
  const type = () => {
    el.textContent = `[${text.slice(0, k)}${k >= text.length ? ']' : ''}`;
    captionTimer = k++ < text.length ? setTimeout(type, 26) : setTimeout(() => el.classList.add('out'), hold);
  };
  type();
}

const typeOf = (i) => kg.meta.types[kg.types[i]];
const typeLabel = (name) => kg.meta.types.find((t) => t.name === name)?.label ?? name;
// DrugBank / Mayo feature text carries markdown and HTML fragments.
const plain = (s) =>
  s
    .replace(/<[^>]+>/g, '')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\*\*|__/g, '')
    .replace(/(^|[\s(])_([^_]+)_(?=[\s).,;:]|$)/g, '$1$2')
    .trim();
const chip = (i, label = kg.names[i]) => `<button class="chip" data-i="${i}" style="--c:${typeOf(i).color}">${esc(label)}</button>`;
const bindChips = (root) => root.querySelectorAll('.chip[data-i]').forEach((b) => b.addEventListener('click', () => inspect(Number(b.dataset.i))));

const nbrCache = new Map();
function neighbors(i) {
  if (!nbrCache.has(i)) {
    if (nbrCache.size > 400) nbrCache.delete(nbrCache.keys().next().value);
    nbrCache.set(i, fetch(`/api/primekg/neighbors/${i}`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return nbrCache.get(i);
}

// ---------- input ----------

$('#run-form').addEventListener('submit', (e) => {
  e.preventDefault();
  submit($('#nct').value);
});
document.querySelectorAll('.examples button').forEach((b) =>
  b.addEventListener('click', () => {
    $('#nct').value = b.dataset.q;
    submit(b.dataset.q);
  }),
);

const PAIR = /^(.+?)\s*(?:->|→|<->|⟷|↔)\s*(.+)$/;
const PAIR_TO = /^(.+?)\s+to\s+(.+)$/i;

async function submit(raw) {
  const q = String(raw || '').trim();
  if (!q || !brain) return;
  if (/^NCT\d{8}$/i.test(q)) {
    history.replaceState(null, '', `?q=${encodeURIComponent(q)}`);
    return run(q.toUpperCase());
  }
  // "A -> B" (or "A to B" when the whole phrase is not itself a node name) connects two nodes.
  let pair = q.match(PAIR);
  if (!pair && PAIR_TO.test(q)) {
    const whole = await find(q);
    if (!whole || whole.score < 3) pair = q.match(PAIR_TO);
  }
  if (pair) {
    const [a, b] = await Promise.all([find(pair[1]), find(pair[2])]);
    if (a && b && a.i !== b.i) return connect(a.i, b.i);
    $('#steps').innerHTML = `<li class="empty">Could not find ${esc(!a ? pair[1] : pair[2])} in PrimeKG.</li>`;
    return;
  }
  history.replaceState(null, '', `?q=${encodeURIComponent(q)}`);
  searchKG(q);
}

const find = (q) =>
  fetch(`/api/primekg/search?q=${encodeURIComponent(q.trim())}&limit=1`)
    .then((r) => r.json())
    .then((h) => (Array.isArray(h) ? h[0] : null))
    .catch(() => null);

function reset() {
  source?.close();
  caption(null);
  paths = [];
  summary = null;
  selectedPath = null;
  pinned = -1;
  brain.clearSignals();
  brain.clearPaths();
  brain.clearLabels();
  brain.clearEgo();
  brain.mark(-1);
  brain.setDim(false);
  telemetry.mark(-1);
  telemetry.idle();
  $('#tab-note').textContent = '';
  $('#trace-note').textContent = '';
}

async function searchKG(q) {
  const connecting = connectFrom >= 0;
  if (!connecting) reset();
  $('#trace-note').textContent = 'Search';
  $('#steps').innerHTML = '<li class="empty">Searching PrimeKG…</li>';
  const hits = await fetch(`/api/primekg/search?q=${encodeURIComponent(q)}&limit=14`).then((r) => r.json()).catch(() => []);
  if (!Array.isArray(hits) || !hits.length) {
    $('#steps').innerHTML = `<li class="empty">No PrimeKG node matches “${esc(q)}”.</li>`;
    return;
  }
  const title = connecting ? `Connect ${esc(kg.names[connectFrom])} to…` : `PrimeKG matches for “${esc(q)}”`;
  $('#steps').innerHTML = `<li class="hits"><div class="title">${title}</div>${hits
    .map((h) => `<button class="hit" data-i="${h.i}"><span class="sw" style="background:${typeOf(h.i).color}"></span><span class="nm">${esc(h.name)}</span><span class="ty">${esc(typeOf(h.i).label)} · ${h.degree.toLocaleString()}</span></button>`)
    .join('')}</li>`;
  document.querySelectorAll('.hit').forEach((b) => b.addEventListener('click', () => inspect(Number(b.dataset.i))));
  brain.fire(hits.map((h) => h.i));
  if (!connecting) inspect(hits[0].i);
}

// ---------- agent run / connect: both stream search events ----------

function run(nct) {
  cancelConnect();
  reset();
  expectSummary = true;
  $('#trace-note').textContent = nct;
  $('#steps').innerHTML = '';
  $('#tab-connections').innerHTML = '<div class="placeholder">Agent is reading the trial…</div>';
  showTab('connections');
  stream(`/api/assemble?nct=${encodeURIComponent(nct)}`);
}

function connect(a, b) {
  cancelConnect();
  reset();
  expectSummary = false;
  history.replaceState(null, '', `?connect=${a},${b}`);
  $('#trace-note').textContent = 'A → B';
  $('#steps').innerHTML = '';
  $('#tab-connections').innerHTML = `<div class="trial pair">${chip(a)}<span>→</span>${chip(b)}</div><div class="placeholder">Firing from both nodes…</div>`;
  bindChips($('#tab-connections'));
  showTab('connections');
  stream(`/api/connect?a=${a}&b=${b}`);
}

function stream(url) {
  $('#run').disabled = true;
  source = new EventSource(url);
  let finished = false;
  source.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    handle(ev);
    if (ev.type === 'done' || ev.type === 'error') {
      finished = true;
      source.close();
      $('#run').disabled = false;
    }
  };
  source.onerror = () => {
    source.close();
    $('#run').disabled = false;
    if (!finished) showError('Lost connection to the agent server. Is `npm run dev` running?');
  };
}

function handle(ev) {
  switch (ev.type) {
    case 'step':
      renderStep(ev);
      break;
    case 'log': {
      const logs = document.querySelector(`#step-${ev.step} .logs`);
      if (!logs) break;
      const d = document.createElement('div');
      d.textContent = ev.text;
      logs.appendChild(d);
      logs.scrollTop = logs.scrollHeight;
      break;
    }
    case 'trial':
      $('#tab-connections').innerHTML = `<div class="trial"><a href="${esc(ev.trial.url)}" target="_blank" rel="noopener">${esc(ev.trial.nct)} ↗</a><div>${esc(ev.trial.title)}</div></div><div class="placeholder">Grounding entities in PrimeKG…</div>`;
      break;
    case 'probe':
      ev.nodes.forEach((i, k) => brain.fire([i], k * 0.03));
      break;
    case 'seeds': {
      brain.setDim(true);
      for (const s of ev.seeds) brain.label(s.i, s.name, `seed ${s.side}`);
      brain.fire(ev.seeds.map((s) => s.i));
      brain.focus(
        ev.seeds.map((s) => s.i),
        { distance: 900 },
      );
      break;
    }
    case 'search':
      telemetry.search(ev);
      break;
    case 'signal':
      brain.wave(ev);
      telemetry.signal(ev);
      break;
    case 'meet':
      brain.fire(ev.nodes, ev.times ?? 0);
      telemetry.meet(ev.times);
      break;
    case 'paths':
      paths = ev.paths;
      brain.lockPaths(paths);
      labelPaths(paths.slice(0, 8));
      if (paths.length) brain.focus([...new Set(paths.flatMap((p) => p.nodes))]);
      telemetry.paths(paths.length);
      $('#tab-note').textContent = `${paths.length} found`;
      renderConnections();
      break;
    case 'summary':
      summary = ev.summary;
      renderConnections();
      break;
    case 'done':
      if (!expectSummary && !paths.length) $('#tab-connections .placeholder').textContent = 'No pathway connects these nodes within four hops.';
      break;
    case 'error':
      document.querySelectorAll('.step.running').forEach((s) => {
        s.classList.replace('running', 'error');
        s.querySelector('.state').textContent = STATE.error;
      });
      showError(ev.message);
      break;
  }
}

function labelPaths(list) {
  for (const p of list) for (const n of p.nodes.slice(1, -1)) brain.label(n, kg.names[n], 'path');
}

const STATE = { running: 'Active', done: 'Done', error: 'Failed' };
function renderStep({ id, title, status, detail }) {
  let li = document.getElementById(`step-${id}`);
  if (!li) {
    li = document.createElement('li');
    li.id = `step-${id}`;
    li.innerHTML = '<div class="head"><span class="title"></span><span class="state"></span></div><div class="detail"></div><div class="logs"></div>';
    $('#steps').appendChild(li);
  }
  li.className = `step ${status}`;
  li.querySelector('.title').textContent = title;
  li.querySelector('.state').textContent = STATE[status] ?? status;
  if (detail) li.querySelector('.detail').textContent = detail;
}

function showError(msg) {
  const d = document.createElement('div');
  d.className = 'error-box';
  d.innerHTML = esc(msg).replace(/`([^`]+)`/g, '<code>$1</code>');
  $('#steps').appendChild(d);
}

// ---------- connections ----------

function pathChain(p) {
  return p.nodes.map((n, k) => `${k ? `<span class="rel">${esc(p.rels[k - 1])} →</span>` : ''}${chip(n)}`).join('');
}

function renderConnections() {
  if (!paths.length && !summary) return;
  const trial = $('#tab-connections .trial')?.outerHTML || '';
  const s = summary;
  const keyIds = new Map((s?.key_paths || []).map((k) => [k.path, k]));
  const card = (p) => {
    const key = keyIds.get(p.id);
    const pct = Math.min(100, Math.round(p.strength * 100));
    return `<div class="path ${selectedPath === p.id ? 'active' : ''} ${key ? 'key' : ''}" data-id="${p.id}">
      <div class="phead"><span class="pid">${p.id}</span><span class="kind">${esc(p.kind)}</span><span class="pct">${pct}%</span></div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      ${key?.title ? `<div class="t">${esc(key.title)}</div>` : ''}
      <div class="chain">${pathChain(p)}</div>
      ${key?.explanation ? `<div class="e prose">${esc(key.explanation)}</div>` : ''}
    </div>`;
  };
  const keyPaths = paths.filter((p) => keyIds.has(p.id));
  const others = paths.filter((p) => !keyIds.has(p.id));
  const synthesis = s
    ? `<div class="syn">
      <span class="conf ${esc(s.confidence)}">Confidence: ${esc(s.confidence || 'n/a')}</span>
      <h2>${esc(s.headline)}</h2>
      <p class="prose">${esc(s.summary)}</p>
      ${s.mechanistic_hypothesis ? `<h3>Mechanistic hypothesis</h3><p class="prose">${esc(s.mechanistic_hypothesis)}</p>` : ''}
    </div>`
    : expectSummary
      ? '<div class="placeholder">Interpreting pathways…</div>'
      : '';
  $('#tab-connections').innerHTML = `
    ${trial}
    ${synthesis}
    ${keyPaths.length ? `<h3 class="sec">Key pathways</h3>${keyPaths.map(card).join('')}` : ''}
    ${others.length ? `<h3 class="sec">All pathways found (${paths.length}) · click to trace</h3>${others.map(card).join('')}` : ''}
    ${s?.novel_hypotheses?.length ? `<h3 class="sec">Hypotheses</h3><ul class="prose">${s.novel_hypotheses.map((h) => `<li>${esc(h)}</li>`).join('')}</ul>` : ''}
    ${s?.gaps?.length ? `<h3 class="sec">Gaps</h3><ul class="prose">${s.gaps.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>` : ''}`;
  document.querySelectorAll('#tab-connections .path').forEach((el) =>
    el.addEventListener('click', (e) => {
      if (e.target.closest('.chip')) return;
      selectPath(el.dataset.id === selectedPath ? null : el.dataset.id);
    }),
  );
  bindChips($('#tab-connections'));
}

function selectPath(id) {
  selectedPath = id;
  document.querySelectorAll('#tab-connections .path').forEach((el) => el.classList.toggle('active', el.dataset.id === id));
  brain.lockPaths(paths, id);
  if (!id) return;
  const p = paths.find((x) => x.id === id);
  for (const n of p.nodes) brain.label(n, kg.names[n], 'path');
  brain.focus(p.nodes);
}

function deselect() {
  if (connectFrom >= 0) return cancelConnect();
  if (selectedPath) selectPath(null);
  unpin();
}

function unpin() {
  pinned = -1;
  brain.clearEgo();
  brain.clearLabels('node');
  brain.mark(-1);
  telemetry.mark(-1);
}

// ---------- connect mode ----------

function startConnect(i) {
  connectFrom = i;
  $('#mode').innerHTML = `<b>Connect</b>${chip(i)}<b>to…</b><span>click any node or search for one · Esc to cancel</span>`;
  $('#mode').classList.remove('hidden');
  bindChips($('#mode'));
  $('#nct').placeholder = 'search the node to connect to';
  $('#nct').value = '';
  $('#nct').focus();
}

function cancelConnect() {
  connectFrom = -1;
  $('#mode').classList.add('hidden');
  $('#nct').placeholder = 'NCT02142738';
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') deselect();
});

// ---------- inspector ----------

function sourceLink(n) {
  const id = String(n.id).split('_')[0];
  const pad = (s) => s.padStart(7, '0');
  switch (n.source) {
    case 'DrugBank': return [`DrugBank ${id}`, `https://go.drugbank.com/drugs/${id}`];
    case 'NCBI': return [`NCBI Gene ${id}`, `https://www.ncbi.nlm.nih.gov/gene/${id}`];
    case 'MONDO':
    case 'MONDO_grouped': return [`MONDO:${pad(id)}`, `https://monarchinitiative.org/MONDO:${pad(id)}`];
    case 'HPO': return [`HP:${pad(id)}`, `https://hpo.jax.org/browse/term/HP:${pad(id)}`];
    case 'GO': return [`GO:${pad(id)}`, `https://amigo.geneontology.org/amigo/term/GO:${pad(id)}`];
    case 'UBERON': return [`UBERON:${pad(id)}`, `https://www.ebi.ac.uk/ols4/ontologies/uberon/classes?obo_id=UBERON:${pad(id)}`];
    case 'REACTOME': return [id, `https://reactome.org/content/detail/${id}`];
    case 'CTD': return [`CTD ${id}`, `https://ctdbase.org/detail.go?type=chem&acc=${id}`];
    default: return null;
  }
}

async function inspect(i) {
  if (connectFrom >= 0) {
    if (i !== connectFrom) connect(connectFrom, i);
    return;
  }
  showTab('inspector');
  pinned = i;
  brain.clearLabels('node');
  brain.clearLabels('focus');
  brain.label(i, kg.names[i], 'focus');
  brain.mark(i);
  brain.focus([i], { distance: 240 });
  telemetry.mark(i);
  $('#tab-inspector').innerHTML = '<div class="placeholder">Reading node…</div>';
  const [info, nb] = await Promise.all([
    fetch(`/api/primekg/node/${i}`)
      .then((r) => r.json())
      .catch(() => null),
    neighbors(i),
  ]);
  if (pinned !== i) return;
  if (!info || info.error) {
    $('#tab-inspector').innerHTML = `<div class="placeholder">${esc(info?.error || 'Agent server offline.')}</div>`;
    return;
  }
  if (nb) {
    brain.showEgo(i, nb.nbrs);
    const named = [...new Set(nb.nbrs)].sort((a, b) => kg.degree[b] - kg.degree[a]).slice(0, 10);
    for (const j of named) brain.label(j, kg.names[j], 'node');
  }

  const t = typeOf(i);
  const link = sourceLink(info);
  const f = Object.fromEntries(Object.entries(info.features || {}).map(([k, v]) => [k, typeof v === 'string' ? plain(v) : v]));
  const facts = [
    ['Description', f.description || f.definition],
    ['Mechanism', f.mechanism],
    ['Indication', f.indication],
    ['Symptoms', f.symptoms],
    ['Prevalence', f.prevalence],
    ['Category', f.category],
    ['Status', f.group],
  ].filter(([, v]) => v);
  const shown = nb ? nb.nbrs.length : 0;
  $('#tab-inspector').innerHTML = `
    <div class="insp">
      <div class="insp-meta"><span class="kind" style="--c:${t.color}">${esc(t.label)}</span>${link ? `<a href="${esc(link[1])}" target="_blank" rel="noopener">${esc(link[0])} ↗</a>` : ''}</div>
      <h2>${esc(info.name)}</h2>
      <div class="stats">
        <span>Connections:</span><b>${info.degree.toLocaleString()}</b>
        <span>Edges drawn:</span><b>${shown < info.degree ? `${shown.toLocaleString()} · sampled per relation` : 'all'}</b>
        <span>Relation kinds:</span><b>${info.byRelation.length}</b>
      </div>
      <button class="connect-btn">Connect to →</button>
      ${facts.map(([k, v]) => `<h3>${esc(k)}</h3><p class="prose">${esc(v)}</p>`).join('')}
      <h3>Connections by relation</h3>
      ${info.byRelation
        .map(
          (g) => `<div class="relgroup"><div class="rg-head"><span>${esc(g.display)}<span class="to"> · ${esc(typeLabel(g.sample[0]?.type))}</span></span><span class="n">${g.count.toLocaleString()}</span></div>
          <div class="chips">${g.sample.map((n) => chip(n.i, n.name)).join('')}${g.count > g.sample.length ? `<span class="more">+${(g.count - g.sample.length).toLocaleString()}</span>` : ''}</div></div>`,
        )
        .join('')}
    </div>`;
  $('#tab-inspector .connect-btn').addEventListener('click', () => startConnect(i));
  bindChips($('#tab-inspector'));
}

// ---------- hover, tabs, legend, controls ----------

function onHover(i, x, y) {
  showTip(i, x, y);
  if (i === hovered) return;
  hovered = i;
  clearTimeout(hoverTimer);
  if (pinned >= 0) return;
  if (i < 0) return brain.clearEgo();
  hoverTimer = setTimeout(async () => {
    const nb = await neighbors(i);
    if (nb && hovered === i && pinned < 0) brain.showEgo(i, nb.nbrs);
  }, 80);
}

function showTip(i, x, y) {
  const tip = $('#tip');
  if (i < 0) return tip.classList.add('hidden');
  const t = typeOf(i);
  const hint = connectFrom >= 0 && i !== connectFrom ? ' · click to connect' : '';
  tip.innerHTML = `<span class="sw" style="background:${t.color}"></span><b>${esc(kg.names[i])}</b><span class="ty">${esc(t.label)} · ${kg.degree[i].toLocaleString()} edge${kg.degree[i] === 1 ? '' : 's'}${hint}</span>`;
  tip.style.transform = `translate(${x + 14}px, ${y + 14}px)`;
  tip.classList.remove('hidden');
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  $('#tab-connections').classList.toggle('hidden', name !== 'connections');
  $('#tab-inspector').classList.toggle('hidden', name !== 'inspector');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));

const SHORT = {
  'gene/protein': 'Gene',
  'effect/phenotype': 'Phenotype',
  biological_process: 'Bio-process',
  molecular_function: 'Mol-function',
  cellular_component: 'Cell-component',
};
function renderLegend() {
  $('#legend').innerHTML = kg.meta.types
    .map((t, k) => `<button data-k="${k}" title="${t.count.toLocaleString()} ${esc(t.label)} nodes · click to hide"><span class="sw" style="background:${t.color}"></span>${esc(SHORT[t.name] || t.label)}<span class="n">${(t.count / 1000).toFixed(1)}k</span></button>`)
    .join('');
  document.querySelectorAll('#legend button').forEach((b) =>
    b.addEventListener('click', () => {
      const off = !b.classList.contains('off');
      b.classList.toggle('off', off);
      brain.setTypeVisible(Number(b.dataset.k), !off);
    }),
  );
}

function toggle(id, fn) {
  const b = $(id);
  b.addEventListener('click', () => {
    const on = !b.classList.contains('on');
    b.classList.toggle('on', on);
    b.textContent = `${b.dataset.label}: ${on ? 'on' : 'off'}`;
    fn(on);
  });
}
toggle('#rotate', (on) => brain.setOrbit(on));
toggle('#labels', (on) => brain.setLabels(on));
$('#home').addEventListener('click', () => brain.home());
$('#clear').addEventListener('click', () => {
  cancelConnect();
  reset();
  brain.home();
});
window.addEventListener('resize', () => {
  if (!brain) return;
  brain.setInsets(insets());
  telemetry.redraw();
});

boot();
