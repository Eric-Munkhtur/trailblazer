import { loadPrimeKG, createBrain } from './brain.js';

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

let kg;
let brain;
let source = null;
let paths = [];
let summary = null;
let selectedPath = null;

// ---------- boot ----------

async function boot() {
  try {
    kg = await loadPrimeKG();
  } catch (e) {
    $('#boot-msg').innerHTML = esc(e.message).replace(/`([^`]+)`/g, '<code>$1</code>');
    $('#boot').classList.add('error');
    return;
  }
  brain = createBrain($('#scene'), kg, { onHover: showTip, onClick: (i) => (i >= 0 ? inspect(i) : deselect()) });
  $('#kg-status').textContent = `PrimeKG · ${kg.N.toLocaleString()} nodes · ${kg.meta.edges.toLocaleString()} edges`;
  renderLegend();
  $('#boot').classList.add('done');
  setTimeout(() => $('#boot').remove(), 1200);

  fetch('/api/health')
    .then((r) => r.json())
    .then(({ llm, primekg }) => {
      const el = $('#llm-status');
      if (!primekg) {
        el.textContent = 'server: PrimeKG not loaded · run npm run primekg and restart';
        el.classList.add('bad');
        return;
      }
      el.textContent = llm.ready ? `agent LLM · ${llm.provider}/${llm.model}` : 'no LLM key set · add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env';
      el.classList.toggle('bad', !llm.ready || llm.provider === 'mock');
    })
    .catch(() => ($('#llm-status').textContent = 'agent server offline'));

  const q = new URLSearchParams(location.search).get('q');
  if (q) {
    $('#nct').value = q;
    submit(q);
  }
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

function submit(raw) {
  const q = String(raw || '').trim();
  if (!q || !brain) return;
  history.replaceState(null, '', `?q=${encodeURIComponent(q)}`);
  if (/^NCT\d{8}$/i.test(q)) run(q.toUpperCase());
  else searchKG(q);
}

function reset() {
  source?.close();
  paths = [];
  summary = null;
  selectedPath = null;
  brain.clearSignals();
  brain.clearPaths();
  brain.clearLabels();
  brain.setDim(false);
}

async function searchKG(q) {
  reset();
  $('#steps').innerHTML = '<li class="empty">Searching PrimeKG…</li>';
  const hits = await fetch(`/api/primekg/search?q=${encodeURIComponent(q)}&limit=14`).then((r) => r.json()).catch(() => []);
  if (!Array.isArray(hits) || !hits.length) {
    $('#steps').innerHTML = `<li class="empty">No PrimeKG node matches “${esc(q)}”.</li>`;
    return;
  }
  $('#steps').innerHTML = `<li class="hits"><div class="title">PrimeKG matches for “${esc(q)}”</div>${hits
    .map((h) => `<button class="hit" data-i="${h.i}"><span class="sw" style="background:${typeOf(h.i).color}"></span><span class="nm">${esc(h.name)}</span><span class="ty">${esc(typeOf(h.i).label)} · ${h.degree}</span></button>`)
    .join('')}</li>`;
  document.querySelectorAll('.hit').forEach((b) => b.addEventListener('click', () => inspect(Number(b.dataset.i))));
  brain.fire(hits.map((h) => h.i));
  inspect(hits[0].i);
}

// ---------- agent run ----------

function run(nct) {
  reset();
  $('#steps').innerHTML = '';
  $('#tab-connections').innerHTML = '<div class="placeholder">Agent is reading the trial…</div>';
  showTab('connections');
  $('#run').disabled = true;

  source = new EventSource(`/api/assemble?nct=${encodeURIComponent(nct)}`);
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
    case 'signal':
      brain.propagate(ev.edges, { side: ev.side, spread: ev.hop === 1 ? 1.6 : 2.8, strength: ev.hop === 1 ? 0.9 : 0.5 });
      break;
    case 'meet':
      brain.fire(ev.nodes);
      break;
    case 'paths':
      paths = ev.paths;
      brain.lockPaths(paths);
      labelPaths(paths.slice(0, 8));
      if (paths.length) brain.focus([...new Set(paths.flatMap((p) => p.nodes))]);
      renderConnections();
      break;
    case 'summary':
      summary = ev.summary;
      renderConnections();
      break;
    case 'error':
      document.querySelectorAll('.step.running').forEach((s) => s.classList.replace('running', 'error'));
      showError(ev.message);
      break;
  }
}

function labelPaths(list) {
  for (const p of list) for (const n of p.nodes.slice(1, -1)) brain.label(n, kg.names[n], 'path');
}

function renderStep({ id, title, status, detail }) {
  let li = document.getElementById(`step-${id}`);
  if (!li) {
    li = document.createElement('li');
    li.id = `step-${id}`;
    li.innerHTML = '<div class="title"></div><div class="detail"></div><div class="logs"></div>';
    $('#steps').appendChild(li);
  }
  li.className = `step ${status}`;
  li.querySelector('.title').textContent = title;
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
    return `<div class="path ${selectedPath === p.id ? 'active' : ''} ${key ? 'key' : ''}" data-id="${p.id}">
      <div class="phead"><span class="kind">${esc(p.kind)}</span><span class="pid">${p.id}</span><span class="bar"><i style="width:${Math.round(p.strength * 100)}%"></i></span></div>
      ${key?.title ? `<div class="t">${esc(key.title)}</div>` : ''}
      <div class="chain">${pathChain(p)}</div>
      ${key?.explanation ? `<div class="e">${esc(key.explanation)}</div>` : ''}
    </div>`;
  };
  const keyPaths = paths.filter((p) => keyIds.has(p.id));
  const others = paths.filter((p) => !keyIds.has(p.id));
  $('#tab-connections').innerHTML = `
    ${trial}
    ${
      s
        ? `<div class="syn">
      <span class="conf ${esc(s.confidence)}">confidence: ${esc(s.confidence || 'n/a')}</span>
      <h2>${esc(s.headline)}</h2>
      <p>${esc(s.summary)}</p>
      ${s.mechanistic_hypothesis ? `<h3>Mechanistic hypothesis</h3><p>${esc(s.mechanistic_hypothesis)}</p>` : ''}
    </div>`
        : '<div class="placeholder">Interpreting pathways…</div>'
    }
    ${keyPaths.length ? `<h3 class="sec">Key pathways</h3>${keyPaths.map(card).join('')}` : ''}
    ${others.length ? `<h3 class="sec">All pathways found (${paths.length}) · click to trace</h3>${others.map(card).join('')}` : ''}
    ${s?.novel_hypotheses?.length ? `<h3 class="sec">Hypotheses</h3><ul>${s.novel_hypotheses.map((h) => `<li>${esc(h)}</li>`).join('')}</ul>` : ''}
    ${s?.gaps?.length ? `<h3 class="sec">Gaps</h3><ul>${s.gaps.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>` : ''}`;
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
  if (selectedPath) selectPath(null);
}

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
  showTab('inspector');
  $('#tab-inspector').innerHTML = '<div class="placeholder">Reading neuron…</div>';
  brain.label(i, kg.names[i], 'focus');
  brain.focus([i], { distance: 260 });
  const info = await fetch(`/api/primekg/node/${i}`).then((r) => r.json()).catch(() => null);
  if (!info || info.error) {
    $('#tab-inspector').innerHTML = `<div class="placeholder">${esc(info?.error || 'Agent server offline.')}</div>`;
    return;
  }
  const edges = info.byRelation.flatMap((g) => g.sample.map((n) => [i, n.i]));
  brain.propagate(edges, { side: 'focus', spread: 1.2, strength: 0.7 });

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
  $('#tab-inspector').innerHTML = `
    <div class="insp">
      <span class="kind" style="color:${t.color};border-color:${t.color}66">${esc(t.label)}</span>
      <h2>${esc(info.name)}</h2>
      <div class="src">${link ? `<a href="${esc(link[1])}" target="_blank" rel="noopener">${esc(link[0])} ↗</a> · ` : ''}${info.degree.toLocaleString()} connections</div>
      ${facts.map(([k, v]) => `<h3>${esc(k)}</h3><p>${esc(v)}</p>`).join('')}
      <h3>Connections by relation</h3>
      ${info.byRelation
        .map(
          (g) => `<div class="relgroup"><div class="rg-head"><span>${esc(g.display)}<span class="to"> · ${esc(typeLabel(g.sample[0]?.type))}</span></span><span class="n">${g.count.toLocaleString()}</span></div>
          <div class="chips">${g.sample.map((n) => chip(n.i, n.name)).join('')}${g.count > g.sample.length ? `<span class="more">+${(g.count - g.sample.length).toLocaleString()}</span>` : ''}</div></div>`,
        )
        .join('')}
    </div>`;
  bindChips($('#tab-inspector'));
}

// ---------- tooltip, tabs, legend, controls ----------

function showTip(i, x, y) {
  const tip = $('#tip');
  if (i < 0) return tip.classList.add('hidden');
  const t = typeOf(i);
  tip.innerHTML = `<span class="sw" style="background:${t.color}"></span><b>${esc(kg.names[i])}</b><span class="ty">${esc(t.label)}</span>`;
  tip.style.transform = `translate(${x + 14}px, ${y + 14}px)`;
  tip.classList.remove('hidden');
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  $('#tab-connections').classList.toggle('hidden', name !== 'connections');
  $('#tab-inspector').classList.toggle('hidden', name !== 'inspector');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));

function renderLegend() {
  $('#legend').innerHTML = kg.meta.types
    .map((t, k) => `<button data-k="${k}" title="${t.count.toLocaleString()} ${esc(t.label)} nodes"><span class="sw" style="background:${t.color};box-shadow:0 0 8px ${t.color}"></span>${esc(t.label)}<span class="n">${(t.count / 1000).toFixed(1)}k</span></button>`)
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
    fn(on);
  });
}
toggle('#ambient', (on) => brain.setAmbient(on));
toggle('#rotate', (on) => brain.setOrbit(on));
toggle('#labels', (on) => brain.setLabels(on));
$('#home').addEventListener('click', () => brain.home());
$('#clear').addEventListener('click', () => {
  reset();
  brain.home();
});

boot();
