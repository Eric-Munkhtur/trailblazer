import { createScene, KINDS, LINK_COLORS, visualKind } from './scene.js';

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const scene = createScene($('#scene'), {
  onNodeClick: (n) => selectNode(n),
  onBackgroundClick: () => {
    scene.highlight(null);
    document.querySelectorAll('.path.active').forEach((p) => p.classList.remove('active'));
  },
});

let source = null;
let summary = null;
let currentNct = null;

// ---------- run ----------

$('#run-form').addEventListener('submit', (e) => {
  e.preventDefault();
  run($('#nct').value);
});
document.querySelectorAll('.examples button').forEach((b) =>
  b.addEventListener('click', () => {
    $('#nct').value = b.dataset.nct;
    run(b.dataset.nct);
  }),
);

function run(raw) {
  const nct = String(raw || '').trim().toUpperCase();
  if (!nct) return;
  source?.close();
  currentNct = nct;
  summary = null;
  scene.clear();
  $('#steps').innerHTML = '';
  $('#tab-synthesis').innerHTML = '<div class="placeholder">Agent is assembling the graph…</div>';
  $('#tab-inspector').innerHTML = '<div class="placeholder">Click any node in the graph to inspect it.</div>';
  showTab('synthesis');
  $('#run').disabled = true;
  history.replaceState(null, '', `?nct=${nct}`);
  updateStats();

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
      if (logs) {
        const d = document.createElement('div');
        d.textContent = ev.text;
        logs.appendChild(d);
        logs.scrollTop = logs.scrollHeight;
      }
      break;
    }
    case 'graph':
      scene.add(ev);
      updateLegend();
      updateStats();
      break;
    case 'summary':
      summary = ev.summary;
      renderSummary();
      break;
    case 'done':
      setTimeout(() => scene.fit(), 1500);
      setTimeout(() => !document.querySelector('.path.active') && $('#tab-inspector').classList.contains('hidden') && scene.fit(), 6000);
      break;
    case 'error':
      document.querySelectorAll('.step.running').forEach((s) => s.classList.replace('running', 'error'));
      showError(ev.message);
      break;
  }
}

function renderStep({ id, title, status, detail }) {
  let li = document.getElementById(`step-${id}`);
  if (!li) {
    li = document.createElement('li');
    li.id = `step-${id}`;
    li.className = 'step';
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
  d.textContent = msg;
  $('#steps').appendChild(d);
  if (!summary) $('#tab-synthesis').innerHTML = `<div class="placeholder">The run stopped before synthesis.</div>`;
}

// ---------- synthesis ----------

function renderSummary() {
  const s = summary;
  const label = (id) => scene.nodeById.get(id)?.label || id;
  $('#tab-synthesis').innerHTML = `
    <div class="syn">
      <span class="conf ${esc(s.confidence)}">confidence: ${esc(s.confidence || 'n/a')}</span>
      <h2>${esc(s.headline)}</h2>
      <p>${esc(s.summary)}</p>
      <h3>Mechanistic hypothesis</h3>
      <p>${esc(s.mechanistic_hypothesis)}</p>
      ${s.key_paths?.length ? `<h3>Key paths · click to trace</h3>` : ''}
      ${(s.key_paths || [])
        .map(
          (p, i) => `<button class="path" data-i="${i}">
            <div class="t">${esc(p.title)}</div>
            <div class="chain">${p.nodes.map((id) => esc(label(id))).join(' → ')}</div>
            <div class="e">${esc(p.explanation)}</div>
          </button>`,
        )
        .join('')}
      ${s.novel_links?.length ? `<h3>Hypothesized links</h3><ul>${s.novel_links.map((l) => `<li><b>${esc(label(l.source))}</b> ⇢ <b>${esc(label(l.target))}</b>: ${esc(l.rationale)}</li>`).join('')}</ul>` : ''}
      ${s.gaps?.length ? `<h3>Gaps</h3><ul>${s.gaps.map((g) => `<li>${esc(g)}</li>`).join('')}</ul>` : ''}
    </div>`;
  document.querySelectorAll('.path').forEach((b) =>
    b.addEventListener('click', () => {
      const p = summary.key_paths[b.dataset.i];
      const active = b.classList.contains('active');
      document.querySelectorAll('.path.active').forEach((x) => x.classList.remove('active'));
      if (active) return scene.highlight(null);
      b.classList.add('active');
      scene.highlight(p.nodes, scene.linksBetween(p.nodes));
      scene.fit((n) => p.nodes.includes(n.id));
    }),
  );
}

// ---------- inspector ----------

function selectNode(n) {
  const nbrs = scene.neighbors(n.id);
  scene.highlight([n.id, ...nbrs.map((x) => x.node.id)], nbrs.map((x) => x.link.id));
  scene.focus(n);
  renderInspector(n, nbrs);
  showTab('inspector');
}

const FIELDS = {
  trial: [['title', 'Title'], ['status', 'Status'], ['phases', 'Phase'], ['studyType', 'Type'], ['sponsor', 'Sponsor'], ['enrollment', 'Enrollment'], ['startDate', 'Start'], ['primaryOutcomes', 'Primary outcomes'], ['summary', 'Summary']],
  intervention: [['originalName', 'As registered'], ['type', 'Modality'], ['role', 'Arm role'], ['drugClass', 'Class'], ['mechanism', 'Mechanism'], ['otherNames', 'Also known as'], ['chembl', 'ChEMBL']],
  disease: [['originalName', 'As registered'], ['category', 'Category'], ['isPrimary', 'Primary'], ['efo', 'Ontology ID'], ['description', 'Description']],
  target: [['name', 'Name'], ['targetName', 'ChEMBL target'], ['targetType', 'Target type'], ['genes', 'Genes'], ['organism', 'Organism'], ['uniprot', 'UniProt'], ['ensembl', 'Ensembl'], ['drugTarget', 'Drug target'], ['diseaseGene', 'Disease gene'], ['inferred', 'LLM-inferred']],
  mesh: [['meshId', 'MeSH ID'], ['treeNumbers', 'Tree numbers'], ['level', 'Depth'], ['supplementary', 'Supplementary']],
  area: [['id', 'ID']],
};

function fmt(v) {
  if (v == null || v === '' || (Array.isArray(v) && !v.length)) return null;
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return v.chemblId ? `${v.chemblId} · ${v.moleculeType || ''} · max phase ${v.maxPhase ?? '?'}${v.firstApproval ? ` · approved ${v.firstApproval}` : ''}` : JSON.stringify(v);
  return String(v);
}

function externalLinks(n) {
  const d = n.data;
  const out = [];
  if (d.url) out.push(['ClinicalTrials.gov', d.url]);
  if (d.chembl) out.push(['ChEMBL', `https://www.ebi.ac.uk/chembl/explore/compound/${d.chembl.chemblId}`]);
  if (d.chemblTarget) out.push(['ChEMBL target', `https://www.ebi.ac.uk/chembl/explore/target/${d.chemblTarget}`]);
  if (d.ensembl) out.push(['Open Targets', `https://platform.opentargets.org/target/${d.ensembl}`]);
  if (d.uniprot) out.push(['UniProt', `https://www.uniprot.org/uniprotkb/${d.uniprot}`]);
  if (d.efo) out.push(['Open Targets', `https://platform.opentargets.org/disease/${d.efo}`]);
  if (d.meshId) out.push(['MeSH', `https://meshb.nlm.nih.gov/record/ui?ui=${d.meshId}`]);
  if (n.kind === 'area') out.push(['Open Targets', `https://platform.opentargets.org/disease/${d.id}`]);
  return out;
}

function renderInspector(n, nbrs) {
  const k = KINDS[visualKind(n)];
  const rows = (FIELDS[n.kind] || [])
    .map(([key, lbl]) => [lbl, fmt(n.data[key])])
    .filter(([, v]) => v != null)
    .map(([l, v]) => `<dt>${esc(l)}</dt><dd>${esc(v)}</dd>`)
    .join('');
  const links = externalLinks(n).map(([l, u]) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(l)} ↗</a>`).join('');
  const nb = nbrs
    .sort((a, b) => (a.link.type === 'BROADER') - (b.link.type === 'BROADER'))
    .map(({ link, node, out }) => {
      const c = KINDS[visualKind(node)].color;
      const rel = `${out ? '' : '← '}${link.type.toLowerCase().replace(/_/g, ' ')}${out ? ' →' : ''}`;
      return `<button class="nbr" data-id="${esc(node.id)}" title="${esc(link.mechanism || link.rationale || link.label || '')}">
        <span class="rel">${esc(rel)}</span><span class="sw" style="background:${c}"></span><span class="lbl">${esc(node.label)}${link.bridge ? ' ✦' : ''}</span></button>`;
    })
    .join('');
  $('#tab-inspector').innerHTML = `
    <div class="insp">
      <span class="kind" style="color:${k.color};border:1px solid ${k.color}55">${esc(k.label)}</span>
      <h2>${esc(n.label)}</h2>
      <div class="src">source: ${esc(n.source || '—')}</div>
      <dl>${rows}</dl>
      ${links ? `<div class="links">${links}</div>` : ''}
      <h3>Connections (${nbrs.length})</h3>
      ${nb}
    </div>`;
  document.querySelectorAll('.nbr').forEach((b) => b.addEventListener('click', () => selectNode(scene.nodeById.get(b.dataset.id))));
}

// ---------- tabs, legend, controls ----------

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  $('#tab-synthesis').classList.toggle('hidden', name !== 'synthesis');
  $('#tab-inspector').classList.toggle('hidden', name !== 'inspector');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));

const hiddenKinds = new Set();
function updateLegend() {
  const counts = {};
  scene.data.nodes.forEach((n) => (counts[visualKind(n)] = (counts[visualKind(n)] || 0) + 1));
  $('#legend').innerHTML = Object.entries(KINDS)
    .map(([key, k]) => `<button data-k="${key}" class="${hiddenKinds.has(key) ? 'off' : ''}"><span class="sw" style="background:${k.color};box-shadow:0 0 8px ${k.color}"></span>${k.label}<span class="n">${counts[key] || 0}</span></button>`)
    .join('');
  document.querySelectorAll('#legend button').forEach((b) =>
    b.addEventListener('click', () => {
      const key = b.dataset.k;
      const hide = !hiddenKinds.has(key);
      hide ? hiddenKinds.add(key) : hiddenKinds.delete(key);
      scene.setHidden(key, hide);
      b.classList.toggle('off', hide);
    }),
  );
}

function updateStats() {
  const bridges = scene.data.links.filter((l) => l.bridge).length;
  $('#stats').textContent = currentNct ? `${scene.data.nodes.length} nodes · ${scene.data.links.length} edges · ${bridges} bridges` : '';
}

function toggle(id, fn) {
  const b = $(id);
  b.addEventListener('click', () => {
    const on = !b.classList.contains('on');
    b.classList.toggle('on', on);
    fn(on);
  });
}
toggle('#layout', (on) => scene.setSpatial(on));
toggle('#labels', (on) => scene.setLabels(on));
toggle('#rotate', (on) => scene.setOrbit(on));
$('#fit').addEventListener('click', () => scene.fit());
$('#export').addEventListener('click', () => {
  const nodes = scene.data.nodes.map(({ id, label, kind, source, data }) => ({ id, label, kind, source, data }));
  const links = scene.data.links.map(({ id, type, label, score, bridge, inferred, mechanism, rationale, provenance, source, target }) => ({ id, type, label, score, bridge, inferred, mechanism, rationale, provenance, source: source.id, target: target.id }));
  const blob = new Blob([JSON.stringify({ nct: currentNct, summary, nodes, links, linkTypes: Object.keys(LINK_COLORS) }, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: `${currentNct || 'graph'}.kg.json` });
  a.click();
  URL.revokeObjectURL(a.href);
});

updateLegend();

fetch('/api/health')
  .then((r) => r.json())
  .then(({ llm }) => {
    const el = $('#llm-status');
    el.textContent = llm.ready ? `agent LLM · ${llm.provider}/${llm.model}` : 'no LLM key set · add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env';
    el.classList.toggle('bad', !llm.ready || llm.provider === 'mock');
  })
  .catch(() => ($('#llm-status').textContent = 'agent server offline'));

const initial = new URLSearchParams(location.search).get('nct');
if (initial) {
  $('#nct').value = initial;
  run(initial);
}
