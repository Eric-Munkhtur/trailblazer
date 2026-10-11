// Downloads PrimeKG (Chandak, Huang & Zitnik, Scientific Data 2023) and compiles it into
// binary CSR graph files for the server plus a force-directed 3D layout for the browser.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { forceAtlas2, spectralInit } from './forcelayout.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RAW = path.join(ROOT, 'data/primekg');
const OUT = path.join(RAW, 'graph');
const WEB = path.join(ROOT, 'public/primekg');
const DATAVERSE = 'https://dataverse.harvard.edu/api/access/datafile';
const FILES = { 'nodes.tab': 6180617, 'edges.csv': 6180616, 'drug_features.tab': 6180619, 'disease_features.tab': 6180618 };

const TYPES = {
  'gene/protein': { label: 'Gene / protein', color: '#ffb547' },
  drug: { label: 'Drug', color: '#38e1ff' },
  disease: { label: 'Disease', color: '#ff4f9a' },
  'effect/phenotype': { label: 'Phenotype / effect', color: '#ff7a59' },
  biological_process: { label: 'Biological process', color: '#7c8cff' },
  molecular_function: { label: 'Molecular function', color: '#b18cff' },
  cellular_component: { label: 'Cellular component', color: '#e08cff' },
  pathway: { label: 'Pathway', color: '#5cf2c0' },
  anatomy: { label: 'Anatomy', color: '#9ae66e' },
  exposure: { label: 'Exposure', color: '#f5e663' },
};
// Tissue expression (1.5M edges) and drug–drug synergy (1.3M) link almost everything to
// everything; the layout uses the other ~1.2M edges plus a few of the most specific excluded
// ones (see layoutEdges).
const LAYOUT_EXCLUDE = new Set(['anatomy_protein_present', 'anatomy_protein_absent', 'drug_drug']);
const FALLBACK_EDGES = 3;
const ANATOMY_LINKS = 3;
// LinLog attraction separates communities (Noack 2007); gravity keeps them from drifting apart.
const FA2 = { iterations: 700, linLog: true, gravity: 5, scaling: 2 };
const RADIUS = 420;
const BG_EDGES = 140000;

const log = (...a) => console.log('[primekg]', ...a);
// Seeded PRNG: the same PrimeKG release always produces the same map.
const random = (() => {
  let seed = 20231;
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
})();
const t0 = Date.now();
const elapsed = () => `${((Date.now() - t0) / 1000).toFixed(1)}s`;

async function download() {
  fs.mkdirSync(RAW, { recursive: true });
  for (const [name, id] of Object.entries(FILES)) {
    const dest = path.join(RAW, name);
    if (fs.existsSync(dest)) continue;
    log(`downloading ${name} from Harvard Dataverse…`);
    const res = await fetch(`${DATAVERSE}/${id}`, { redirect: 'follow' });
    if (!res.ok) throw new Error(`Download failed for ${name}: HTTP ${res.status}`);
    await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(`${dest}.part`));
    fs.renameSync(`${dest}.part`, dest);
  }
}

// Quote-aware TSV parser (fields may contain quoted tabs/newlines).
function* parseTSV(text) {
  const n = text.length;
  let i = 0;
  while (i < n) {
    const row = [];
    for (;;) {
      let field = '';
      if (text[i] === '"') {
        i++;
        for (;;) {
          const q = text.indexOf('"', i);
          if (q === -1) {
            field += text.slice(i);
            i = n;
            break;
          }
          field += text.slice(i, q);
          if (text[q + 1] === '"') {
            field += '"';
            i = q + 2;
          } else {
            i = q + 1;
            break;
          }
        }
      } else {
        let j = i;
        while (j < n && text[j] !== '\t' && text[j] !== '\n' && text[j] !== '\r') j++;
        field = text.slice(i, j);
        i = j;
      }
      row.push(field);
      if (text[i] === '\t') {
        i++;
        continue;
      }
      if (text[i] === '\r') i++;
      if (text[i] === '\n') i++;
      break;
    }
    yield row;
  }
}

function readNodes() {
  const rows = parseTSV(fs.readFileSync(path.join(RAW, 'nodes.tab'), 'utf8'));
  rows.next();
  const typeNames = Object.keys(TYPES);
  const nodes = { ids: [], types: [], names: [], sources: [] };
  for (const r of rows) {
    if (r.length < 5) continue;
    const i = Number(r[0]);
    let t = typeNames.indexOf(r[2]);
    if (t < 0) t = typeNames.push(r[2]) - 1;
    nodes.ids[i] = r[1];
    nodes.types[i] = t;
    nodes.names[i] = r[3].replace(/\s+/g, ' ').trim();
    nodes.sources[i] = r[4];
  }
  return { ...nodes, typeNames };
}

async function readEdges(N) {
  let cap = 8_200_000;
  let xs = new Uint32Array(cap);
  let ys = new Uint32Array(cap);
  let rs = new Uint8Array(cap);
  const relIndex = new Map();
  const relations = [];
  let E = 0;
  const rl = readline.createInterface({ input: fs.createReadStream(path.join(RAW, 'edges.csv')), crlfDelay: Infinity });
  let header = true;
  for await (const line of rl) {
    if (header) {
      header = false;
      continue;
    }
    const a = line.split(',');
    if (a.length < 4) continue;
    const key = `${a[0]}|${a[1]}`;
    let code = relIndex.get(key);
    if (code === undefined) {
      code = relations.push({ relation: a[0], display: a[1], count: 0 }) - 1;
      relIndex.set(key, code);
    }
    if (E === cap) {
      cap = Math.ceil(cap * 1.5);
      const grow = (arr, T) => {
        const b = new T(cap);
        b.set(arr);
        return b;
      };
      xs = grow(xs, Uint32Array);
      ys = grow(ys, Uint32Array);
      rs = grow(rs, Uint8Array);
    }
    xs[E] = Number(a[2]);
    ys[E] = Number(a[3]);
    rs[E] = code;
    relations[code].count++;
    E++;
  }
  return { xs: xs.subarray(0, E), ys: ys.subarray(0, E), rs: rs.subarray(0, E), E, relations };
}

function buildCSR(N, xs, ys, rs) {
  const E = xs.length;
  const offsets = new Uint32Array(N + 1);
  for (let e = 0; e < E; e++) offsets[xs[e] + 1]++;
  for (let i = 0; i < N; i++) offsets[i + 1] += offsets[i];
  const cursor = offsets.slice(0, N);
  const nbrs = new Uint32Array(E);
  const rels = new Uint8Array(E);
  for (let e = 0; e < E; e++) {
    const p = cursor[xs[e]]++;
    nbrs[p] = ys[e];
    rels[p] = rs[e];
  }
  return { offsets, nbrs, rels };
}

function isSymmetric({ offsets, nbrs }, xs, ys) {
  let ok = 0;
  const samples = 3000;
  for (let s = 0; s < samples; s++) {
    const e = Math.floor(Math.random() * xs.length);
    const x = xs[e];
    const y = ys[e];
    for (let p = offsets[y]; p < offsets[y + 1]; p++) {
      if (nbrs[p] === x) {
        ok++;
        break;
      }
    }
  }
  return ok / samples > 0.99;
}

function readFeatures() {
  const features = {};
  const clip = (s, n) => (s && s.length > n ? `${s.slice(0, n).replace(/\s+\S*$/, '')}…` : s || undefined);
  const drugRows = parseTSV(fs.readFileSync(path.join(RAW, 'drug_features.tab'), 'utf8'));
  const dh = drugRows.next().value;
  const col = (h, name) => h.indexOf(name);
  for (const r of drugRows) {
    if (r.length < dh.length) continue;
    features[r[0]] = {
      description: clip(r[col(dh, 'description')], 700),
      mechanism: clip(r[col(dh, 'mechanism_of_action')], 700),
      indication: clip(r[col(dh, 'indication')], 400),
      category: clip(r[col(dh, 'category')], 200),
      group: r[col(dh, 'group')] || undefined,
      atc: r[col(dh, 'atc_4')] || undefined,
    };
  }
  const disRows = parseTSV(fs.readFileSync(path.join(RAW, 'disease_features.tab'), 'utf8'));
  const sh = disRows.next().value;
  for (const r of disRows) {
    if (r.length < sh.length || features[r[0]]) continue;
    const definition = r[col(sh, 'mondo_definition')] || r[col(sh, 'umls_description')] || r[col(sh, 'orphanet_definition')];
    const f = {
      definition: clip(definition, 700),
      prevalence: clip(r[col(sh, 'orphanet_prevalence')], 120),
      symptoms: clip(r[col(sh, 'mayo_symptoms')], 400),
    };
    if (f.definition || f.symptoms) features[r[0]] = f;
  }
  return features;
}

// ---------- layout ----------

// The layout's edges: every edge of a non-excluded relation, plus a few of the most specific
// excluded ones (lowest-degree neighbours that do have other edges) so that anatomy terms
// attach to their most tissue-specific genes and nodes with no other edges are not left floating.
function layoutEdges(N, { offsets, nbrs, rels }, relations, nodes, useRel) {
  const expr = relations.findIndex((r) => r.relation === 'anatomy_protein_present');
  const anatomy = nodes.typeNames.indexOf('anatomy');
  const degreeIn = (keep) => {
    const d = new Uint32Array(N);
    for (let i = 0; i < N; i++) for (let p = offsets[i]; p < offsets[i + 1]; p++) if (keep(rels[p])) d[i]++;
    return d;
  };
  const exprDegree = degreeIn((r) => r === expr);
  const degree = degreeIn(() => true);
  const semantic = degreeIn((r) => useRel[r] === 1);
  const mostSpecific = (i, keep, deg, k) => {
    const out = [];
    for (let p = offsets[i]; p < offsets[i + 1]; p++) if (keep(p)) out.push(nbrs[p]);
    return out.sort((a, b) => (semantic[a] ? 0 : 1) - (semantic[b] ? 0 : 1) || deg[a] - deg[b] || a - b).slice(0, k);
  };
  const src = [];
  const dst = [];
  const add = (a, b) => {
    src.push(a);
    dst.push(b);
  };
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let p = offsets[i]; p < offsets[i + 1]; p++) {
      if (!useRel[rels[p]]) continue;
      d++;
      if (nbrs[p] > i) add(i, nbrs[p]);
    }
    if (nodes.types[i] === anatomy) for (const g of mostSpecific(i, (p) => rels[p] === expr, exprDegree, ANATOMY_LINKS)) add(i, g);
    else if (!d) for (const j of mostSpecific(i, () => true, degree, FALLBACK_EDGES)) add(i, j);
  }
  return { src: Uint32Array.from(src), dst: Uint32Array.from(dst) };
}

// Centre on the median node and scale so that 95% of nodes fall within RADIUS; the outer 5%
// are compressed radially (monotonically) to at most 1.4 × RADIUS so stragglers stay in view.
function normalize(N, xyz) {
  const med = [0, 1, 2].map((d) => Float64Array.from({ length: N }, (_, i) => xyz[i * 3 + d]).sort()[N >> 1]);
  const r = Float64Array.from({ length: N }, (_, i) => Math.hypot(xyz[i * 3] - med[0], xyz[i * 3 + 1] - med[1], xyz[i * 3 + 2] - med[2]));
  const r95 = Float64Array.from(r).sort()[Math.floor(N * 0.95)] || 1;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const x = r[i] / r95;
    const s = (x > 1 ? 1 + 0.4 * Math.tanh((x - 1) / 0.4) : x) / (x || 1);
    for (let d = 0; d < 3; d++) pos[i * 3 + d] = ((xyz[i * 3 + d] - med[d]) / r95) * RADIUS * s;
  }
  return pos;
}

// Background fibres: a sample of the layout edges, square-root stratified so that small
// relation types (indications, pathways) stay visible next to protein interactions.
function sampleEdges({ offsets, nbrs, rels }, relations, N, useRel) {
  const weights = relations.map((r, k) => (useRel[k] ? Math.sqrt(r.count / 2) : 0));
  const total = weights.reduce((a, b) => a + b, 0);
  const prob = relations.map((r, k) => Math.min(1, (BG_EDGES * weights[k]) / total / (r.count / 2)));
  const pairs = [];
  const prel = [];
  for (let i = 0; i < N; i++) {
    for (let p = offsets[i]; p < offsets[i + 1]; p++) {
      const j = nbrs[p];
      if (j <= i) continue;
      if (random() < prob[rels[p]]) {
        pairs.push(i, j);
        prel.push(rels[p]);
      }
    }
  }
  return { pairs: Uint32Array.from(pairs), rels: Uint8Array.from(prel) };
}

// ---------- main ----------

const write = (dir, name, data) => fs.writeFileSync(path.join(dir, name), data instanceof Uint8Array || ArrayBuffer.isView(data) ? Buffer.from(data.buffer, data.byteOffset, data.byteLength) : data);

await download();
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(WEB, { recursive: true });

const nodes = readNodes();
const N = nodes.names.length;
log(`${N.toLocaleString()} nodes · ${nodes.typeNames.length} types (${elapsed()})`);

const { xs, ys, rs, E, relations } = await readEdges(N);
log(`${E.toLocaleString()} directed edge rows · ${relations.length} relation types (${elapsed()})`);

let csr = buildCSR(N, xs, ys, rs);
if (!isSymmetric(csr, xs, ys)) {
  log('edge list is not symmetric, adding reverse edges');
  const cat = (a, b, T) => {
    const o = new T(a.length + b.length);
    o.set(a);
    o.set(b, a.length);
    return o;
  };
  csr = buildCSR(N, cat(xs, ys, Uint32Array), cat(ys, xs, Uint32Array), cat(rs, rs, Uint8Array));
  relations.forEach((r) => (r.count *= 2));
}
log(`CSR adjacency built (${elapsed()})`);

const useRel = Uint8Array.from(relations, (r) => (LAYOUT_EXCLUDE.has(r.relation) ? 0 : 1));
const ledges = layoutEdges(N, csr, relations, nodes, useRel);
const layoutDegree = new Float64Array(N);
for (let e = 0; e < ledges.src.length; e++) {
  layoutDegree[ledges.src[e]]++;
  layoutDegree[ledges.dst[e]]++;
}
const init = spectralInit(N, csr, (p) => useRel[csr.rels[p]] === 1, { random });
// Spectral axes are unit vectors over all nodes; bring them to FA2's working scale.
const spread = 1000 * Math.sqrt(N);
for (let k = 0; k < init.length; k++) init[k] = init[k] * spread + (random() - 0.5) * 10;
log(`${ledges.src.length.toLocaleString()} layout edges · spectral start (${elapsed()})`);
const xyz = await forceAtlas2(init, ledges, Float64Array.from(layoutDegree, (d) => d + 1), {
  ...FA2,
  random,
  onIteration: (it) => {
    if ((it + 1) % 100 === 0) log(`ForceAtlas2 iteration ${it + 1}/${FA2.iterations} (${elapsed()})`);
  },
});
const positions = normalize(N, xyz);
const bg = sampleEdges(csr, relations, N, useRel);
log(`layout + ${bg.rels.length.toLocaleString()} background edges (${elapsed()})`);

const features = readFeatures();
log(`${Object.keys(features).length.toLocaleString()} drug/disease feature records (${elapsed()})`);

write(OUT, 'offsets.u32', csr.offsets);
write(OUT, 'nbrs.u32', csr.nbrs);
write(OUT, 'rels.u8', csr.rels);
write(OUT, 'nodes.json', JSON.stringify(nodes));
write(OUT, 'relations.json', JSON.stringify(relations));
write(OUT, 'features.json', JSON.stringify(features));

const typeCounts = nodes.typeNames.map(() => 0);
nodes.types.forEach((t) => typeCounts[t]++);
const degree = Uint32Array.from({ length: N }, (_, i) => csr.offsets[i + 1] - csr.offsets[i]);
const meta = {
  source: 'PrimeKG (Chandak, Huang & Zitnik, Scientific Data 2023) · doi:10.7910/DVN/IXA7BM',
  nodes: N,
  edges: csr.nbrs.length / 2,
  layout: { method: 'ForceAtlas2 (3D, Barnes–Hut, LinLog)', ...FA2, edges: ledges.src.length, radius: RADIUS },
  types: nodes.typeNames.map((name, k) => ({ name, label: TYPES[name]?.label || name, color: TYPES[name]?.color || '#cccccc', count: typeCounts[k] })),
  relations: relations.map((r) => ({ relation: r.relation, display: r.display, count: r.count / 2 })),
  backgroundEdges: bg.rels.length,
};
write(WEB, 'meta.json', JSON.stringify(meta));
write(WEB, 'positions.f32', positions);
write(WEB, 'types.u8', Uint8Array.from(nodes.types));
write(WEB, 'degree.u32', degree);
write(WEB, 'names.txt', nodes.names.join('\n'));
write(WEB, 'edges.u32', bg.pairs);
write(WEB, 'edges-rel.u8', bg.rels);
log(`done in ${elapsed()} → data/primekg/graph (server) + public/primekg (browser)`);
