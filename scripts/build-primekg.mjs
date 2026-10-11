// Downloads PrimeKG (Chandak, Huang & Zitnik, Scientific Data 2023) and compiles it into
// binary CSR graph files for the server plus a precomputed 3D "brain" layout for the browser.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RAW = path.join(ROOT, 'data/primekg');
const OUT = path.join(RAW, 'graph');
const WEB = path.join(ROOT, 'public/primekg');
const DATAVERSE = 'https://dataverse.harvard.edu/api/access/datafile';
const FILES = { 'nodes.tab': 6180617, 'edges.csv': 6180616, 'drug_features.tab': 6180619, 'disease_features.tab': 6180618 };

// Region of the brain each node type occupies. Genes/proteins form the inner core; every other
// type is a patch of the outer cortex shell, centred on `dir` (x right, y up, z front).
const TYPES = {
  'gene/protein': { label: 'Gene / protein', color: '#ffb547', core: true },
  drug: { label: 'Drug', color: '#38e1ff', dir: [0, 0.5, -0.87] },
  disease: { label: 'Disease', color: '#ff4f9a', dir: [-0.96, 0.12, -0.1] },
  'effect/phenotype': { label: 'Phenotype / effect', color: '#ff7a59', dir: [0.96, 0.12, -0.1] },
  biological_process: { label: 'Biological process', color: '#7c8cff', dir: [0, 0.8, 0.6] },
  molecular_function: { label: 'Molecular function', color: '#b18cff', dir: [-0.62, 0.3, 0.72] },
  cellular_component: { label: 'Cellular component', color: '#e08cff', dir: [0.62, 0.3, 0.72] },
  pathway: { label: 'Pathway', color: '#5cf2c0', dir: [0, 0.98, -0.2] },
  anatomy: { label: 'Anatomy', color: '#9ae66e', dir: [0, -0.8, -0.6] },
  exposure: { label: 'Exposure', color: '#f5e663', dir: [0.45, 0.05, -0.89] },
};
// Very dense relation types that would swamp the structural layout.
const LAYOUT_EXCLUDE = new Set(['anatomy_protein_present', 'anatomy_protein_absent', 'drug_drug']);
const RADIUS = 300;
const SCALE = [1.12, 0.9, 1.3];
const BG_EDGES = 90000;

const log = (...a) => console.log('[primekg]', ...a);
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

function components(N, { offsets, nbrs, rels }, useRel) {
  const comp = new Int32Array(N).fill(-1);
  const stack = new Uint32Array(N);
  let c = 0;
  for (let s = 0; s < N; s++) {
    if (comp[s] >= 0) continue;
    let top = 0;
    stack[top++] = s;
    comp[s] = c;
    while (top) {
      const u = stack[--top];
      for (let p = offsets[u]; p < offsets[u + 1]; p++) {
        if (!useRel[rels[p]]) continue;
        const v = nbrs[p];
        if (comp[v] < 0) {
          comp[v] = c;
          stack[top++] = v;
        }
      }
    }
    c++;
  }
  return { comp, count: c };
}

// Diffusion coordinates: a few rounds of lazy random-walk smoothing with per-component centring
// and Gram-Schmidt, i.e. orthogonal iteration towards the graph's low-frequency eigenvectors.
function diffusionCoords(N, csr, useRel, iters = 30) {
  const { offsets, nbrs, rels } = csr;
  const { comp, count } = components(N, csr, useRel);
  let X = new Float64Array(N * 3);
  let Y = new Float64Array(N * 3);
  for (let k = 0; k < X.length; k++) X[k] = Math.random() * 2 - 1;
  const sums = new Float64Array(count * 3);
  const sizes = new Float64Array(count);
  for (let i = 0; i < N; i++) sizes[comp[i]]++;

  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < N; i++) {
      let s0 = 0;
      let s1 = 0;
      let s2 = 0;
      let c = 0;
      for (let p = offsets[i]; p < offsets[i + 1]; p++) {
        if (!useRel[rels[p]]) continue;
        const j = nbrs[p] * 3;
        s0 += X[j];
        s1 += X[j + 1];
        s2 += X[j + 2];
        c++;
      }
      const k = i * 3;
      if (c) {
        Y[k] = 0.5 * X[k] + (0.5 * s0) / c;
        Y[k + 1] = 0.5 * X[k + 1] + (0.5 * s1) / c;
        Y[k + 2] = 0.5 * X[k + 2] + (0.5 * s2) / c;
      } else {
        Y[k] = X[k];
        Y[k + 1] = X[k + 1];
        Y[k + 2] = X[k + 2];
      }
    }
    sums.fill(0);
    for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) sums[comp[i] * 3 + d] += Y[i * 3 + d];
    for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) Y[i * 3 + d] -= sums[comp[i] * 3 + d] / sizes[comp[i]];
    for (let d = 0; d < 3; d++) {
      for (let q = 0; q < d; q++) {
        let dot = 0;
        for (let i = 0; i < N; i++) dot += Y[i * 3 + d] * Y[i * 3 + q];
        for (let i = 0; i < N; i++) Y[i * 3 + d] -= dot * Y[i * 3 + q];
      }
      let norm = 0;
      for (let i = 0; i < N; i++) norm += Y[i * 3 + d] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < N; i++) Y[i * 3 + d] /= norm;
    }
    [X, Y] = [Y, X];
  }
  return X;
}

function rankUniform(values) {
  const idx = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const out = new Float64Array(values.length);
  idx.forEach((k, r) => (out[k] = ((r + 0.5) / values.length) * 2 - 1));
  return out;
}

const cubeToBall = (x, y, z) => [
  x * Math.sqrt(Math.max(0, 1 - (y * y) / 2 - (z * z) / 2 + (y * y * z * z) / 3)),
  y * Math.sqrt(Math.max(0, 1 - (z * z) / 2 - (x * x) / 2 + (z * z * x * x) / 3)),
  z * Math.sqrt(Math.max(0, 1 - (x * x) / 2 - (y * y) / 2 + (x * x * y * y) / 3)),
];
const squareToDisc = (x, y) => [x * Math.sqrt(1 - (y * y) / 2), y * Math.sqrt(1 - (x * x) / 2)];
const normalize = (v) => {
  const l = Math.hypot(...v) || 1;
  return v.map((c) => c / l);
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function brainLayout(N, nodes, coords) {
  const pos = new Float32Array(N * 3);
  const byType = new Map();
  for (let i = 0; i < N; i++) {
    const t = nodes.typeNames[nodes.types[i]];
    if (!byType.has(t)) byType.set(t, []);
    byType.get(t).push(i);
  }
  const shellTotal = [...byType].filter(([t]) => !TYPES[t]?.core).reduce((s, [, l]) => s + l.length, 0);
  const jitter = () => (Math.random() - 0.5) * 0.35;

  for (const [t, list] of byType) {
    const cfg = TYPES[t] || { dir: [0, -1, 0] };
    const u = [0, 1, 2].map((d) => rankUniform(Float64Array.from(list, (i) => coords[i * 3 + d] + jitter() * 1e-9)));
    const mix = (k, d) => Math.max(-0.999, Math.min(0.999, u[d][k] * 0.85 + jitter()));
    if (cfg.core) {
      list.forEach((i, k) => {
        const [x, y, z] = cubeToBall(mix(k, 0), mix(k, 1), mix(k, 2));
        const r = 0.5;
        pos.set([x * r, y * r, z * r], i * 3);
      });
      continue;
    }
    const c = normalize(cfg.dir);
    const e1 = normalize(cross(Math.abs(c[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0], c));
    const e2 = cross(c, e1);
    const alpha = Math.min(1.0, Math.sqrt(2.6 * (list.length / shellTotal)));
    list.forEach((i, k) => {
      const [a, b] = squareToDisc(mix(k, 0), mix(k, 1));
      const ta = Math.tan(a * alpha);
      const tb = Math.tan(b * alpha);
      const dir = normalize([c[0] + ta * e1[0] + tb * e2[0], c[1] + ta * e1[1] + tb * e2[1], c[2] + ta * e1[2] + tb * e2[2]]);
      const r = 0.78 + 0.22 * ((mix(k, 2) + 1) / 2);
      pos.set([dir[0] * r, dir[1] * r, dir[2] * r], i * 3);
    });
  }
  for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) pos[i * 3 + d] *= RADIUS * SCALE[d];
  return pos;
}

function sampleEdges({ offsets, nbrs, rels }, relations, N) {
  const weights = relations.map((r) => Math.sqrt(r.count / 2));
  const total = weights.reduce((a, b) => a + b, 0);
  const prob = relations.map((r, k) => Math.min(1, (BG_EDGES * weights[k]) / total / (r.count / 2)));
  const pairs = [];
  const prel = [];
  for (let i = 0; i < N; i++) {
    for (let p = offsets[i]; p < offsets[i + 1]; p++) {
      const j = nbrs[p];
      if (j <= i) continue;
      if (Math.random() < prob[rels[p]]) {
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
const coords = diffusionCoords(N, csr, useRel);
log(`diffusion coordinates computed (${elapsed()})`);
const positions = brainLayout(N, nodes, coords);
const bg = sampleEdges(csr, relations, N);
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
const meta = {
  source: 'PrimeKG (Chandak, Huang & Zitnik, Scientific Data 2023) · doi:10.7910/DVN/IXA7BM',
  nodes: N,
  edges: csr.nbrs.length / 2,
  types: nodes.typeNames.map((name, k) => ({ name, label: TYPES[name]?.label || name, color: TYPES[name]?.color || '#cccccc', count: typeCounts[k] })),
  relations: relations.map((r) => ({ relation: r.relation, display: r.display, count: r.count / 2 })),
  backgroundEdges: bg.rels.length,
  radius: RADIUS,
  scale: SCALE,
};
write(WEB, 'meta.json', JSON.stringify(meta));
write(WEB, 'positions.f32', positions);
write(WEB, 'types.u8', Uint8Array.from(nodes.types));
write(WEB, 'names.txt', nodes.names.join('\n'));
write(WEB, 'edges.u32', bg.pairs);
write(WEB, 'edges-rel.u8', bg.rels);
log(`done in ${elapsed()} → data/primekg/graph (server) + public/primekg (browser)`);
