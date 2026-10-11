import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../data/primekg/graph');

export const norm = (s) =>
  String(s || '')
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
const STOP = new Set(['with', 'without', 'of', 'the', 'and', 'or', 'in', 'to', 'for', 'due', 'an', 'on', 'by', 'at', 'as', 'from']);
const tokenize = (s) =>
  norm(s)
    .split(' ')
    .filter((t) => (t.length > 1 || /\d/.test(t)) && !STOP.has(t))
    .map((t) => (t.length > 3 && /[^su]s$/.test(t) ? t.slice(0, -1) : t));

let KG;

export function primekg() {
  if (KG === undefined) KG = load();
  return KG;
}

function typed(file, T) {
  const b = fs.readFileSync(path.join(DIR, file));
  return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

function load() {
  if (!fs.existsSync(path.join(DIR, 'nodes.json'))) return null;
  const t = Date.now();
  const nodes = JSON.parse(fs.readFileSync(path.join(DIR, 'nodes.json'), 'utf8'));
  const relations = JSON.parse(fs.readFileSync(path.join(DIR, 'relations.json'), 'utf8'));
  const features = JSON.parse(fs.readFileSync(path.join(DIR, 'features.json'), 'utf8'));
  const offsets = typed('offsets.u32', Uint32Array);
  const nbrs = typed('nbrs.u32', Uint32Array);
  const rels = typed('rels.u8', Uint8Array);
  const N = nodes.names.length;

  const exact = new Map();
  const postings = new Map();
  const nameTokens = new Array(N);
  for (let i = 0; i < N; i++) {
    const n = norm(nodes.names[i]);
    if (!exact.has(n)) exact.set(n, []);
    exact.get(n).push(i);
    const toks = [...new Set(tokenize(nodes.names[i]))];
    nameTokens[i] = toks.length;
    for (const tok of toks) {
      let list = postings.get(tok);
      if (!list) postings.set(tok, (list = []));
      list.push(i);
    }
  }

  const kg = {
    N,
    E: nbrs.length / 2,
    nodes,
    relations,
    features,
    offsets,
    nbrs,
    rels,
    typeOf: (i) => nodes.typeNames[nodes.types[i]],
    degree: (i) => offsets[i + 1] - offsets[i],
    node: (i) => ({ i, id: nodes.ids[i], name: nodes.names[i], type: nodes.typeNames[nodes.types[i]], source: nodes.sources[i] }),
    search(q, { types, limit = 10 } = {}) {
      const qn = norm(q);
      const qt = [...new Set(tokenize(q))];
      if (!qn) return [];
      const typeOk = types ? new Set(types) : null;
      const scores = new Map();
      for (const i of exact.get(qn) || []) scores.set(i, 3);
      const overlap = new Map();
      for (const tok of qt) {
        const list = postings.get(tok);
        if (!list || list.length > 25000) continue;
        for (const i of list) overlap.set(i, (overlap.get(i) || 0) + 1);
      }
      for (const [i, o] of overlap) {
        const jac = o / (qt.length + nameTokens[i] - o);
        const n = norm(nodes.names[i]);
        // Also reward names that are the head of the query ("heart failure" in "heart failure with reduced EF").
        const bonus = n.startsWith(qn) ? 0.25 : n.includes(qn) ? 0.15 : qn.startsWith(`${n} `) ? 0.2 : 0;
        scores.set(i, Math.max(scores.get(i) || 0, jac + bonus));
      }
      return [...scores]
        .filter(([i]) => !typeOk || typeOk.has(kg.typeOf(i)))
        .sort((a, b) => b[1] - a[1] || kg.degree(b[0]) - kg.degree(a[0]))
        .slice(0, limit)
        .map(([i, score]) => ({ ...kg.node(i), score: +score.toFixed(3), degree: kg.degree(i) }));
    },
    info(i, perRelation = 12) {
      const groups = new Map();
      for (let p = offsets[i]; p < offsets[i + 1]; p++) {
        const r = rels[p];
        if (!groups.has(r)) groups.set(r, []);
        groups.get(r).push(nbrs[p]);
      }
      const byRelation = [...groups]
        .map(([r, list]) => ({
          relation: relations[r].relation,
          display: relations[r].display,
          count: list.length,
          sample: list
            .sort((a, b) => kg.degree(b) - kg.degree(a))
            .slice(0, perRelation)
            .map((j) => ({ i: j, name: nodes.names[j], type: kg.typeOf(j) })),
        }))
        .sort((a, b) => b.count - a.count);
      return { ...kg.node(i), degree: kg.degree(i), byRelation, features: features[i] || null };
    },
    /** All neighbours of i, or an evenly spaced sample of each relation if there are more than `limit`. */
    neighbors(i, limit = 1500) {
      const lo = offsets[i];
      const degree = offsets[i + 1] - lo;
      const byRel = new Map();
      for (let p = lo; p < lo + degree; p++) {
        if (!byRel.has(rels[p])) byRel.set(rels[p], []);
        byRel.get(rels[p]).push(p);
      }
      const out = [];
      const outRels = [];
      for (const [r, list] of byRel) {
        const take = degree <= limit ? list.length : Math.max(1, Math.round((list.length / degree) * limit));
        for (let k = 0; k < take; k++) {
          out.push(nbrs[list[Math.floor((k * list.length) / take)]]);
          outRels.push(r);
        }
      }
      return { i, degree, nbrs: out, rels: outRels };
    },
  };
  console.log(`[primekg] loaded ${N.toLocaleString()} nodes · ${kg.E.toLocaleString()} edges in ${Date.now() - t}ms`);
  return kg;
}
