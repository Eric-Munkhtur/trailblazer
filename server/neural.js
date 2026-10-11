// Bidirectional "neural" search over PrimeKG: activation spreads outward from the intervention
// seeds and the disease seeds; where the two wavefronts meet we reconstruct connecting pathways.

// Traversal cost per (relation|display). Relations not listed are not traversed (e.g. anatomy
// expression and drug-drug synergy edges, which connect almost everything to everything).
const COST = {
  'indication|indication': 0.6,
  'off-label use|off-label use': 0.8,
  'contraindication|contraindication': 1.0,
  'drug_protein|target': 0.8,
  'drug_protein|enzyme': 1.6,
  'drug_protein|transporter': 1.6,
  'drug_protein|carrier': 1.7,
  'disease_protein|associated with': 1.0,
  'protein_protein|ppi': 1.35,
  'pathway_protein|interacts with': 1.3,
  'bioprocess_protein|interacts with': 1.6,
  'molfunc_protein|interacts with': 1.7,
  'drug_effect|side effect': 1.5,
  'disease_phenotype_positive|phenotype present': 1.3,
  'phenotype_protein|associated with': 1.2,
  'disease_disease|parent-child': 1.0,
  'exposure_disease|linked to': 1.4,
  'exposure_protein|interacts with': 1.4,
};
// Extra cost for routing through another drug or disease, so mechanistic routes (targets,
// interactions, pathways) compete with "therapeutic analogy" routes via other drugs.
const TYPE_PENALTY = { drug: 0.9, disease: 0.3 };
const KIND_CAP = { 'known indication': 6, 'off-label use': 6, contraindication: 4, 'direct link': 4, 'therapeutic analogy': 3 };
const DEFAULT_KIND_CAP = 4;
const HUB_EXPAND = 3000;
// Replay timing: a signal takes WAVE_SECONDS per unit of path cost to reach a node, so the
// cheapest (strongest) connections light up first, in the order the search ranked them.
const WAVE_SECONDS = 0.9;
const DRAW_CAP = [0, 600, 1400];
const LIT_CAP = 30000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let cache = null;
function prepare(kg) {
  if (cache?.kg === kg) return cache;
  const relCost = Float32Array.from(kg.relations, (r) => COST[`${r.relation}|${r.display}`] || 0);
  const searchDeg = new Uint32Array(kg.N);
  for (let i = 0; i < kg.N; i++) {
    let d = 0;
    for (let p = kg.offsets[i]; p < kg.offsets[i + 1]; p++) if (relCost[kg.rels[p]]) d++;
    searchDeg[i] = d;
  }
  const hub = Float32Array.from(searchDeg, (d, i) => 0.32 * Math.log10(1 + d) + (TYPE_PENALTY[kg.typeOf(i)] || 0));
  cache = { kg, relCost, searchDeg, hub };
  return cache;
}

function expand(kg, P, seeds, hops) {
  const cost = new Map();
  const parent = new Map();
  const hop = new Map();
  for (const s of seeds) {
    cost.set(s, 0);
    hop.set(s, 0);
  }
  const layers = [];
  let frontier = [...new Set(seeds)];
  for (let h = 1; h <= hops; h++) {
    const best = new Map();
    for (const u of frontier) {
      if (h > 1 && P.searchDeg[u] > HUB_EXPAND) continue;
      const cu = cost.get(u);
      for (let p = kg.offsets[u]; p < kg.offsets[u + 1]; p++) {
        const rc = P.relCost[kg.rels[p]];
        if (!rc) continue;
        const v = kg.nbrs[p];
        if (cost.has(v)) continue;
        const c = cu + rc + P.hub[v];
        const b = best.get(v);
        if (!b || c < b[0]) best.set(v, [c, u, kg.rels[p]]);
      }
    }
    const layer = [];
    for (const [v, [c, u, r]] of best) {
      cost.set(v, c);
      parent.set(v, [u, r]);
      hop.set(v, h);
      layer.push(v);
    }
    layer.sort((a, b) => cost.get(a) - cost.get(b));
    layers.push(layer);
    frontier = layer;
  }
  return { cost, parent, hop, layers };
}

function trace(side, m) {
  const nodes = [m];
  const rels = [];
  let x = m;
  while (side.parent.has(x)) {
    const [u, r] = side.parent.get(x);
    rels.push(r);
    nodes.push(u);
    x = u;
  }
  return { nodes, rels };
}

function classify(rels, interiorTypes) {
  if (rels.length === 1) {
    const r = rels[0];
    return r === 'indication' ? 'known indication' : r === 'off-label use' ? 'off-label use' : r === 'contraindication' ? 'contraindication' : 'direct link';
  }
  if (rels.length === 2 && rels[0] === 'drug_protein' && rels[1] === 'disease_protein') return 'shared target';
  if (interiorTypes.includes('drug')) return 'therapeutic analogy';
  if (interiorTypes.includes('disease') && rels.filter((r) => r === 'disease_protein').length >= 2) return 'shared disease genetics';
  if (rels.includes('protein_protein')) return 'protein interaction bridge';
  if (rels.includes('drug_effect') && rels.includes('disease_phenotype_positive')) return 'phenotype overlap';
  if (rels.includes('pathway_protein')) return 'pathway bridge';
  if (rels.includes('disease_disease')) return 'disease hierarchy';
  if (rels.includes('bioprocess_protein') || rels.includes('molfunc_protein')) return 'functional bridge';
  return 'multi-hop';
}

export function neuralSearch(kg, seedsA, seedsB, { hops = 2, maxPaths = 24, sides = ['intervention', 'disease'] } = {}) {
  const t0 = performance.now();
  const P = prepare(kg);
  const A = expand(kg, P, seedsA, seedsA.length ? hops : 0);
  const B = expand(kg, P, seedsB, seedsB.length ? hops : 0);

  const meets = [];
  if (seedsA.length && seedsB.length) {
    for (const [m, ca] of A.cost) {
      const cb = B.cost.get(m);
      if (cb === undefined) continue;
      const interior = A.hop.get(m) > 0 && B.hop.get(m) > 0;
      if (!interior && A.hop.get(m) === 0 && B.hop.get(m) === 0) continue;
      meets.push([m, ca + cb - (interior ? P.hub[m] : 0)]);
    }
    meets.sort((a, b) => a[1] - b[1]);
  }

  const paths = [];
  const seen = new Set();
  const usage = new Map();
  const pairUse = new Map();
  const kindUse = new Map();
  for (const [m, total] of meets) {
    if (paths.length >= maxPaths) break;
    const a = trace(A, m);
    const b = trace(B, m);
    const nodes = [...a.nodes.slice().reverse(), ...b.nodes.slice(1)];
    const rels = [...a.rels.slice().reverse(), ...b.rels];
    const key = nodes.join('>');
    if (seen.has(key) || new Set(nodes).size !== nodes.length) continue;
    const interior = nodes.slice(1, -1);
    if (interior.some((n) => (usage.get(n) || 0) >= 2)) continue;
    const pair = `${nodes[0]}>${nodes[nodes.length - 1]}`;
    if ((pairUse.get(pair) || 0) >= 10) continue;
    const kind = classify(
      rels.map((r) => kg.relations[r].relation),
      interior.map((n) => kg.typeOf(n)),
    );
    if ((kindUse.get(kind) || 0) >= (KIND_CAP[kind] ?? DEFAULT_KIND_CAP)) continue;
    seen.add(key);
    interior.forEach((n) => usage.set(n, (usage.get(n) || 0) + 1));
    pairUse.set(pair, (pairUse.get(pair) || 0) + 1);
    kindUse.set(kind, (kindUse.get(kind) || 0) + 1);
    paths.push({
      id: `P${paths.length + 1}`,
      nodes,
      rels: rels.map((r) => kg.relations[r].display),
      cost: +total.toFixed(2),
      strength: +Math.exp(-total / 4).toFixed(3),
      kind,
    });
  }

  const ms = performance.now() - t0;

  // Replay frames, one per hop and wavefront. Every reached node is lit at its arrival time;
  // the cheapest edges of each hop (plus all edges on a reported pathway) are drawn as signals.
  const onPath = new Set();
  for (const p of paths) for (const n of p.nodes) onPath.add(n);
  const at = (side, v) => +(side.cost.get(v) * WAVE_SECONDS).toFixed(2);
  let duration = 0;
  const framesFor = (side, label) =>
    side.layers.map((layer, h) => {
      const keep = layer.filter((v) => onPath.has(v));
      const rest = layer.filter((v) => !onPath.has(v)).slice(0, DRAW_CAP[h + 1] || 600);
      const lit = layer.slice(0, LIT_CAP);
      const edges = [...keep, ...rest].map((v) => {
        const [u, r] = side.parent.get(v);
        duration = Math.max(duration, at(side, v));
        return [u, v, r, at(side, u), at(side, v)];
      });
      return { side: label, hop: h + 1, reached: layer.length, edges, nodes: lit, times: lit.map((v) => at(side, v)) };
    });
  const fa = framesFor(A, sides[0]);
  const fb = framesFor(B, sides[1]);
  const frames = [];
  for (let h = 0; h < Math.max(fa.length, fb.length); h++) {
    if (fa[h]) frames.push(fa[h]);
    if (fb[h]) frames.push(fb[h]);
  }
  const shownMeets = meets.slice(0, 60).map(([m]) => m);

  return {
    frames,
    meets: shownMeets,
    meetTimes: shownMeets.map((m) => Math.max(at(A, m), at(B, m))),
    meetCount: meets.length,
    paths,
    reached: { [sides[0]]: A.cost.size, [sides[1]]: B.cost.size },
    ms,
    duration,
  };
}

/** Runs a search and streams it as `signal`, `meet` and `paths` events, replayed in cost time. */
export async function streamSearch(kg, seedsA, seedsB, emit, log, opts) {
  const s = neuralSearch(kg, seedsA, seedsB, opts);
  const explored = Object.values(s.reached).reduce((a, b) => a + b, 0);
  log(`searched ${explored.toLocaleString()} nodes in ${s.ms < 10 ? s.ms.toFixed(1) : Math.round(s.ms)} ms · replayed at ${WAVE_SECONDS}s per unit of path cost`);
  for (const f of s.frames) {
    emit({ type: 'signal', ...f });
    log(`${f.side} wave ${f.hop}: ${f.reached.toLocaleString()} nodes reached`);
  }
  if (s.meetCount) {
    emit({ type: 'meet', nodes: s.meets, times: s.meetTimes });
    log(`wavefronts met at ${s.meetCount.toLocaleString()} nodes`);
  }
  await sleep(s.duration * 1000 + 400);
  emit({ type: 'paths', paths: s.paths });
  return s;
}
