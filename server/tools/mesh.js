import { getJSON, mapLimit } from './http.js';

export const MESH_CATEGORIES = {
  A: 'Anatomy',
  B: 'Organisms',
  C: 'Diseases',
  D: 'Chemicals and Drugs',
  E: 'Analytical, Diagnostic and Therapeutic Techniques',
  F: 'Psychiatry and Psychology',
  G: 'Phenomena and Processes',
  N: 'Health Care',
};

async function lookup(id) {
  try {
    const d = await getJSON(`https://id.nlm.nih.gov/mesh/${id}.json`, { timeout: 15000 });
    const strip = (u) => u.replace('http://id.nlm.nih.gov/mesh/', '');
    const asArr = (v) => (v == null ? [] : Array.isArray(v) ? v : [v]);
    return {
      id,
      treeNumbers: asArr(d.treeNumber).map(strip),
      mappedTo: asArr(d.preferredMappedTo).map(strip),
    };
  } catch {
    return { id, treeNumbers: [], mappedTo: [] };
  }
}

/**
 * Reconstructs the MeSH polyhierarchy for a flat set of terms using tree numbers.
 * Returns nodes (with level) and child->parent edges restricted to the given set,
 * plus category roots (e.g. "C" Diseases).
 */
export async function buildLineage(terms) {
  const termById = new Map(terms.map((t) => [t.id, t.term]));
  const infos = new Map();
  let queue = [...termById.keys()];

  // Supplementary concepts (C-prefixed IDs) map to descriptors; pull those in too.
  for (let round = 0; round < 2 && queue.length; round++) {
    const results = await mapLimit(queue, 8, lookup);
    queue = [];
    for (const r of results) {
      infos.set(r.id, r);
      for (const m of r.mappedTo) {
        if (!infos.has(m) && !queue.includes(m)) queue.push(m);
      }
    }
  }

  const missingLabels = [...infos.keys()].filter((id) => !termById.has(id));
  await mapLimit(missingLabels, 6, async (id) => {
    try {
      const d = await getJSON(`https://id.nlm.nih.gov/mesh/lookup/label?resource=${id}`);
      termById.set(id, Array.isArray(d) ? d[0] : d);
    } catch {
      termById.set(id, id);
    }
  });

  const treeOwner = new Map();
  for (const info of infos.values()) for (const tn of info.treeNumbers) treeOwner.set(tn, info.id);

  const nodes = [];
  const edges = [];
  const roots = new Set();
  for (const info of infos.values()) {
    const parents = new Set();
    let level = 99;
    for (const tn of info.treeNumbers) {
      const segs = tn.split('.');
      level = Math.min(level, segs.length);
      let found = false;
      for (let k = segs.length - 1; k >= 1; k--) {
        const owner = treeOwner.get(segs.slice(0, k).join('.'));
        if (owner && owner !== info.id) {
          parents.add(owner);
          found = true;
          break;
        }
      }
      if (!found) {
        const cat = tn[0];
        roots.add(cat);
        parents.add(`cat:${cat}`);
      }
    }
    for (const m of info.mappedTo) parents.add(m);
    nodes.push({
      id: info.id,
      term: termById.get(info.id) || info.id,
      treeNumbers: info.treeNumbers,
      level: level === 99 ? 1 : level,
      supplementary: info.treeNumbers.length === 0,
    });
    for (const p of parents) edges.push({ child: info.id, parent: p });
  }

  for (const n of nodes) if (n.supplementary) n.level = 1 + Math.max(0, ...edges.filter((e) => e.child === n.id).map((e) => nodes.find((x) => x.id === e.parent)?.level || 0));

  return {
    nodes,
    edges,
    roots: [...roots].map((c) => ({ id: `cat:${c}`, letter: c, term: MESH_CATEGORIES[c] || `MeSH ${c}` })),
  };
}
