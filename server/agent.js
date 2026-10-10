import { GraphBuilder } from './graph.js';
import { askJSON, llmConfig } from './llm.js';
import { fetchTrial } from './tools/ctgov.js';
import { resolveMolecule, getMechanisms } from './tools/chembl.js';
import { buildLineage } from './tools/mesh.js';
import { findDisease, diseaseProfile, mapGenes } from './tools/opentargets.js';
import { mapLimit } from './tools/http.js';

const LOOKUP_TYPES = new Set(['small_molecule', 'biologic', 'vaccine', 'cell_therapy', 'gene_therapy', 'dietary', 'other']);

const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
const tokens = (s) => new Set(String(s || '').toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 2));

function similarity(a, b) {
  const A = tokens(a);
  const B = tokens(b);
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  const al = String(a).toLowerCase();
  const bl = String(b).toLowerCase();
  const substr = al.includes(bl) || bl.includes(al) ? 0.5 : 0;
  return inter / (A.size + B.size - inter) + substr;
}

function bestMatch(term, candidates, namesOf) {
  let best = null;
  let score = 0;
  for (const c of candidates) {
    const s = Math.max(...namesOf(c).map((n) => similarity(term, n)));
    if (s > score) [best, score] = [c, s];
  }
  if (!best && candidates.length === 1) return candidates[0];
  return best;
}

function armRole(trial, name) {
  const types = trial.armGroups
    .filter((a) => a.interventions.some((i) => i.toLowerCase().includes(name.toLowerCase())))
    .map((a) => a.type);
  if (types.includes('EXPERIMENTAL')) return 'experimental';
  if (types.includes('ACTIVE_COMPARATOR')) return 'active_comparator';
  if (types.includes('PLACEBO_COMPARATOR') || types.includes('SHAM_COMPARATOR')) return 'placebo';
  return 'experimental';
}

// ---------- LLM prompts ----------

const EXTRACT_SYSTEM = `You are a biomedical knowledge-graph curation agent.
Given a ClinicalTrials.gov record, extract and normalize the interventions and diseases so they can be resolved against ChEMBL, MeSH and Open Targets.
Rules:
- One entry per distinct intervention. Use the INN / generic name as "canonical" (no doses, routes, brand names, or formulation words).
- type is one of: small_molecule, biologic, vaccine, cell_therapy, gene_therapy, device, procedure, behavioral, dietary, radiation, placebo, other.
- role is one of: experimental, active_comparator, placebo, background.
- For fixed-dose combinations or code names (e.g. LCZ696), set canonical to the common name and list each active ingredient's INN in "components".
- putative_targets are HGNC gene symbols of the primary molecular targets (empty if not applicable). Only list targets you are confident about.
- diseases: normalize to standard disease names (as used by MONDO/EFO). Mark the main studied indication with is_primary=true.
- relations connect intervention canonical names to disease canonical names.
Return JSON:
{"interventions":[{"name":"","canonical":"","components":[],"type":"","role":"","drug_class":"","mechanism":"","putative_targets":[]}],
 "diseases":[{"name":"","canonical":"","category":"","is_primary":true}],
 "relations":[{"intervention":"","disease":"","relation":"treats|prevents|diagnoses|manages_symptom_of|studied_in","rationale":""}]}`;

function extractPrompt(t) {
  return JSON.stringify(
    {
      nct: t.nct,
      title: t.briefTitle,
      officialTitle: t.officialTitle,
      phases: t.phases,
      summary: t.summary.slice(0, 2000),
      conditions: t.conditions,
      keywords: t.keywords,
      interventions: t.interventions,
      armGroups: t.armGroups,
      primaryOutcomes: t.primaryOutcomes.slice(0, 5),
      conditionMesh: t.conditionMesh.map((m) => m.term),
      interventionMesh: t.interventionMesh.map((m) => m.term),
    },
    null,
    1,
  );
}

function mockExtract(t) {
  const typeMap = { DRUG: 'small_molecule', BIOLOGICAL: 'biologic', DEVICE: 'device', PROCEDURE: 'procedure', BEHAVIORAL: 'behavioral', DIETARY_SUPPLEMENT: 'dietary', RADIATION: 'radiation', GENETIC: 'gene_therapy' };
  const interventions = t.interventions.map((i) => {
    const placebo = /placebo|sham/i.test(i.name);
    const canonical = i.name
      .replace(/\(.*?\)/g, '')
      .replace(/\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|iu|mg\/kg)\b.*$/i, '')
      .replace(/\b(injection|tablets?|capsules?|infusion|oral|solution|iv)\b/gi, '')
      .trim();
    return { name: i.name, canonical: canonical || i.name, type: placebo ? 'placebo' : typeMap[i.type] || 'other', role: placebo ? 'placebo' : armRole(t, i.name), drug_class: null, mechanism: null, putative_targets: [] };
  });
  const diseases = t.conditions.map((c, k) => ({ name: c, canonical: c, category: null, is_primary: k === 0 }));
  const relations = interventions
    .filter((i) => i.role === 'experimental')
    .map((i) => ({ intervention: i.canonical, disease: diseases[0]?.canonical, relation: 'studied_in', rationale: 'Mock mode: derived from trial arm design.' }));
  return { interventions, diseases, relations };
}

const SYNTH_SYSTEM = `You are a translational-medicine reasoning agent. You receive a knowledge graph assembled from ClinicalTrials.gov, ChEMBL, MeSH and Open Targets around one clinical trial.
Explain how the interventions structurally connect to the disease biology. Only use node ids that appear in the graph for paths.
Return JSON:
{"headline":"<=12 words",
 "summary":"2-3 sentences on what the trial tests",
 "mechanistic_hypothesis":"3-5 sentences tracing drug -> target -> pathway/biology -> disease",
 "key_paths":[{"title":"","nodes":["node-id", "..."],"explanation":""}],
 "novel_links":[{"source":"node-id","target":"node-id","relation":"","rationale":""}],
 "gaps":["missing or uncertain links worth investigating"],
 "confidence":"low|medium|high"}
Give 2-4 key_paths (each 3-6 nodes, following edges where possible) and 0-3 novel_links (plausible hypotheses NOT already edges).`;

function synthPrompt(trial, graph) {
  const { nodes, links } = graph.snapshot();
  const keep = nodes.filter((n) => n.kind !== 'mesh' || n.data.level >= 3);
  const ids = new Set(keep.map((n) => n.id));
  const lines = [
    `Trial ${trial.nct}: ${trial.briefTitle} (${trial.phases.join('/') || 'n/a'}, ${trial.status})`,
    '',
    'NODES (id | kind | label | notes):',
    ...keep.slice(0, 160).map((n) => `${n.id} | ${n.kind} | ${n.label}${n.data.mechanism ? ` | ${n.data.mechanism}` : ''}${n.data.role ? ` | ${n.data.role}` : ''}`),
    '',
    'EDGES:',
    ...links
      .filter((l) => ids.has(l.source) && ids.has(l.target))
      .slice(0, 220)
      .map((l) => `${l.source} -${l.type}${l.score != null ? `(${l.score.toFixed(2)})` : ''}${l.label ? `[${l.label}]` : ''}-> ${l.target}`),
  ];
  return lines.join('\n');
}

function mockSynth(trial, graph) {
  const { nodes, links } = graph.snapshot();
  const ints = nodes.filter((n) => n.kind === 'intervention' && n.data.role !== 'placebo');
  const dis = nodes.find((n) => n.kind === 'disease' && n.data.isPrimary) || nodes.find((n) => n.kind === 'disease');
  const key_paths = [];
  for (const i of ints.slice(0, 3)) {
    const t = links.find((l) => l.source === i.id && l.type === 'TARGETS');
    const bridge = t && links.find((l) => l.target === t.target && l.source === dis?.id);
    key_paths.push({
      title: `${i.label} → ${dis?.label}`,
      nodes: bridge ? [i.id, t.target, dis.id] : [`trial:${trial.nct}`, i.id, dis?.id].filter(Boolean),
      explanation: bridge ? `${i.label} acts on ${graph.nodes.get(t.target).label}, which Open Targets associates with ${dis.label}.` : `${i.label} is tested in this trial for ${dis?.label}.`,
    });
  }
  return {
    headline: trial.briefTitle.slice(0, 80),
    summary: `Mock synthesis (no LLM). ${trial.nct} is a ${trial.phases.join('/') || ''} ${trial.studyType?.toLowerCase() || ''} study sponsored by ${trial.sponsor}.`,
    mechanistic_hypothesis: 'Set OPENAI_API_KEY or ANTHROPIC_API_KEY to get an LLM-generated mechanistic hypothesis.',
    key_paths,
    novel_links: [],
    gaps: ['Running in mock mode — LLM reasoning disabled.'],
    confidence: 'low',
  };
}

// ---------- Agent ----------

export async function runAgent(nct, emit) {
  const llm = llmConfig();
  const graph = new GraphBuilder(emit);
  const step = (id, title, status, detail) => emit({ type: 'step', id, title, status, detail });
  const log = (id, text) => emit({ type: 'log', step: id, text });

  emit({ type: 'meta', llm: { provider: llm.provider, model: llm.model } });
  if (!llm.ready) throw new Error('No LLM configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env and restart the server.');

  // 1. Retrieve
  step('retrieve', 'Retrieve trial record', 'running', 'ClinicalTrials.gov API v2');
  const trial = await fetchTrial(nct);
  const trialId = `trial:${nct}`;
  graph.node(trialId, {
    kind: 'trial',
    label: trial.acronym ? `${nct} · ${trial.acronym}` : nct,
    source: 'ClinicalTrials.gov',
    data: {
      title: trial.briefTitle,
      status: trial.status,
      phases: trial.phases,
      sponsor: trial.sponsor,
      enrollment: trial.enrollment,
      startDate: trial.startDate,
      studyType: trial.studyType,
      summary: trial.summary.slice(0, 600),
      primaryOutcomes: trial.primaryOutcomes.slice(0, 3),
      url: trial.url,
    },
  });
  graph.flush();
  log('retrieve', `"${trial.briefTitle}"`);
  log('retrieve', `${trial.interventions.length} interventions · ${trial.conditions.length} conditions · ${trial.status}`);
  step('retrieve', 'Retrieve trial record', 'done');

  // 2. Extract
  step('extract', 'Extract & normalize entities', 'running', `LLM · ${llm.provider}/${llm.model}`);
  const ex = await askJSON({ system: EXTRACT_SYSTEM, user: extractPrompt(trial), mock: () => mockExtract(trial) });
  const interventions = (ex.interventions || []).filter((i) => i?.canonical || i?.name).slice(0, 12);
  const diseases = (ex.diseases || []).filter((d) => d?.canonical || d?.name).slice(0, 6);
  if (!interventions.length && !diseases.length) throw new Error('The LLM did not extract any interventions or diseases.');

  for (const i of interventions) {
    i.canonical ||= i.name;
    i.id = `int:${slug(i.canonical)}`;
    const ct = trial.interventions.find((x) => similarity(x.name, i.name || i.canonical) > 0.3);
    graph.node(i.id, {
      kind: 'intervention',
      label: i.canonical,
      source: 'ClinicalTrials.gov + LLM',
      data: { originalName: i.name, type: i.type, role: i.role, drugClass: i.drug_class, mechanism: i.mechanism, otherNames: ct?.otherNames || [], description: ct?.description },
    });
    graph.link(trialId, i.id, 'TESTS', { label: i.role });
    log('extract', `intervention → ${i.canonical} [${i.type}, ${i.role}]`);
  }
  for (const d of diseases) {
    d.canonical ||= d.name;
    d.id = `dis:${slug(d.canonical)}`;
    graph.node(d.id, { kind: 'disease', label: d.canonical, source: 'ClinicalTrials.gov + LLM', data: { originalName: d.name, category: d.category, isPrimary: !!d.is_primary } });
    graph.link(trialId, d.id, 'STUDIES', { label: d.is_primary ? 'primary' : 'secondary' });
    log('extract', `disease → ${d.canonical}${d.is_primary ? ' (primary)' : ''}`);
  }
  const findInt = (name) => interventions.find((i) => i.canonical.toLowerCase() === String(name).toLowerCase()) || bestMatch(name, interventions, (i) => [i.canonical, i.name]);
  const findDis = (name) => diseases.find((d) => d.canonical.toLowerCase() === String(name).toLowerCase()) || bestMatch(name, diseases, (d) => [d.canonical, d.name]);
  for (const r of ex.relations || []) {
    const i = findInt(r.intervention);
    const d = findDis(r.disease);
    if (i && d) graph.link(i.id, d.id, r.relation?.toUpperCase() || 'TREATS', { source: 'LLM', inferred: true, rationale: r.rationale });
  }
  graph.flush();
  step('extract', 'Extract & normalize entities', 'done', `${interventions.length} interventions · ${diseases.length} diseases`);

  // 3. ChEMBL
  step('chembl', 'Resolve molecules & mechanisms', 'running', 'ChEMBL');
  const drugTargets = new Set();
  const lookups = interventions.filter((i) => LOOKUP_TYPES.has(i.type) && i.role !== 'placebo');
  const resolve = async (names) => {
    for (const n of names) {
      const mol = await resolveMolecule(n.replace(/[®™]/g, '')).catch(() => null);
      if (mol) return { mol, mechs: await getMechanisms(mol.parentId).catch(() => []) };
    }
    return { mol: null, mechs: [] };
  };
  await mapLimit(lookups, 3, async (i) => {
    const names = [...new Set([i.canonical, i.name, ...(graph.nodes.get(i.id).data.otherNames || []).slice(0, 2)])].filter(Boolean);
    const { mol, mechs } = await resolve(names);
    if (mol) {
      graph.node(i.id, { kind: 'intervention', data: { chembl: mol } });
      log('chembl', `${i.canonical} → ${mol.chemblId} (${mol.moleculeType || '?'}, max phase ${mol.maxPhase ?? '?'})`);
    } else {
      log('chembl', `${i.canonical} → no ChEMBL match`);
    }
    if (Array.isArray(i.components) && i.components.length > 1) {
      for (const comp of i.components.slice(0, 4)) {
        const r = await resolve([comp]);
        log('chembl', `  component ${comp} → ${r.mol ? r.mol.chemblId : 'no match'}`);
        mechs.push(...r.mechs);
      }
    }
    const seen = new Set();
    for (const m of mechs) {
      const t = m.target;
      if (!t || seen.has(t.chemblId)) continue;
      seen.add(t.chemblId);
      if (t.genes.length === 1) {
        const g = t.genes[0];
        const gid = `gene:${g.symbol}`;
        graph.node(gid, { kind: 'target', label: g.symbol, source: 'ChEMBL', data: { name: g.name, uniprot: g.accession, chemblTarget: t.chemblId, targetName: t.name, organism: t.organism, drugTarget: true } });
        graph.link(i.id, gid, 'TARGETS', { label: m.actionType?.toLowerCase(), mechanism: m.mechanism, source: 'ChEMBL' });
        drugTargets.add(g.symbol);
        log('chembl', `  ${m.mechanism} → ${g.symbol}`);
      } else {
        // Protein families, complexes and non-protein targets (e.g. DNA) stay as one node.
        const tid = `tgt:${t.chemblId}`;
        graph.node(tid, { kind: 'target', label: t.name, source: 'ChEMBL', data: { chemblTarget: t.chemblId, targetType: t.type, organism: t.organism, genes: t.genes.map((g) => g.symbol), drugTarget: true } });
        graph.link(i.id, tid, 'TARGETS', { label: m.actionType?.toLowerCase(), mechanism: m.mechanism, source: 'ChEMBL' });
        log('chembl', `  ${m.mechanism} → ${t.name}${t.genes.length ? ` (${t.genes.length} genes)` : ''}`);
      }
    }
    if (!mechs.some((m) => m.target)) {
      for (const sym of (i.putative_targets || []).slice(0, 4)) {
        const s = String(sym).toUpperCase().trim();
        if (!/^[A-Z0-9-]{2,15}$/.test(s)) continue;
        const gid = `gene:${s}`;
        graph.node(gid, { kind: 'target', label: s, source: 'LLM', data: { drugTarget: true, inferred: true } });
        graph.link(i.id, gid, 'TARGETS', { label: 'putative', mechanism: i.mechanism, source: 'LLM', inferred: true });
        drugTargets.add(s);
        log('chembl', `  ${i.canonical} ⇢ ${s} (LLM-inferred)`);
      }
    }
    graph.flush();
  });
  const targetCount = [...graph.nodes.values()].filter((n) => n.kind === 'target').length;
  step('chembl', 'Resolve molecules & mechanisms', 'done', `${targetCount} molecular targets`);

  // 4. MeSH
  step('mesh', 'Map into MeSH ontology', 'running', 'NLM MeSH tree numbers');
  const meshSides = [
    { side: 'disease', leaves: trial.conditionMesh, all: [...trial.conditionMesh, ...trial.conditionMeshAncestors], entities: diseases, names: (d) => [d.canonical, d.name] },
    { side: 'chem', leaves: trial.interventionMesh, all: [...trial.interventionMesh, ...trial.interventionMeshAncestors], entities: interventions, names: (i) => [i.canonical, i.name, ...(graph.nodes.get(i.id).data.otherNames || [])] },
  ];
  let meshCount = 0;
  for (const s of meshSides) {
    if (!s.all.length) continue;
    const lin = await buildLineage(s.all);
    for (const r of lin.roots) graph.node(`mesh:${r.id}`, { kind: 'mesh', label: `${r.term} [${r.letter}]`, source: 'MeSH', data: { side: s.side, level: 0, root: true } });
    for (const n of lin.nodes) {
      graph.node(`mesh:${n.id}`, { kind: 'mesh', label: n.term, source: 'MeSH', data: { side: s.side, level: n.level, meshId: n.id, treeNumbers: n.treeNumbers, supplementary: n.supplementary } });
    }
    for (const e of lin.edges) graph.link(`mesh:${e.child}`, `mesh:${e.parent}`, 'BROADER', { source: 'MeSH' });
    for (const leaf of s.leaves) {
      const ent = bestMatch(leaf.term, s.entities, s.names);
      if (ent) {
        graph.link(ent.id, `mesh:${leaf.id}`, 'MAPS_TO', { source: 'MeSH' });
        log('mesh', `${ent.canonical} ≡ MeSH ${leaf.id} "${leaf.term}"`);
      }
    }
    meshCount += lin.nodes.length + lin.roots.length;
    graph.flush();
  }
  step('mesh', 'Map into MeSH ontology', 'done', `${meshCount} ontology concepts`);

  // 5. Open Targets
  step('ot', 'Link disease genetics', 'running', 'Open Targets Platform');
  const geneIds = await mapGenes([...drugTargets]).catch(() => ({}));
  for (const [sym, ens] of Object.entries(geneIds)) graph.node(`gene:${sym}`, { kind: 'target', data: { ensembl: ens } });
  const ensToSym = Object.fromEntries(Object.entries(geneIds).map(([s, e]) => [e, s]));
  let bridges = 0;
  for (const d of diseases.slice(0, 4)) {
    const queries = [...new Set([d.canonical, d.name].filter(Boolean))];
    const hits = (await Promise.all(queries.map((q) => findDisease(q).then((h) => h && { ...h, fit: similarity(q, h.name) }).catch(() => null)))).filter(Boolean);
    const hit = hits.sort((a, b) => b.fit - a.fit)[0];
    if (!hit) {
      log('ot', `${d.canonical} → no Open Targets match`);
      continue;
    }
    const prof = await diseaseProfile(hit.id, Object.values(geneIds)).catch(() => null);
    if (!prof) continue;
    graph.node(d.id, { kind: 'disease', data: { efo: prof.id, otName: prof.name, description: prof.description } });
    log('ot', `${d.canonical} ≡ ${prof.id} "${prof.name}"`);
    for (const a of prof.areas.filter((a) => a.id !== 'EFO_0000651')) {
      graph.node(`area:${a.id}`, { kind: 'area', label: a.name, source: 'Open Targets', data: { id: a.id } });
      graph.link(d.id, `area:${a.id}`, 'IN_AREA', { source: 'Open Targets' });
    }
    for (const g of prof.top.slice(0, 6)) {
      const gid = `gene:${g.symbol}`;
      graph.node(gid, { kind: 'target', label: g.symbol, source: graph.has(gid) ? graph.nodes.get(gid).source : 'Open Targets', data: { name: g.name, ensembl: g.ensembl, diseaseGene: true } });
      graph.link(d.id, gid, 'ASSOCIATED_WITH', { score: g.score, label: g.score.toFixed(2), source: 'Open Targets' });
    }
    for (const g of prof.focused) {
      const sym = ensToSym[g.ensembl] || g.symbol;
      const gid = `gene:${sym}`;
      if (!graph.has(gid)) continue;
      const bridge = g.score >= 0.1;
      graph.link(d.id, gid, 'ASSOCIATED_WITH', { score: g.score, label: g.score.toFixed(2), source: 'Open Targets', bridge });
      if (bridge) bridges++;
      log('ot', `  ${bridge ? 'bridge' : 'weak link'}: ${sym} ↔ ${d.canonical} (score ${g.score.toFixed(2)})`);
    }
    graph.flush();
  }
  step('ot', 'Link disease genetics', 'done', `${bridges} drug-target ↔ disease bridges`);

  // 6. Synthesize
  step('synth', 'Synthesize mechanistic map', 'running', `LLM · ${llm.provider}/${llm.model}`);
  const syn = await askJSON({ system: SYNTH_SYSTEM, user: synthPrompt(trial, graph), mock: () => mockSynth(trial, graph) });
  const key_paths = (syn.key_paths || [])
    .map((p) => ({ ...p, nodes: (p.nodes || []).filter((id) => graph.has(id)) }))
    .filter((p) => p.nodes.length >= 2);
  for (const nl of syn.novel_links || []) {
    if (graph.has(nl.source) && graph.has(nl.target)) {
      graph.link(nl.source, nl.target, 'HYPOTHESIS', { label: nl.relation, rationale: nl.rationale, source: 'LLM', inferred: true, hypothesis: true });
      log('synth', `hypothesis: ${graph.nodes.get(nl.source).label} ⇢ ${graph.nodes.get(nl.target).label}`);
    }
  }
  graph.flush();
  emit({ type: 'summary', summary: { ...syn, key_paths } });
  step('synth', 'Synthesize mechanistic map', 'done', `${key_paths.length} key paths · confidence ${syn.confidence || '?'}`);

  const snap = graph.snapshot();
  emit({ type: 'done', stats: { nodes: snap.nodes.length, links: snap.links.length } });
}
