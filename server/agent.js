import { askJSON, llmConfig } from './llm.js';
import { fetchTrial } from './tools/ctgov.js';
import { primekg } from './primekg.js';
import { streamSearch } from './neural.js';

const LOOKUP_TYPES = new Set(['small_molecule', 'biologic', 'vaccine', 'cell_therapy', 'gene_therapy', 'dietary', 'other']);
function armRole(trial, name) {
  const types = trial.armGroups
    .filter((a) => a.interventions.some((i) => i.toLowerCase().includes(name.toLowerCase())))
    .map((a) => a.type);
  if (types.includes('EXPERIMENTAL')) return 'experimental';
  if (types.includes('ACTIVE_COMPARATOR')) return 'active_comparator';
  if (types.includes('PLACEBO_COMPARATOR') || types.includes('SHAM_COMPARATOR')) return 'placebo';
  return 'experimental';
}

// ---------- 1. extraction ----------

const EXTRACT_SYSTEM = `You are a biomedical knowledge-graph curation agent.
Given a ClinicalTrials.gov record, extract and normalize the interventions and diseases so they can be grounded in PrimeKG (DrugBank drugs, MONDO diseases, NCBI genes).
Rules:
- One entry per distinct intervention. Use the INN / generic name as "canonical" (no doses, routes, brand names, or formulation words).
- For fixed-dose combinations or code names (e.g. LCZ696), set canonical to the common name and list each active ingredient's INN in "components".
- type is one of: small_molecule, biologic, vaccine, cell_therapy, gene_therapy, device, procedure, behavioral, dietary, radiation, placebo, other.
- role is one of: experimental, active_comparator, placebo, background.
- putative_targets are HGNC gene symbols of the primary molecular targets (empty if not applicable). Only list targets you are confident about.
- diseases: normalize to standard disease names (as used by MONDO). Mark the main studied indication with is_primary=true.
Return JSON:
{"interventions":[{"name":"","canonical":"","components":[],"type":"","role":"","drug_class":"","mechanism":"","putative_targets":[]}],
 "diseases":[{"name":"","canonical":"","is_primary":true}]}`;

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
      conditionMesh: t.conditionMesh.map((m) => m.term),
      interventionMesh: t.interventionMesh.map((m) => m.term),
    },
    null,
    1,
  );
}

function mockExtract(t) {
  const typeMap = { DRUG: 'small_molecule', BIOLOGICAL: 'biologic', DEVICE: 'device', PROCEDURE: 'procedure', BEHAVIORAL: 'behavioral', DIETARY_SUPPLEMENT: 'dietary', RADIATION: 'radiation', GENETIC: 'gene_therapy' };
  // Code-named products (e.g. LCZ696) take their ingredients from CT.gov's MeSH "drug combination" terms.
  const combos = t.interventionMesh
    .map((m) => m.term)
    .filter((s) => /drug combination/i.test(s))
    .map((s) =>
      s
        .replace(/drug combination/i, '')
        .replace(/\b(sodium|potassium|calcium|hydrate|hydrochloride|anhydrous)\b/gi, '')
        .split(/\s*(?:,|\band\b)\s*/)
        .map((x) => x.trim())
        .filter(Boolean),
    );
  const interventions = t.interventions.map((i) => {
    const placebo = /placebo|sham/i.test(i.name);
    const canonical = i.name
      .replace(/\(.*?\)/g, '')
      .replace(/\b\d+(\.\d+)?\s?(mg|mcg|µg|g|ml|iu|mg\/kg)\b.*$/i, '')
      .replace(/\b(injection|tablets?|capsules?|infusion|oral|solution|iv)\b/gi, '')
      .trim();
    const components = /^[A-Z]{2,5}[- ]?\d{2,6}[A-Z]?$/.test(canonical) && combos.length ? combos.shift() : [];
    return { name: i.name, canonical: canonical || i.name, components, type: placebo ? 'placebo' : typeMap[i.type] || 'other', role: placebo ? 'placebo' : armRole(t, i.name), putative_targets: [] };
  });
  const diseases = t.conditions.map((c, k) => ({ name: c, canonical: c, is_primary: k === 0 }));
  return { interventions, diseases };
}

// ---------- 2. grounding ----------

const GROUND_SYSTEM = `You ground clinical-trial entities to nodes of PrimeKG, a precision-medicine knowledge graph.
For each entity pick the candidate node that denotes the same concept:
- drugs: the same active ingredient (ignore salts/formulations); never pick a different drug.
- diseases: the same disease; if no exact match, the closest more general disease that still contains it.
Use null when no candidate fits.
Return JSON {"matches":[{"key":"<entity key>","node":<candidate i or null>,"reason":"<=12 words"}]}`;

function mockGround(entities) {
  const matches = [];
  for (const e of entities) {
    const top = e.candidates[0];
    if (!(top?.score >= 0.45)) continue;
    matches.push({ key: e.key, node: top.i, reason: 'top lexical match' });
    // A phenotype-only grounding misses indication edges, so also seed the closest disease concept.
    const dz = top.type !== 'disease' && e.side === 'disease' && e.candidates.find((c) => c.type === 'disease' && c.score >= 0.45);
    if (dz) matches.push({ key: e.key, node: dz.i, reason: 'closest disease concept' });
  }
  return { matches };
}

function candidatesFor(kg, names, types, limit = 8) {
  const merged = new Map();
  for (const n of names.filter(Boolean)) {
    for (const c of kg.search(n, { types, limit })) {
      const prev = merged.get(c.i);
      if (!prev || c.score > prev.score) merged.set(c.i, c);
    }
  }
  return [...merged.values()].sort((a, b) => b.score - a.score || b.degree - a.degree).slice(0, limit);
}

// ---------- 4. synthesis ----------

const SYNTH_SYSTEM = `You are a translational-medicine reasoning agent working over PrimeKG.
You receive a clinical trial, its interventions and diseases grounded in PrimeKG, and pathways ranked by a manually weighted bidirectional graph search (lower cost ranks earlier; it is not a probability or validated evidence strength).
Explain how the interventions connect to the disease biology. Prefer mechanistic pathways (targets, protein interactions, pathways, phenotypes) over therapeutic analogies via other drugs. Only cite pathway ids that are given.
Return JSON:
{"headline":"<=12 words",
 "summary":"2-3 sentences on what the trial tests",
 "mechanistic_hypothesis":"3-5 sentences tracing drug -> target -> biology -> disease",
 "key_paths":[{"path":"P1","title":"","explanation":""}],
 "novel_hypotheses":["testable hypotheses suggested by the graph"],
 "gaps":["missing or uncertain links"],
 "confidence":"low|medium|high"}
Give 2-5 key_paths.`;

function describePath(kg, p) {
  return p.nodes.map((n, k) => `${k ? ` -[${p.rels[k - 1]}]-> ` : ''}${kg.nodes.names[n]} (${kg.typeOf(n)})`).join('');
}

function synthPrompt(kg, trial, seeds, search) {
  const feat = (i) => kg.features[i] || {};
  const lines = [
    `Trial ${trial.nct}: ${trial.briefTitle}`,
    `Phase: ${trial.phases.join('/') || 'n/a'} · sponsor ${trial.sponsor}`,
    `Summary: ${trial.summary.slice(0, 800)}`,
    '',
    'Grounded seeds:',
    ...seeds.map((s) => {
      const f = feat(s.i);
      const extra = f.mechanism || f.definition || '';
      return `- [${s.side}] ${s.name} (${s.type}) for "${s.entity}"${extra ? ` — ${extra.slice(0, 350)}` : ''}`;
    }),
    '',
    `Graph search reached ${search.reached.intervention} nodes from the intervention side and ${search.reached.disease} from the disease side; ${search.meetCount} meeting points.`,
    'Pathways:',
    ...search.paths.map((p) => `${p.id} (${p.kind}, cost ${p.cost}): ${describePath(kg, p)}`),
  ];
  return lines.join('\n');
}

const MOCK_KINDS = ['target–disease association', 'pathway bridge', 'protein interaction bridge', 'known indication', 'phenotype overlap', 'shared disease genetics', 'functional bridge', 'off-label use'];

function mockSynth(kg, trial, search) {
  const picks = MOCK_KINDS.map((k) => search.paths.find((p) => p.kind === k)).filter(Boolean).slice(0, 4);
  if (!picks.length) picks.push(...search.paths.slice(0, 4));
  return {
    headline: trial.briefTitle.length > 90 ? `${trial.briefTitle.slice(0, 90).replace(/\s+\S*$/, '')}…` : trial.briefTitle,
    summary: `Structured-data mode (no LLM). ${trial.nct} is a ${trial.phases.join('/') || ''} study sponsored by ${trial.sponsor}. Graph search found ${search.paths.length} pathways across ${search.meetCount} meeting points in PrimeKG.`,
    mechanistic_hypothesis: 'Inspect the individual relationships below. Graph connections provide biological context; they do not establish a trial outcome or a causal mechanism.',
    key_paths: picks.map((p) => ({ path: p.id, title: p.kind, explanation: describePath(kg, p) })),
    novel_hypotheses: [],
    gaps: ['Entities were matched by name. Review matches before interpreting paths.', 'The Prediction tab runs HINT separately. These PrimeKG paths are biological context, not inputs to that model.'],
    confidence: 'low',
  };
}

// ---------- agent ----------

export async function runAgent(nct, emit) {
  const llm = llmConfig();
  const step = (id, title, status, detail) => emit({ type: 'step', id, title, status, detail });
  const log = (id, text) => emit({ type: 'log', step: id, text });

  emit({ type: 'meta', llm: { provider: llm.provider, model: llm.model } });
  const kg = primekg();
  if (!kg) throw new Error('PrimeKG is not built yet. Run `npm run primekg` once, then restart the server.');
  if (!llm.ready) throw new Error('No LLM configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env and restart the server.');

  // 1. Retrieve
  step('retrieve', 'Retrieve trial record', 'running', 'ClinicalTrials.gov API v2');
  const trial = await fetchTrial(nct);
  emit({ type: 'trial', trial: { nct, title: trial.briefTitle, acronym: trial.acronym, status: trial.status, phases: trial.phases, sponsor: trial.sponsor, url: trial.url, conditions: trial.conditions, interventions: trial.interventions.map((i) => ({ name: i.name, roles: trial.armGroups.filter((a) => i.arms.includes(a.label)).map((a) => a.type) })) } });
  log('retrieve', `"${trial.briefTitle}"`);
  log('retrieve', `${trial.interventions.length} interventions · ${trial.conditions.length} conditions · ${trial.status}`);
  step('retrieve', 'Retrieve trial record', 'done');

  // 2. Extract
  step('extract', 'Extract & normalize entities', 'running', `LLM · ${llm.provider}/${llm.model}`);
  const ex = await askJSON({ system: EXTRACT_SYSTEM, user: extractPrompt(trial), mock: () => mockExtract(trial) });
  const interventions = (ex.interventions || []).filter((i) => i?.canonical || i?.name).slice(0, 12);
  const diseases = (ex.diseases || []).filter((d) => d?.canonical || d?.name).slice(0, 6);
  if (!interventions.length && !diseases.length) throw new Error('The LLM did not extract any interventions or diseases.');
  for (const i of interventions) log('extract', `intervention → ${(i.canonical ||= i.name)} [${i.type}, ${i.role}]`);
  for (const d of diseases) log('extract', `disease → ${(d.canonical ||= d.name)}${d.is_primary ? ' (primary)' : ''}`);
  step('extract', 'Extract & normalize entities', 'done', `${interventions.length} interventions · ${diseases.length} diseases`);

  // 3. Ground in PrimeKG
  step('ground', 'Ground entities in PrimeKG', 'running', `${kg.N.toLocaleString()} nodes · ${kg.E.toLocaleString()} edges`);
  const entities = [];
  interventions
    .filter((i) => LOOKUP_TYPES.has(i.type) && i.role !== 'placebo')
    .forEach((i, k) => {
      const ct = trial.interventions.find((x) => x.name === i.name);
      entities.push({ key: `I${k + 1}`, side: 'intervention', entity: i.canonical, names: [i.canonical, i.name, ...(ct?.otherNames || []).slice(0, 3)], types: ['drug'] });
      (Array.isArray(i.components) ? i.components : []).slice(0, 4).forEach((c, j) => {
        entities.push({ key: `I${k + 1}.${j + 1}`, side: 'intervention', entity: `${i.canonical} › ${c}`, names: [c], types: ['drug'] });
      });
    });
  diseases.forEach((d, k) => entities.push({ key: `D${k + 1}`, side: 'disease', entity: d.canonical, names: [d.canonical, d.name], types: ['disease', 'effect/phenotype'] }));
  for (const e of entities) e.candidates = candidatesFor(kg, e.names, e.types);
  log('ground', `${entities.length} entities · ${entities.reduce((s, e) => s + e.candidates.length, 0)} candidate nodes probed`);

  const picked = await askJSON({
    system: GROUND_SYSTEM,
    user: JSON.stringify(entities.map((e) => ({ key: e.key, kind: e.side, entity: e.entity, aliases: e.names, candidates: e.candidates.map((c) => ({ i: c.i, name: c.name, type: c.type })) }))),
    mock: () => mockGround(entities),
  });
  const seeds = [];
  for (const m of picked.matches || []) {
    const e = entities.find((x) => x.key === m.key);
    const c = e?.candidates.find((x) => x.i === Number(m.node));
    if (!e || !c) continue;
    if (seeds.some((s) => s.i === c.i)) continue;
    seeds.push({ i: c.i, name: c.name, type: c.type, side: e.side, entity: e.entity, via: m.reason });
    log('ground', `${e.entity} ≡ ${c.name} [${c.type} · ${kg.nodes.sources[c.i]} ${kg.nodes.ids[c.i]}]`);
  }
  for (const e of entities) if (!seeds.some((s) => s.entity === e.entity)) log('ground', `${e.entity} → no PrimeKG match`);

  // Interventions without a DrugBank node (e.g. many biologics/vaccines) are seeded at their targets.
  for (const i of interventions) {
    if (i.role === 'placebo' || seeds.some((s) => s.side === 'intervention' && s.entity.startsWith(i.canonical))) continue;
    for (const sym of (i.putative_targets || []).slice(0, 4)) {
      const hit = kg.search(sym, { types: ['gene/protein'], limit: 1 })[0];
      if (hit?.score >= 3 && !seeds.some((s) => s.i === hit.i)) {
        seeds.push({ i: hit.i, name: hit.name, type: hit.type, side: 'intervention', entity: i.canonical, via: 'LLM-proposed target' });
        log('ground', `${i.canonical} ⇢ target ${hit.name} (LLM-proposed)`);
      }
    }
  }
  emit({ type: 'seeds', seeds });
  const seedsA = seeds.filter((s) => s.side === 'intervention').map((s) => s.i);
  const seedsB = seeds.filter((s) => s.side === 'disease').map((s) => s.i);
  if (!seeds.length) throw new Error('None of the trial entities could be grounded in PrimeKG.');
  step('ground', 'Ground entities in PrimeKG', 'done', `${seedsA.length} intervention seeds · ${seedsB.length} disease seeds`);

  // 4. Neural search
  step('signal', 'Explore graph connections', 'running', 'bidirectional traversal over PrimeKG');
  const search = await streamSearch(kg, seedsA, seedsB, emit, (text) => log('signal', text));
  const kinds = [...new Set(search.paths.map((p) => p.kind))];
  for (const p of search.paths.slice(0, 6)) log('signal', `${p.id} ${p.kind}: ${describePath(kg, p).replace(/ \([^)]*\)/g, '')}`);
  step('signal', 'Explore graph connections', 'done', `${search.paths.length} pathways · ${kinds.length} kinds`);

  // 5. Synthesize
  step('synth', 'Interpret the connections', 'running', `LLM · ${llm.provider}/${llm.model}`);
  const syn = search.paths.length
    ? await askJSON({ system: SYNTH_SYSTEM, user: synthPrompt(kg, trial, seeds, search), mock: () => mockSynth(kg, trial, search) })
    : { headline: 'No connecting pathway found', summary: 'The neural search did not find a pathway within four hops between the grounded interventions and diseases.', key_paths: [], gaps: [], confidence: 'low' };
  const ids = new Set(search.paths.map((p) => p.id));
  syn.key_paths = (syn.key_paths || []).filter((k) => ids.has(k.path));
  emit({ type: 'summary', summary: syn });
  step('synth', 'Interpret the connections', 'done', `${syn.key_paths.length} key pathways · confidence ${syn.confidence || '?'}`);

  emit({ type: 'done', stats: { seeds: seeds.length, paths: search.paths.length, reached: search.reached } });
}
