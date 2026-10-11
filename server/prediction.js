import fs from 'node:fs';
import { fetchTrial } from './tools/ctgov.js';
import { getJSON } from './tools/http.js';

let evidence;
let diseaseMap;
try {
  evidence = JSON.parse(fs.readFileSync(new URL('../data/predictions/benchmark.json', import.meta.url), 'utf8'));
  diseaseMap = JSON.parse(fs.readFileSync(new URL('../data/predictions/disease-map.json', import.meta.url), 'utf8'));
} catch { /* Graph exploration remains available without model artifacts. */ }
const worker = 'http://127.0.0.1:8791';
const cached = new Map();
const pending = new Map();
let queue = Promise.resolve();
const normal = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim().replace(/\s+/g, ' ');

export async function predictionHealth() {
  try {
    const r = await fetch(`${worker}/health`, { signal: AbortSignal.timeout(1500) });
    const health = await r.json();
    return { ready: !!evidence && r.ok && health.ready === true, mode: 'live_inference', device: 'cpu', busy: !!health.busy, savedTrials: evidence?.nTest || 0 };
  } catch {
    return { ready: false, mode: 'live_inference', savedTrials: evidence?.nTest || 0 };
  }
}

export function benchmarkFor(nct) {
  if (!evidence) return null;
  return {
    dataset: evidence.dataset, nTest: evidence.nTest, recordedDate: evidence.recordedDate,
    models: evidence.models, conclusion: evidence.conclusion,
    upstreamCommit: evidence.upstreamCommit, upstreamUrl: evidence.upstreamUrl,
    inputSha256: evidence.inputSha256,
    historicalScores: evidence.records[nct] || null,
  };
}

function placebo(i) {
  return /^(?:matching\s+)?(?:placebo|sham)\b/i.test(i.name);
}

async function compoundFor(intervention, deadline) {
  const cleaned = intervention.name
    .replace(/\([^)]*\)/g, ' ')
    .replace(/\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml)\b.*$/i, '')
    .replace(/\b(?:oral|tablets?|capsules?|injection|infusion|solution)\b/gi, ' ')
    .replace(/\s+/g, ' ').trim();
  // Registry names/aliases only; no fuzzy graph match is treated as a molecule.
  const names = [...new Set([intervention.name, cleaned, ...intervention.otherNames.slice(0, 2)])].filter((s) => s && s.length <= 180);
  for (const name of names) {
    if (Date.now() > deadline) throw new Error('Drug resolution took too long. Please retry the prediction.');
    try {
      const url = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(name)}/property/IsomericSMILES,Title/JSON`;
      const data = await getJSON(url, { timeout: 12000 });
      const choices = data.PropertyTable?.Properties || [];
      if (choices.length !== 1) continue;
      const c = choices[0];
      const smiles = c.SMILES || c.IsomericSMILES;
      if (!smiles || !Number.isInteger(c.CID)) continue;
      return { intervention: intervention.name, matchedName: name, title: c.Title,
        cid: c.CID, smiles, source: `https://pubchem.ncbi.nlm.nih.gov/compound/${c.CID}` };
    } catch (error) {
      if (error.status !== 404 && error.status !== 400) throw new Error('PubChem could not be reached. Please retry the prediction.');
    }
  }
  return null;
}

// Deduplicate a trial and serialize CPU jobs instead of rejecting every second
// visitor while the first inference is still running. The queue is bounded.
export function predictTrial(nct) {
  const entry = cached.get(nct);
  if (entry && entry.expires > Date.now()) return Promise.resolve({ ...entry.value, cached: true });
  if (pending.has(nct)) return pending.get(nct);
  if (pending.size >= 3) return Promise.resolve({ available: false, nct, benchmark: benchmarkFor(nct), reasonCode: 'busy',
    reason: 'Three predictions are already queued. Please retry shortly.', retryable: true });
  const job = queue.then(() => predictUncached(nct)).finally(() => pending.delete(nct));
  queue = job.catch(() => {});
  pending.set(nct, job);
  return job;
}

async function predictUncached(nct) {
  const benchmark = benchmarkFor(nct);
  const reject = (reason, extras = {}) => {
    console.info(`[prediction] ${nct}: ${extras.reasonCode || 'unavailable'} · ${reason}`);
    return { available: false, nct, reason, benchmark, ...extras };
  };
  if (!evidence) return reject('Model assets are unavailable on this server.');
  const entry = cached.get(nct);
  if (entry && entry.expires > Date.now()) return { ...entry.value, cached: true };
  try {
    const trial = await fetchTrial(nct);
    const trialInfo = { title: trial.briefTitle, studyType: trial.studyType, phases: trial.phases, url: trial.url };
    if (trial.studyType !== 'INTERVENTIONAL') return reject(
      'This is an observational study of existing medical records or exposures. Our model predicts outcomes for Phase II interventional drug trials, so it cannot assign a trial-success score to this study.',
      { reasonCode: 'study_type', trial: trialInfo });
    if (!trial.phases.includes('PHASE2')) return reject(
      `This is ${trial.phases.length ? trial.phases.map((p) => p.replace('PHASE', 'Phase ')).join(' / ') : 'a study without a drug-trial phase'}. Our trained checkpoint covers Phase II. You can still map its biological connections.`,
      { reasonCode: 'phase', trial: trialInfo });
    const missing = trial.conditions.filter((c) => !diseaseMap[normal(c)]);
    if (!trial.conditions.length || missing.length) return reject(`We could not match ${missing.join(', ') || 'the conditions'} to the model's disease-code dictionary. The graph search can still run.`, { reasonCode: 'disease_mapping', trial: trialInfo });
    if (!trial.eligibilityCriteria.trim()) return reject('The registry record has no eligibility text for the model.', { trial: trialInfo });
    const interventions = trial.interventions.filter((i) => !placebo(i));
    if (!interventions.length || interventions.length > 12 || interventions.some((i) => i.type !== 'DRUG')) {
      return reject('Live prediction needs molecular structures for the trial interventions. This record includes a biologic, device, procedure, or other intervention that this pipeline cannot represent.', { reasonCode: 'intervention_type', trial: trialInfo });
    }
    if (!(await predictionHealth()).ready) return reject('The model worker is starting or unavailable. The graph and saved benchmark results remain available.', { retryable: true, trial: trialInfo });
    const compounds = [];
    const deadline = Date.now() + 30000;
    for (const intervention of interventions) {
      const compound = await compoundFor(intervention, deadline);
      if (!compound) return reject(`Could not resolve a unique molecular structure for “${intervention.name}”. No partial-input score was produced.`, { trial: trialInfo });
      compounds.push(compound);
    }
    const response = await fetch(`${worker}/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(120000),
      body: JSON.stringify({ nct, phases: trial.phases, conditions: trial.conditions,
        eligibilityCriteria: trial.eligibilityCriteria, compounds }),
    });
    const result = await response.json();
    if (!response.ok) return reject(result.error || 'Inference failed; no score was produced.', { retryable: true, trial: trialInfo });
    const value = { ...result, nct, trial: trialInfo, benchmark, cached: false };
    if (result.available) {
      while (cached.size >= 32) cached.delete(cached.keys().next().value);
      cached.set(nct, { expires: Date.now() + 15 * 60000, value });
    }
    return value;
  } catch (error) {
    const reason = error.name === 'TimeoutError' ? 'Prediction took too long. Please retry shortly.' : error.message;
    return reject(reason, { retryable: true });
  }
}
