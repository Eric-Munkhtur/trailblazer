import { getJSON, mapLimit } from './http.js';

const BASE = 'https://www.ebi.ac.uk/chembl/api/data';

export async function resolveMolecule(name) {
  const q = encodeURIComponent(name);
  let mols = (await getJSON(`${BASE}/molecule.json?pref_name__iexact=${q}&limit=5`)).molecules;
  if (!mols?.length) {
    mols = (await getJSON(`${BASE}/molecule/search.json?q=${q}&limit=5`, { timeout: 30000 })).molecules;
  }
  if (!mols?.length) return null;
  const exact = mols.find((m) => m.pref_name?.toLowerCase() === name.toLowerCase());
  const m = exact || mols[0];
  const parentId = m.molecule_hierarchy?.parent_chembl_id || m.molecule_chembl_id;
  return {
    chemblId: m.molecule_chembl_id,
    parentId,
    prefName: m.pref_name,
    maxPhase: m.max_phase,
    moleculeType: m.molecule_type,
    firstApproval: m.first_approval,
    formula: m.molecule_properties?.full_molformula,
    mw: m.molecule_properties?.full_mwt,
  };
}

export async function getMechanisms(chemblId) {
  const data = await getJSON(`${BASE}/mechanism.json?molecule_chembl_id=${chemblId}&limit=20`);
  const mechs = data.mechanisms || [];
  const targets = await mapLimit(
    [...new Set(mechs.map((m) => m.target_chembl_id).filter(Boolean))],
    4,
    (tid) => getTarget(tid).catch(() => null),
  );
  const byId = Object.fromEntries(targets.filter(Boolean).map((t) => [t.chemblId, t]));
  return mechs.map((m) => ({
    mechanism: m.mechanism_of_action,
    actionType: m.action_type,
    target: byId[m.target_chembl_id] || null,
  }));
}

async function getTarget(tid) {
  const t = await getJSON(`${BASE}/target/${tid}.json`);
  const genes = [];
  for (const c of t.target_components || []) {
    const sym = (c.target_component_synonyms || []).find((s) => s.syn_type === 'GENE_SYMBOL');
    if (sym) genes.push({ symbol: sym.component_synonym.toUpperCase(), accession: c.accession, name: c.component_description });
  }
  return { chemblId: tid, name: t.pref_name, type: t.target_type, organism: t.organism, genes };
}
