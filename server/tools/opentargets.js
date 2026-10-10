import { getJSON } from './http.js';

const URL = 'https://api.platform.opentargets.org/api/v4/graphql';

async function gql(query, variables) {
  const body = JSON.stringify({ query, variables });
  const res = await getJSON(URL, {
    init: { method: 'POST', body, headers: { 'Content-Type': 'application/json' } },
    cacheKey: body,
  });
  if (res.errors?.length) throw new Error(`Open Targets: ${res.errors[0].message}`);
  return res.data;
}

export async function findDisease(name) {
  const d = await gql(
    `query($q:String!){search(queryString:$q,entityNames:["disease"],page:{index:0,size:1}){hits{id name}}}`,
    { q: name },
  );
  return d.search.hits[0] || null;
}

export async function diseaseProfile(efoId, geneIds, topN = 8) {
  const d = await gql(
    `query($id:String!,$bs:[String!],$n:Int!){
      disease(efoId:$id){
        id name description
        therapeuticAreas{ id name }
        top: associatedTargets(page:{index:0,size:$n}){ rows{ score target{ id approvedSymbol approvedName } } }
        focused: associatedTargets(Bs:$bs,page:{index:0,size:50}){ rows{ score target{ id approvedSymbol approvedName } } }
      }}`,
    { id: efoId, bs: geneIds.length ? geneIds : ['ENSG00000000000'], n: topN },
  );
  if (!d.disease) return null;
  const pick = (r) => ({ ensembl: r.target.id, symbol: r.target.approvedSymbol, name: r.target.approvedName, score: r.score });
  return {
    id: d.disease.id,
    name: d.disease.name,
    description: d.disease.description,
    areas: d.disease.therapeuticAreas,
    top: d.disease.top.rows.map(pick),
    focused: geneIds.length ? d.disease.focused.rows.map(pick) : [],
  };
}

export async function mapGenes(symbols) {
  if (!symbols.length) return {};
  const d = await gql(
    `query($t:[String!]!){mapIds(queryTerms:$t,entityNames:["target"]){mappings{term hits{id name}}}}`,
    { t: symbols },
  );
  const out = {};
  for (const m of d.mapIds.mappings) {
    const hit = m.hits.find((h) => h.name.toUpperCase() === m.term.toUpperCase()) || m.hits[0];
    if (hit) out[m.term] = hit.id;
  }
  return out;
}
