import { getJSON } from './http.js';

export async function fetchTrial(nct) {
  let raw;
  try {
    raw = await getJSON(`https://clinicaltrials.gov/api/v2/studies/${nct}`);
  } catch (e) {
    if (e.status === 404) throw new Error(`${nct} was not found on ClinicalTrials.gov`);
    throw e;
  }
  const p = raw.protocolSection || {};
  const d = raw.derivedSection || {};
  const id = p.identificationModule || {};
  const status = p.statusModule || {};
  const design = p.designModule || {};
  const arms = p.armsInterventionsModule || {};

  return {
    nct,
    url: `https://clinicaltrials.gov/study/${nct}`,
    briefTitle: id.briefTitle,
    officialTitle: id.officialTitle,
    acronym: id.acronym,
    status: status.overallStatus,
    startDate: status.startDateStruct?.date,
    completionDate: status.completionDateStruct?.date,
    sponsor: p.sponsorCollaboratorsModule?.leadSponsor?.name,
    studyType: design.studyType,
    phases: design.phases || [],
    enrollment: design.enrollmentInfo?.count,
    summary: p.descriptionModule?.briefSummary || '',
    conditions: p.conditionsModule?.conditions || [],
    eligibilityCriteria: p.eligibilityModule?.eligibilityCriteria || '',
    keywords: p.conditionsModule?.keywords || [],
    interventions: (arms.interventions || []).map((i) => ({
      type: i.type,
      name: i.name,
      description: i.description,
      otherNames: i.otherNames || [],
      arms: i.armGroupLabels || [],
    })),
    armGroups: (arms.armGroups || []).map((a) => ({
      label: a.label,
      type: a.type,
      interventions: a.interventionNames || [],
    })),
    primaryOutcomes: (p.outcomesModule?.primaryOutcomes || []).map((o) => o.measure),
    conditionMesh: d.conditionBrowseModule?.meshes || [],
    conditionMeshAncestors: d.conditionBrowseModule?.ancestors || [],
    interventionMesh: d.interventionBrowseModule?.meshes || [],
    interventionMeshAncestors: d.interventionBrowseModule?.ancestors || [],
  };
}
