"""ClinicalTrials.gov API v2 client.

Fetches PROTOCOL-stage fields only: identification, sponsor, status, phase,
condition, interventions, primary outcome.

DISCIPLINE RULE (the core methodological claim of this project):
we intentionally never request or parse `resultsSection`. "Predicting prior to
public results" is enforced in code — there is no code path that reads outcomes.
"""

from __future__ import annotations

import json
import os

import requests

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
TIMEOUT = 30
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "demo_cache")


def get_trial(nct_id: str, use_cache: bool = True) -> dict:
    """Fetch a trial by NCT id and return a compact protocol dict.

    Results-section discipline: we request the whole record but extract
    protocolSection only — no code path reads outcomes. Cached trials make
    the demo work offline at the judging table.
    """
    nct_id = nct_id.strip().upper()
    if not nct_id.startswith("NCT"):
        raise ValueError(f"Expected an NCT id like NCT01234567, got {nct_id!r}")

    cache_path = os.path.join(CACHE_DIR, f"{nct_id}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    resp = requests.get(f"{API_BASE}/{nct_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    trial = _extract_protocol(resp.json()["protocolSection"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(trial, f, indent=2)
    return trial


def _extract_protocol(p: dict) -> dict:
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    conditions = p.get("conditionsModule", {})
    arms = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    elig = p.get("eligibilityModule", {})

    interventions = [
        i.get("name", "") for i in arms.get("interventions", []) if i.get("name")
    ]
    primary = [
        o.get("measure", "") for o in outcomes.get("primaryOutcomes", []) if o.get("measure")
    ]

    return {
        "nct_id": ident.get("nctId", ""),
        "title": ident.get("briefTitle", ""),
        "sponsor": p.get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {})
        .get("name", ""),
        "status": status.get("overallStatus", ""),
        "phases": design.get("phases", []),
        "conditions": conditions.get("conditions", []),
        "mesh_terms": conditions.get("meshTerms", []),
        "interventions": interventions,
        "primary_outcomes": primary,
        "eligibility_criteria": elig.get("eligibilityCriteria", ""),
        "sex": elig.get("sex", ""),
        "min_age": elig.get("minimumAge", ""),
        "max_age": elig.get("maximumAge", ""),
        "healthy_volunteers": elig.get("healthyVolunteers", ""),
    }


def search_trials(query: str, max_count: int = 20, ongoing_only: bool = True) -> list[dict]:
    """Search trials (used to hand-pick demo candidates). Returns brief dicts."""
    params: dict = {"query.term": query, "pageSize": max_count}
    if ongoing_only:
        # RECRUITING / ACTIVE, not recruiting / ENROLLING BY INVITATION
        params["filter.overallStatus"] = "RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION"
    resp = requests.get(API_BASE, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    studies = resp.json().get("studies", [])
    out = []
    for s in studies:
        p = s.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        out.append(
            {
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": p.get("statusModule", {}).get("overallStatus", ""),
            }
        )
    return out


if __name__ == "__main__":
    import json
    import sys

    for nct in sys.argv[1:] or ["NCT04437511"]:
        print(json.dumps(get_trial(nct), indent=2))
