"""Entity resolver: trial text names -> PrimeKG node ids.

The unglamorous 40% of the project. Strategy (in order):
  1. exact match on normalized node names (case/punctuation-insensitive)
  2. fuzzy match (rapidfuzz token_set_ratio, cutoff >= 87) — handles
     "Alzheimer Disease" vs "Alzheimer's disease", dose suffixes, word order
  3. partial-token fallback (longest informative token, e.g. brand codes)
  4. honest failure: Resolution(status=UNASSESSABLE) — a first-class output

Excludes non-drug interventions (Placebo etc.). Designed against the artifacts
produced by prep/01_build_graph.ipynb (nodes_drug.csv, nodes_disease.csv, ...).
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    from rapidfuzz import fuzz, process as rf_process
    HAVE_RAPIDFUZZ = True
except ImportError:  # graceful degrade to stdlib difflib
    import difflib
    HAVE_RAPIDFUZZ = False

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")

PUNCT_TABLE = str.maketrans("", "", string.punctuation)
NON_DRUG_TOKENS = {"placebo", "sham", "vehicle", "standard of care", "control", "usual care"}

FUZZY_CUTOFF = 87.0

# Generic words that must never anchor a partial-token match ("Disease From
# The Moon" must not resolve to "Parkinson disease" via the word "disease").
GENERIC_TOKENS = {
    "disease", "diseases", "syndrome", "disorder", "disorders", "type", "stage",
    "cancer", "carcinoma", "failure", "chronic", "acute", "severe", "malignant",
    "benign", "infection", "infections", "inflammatory", "disease,", "study",
    "patients", "treatment", "therapy", "therapy", "with", "without",
}


@dataclass
class Resolution:
    status: str                       # "RESOLVED" | "UNASSESSABLE"
    query: str = ""
    node_type: str = ""
    node_id: str = ""
    node_name: str = ""
    method: str = ""                  # exact | fuzzy | partial
    score: float = 0.0
    candidates: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "RESOLVED"


def _norm(s: str) -> str:
    s = str(s).lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()


def _informative_tokens(name: str) -> list[str]:
    """Tokens worth trying, longest first; drops dose/route noise."""
    noise = {"mg", "mcg", "ml", "iv", "sc", "oral", "tablet", "injection", "solution",
             "infusion", "capsule", "subcutaneous", "intravenous"}
    toks = [t for t in _norm(name).split() if t not in noise and len(t) > 2 and not t.isdigit()]
    return sorted(toks, key=len, reverse=True)


class Resolver:
    """Loads per-type node name indexes and resolves trial strings onto them."""

    FILE_BY_TYPE = {
        "drug": "nodes_drug.csv",
        "disease": "nodes_disease.csv",
        "gene/protein": "nodes_gene_protein.csv",
        "biological_process": "nodes_biological_process.csv",
        "pathway": "nodes_pathway.csv",
        "effect/phenotype": "nodes_effect_phenotype.csv",
    }

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        self.dir = artifacts_dir
        self.index: dict[str, pd.DataFrame] = {}     # type -> df[node_id, node_name, _norm]
        self.by_norm: dict[str, dict[str, tuple]] = {}  # type -> {norm: (id, name)}
        for ntype, fname in self.FILE_BY_TYPE.items():
            path = os.path.join(artifacts_dir, fname)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df["_norm"] = df["node_name"].map(_norm)
            self.index[ntype] = df.drop_duplicates("_norm")
            self.by_norm[ntype] = dict(zip(df["_norm"], zip(df["node_id"], df["node_name"])))

    # ------------------------------------------------------------------ core

    def resolve(self, query: str, node_type: str) -> Resolution:
        q = _norm(query)
        if not q or node_type not in self.index:
            return Resolution("UNASSESSABLE", query=query, node_type=node_type)

        # 1) exact
        if q in self.by_norm[node_type]:
            nid, name = self.by_norm[node_type][q]
            return Resolution("RESOLVED", query, node_type, str(nid), name, "exact", 100.0)

        # 2) fuzzy (full query)
        res = self._fuzzy(q, node_type)
        if res is not None:
            return Resolution("RESOLVED", query, node_type, res[2], res[1], "fuzzy", res[0],
                              candidates=self._candidate_list(q, node_type))

        # 3) informative-token fallback (handles "Donanemab 700 mg (LY3002813)").
        #    Uses two-sided ratio only — token_set_ratio lets generic words like
        #    "disease" subset-match long names (self-test caught this).
        for tok in _informative_tokens(query):
            if len(tok) < 5 or tok in GENERIC_TOKENS:
                continue
            res = self._fuzzy(tok, node_type, two_sided=True)
            if res is not None and res[0] >= 90.0:
                return Resolution("RESOLVED", query, node_type, res[2], res[1], "partial",
                                  res[0], candidates=self._candidate_list(q, node_type))

        return Resolution("UNASSESSABLE", query=query, node_type=node_type,
                          candidates=self._candidate_list(q, node_type))

    def resolve_drug(self, intervention_names: list[str]) -> Resolution:
        """Pick the first resolvable non-placebo intervention."""
        for name in intervention_names:
            if _norm(name) in NON_DRUG_TOKENS:
                continue
            r = self.resolve(name, "drug")
            if r.ok:
                return r
        return Resolution("UNASSESSABLE", query=" | ".join(intervention_names), node_type="drug")

    def resolve_disease(self, condition_names: list[str]) -> Resolution:
        """Try each condition; prefer exact, then best fuzzy across all."""
        best: Optional[Resolution] = None
        for name in condition_names:
            r = self.resolve(name, "disease")
            if r.ok:
                if r.method == "exact":
                    return r
                if best is None or r.score > best.score:
                    best = r
        return best or Resolution("UNASSESSABLE", query=" | ".join(condition_names),
                                  node_type="disease")

    # ------------------------------------------------------------- internals

    def _fuzzy(self, q: str, node_type: str, two_sided: bool = False):
        df = self.index[node_type]
        scorer = fuzz.ratio if two_sided else fuzz.token_set_ratio
        cutoff = 90.0 if two_sided else FUZZY_CUTOFF
        if HAVE_RAPIDFUZZ:
            choice = df["_norm"].tolist()
            out = rf_process.extractOne(q, choice, scorer=scorer, score_cutoff=cutoff)
            if out is None:
                return None
            score, idx = out[1], out[2]
            row = df.iloc[idx]
            return float(score), row["node_name"], str(row["node_id"])
        # difflib fallback
        matches = difflib.get_close_matches(q, df["_norm"].tolist(), n=1, cutoff=0.85)
        if not matches:
            return None
        row = df[df["_norm"] == matches[0]].iloc[0]
        return 90.0, row["node_name"], str(row["node_id"])

    def _candidate_list(self, q: str, node_type: str, k: int = 5) -> list:
        try:
            return self._fuzzy_all(q, node_type, k)
        except Exception:
            return []

    def _fuzzy_all(self, q: str, node_type: str, k: int) -> list:
        df = self.index[node_type]
        if HAVE_RAPIDFUZZ:
            out = rf_process.extract(q, df["_norm"].tolist(),
                                     scorer=fuzz.token_set_ratio, limit=k)
            return [{"name": df.iloc[i]["node_name"], "score": s}
                    for s, i in [(o[1], o[2]) for o in out]]
        return []


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Self-test against mock artifacts (no PrimeKG needed) — validates logic.
    import tempfile

    tmp = tempfile.mkdtemp()
    pd.DataFrame({
        "node_id": ["DB00331", "DB01076"],
        "node_name": ["Metformin", "Atorvastatin"],
    }).to_csv(os.path.join(tmp, "nodes_drug.csv"), index=False)
    pd.DataFrame({
        "node_id": ["MONDO:0005180", "MONDO:0004976", "MONDO:0005148"],
        "node_name": ["Parkinson disease", "Alzheimer disease", "type 2 diabetes mellitus"],
    }).to_csv(os.path.join(tmp, "nodes_disease.csv"), index=False)

    r = Resolver(artifacts_dir=tmp)
    cases = [
        (["Metformin"], ["Alzheimer Disease"]),                 # exact + word-order
        (["metformin 500 mg"], ["Alzheimer's disease"]),        # dose strip + apostrophe
        (["Placebo", "Atorvastatin 10 MG"], ["Diabetes Mellitus, Type 2"]),  # skip placebo
        (["Unobtainium-X"], ["Disease From The Moon"]),         # unassessable
    ]
    for drugs, dis in cases:
        rd = r.resolve_drug(drugs)
        rc = r.resolve_disease(dis)
        print(f"drugs={drugs!r:45} -> {rd.status:12} {rd.node_name!r:20} ({rd.method})")
        print(f"diseases={dis!r:45} -> {rc.status:12} {rc.node_name!r:28} ({rc.method})")
        print("-" * 100)
