"""Agentic Assembly: NCT number -> entities pinned on the PrimeKG graph.

Three moves (the kit's bullet #2, nothing more):
  1. LOOK UP — fetch the trial's protocol fields (ctgov.py, cached for offline demo)
  2. FIND    — locate the intervention (drug) and condition (disease) in PrimeKG
  3. PIN     — return a MappingReport with both nodes + a visible decision trace

The trace records every step, decision, and honest failure, so the UI can
show the agent thinking. UNASSESSABLE is a first-class outcome.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:                                    # package import (repo root)
    from . import ctgov
    from . import resolver as rez
except ImportError:                     # script import (inside app/)
    import ctgov
    import resolver as rez

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")


@dataclass
class Step:
    n: int
    name: str
    status: str          # ok | warn | fail
    detail: str

    def __str__(self) -> str:
        icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.status]
        return f"  {self.n}. {icon} {self.name:<10} {self.detail}"


@dataclass
class MappingReport:
    nct_id: str
    trial: dict = field(default_factory=dict)
    drug: Optional[rez.Resolution] = None
    disease: Optional[rez.Resolution] = None
    steps: list = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.trial:
            return "FETCH_FAILED"
        if self.drug and self.drug.ok and self.disease and self.disease.ok:
            return "MAPPED"
        if (self.drug and self.drug.ok) or (self.disease and self.disease.ok):
            return "PARTIAL"
        return "UNASSESSABLE"

    def summary(self) -> str:
        lines = [f"🤖 ASSEMBLY TRACE — {self.nct_id}  [{self.status}]"]
        lines += [str(s) for s in self.steps]
        return "\n".join(lines)


class AssemblyAgent:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR, use_cache: bool = True):
        self.resolver = rez.Resolver(artifacts_dir=artifacts_dir)
        self.use_cache = use_cache
        self.has_graph = bool(self.resolver.index)   # any node type loaded?

    def analyze(self, nct_id: str) -> MappingReport:
        rep = MappingReport(nct_id=nct_id.strip().upper())
        step = 0

        # -- 1. LOOK UP -----------------------------------------------------
        step += 1
        try:
            rep.trial = ctgov.get_trial(rep.nct_id, use_cache=self.use_cache)
            rep.steps.append(Step(step, "LOOK UP", "ok",
                                  f'"{rep.trial["title"][:60]}" · '
                                  f'{",".join(rep.trial["phases"] or ["?"])}'))
        except Exception as e:
            rep.steps.append(Step(step, "LOOK UP", "fail", f"registry error: {e}"))
            return rep

        # -- 2. FIND drug ----------------------------------------------------
        step += 1
        if not self.has_graph:
            rep.steps.append(Step(step, "FIND drug", "warn",
                                  "graph artifacts missing — run prep/01_build_graph.ipynb "
                                  "and drop files into artifacts/"))
            return rep

        drug, tried = None, []
        for name in rep.trial["interventions"]:
            if rez._norm(name) in rez.NON_DRUG_TOKENS:
                tried.append(f"skip placebo-arm '{name}'")
                continue
            r = self.resolver.resolve(name, "drug")
            if r.ok:
                drug = r
                tried.append(f"'{name}' -> pinned")
                break
            tried.append(f"'{name}' not found")
        rep.drug = drug or rez.Resolution("UNASSESSABLE",
                                          query=" | ".join(rep.trial["interventions"]),
                                          node_type="drug")
        if drug:
            rep.steps.append(Step(step, "FIND drug", "ok",
                                  f"{drug.node_name} ({drug.method} match); "
                                  + "; ".join(tried[:3])))
        else:
            rep.steps.append(Step(step, "FIND drug", "fail",
                                  "no intervention pinned; " + "; ".join(tried[:3])))

        # -- 3. FIND disease --------------------------------------------------
        step += 1
        dis = self.resolver.resolve_disease(rep.trial["conditions"])
        rep.disease = dis
        if dis and dis.ok:
            rep.steps.append(Step(step, "FIND disease", "ok",
                                  f"{dis.node_name} ({dis.method} match, {dis.score:.0f})"))
        else:
            rep.steps.append(Step(step, "FIND disease", "fail",
                                  f"no condition pinned among {rep.trial['conditions']}"))

        # -- 4. PIN -----------------------------------------------------------
        step += 1
        if rep.status == "MAPPED":
            rep.steps.append(Step(step, "PIN", "ok",
                                  f"both entities on graph: "
                                  f"[{rep.drug.node_id}] {rep.drug.node_name}  <->  "
                                  f"[{rep.disease.node_id}] {rep.disease.node_name}"))
        elif rep.status == "PARTIAL":
            rep.steps.append(Step(step, "PIN", "warn", "partial mapping — one entity missing"))
        else:
            rep.steps.append(Step(step, "PIN", "fail",
                                  "UNASSESSABLE — honest refusal, candidates logged"))
        return rep


if __name__ == "__main__":
    # Self-test: mock graph + a REAL cached trial (fetched offline-safe).
    import tempfile

    import pandas as pd

    tmp = tempfile.mkdtemp()
    pd.DataFrame({"node_id": ["DB06672", "DB00331"],
                  "node_name": ["Donanemab", "Metformin"]}).to_csv(
        os.path.join(tmp, "nodes_drug.csv"), index=False)
    pd.DataFrame({"node_id": ["MONDO:0004976"],
                  "node_name": ["Alzheimer disease"]}).to_csv(
        os.path.join(tmp, "nodes_disease.csv"), index=False)

    agent = AssemblyAgent(artifacts_dir=tmp)
    for nct in ["NCT04437511"]:                # donanemab / Alzheimer's
        print(agent.analyze(nct).summary())

    # a case that cannot resolve: drug absent from mock graph
    agent2 = AssemblyAgent(artifacts_dir=tmp)
    rep2 = agent2.analyze("NCT07091175")       # dupilumab trial (nephrotic syndrome)
    print()
    print(rep2.summary())
