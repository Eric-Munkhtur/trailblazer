> Historical prototype log. See the root README for the deployed application and measured results.

# Dev Log

Build-window evidence for judges: everything below was written during HackRU Fall 2026
(Sat Oct 10, 12:00 PM ET → Sun Oct 11, 10:00 AM ET). `git log` is the source of truth.

---

## Sat Oct 10, 2026 (evening)

- **Repo + scaffold.** README (pitch, architecture, evaluation plan), MIT license,
  module skeleton, Colab prep layout. Chose the name *Verdia* — it blazes the
  trail from drug to disease through the knowledge graph.
- Design rules locked in: the agent never parses the trial's results section
  ("prior to public results" enforced in code, not by promise); the LLM narrates
  but never scores; UNASSESSABLE is a valid output.
- Next: CT.gov API client, then PrimeKG subset notebook (highest data risk — do it
  while fresh).

- **ctgov.py live.** Tested against the real API: NCT04437511 (donanemab,
  TRAILBLAZER-ALZ 2 — fitting name) returns clean protocol fields; search used to
  shortlist recruiting demo candidates. One observation: `meshTerms` comes back
  empty for some trials — name matching must not rely on it (exact + fuzzy on
  conditions/interventions is the primary path).

- **Plan merge (late evening).** Reviewed an external GNN-centric research plan.
  Adopted: PyTDC data loader, HINT published F1s (Ph I/II/III = 0.665/0.620/0.847)
  as prior-art rows, TxGNN (Nat Med 2024, same lab as PrimeKG) as the explanation
  template + citation, prospective demo cases (trials with results after the data
  cutoff), and drug-level split discipline. Rejected as spine: heterogeneous
  GNN training (solo, ~10h — too risky). Optional unattended R-GCN lane with a
  hour-6 kill-switch if time allows.

- **prep/01 written.** PrimeKG download (official Dataverse API, file 6180620),
  defensive column mapping, filter to 9 reasoning node types, node indexes for
  the resolver, sanity checks (metformin/Alzheimer/dupilumab), zipped artifacts.

- **Renamed: Trailblazer -> Verdia.** verdict + via (Latin: the road) — the
  system that finds the road through biology and delivers the verdict.
