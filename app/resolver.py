"""Entity resolver: trial text names -> PrimeKG node ids.

The unglamorous 40% of the project. Strategy (in order):
  1. exact match on node display names / synonyms (drugbank, MONDO labels)
  2. rapidfuzz fuzzy match with cutoff ~90
  3. MeSH terms from the CT.gov record as extra candidates
  4. honest failure: ResolutionResult(status=UNASSESSABLE)

Built tonight against artifacts/graph.pkl produced by prep/01_build_graph.ipynb.
"""
