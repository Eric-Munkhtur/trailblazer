"""Native graph reasoning over PrimeKG structure.

Everything here operates on the graph object (NetworkX) — never on text about
the graph. Reasoning acts:
  1. enumerate structure   : k shortest paths drug->disease (cap hops)
  2. read edge semantics   : mechanistic (inhibits/participates/drives) vs weak
  3. compose the chain     : drug -> target -> pathway -> disease
  4. contradiction check   : contraindication / off-pathway-toxicity flags
  5. quantify over geometry: features for the transparent scorer
"""
