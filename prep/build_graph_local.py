"""Build the compact PrimeKG artifacts locally (no Colab needed).

Mirrors prep/01_build_graph.ipynb: load kg.csv, keep the reasoning node
types, write edges.csv.gz + per-type node name lists into artifacts/.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
KG = os.path.join(ART, "kg.csv")

KEEP = {
    "drug", "disease", "gene/protein", "molecular_function",
    "biological_process", "cellular_component", "pathway", "anatomy",
    "effect/phenotype",
}


def main() -> None:
    if not os.path.exists(KG):
        sys.exit("kg.csv missing — download it into artifacts/ first")
    print("loading kg.csv ...", flush=True)
    kg = pd.read_csv(
        KG,
        low_memory=False,
        usecols=["relation", "display_relation", "x_id", "x_type", "x_name",
                 "y_id", "y_type", "y_name"],
    )
    print(f"loaded {len(kg):,} edges")

    print("\nnode types present:")
    present = set(kg["x_type"].dropna().unique()) | set(kg["y_type"].dropna().unique())
    print(sorted(present))
    keep = KEEP & present
    dropped = KEEP - keep
    if dropped:
        print(f"WARNING: expected types missing from file: {dropped}")

    m = kg["x_type"].isin(keep) & kg["y_type"].isin(keep)
    edges = kg[m].copy()
    edges.columns = ["relation", "display_relation", "src_id", "src_type", "src_name",
                     "dst_id", "dst_type", "dst_name"]
    edges = edges.drop_duplicates()
    print(f"\nkept {len(edges):,} edges between reasoning node types")

    nodes = pd.concat([
        edges[["src_type", "src_id", "src_name"]].rename(
            columns={"src_type": "node_type", "src_id": "node_id", "src_name": "node_name"}),
        edges[["dst_type", "dst_id", "dst_name"]].rename(
            columns={"dst_type": "node_type", "dst_id": "node_id", "dst_name": "node_name"}),
    ]).drop_duplicates()
    print(f"{len(nodes):,} unique nodes")
    print(nodes["node_type"].value_counts())

    edges.to_csv(os.path.join(ART, "edges.csv.gz"), index=False, compression="gzip")
    for t, grp in nodes.groupby("node_type"):
        safe = t.replace("/", "_")
        grp[["node_id", "node_name"]].drop_duplicates().to_csv(
            os.path.join(ART, f"nodes_{safe}.csv"), index=False)

    print("\n--- sanity lookups ---")
    for q, t in [("metformin", "drug"), ("dupilumab", "drug"), ("donanemab", "drug"),
                 ("alzheimer", "disease"), ("eczema", "disease"),
                 ("nephrotic", "disease")]:
        hit = nodes[(nodes["node_type"] == t)
                    & nodes["node_name"].str.lower().str.contains(q, na=False)]
        names = hit["node_name"].drop_duplicates().head(3).tolist()
        print(f"  {t:10} '{q}' -> {names if len(names) else 'NOT FOUND'}")

    print("\nartifacts written:", sorted(
        f for f in os.listdir(ART) if f.startswith(("edges", "nodes"))))


if __name__ == "__main__":
    main()
