"""Agent 2 core: pull the subgraph + connecting routes between two pinned nodes.

Works directly on the compact edge table (8.09M edges) with vectorized pandas
lookups — no giant in-memory graph object needed. Finds:
  - direct 1-hop link (if any)
  - 2-hop routes through common neighbors (drug -> X -> disease)
  - one representative 3-hop route (drug -> A -> B -> disease)
plus 1-hop neighborhoods of both pins for the 3D view. Everything bounded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")

# intermediates we're willing to travel through on a route
ROUTE_TYPES = {"gene/protein", "biological_process", "pathway"}
NEIGHBOR_TYPES = {"drug", "disease", "gene/protein", "biological_process",
                  "pathway", "effect/phenotype", "molecular_function"}


@dataclass
class Subgraph:
    drug: str = ""
    disease: str = ""
    routes: list = field(default_factory=list)      # [{hops, chain:[{name,type,rel}...]}]
    nodes: list = field(default_factory=list)       # [{id,name,type,role}]
    links: list = field(default_factory=list)       # [{source,target,relation,on_route}]

    @property
    def found_routes(self) -> int:
        return len(self.routes)


class GraphView:
    def __init__(self, edges_path: str = None):
        path = edges_path or os.path.join(ART, "edges.csv.gz")
        self.df = pd.read_csv(path, compression="gzip", low_memory=False)
        self.df = self.df.dropna(subset=["src_name", "dst_name"])

    # ------------------------------------------------------------ neighbors
    def _neighbors(self, name: str, types=None, cap: int = 40) -> pd.DataFrame:
        m = (self.df["src_name"] == name) | (self.df["dst_name"] == name)
        sub = self.df[m]
        if sub.empty:
            return pd.DataFrame(columns=["name", "type", "relation"])
        other_name = sub["dst_name"].where(sub["src_name"] == name, sub["src_name"])
        other_type = sub["dst_type"].where(sub["src_name"] == name, sub["src_type"])
        out = pd.DataFrame({"name": other_name, "type": other_type,
                            "relation": sub["display_relation"]})
        if types:
            out = out[out["type"].isin(types)]
        out = out[out["name"] != name].drop_duplicates("name")
        return out.head(cap)

    def _edge_rel(self, a: str, b: str) -> str:
        m = ((self.df["src_name"] == a) & (self.df["dst_name"] == b)) | \
            ((self.df["src_name"] == b) & (self.df["dst_name"] == a))
        if m.any():
            return str(self.df.loc[m.idxmax(), "display_relation"])
        return "related_to"

    # ------------------------------------------------------------ main
    def subgraph_for(self, drug: str, disease: str) -> Subgraph:
        sg = Subgraph(drug=drug, disease=disease)

        # 1-hop neighborhoods (for display + route seeds)
        nd = self._neighbors(drug, NEIGHBOR_TYPES, cap=25)
        nn = self._neighbors(disease, NEIGHBOR_TYPES, cap=25) if drug != disease else nd

        # neighborhood-only mode (single pin, PARTIAL mappings)
        if drug == disease:
            sg.nodes = [{"id": drug, "name": drug, "type": "drug", "role": "pin_drug"}]
            for _, row in nd.head(20).iterrows():
                sg.nodes.append({"id": row["name"], "name": row["name"],
                                 "type": row["type"], "role": "context"})
                sg.links.append({"source": drug, "target": row["name"],
                                 "relation": row["relation"], "on_route": False})
            return sg

        # ---- direct link?
        if (nd["name"] == disease).any():
            sg.routes.append({
                "hops": 1,
                "chain": [drug, f"[{self._edge_rel(drug, disease)}]", disease],
            })

        # ---- 2-hop routes via common neighbors
        common = set(nd[nd["type"].isin(ROUTE_TYPES)]["name"]) & \
                 set(nn[nn["type"].isin(ROUTE_TYPES)]["name"])
        for x in sorted(common)[:5]:
            sg.routes.append({
                "hops": 2,
                "chain": [drug, f"[{self._edge_rel(drug, x)}]", x,
                          f"[{self._edge_rel(x, disease)}]", disease],
            })

        # ---- one 3-hop route (BFS depth 2 from drug, hit neighbors of disease)
        if not sg.routes:
            dn = set(self._neighbors(disease, ROUTE_TYPES | {"disease"}, cap=300)["name"])
            l1 = self._neighbors(drug, ROUTE_TYPES, cap=250)
            if not l1.empty:
                m = (self.df["src_name"].isin(l1["name"])) | \
                    (self.df["dst_name"].isin(l1["name"]))
                sub = self.df[m]
                for _, row in sub.head(4000).iterrows():
                    a = row["src_name"] if row["src_name"] in set(l1["name"]) else row["dst_name"]
                    b = row["dst_name"] if a == row["src_name"] else row["src_name"]
                    if b in dn and b != drug and b != a:
                        sg.routes.append({
                            "hops": 3,
                            "chain": [drug, f"[{self._edge_rel(drug, a)}]", a,
                                      f"[{self._edge_rel(a, b)}]", b,
                                      f"[{self._edge_rel(b, disease)}]", disease],
                        })
                        break

        # ---- build display nodes/links
        keep_names = {drug, disease}
        for r in sg.routes:
            keep_names.update(n for n in r["chain"] if not n.startswith("["))
        for n in list(nd["name"])[:12]:
            keep_names.add(n)
        for n in list(nn["name"])[:12]:
            keep_names.add(n)

        types = {}
        for df_ in (nd, nn):
            for _, row in df_.iterrows():
                types[row["name"]] = row["type"]
        types[drug] = "drug"
        types[disease] = "disease"

        for i, n in enumerate(keep_names):
            role = "pin_drug" if n == drug else ("pin_disease" if n == disease else "context")
            sg.nodes.append({"id": n, "name": n, "type": types.get(n, "other"), "role": role})

        on_route_pairs = set()
        for r in sg.routes:
            names = [n for n in r["chain"] if not n.startswith("[")]
            for a, b in zip(names, names[1:]):
                on_route_pairs.add(frozenset((a, b)))

        nl = list(keep_names)
        for a in nl:
            for b in nl:
                if a >= b or frozenset((a, b)) not in on_route_pairs:
                    continue
                sg.links.append({"source": a, "target": b,
                                 "relation": self._edge_rel(a, b), "on_route": True})
        # context links: from pins to their 1-hop neighbors
        for pin in (drug, disease):
            for _, row in (nd if pin == drug else nn).head(12).iterrows():
                nb = row["name"]
                if nb in keep_names and nb not in (drug, disease) and \
                        not any(l["source"] == pin and l["target"] == nb or
                                l["target"] == pin and l["source"] == nb for l in sg.links):
                    sg.links.append({"source": pin, "target": nb,
                                     "relation": row["relation"], "on_route": False})
        return sg


if __name__ == "__main__":
    gv = GraphView()
    for drug, dis in [("Ibrutinib", "chronic leukemia"),
                      ("Dupilumab", "nephrotic syndrome,")]:
        sg = gv.subgraph_for(drug, dis)
        print(f"\n=== {drug} -> {dis}: {sg.found_routes} route(s), "
              f"{len(sg.nodes)} nodes, {len(sg.links)} links")
        for r in sg.routes[:4]:
            print("  " + " ".join(r["chain"]))
