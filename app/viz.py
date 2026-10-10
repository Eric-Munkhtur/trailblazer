"""Render the subgraph as an interactive 3D force-graph (offline, library inlined)."""

from __future__ import annotations

import json
import os

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "static", "3d-force-graph.min.js")

TYPE_COLORS = {
    "drug": "#8b949e", "disease": "#d29922", "gene/protein": "#58a6ff",
    "biological_process": "#a371f7", "pathway": "#f778ba",
    "molecular_function": "#7ee787", "effect/phenotype": "#ffa657",
    "anatomy": "#79c0ff", "other": "#6e7681",
}


def build_html(nodes: list, links: list, height: int = 620) -> str:
    with open(LIB_PATH, encoding="utf-8") as f:
        lib = f.read()

    for n in nodes:
        if n.get("role") == "pin_drug":
            n["color"] = "#39d353"; n["val"] = 12
        elif n.get("role") == "pin_disease":
            n["color"] = "#ff6b6b"; n["val"] = 12
        else:
            n["color"] = TYPE_COLORS.get(n.get("type", "other"), "#6e7681")
            n["val"] = 3
    for l in links:
        l["color"] = "#ffcf5c" if l.get("on_route") else "#30363d"
        l["width"] = 2.4 if l.get("on_route") else 0.35

    data = json.dumps({"nodes": nodes, "links": links})

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html,body {{ margin:0; height:100%; background:#0d1117; overflow:hidden;
               font-family:-apple-system,sans-serif; }}
  #g {{ width:100%; height:{height}px; }}
  #legend {{ position:absolute; top:10px; left:12px; color:#8b949e;
             font-size:11px; line-height:1.6; pointer-events:none;
             background:rgba(13,17,23,.75); padding:6px 10px; border-radius:8px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%;
          margin-right:5px; }}
</style></head>
<body>
<div id="g"></div>
<div id="legend">
  <div><span class="dot" style="background:#39d353"></span>trial drug
       <span class="dot" style="background:#ff6b6b;margin-left:10px"></span>trial disease</div>
  <div><span class="dot" style="background:#ffcf5c"></span>traced route
       <span class="dot" style="background:#30363d;margin-left:10px"></span>neighborhood</div>
  <div style="color:#6e7681">drag to spin · scroll to zoom · click a node</div>
</div>
<script>{lib}</script>
<script>
const DATA = {data};
const Graph = ForceGraph3D()(document.getElementById('g'))
  .graphData(DATA)
  .backgroundColor('#0d1117')
  .nodeLabel(n => n.name + '  (' + n.type + ')')
  .nodeColor(n => n.color)
  .nodeVal(n => n.val)
  .nodeOpacity(0.95)
  .linkColor(l => l.color)
  .linkWidth(l => l.width)
  .linkLabel(l => l.relation)
  .linkOpacity(0.85)
  .onNodeClick(n => Graph.centerAt({{x:n.x, y:n.y, z:n.z}}, 500));
Graph.d3Force('charge').strength(-90);
</script>
</body></html>"""
