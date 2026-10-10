"""3D view v2: the FULL graph as a star-field backdrop + the agent's live search.

Two layers:
  - dust: all ~128k PrimeKG nodes, precomputed positions (type-clustered
    'galaxy'), rendered as tiny fixed points — the whole map is always visible
  - action: the trial's pins, neighborhood, and traced route — glowing nodes
    and gold links, revealed by an animated camera fly-through (the agent
    'searching the graph live')

Fully offline: the 3d-force-graph library is vendored in app/static/.
"""

from __future__ import annotations

import json
import os

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "static", "3d-force-graph.min.js")

TYPE_COLORS = {
    "drug": "#8b949e", "disease": "#d29922", "gene/protein": "#58a6ff",
    "biological_process": "#a371f7", "molecular_function": "#7ee787",
    "cellular_component": "#7ee787", "pathway": "#f778ba",
    "anatomy": "#79c0ff", "effect/phenotype": "#ffa657", "other": "#6e7681",
}
TYPES = list(TYPE_COLORS)


def build_html(action_nodes: list, action_links: list, dust: dict | None = None,
               height: int = 700, animate: bool = True) -> str:
    """dust: {'xs':[int], 'ys':[int], 'zs':[int], 'ts':[type-index int]}"""
    with open(LIB_PATH, encoding="utf-8") as f:
        lib = f.read()

    for n in action_nodes:
        if n.get("role") == "pin_drug":
            n["color"], n["val"] = "#39d353", 14
        elif n.get("role") == "pin_disease":
            n["color"], n["val"] = "#ff6b6b", 14
        else:
            n["color"] = TYPE_COLORS.get(n.get("type", "other"), "#6e7681")
            n["val"] = 3.5
    for l in action_links:
        l["color"] = "#ffcf5c" if l.get("on_route") else "#30363d"
        l["width"] = 2.4 if l.get("on_route") else 0.35

    payload = {
        "nodes": action_nodes,
        "links": action_links,
        "dust": dust or {"xs": [], "ys": [], "zs": [], "ts": []},
        "palette": [TYPE_COLORS[t] for t in TYPES],
        "animate": animate,
    }
    data = json.dumps(payload, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html,body {{ margin:0; height:100%; background:#0d1117; overflow:hidden;
               font-family:-apple-system,sans-serif; }}
  #g {{ width:100%; height:{height}px; }}
  #legend {{ position:absolute; top:10px; left:12px; color:#8b949e; font-size:11px;
             line-height:1.65; pointer-events:none;
             background:rgba(13,17,23,.78); padding:6px 10px; border-radius:8px; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}
  #btns {{ position:absolute; top:10px; right:12px; }}
  #btns button {{ background:#21262d; color:#c9d1d9; border:1px solid #30363d;
                  border-radius:6px; padding:4px 10px; font-size:11px; margin-left:6px;
                  cursor:pointer; }}
  #status {{ position:absolute; bottom:10px; left:12px; color:#58a6ff; font-size:12px;
             font-family:ui-monospace,monospace; }}
</style></head>
<body>
<div id="g"></div>
<div id="legend">
  <div><span class="dot" style="background:#39d353"></span>trial drug
       <span class="dot" style="background:#ff6b6b;margin-left:8px"></span>trial disease
       <span class="dot" style="background:#ffcf5c;margin-left:8px"></span>traced route</div>
  <div style="color:#6e7681">galaxy = 128k PrimeKG nodes, clustered by type ·
       drag to spin · scroll to zoom</div>
</div>
<div id="btns">
  <button onclick="replay()">▶ replay search</button>
  <button onclick="toggleDust()">galaxy on/off</button>
</div>
<div id="status">agent idle</div>
<script>{lib}</script>
<script>
const DATA = {data};

// ---- dust layer: every node in the knowledge graph, fixed positions ----
const dust = [];
for (let i = 0; i < DATA.dust.xs.length; i++) {{
  const x = DATA.dust.xs[i], y = DATA.dust.ys[i], z = DATA.dust.zs[i];
  dust.push({{ id: 'd' + i, x, y, z, fx: x, fy: y, fz: z,
               color: DATA.palette[DATA.dust.ts[i]] || '#6e7681', val: 0.3, dust: true }});
}}

// ---- action layer: pins + neighborhood + routes ----
const action = DATA.nodes.map(n => ({{ ...n, fx: n.x, fy: n.y, fz: n.z }}));
const all = dust.concat(action);

const Graph = ForceGraph3D()(document.getElementById('g'))
  .graphData({{ nodes: all, links: [] }})
  .backgroundColor('#0d1117')
  .nodeVisibility(n => showDust || !n.dust)
  .nodeLabel(n => n.dust ? null : (n.name + '  (' + n.type + ')'))
  .nodeColor(n => n.color)
  .nodeVal(n => n.val)
  .nodeOpacity(0.95)
  .linkColor(l => l.color)
  .linkWidth(l => l.width)
  .linkLabel(l => l.relation)
  .linkOpacity(0.9)
  .onNodeClick(n => Graph.cameraPosition({{ x: n.x, y: n.y, z: n.z + 300 }}, n, 600));

// full map is pre-laid-out: no physics needed
['charge', 'link', 'center', 'radial'].forEach(k => Graph.d3Force(k, null));

let showDust = true;
function toggleDust() {{ showDust = !showDust; Graph.nodeVisibility(Graph.nodeVisibility()); }}

const drug  = action.find(n => n.role === 'pin_drug');
const dis   = action.find(n => n.role === 'pin_disease');
const status = document.getElementById('status');

function say(t) {{ status.textContent = t; }}
function fly(node, dist, ms) {{
  Graph.cameraPosition({{ x: node.x + dist * .45, y: node.y + dist * .35, z: node.z + dist }},
                       node, ms);
}}

function replay() {{
  if (!drug && !dis) return;
  Graph.graphData({{ nodes: all, links: [] }});            // hide route
  Graph.cameraPosition({{ x: 0, y: 300, z: 4600 }}, {{ x: 0, y: 0, z: 0 }}, 0);  // overview
  say('🛰 agent 1: trial found in registry');
  setTimeout(() => {{ if (drug) fly(drug, 620, 1700); say('🛰 agent 2: scanning galaxy → drug district…'); }}, 900);
  setTimeout(() => {{ if (drug) drug.color = '#39d353'; Graph.refresh(); say('🛰 drug pinned'); }}, 2600);
  setTimeout(() => {{
    Graph.graphData({{ nodes: all, links: DATA.links.map(l => ({{ ...l }})) }});
    say(DATA.links.some(l => l.on_route)
        ? '🛰 route traced through the knowledge graph'
        : '🛰 no route exists — mechanistically uncharted');
  }}, 3300);
  setTimeout(() => {{
    const target = dis || drug; if (!target) return;
    if (dis) fly({{ x: (drug.x + dis.x) / 2, y: (drug.y + dis.y) / 2, z: (drug.z + dis.z) / 2 }}, 900, 1600);
    say('🛰 mapping complete');
  }}, 4300);
}}

if (DATA.animate) replay(); else Graph.graphData({{ nodes: all, links: DATA.links }});
</script>
</body></html>"""
