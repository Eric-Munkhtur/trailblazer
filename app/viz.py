"""3D view v3: FAST full-graph galaxy.

Perf fix: the 128k-node backdrop is ONE THREE.Points cloud (single draw
call) instead of individual node objects (~1000x fewer draw calls). Only
the action layer (pins, neighborhood, route) are real graph nodes. Fully
offline: 3d-force-graph + three.js vendored in app/static/.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "static", "3d-force-graph.min.js")
THREE_PATH = os.path.join(HERE, "static", "three.min.js")

TYPE_COLORS = {
    "drug": "#8b949e", "disease": "#d29922", "gene/protein": "#58a6ff",
    "biological_process": "#a371f7", "molecular_function": "#7ee787",
    "cellular_component": "#7ee787", "pathway": "#f778ba",
    "anatomy": "#79c0ff", "effect/phenotype": "#ffa657", "other": "#6e7681",
}
TYPES = list(TYPE_COLORS)


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def build_html(action_nodes: list, action_links: list, dust: dict | None = None,
               height: int = 700, animate: bool = True) -> str:
    with open(LIB_PATH, encoding="utf-8") as f:
        lib = f.read()
    with open(THREE_PATH, encoding="utf-8") as f:
        three = f.read()

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

    dust = dust or {"xs": [], "ys": [], "zs": [], "ts": []}
    payload = {
        "nodes": action_nodes,
        "links": action_links,
        "dust": dust,
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
  <div style="color:#6e7681">galaxy = 128k PrimeKG nodes, one point-cloud ·
       drag to spin · scroll to zoom</div>
</div>
<div id="btns">
  <button onclick="replay()">▶ replay search</button>
  <button onclick="toggleGalaxy()">galaxy on/off</button>
</div>
<div id="status">agent idle</div>
<script>{three}</script>
<script>{lib}</script>
<script>
const DATA = {data};

// ---- action layer: only real nodes are graph nodes (pins, neighborhood) ----
const action = DATA.nodes.map(n => ({{ ...n, fx: n.x, fy: n.y, fz: n.z }}));

const Graph = ForceGraph3D()(document.getElementById('g'))
  .graphData({{ nodes: action, links: DATA.animate ? [] : DATA.links }})
  .backgroundColor('#0d1117')
  .nodeLabel(n => n.name + '  (' + n.type + ')')
  .nodeColor(n => n.color)
  .nodeVal(n => n.val)
  .nodeOpacity(0.95)
  .linkColor(l => l.color)
  .linkWidth(l => l.width)
  .linkLabel(l => l.relation)
  .linkOpacity(0.9)
  .pixelRatio(Math.min(window.devicePixelRatio, 1.5))
  .onNodeClick(n => Graph.cameraPosition({{ x: n.x, y: n.y, z: n.z + 300 }}, n, 600));
['charge', 'link', 'center', 'radial'].forEach(k => Graph.d3Force(k, null));

// ---- galaxy: ALL 128k nodes as ONE point cloud (single draw call) ----
const n = DATA.dust.xs.length;
const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
const palette = DATA.palette.map(h => {{
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
}});
const dim = 0.55;
for (let i = 0; i < n; i++) {{
  pos[i * 3] = DATA.dust.xs[i]; pos[i * 3 + 1] = DATA.dust.ys[i]; pos[i * 3 + 2] = DATA.dust.zs[i];
  const c = palette[DATA.dust.ts[i]] || palette[palette.length - 1];
  col[i * 3] = c[0] * dim; col[i * 3 + 1] = c[1] * dim; col[i * 3 + 2] = c[2] * dim;
}}
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
const cloud = new THREE.Points(geo, new THREE.PointsMaterial({{
  size: 1.6, vertexColors: true, sizeAttenuation: false,
  transparent: true, opacity: 0.8 }}));
Graph.scene().add(cloud);

let galaxyOn = true;
function toggleGalaxy() {{
  galaxyOn = !galaxyOn; cloud.visible = galaxyOn;
}}

// ---- the agent's live search, performed by the camera ----
const drug = action.find(x => x.role === 'pin_drug');
const dis  = action.find(x => x.role === 'pin_disease');
const status = document.getElementById('status');
const say = t => status.textContent = t;
function fly(node, dist, ms) {{
  Graph.cameraPosition({{ x: node.x + dist * .45, y: node.y + dist * .35, z: node.z + dist }},
                       node, ms);
}}
function replay() {{
  if (!drug && !dis) return;
  Graph.graphData({{ nodes: action, links: [] }});
  Graph.cameraPosition({{ x: 0, y: 300, z: 4600 }}, {{ x: 0, y: 0, z: 0 }}, 0);
  say('🛰 agent 1: trial found in registry');
  setTimeout(() => {{ if (drug) fly(drug, 620, 1700);
                      say('🛰 agent 2: scanning galaxy → drug district…'); }}, 900);
  setTimeout(() => {{ if (drug) Graph.refresh(); say('🛰 drug pinned'); }}, 2600);
  setTimeout(() => {{
    Graph.graphData({{ nodes: action, links: DATA.links.map(l => ({{ ...l }})) }});
    say(DATA.links.some(l => l.on_route)
        ? '🛰 route traced through the knowledge graph'
        : '🛰 no route exists — mechanistically uncharted');
  }}, 3300);
  setTimeout(() => {{
    if (drug && dis) fly({{ x: (drug.x + dis.x) / 2, y: (drug.y + dis.y) / 2,
                          z: (drug.z + dis.z) / 2 }}, 900, 1600);
    say('🛰 mapping complete');
  }}, 4300);
}}
if (DATA.animate) replay();
</script>
</body></html>"""
