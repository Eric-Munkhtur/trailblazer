import * as THREE from 'three';
import ForceGraph3D from '3d-force-graph';
import SpriteText from 'three-spritetext';
import { forceX, forceY, forceZ } from 'd3-force-3d';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

export const KINDS = {
  trial: { label: 'Trial', color: '#f5f7ff', size: 9 },
  intervention: { label: 'Intervention', color: '#38e1ff', size: 6 },
  disease: { label: 'Disease', color: '#ff4f9a', size: 6.5 },
  target: { label: 'Molecular target', color: '#ffb547', size: 4 },
  area: { label: 'Therapeutic area', color: '#a78bfa', size: 4.5 },
  meshChem: { label: 'MeSH · chemicals', color: '#4fc3e8', size: 3.2 },
  meshDisease: { label: 'MeSH · diseases', color: '#e06aa0', size: 3.2 },
};

export const LINK_COLORS = {
  TESTS: '#38e1ff',
  STUDIES: '#ff4f9a',
  TARGETS: '#ffb547',
  ASSOCIATED_WITH: '#ff8a5c',
  MAPS_TO: '#8fa3bf',
  BROADER: '#5d7190',
  IN_AREA: '#a78bfa',
  HYPOTHESIS: '#5cf2c0',
};

export const visualKind = (n) => (n.kind === 'mesh' ? (n.data.side === 'disease' ? 'meshDisease' : 'meshChem') : n.kind);

const GEOMETRIES = {
  trial: (s) => new THREE.IcosahedronGeometry(s, 1),
  intervention: (s) => new THREE.OctahedronGeometry(s),
  disease: (s) => new THREE.DodecahedronGeometry(s),
  target: (s) => new THREE.SphereGeometry(s, 20, 14),
  area: (s) => new THREE.TorusGeometry(s, s * 0.32, 10, 28),
  meshChem: (s) => new THREE.TetrahedronGeometry(s),
  meshDisease: (s) => new THREE.TetrahedronGeometry(s),
};

// Spatial frame: interventions and their chemical ontology extend to -x, diseases and their
// disease ontology to +x, molecular targets bridge them underneath, the trial sits above.
function anchor(n) {
  const k = visualKind(n);
  const lvl = Math.min(n.data.level ?? 6, 9);
  switch (k) {
    case 'trial': return { x: 0, y: 120, z: 0 };
    case 'intervention': return { x: -170, y: 40 };
    case 'disease': return { x: 170, y: 40 };
    case 'target': return { x: 0, y: -90 };
    case 'area': return { x: 230, y: -170 };
    // Ontology wings: broader concepts sit further out and curve back in depth.
    case 'meshChem': return { x: -(250 + (9 - lvl) * 16), y: 20 + (9 - lvl) * 8, z: -(9 - lvl) * 24 };
    case 'meshDisease': return { x: 250 + (9 - lvl) * 16, y: 20 + (9 - lvl) * 8, z: -(9 - lvl) * 24 };
    default: return {};
  }
}

const easeOutBack = (t) => 1 + 2.7 * Math.pow(t - 1, 3) + 1.7 * Math.pow(t - 1, 2);

export function createScene(el, { onNodeClick, onBackgroundClick }) {
  const state = {
    spatial: true,
    labels: true,
    hidden: new Set(),
    hlNodes: null,
    hlLinks: null,
    hover: null,
  };
  const data = { nodes: [], links: [] };
  const nodeById = new Map();
  const linkById = new Map();

  const graph = new ForceGraph3D(el, { controlType: 'orbit' })
    .backgroundColor('#04070d')
    .showNavInfo(false)
    .nodeId('id')
    .nodeThreeObject(makeNodeObject)
    .nodeLabel((n) => tooltip(n))
    .nodeVisibility((n) => !state.hidden.has(visualKind(n)))
    .linkVisibility((l) => !state.hidden.has(visualKind(l.source)) && !state.hidden.has(visualKind(l.target)))
    .linkColor((l) => linkColor(l))
    .linkOpacity(0.5)
    .linkWidth((l) => (isHl(l) ? 1.8 : l.bridge ? 1.4 : l.type === 'BROADER' ? 0.4 : 0.6))
    .linkCurvature((l) => (l.type === 'HYPOTHESIS' ? 0.35 : 0))
    .linkDirectionalParticles((l) => (isHl(l) ? 4 : l.bridge || l.type === 'TARGETS' || l.type === 'HYPOTHESIS' ? 2 : 0))
    .linkDirectionalParticleWidth((l) => (isHl(l) ? 2.4 : 1.4))
    .linkDirectionalParticleSpeed(0.006)
    .linkDirectionalParticleColor((l) => (l.bridge ? '#5cf2c0' : LINK_COLORS[l.type] || '#fff'))
    .linkLabel((l) => linkTooltip(l))
    .onNodeHover((n) => {
      el.style.cursor = n ? 'pointer' : 'grab';
      state.hover = n;
    })
    .onNodeClick((n) => onNodeClick?.(n))
    .onBackgroundClick(() => onBackgroundClick?.())
    .d3VelocityDecay(0.3)
    .warmupTicks(0)
    .cooldownTime(15000);

  graph.d3Force('charge').strength((n) => (n.kind === 'mesh' ? -18 : -140));
  graph.d3Force('link').distance((l) => (l.type === 'BROADER' ? 26 : l.type === 'MAPS_TO' ? 50 : 70)).strength((l) => (l.type === 'BROADER' ? 0.6 : 0.25));
  applyLayoutForces();

  const scene = graph.scene();
  scene.fog = new THREE.FogExp2(0x04070d, 0.0009);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.PointLight(0xffffff, 1.2, 0, 0);
  key.position.set(0, 400, 300);
  scene.add(key);
  scene.add(starfield());
  const frame = spatialFrame();
  scene.add(frame);

  const bloom = new UnrealBloomPass(new THREE.Vector2(el.clientWidth, el.clientHeight), 0.6, 0.45, 0.3);
  graph.postProcessingComposer().addPass(bloom);

  const controls = graph.controls();
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.enableDamping = true;
  let resumeTimer;
  let orbit = true;
  controls.addEventListener('start', () => {
    controls.autoRotate = false;
    clearTimeout(resumeTimer);
  });
  controls.addEventListener('end', () => {
    clearTimeout(resumeTimer);
    resumeTimer = setTimeout(() => (controls.autoRotate = orbit), 6000);
  });

  graph.cameraPosition({ x: 0, y: 160, z: 720 });

  window.addEventListener('resize', () => {
    graph.width(window.innerWidth).height(window.innerHeight);
    bloom.setSize(window.innerWidth, window.innerHeight);
  });

  // Assembly animation: new nodes grow in from zero scale.
  const tick = () => {
    const now = performance.now();
    for (const n of data.nodes) {
      const obj = n.__threeObj;
      if (!obj) continue;
      const t = Math.min(1, (now - (n.__born || 0)) / 900);
      const base = obj.userData.mesh;
      if (t < 1) obj.scale.setScalar(Math.max(0.001, easeOutBack(t)));
      else if (obj.scale.x !== 1) obj.scale.setScalar(1);
      if (base) base.rotation.y += n.kind === 'trial' ? 0.006 : 0.003;
      if (obj.userData.halo) obj.userData.halo.rotation.x += 0.004;
    }
    requestAnimationFrame(tick);
  };
  tick();

  function makeNodeObject(n) {
    const k = visualKind(n);
    const cfg = KINDS[k] || KINDS.target;
    const size = n.data.root ? 4 : cfg.size;
    const group = new THREE.Group();
    const color = new THREE.Color(cfg.color);
    const mat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: n.kind === 'mesh' ? 0.6 : n.kind === 'trial' ? 0.35 : 0.75,
      roughness: 0.35,
      metalness: 0.2,
      transparent: true,
      opacity: n.data.inferred ? 0.6 : 0.95,
    });
    const mesh = new THREE.Mesh(GEOMETRIES[k](size), mat);
    group.add(mesh);
    group.userData = { mesh, mats: [mat] };

    if (n.kind === 'trial' || n.data.isPrimary || (n.kind === 'target' && n.data.drugTarget && n.data.diseaseGene)) {
      const halo = new THREE.Mesh(
        new THREE.IcosahedronGeometry(size * 1.7, 1),
        new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.25 }),
      );
      group.add(halo);
      group.userData.halo = halo;
      group.userData.mats.push(halo.material);
    }

    const isMesh = n.kind === 'mesh';
    const label = new SpriteText(n.label, isMesh ? 3 : n.kind === 'trial' ? 6 : 4.2, isMesh ? '#a9b8cc' : '#eaf2ff');
    label.fontFace = 'Space Grotesk, sans-serif';
    label.fontWeight = n.kind === 'trial' || n.kind === 'intervention' || n.kind === 'disease' ? '600' : '400';
    label.material.depthWrite = false;
    label.material.transparent = true;
    label.position.y = (group.userData.halo ? size * 1.7 : size) + (isMesh ? 3 : 6);
    label.visible = state.labels;
    group.add(label);
    group.userData.label = label;
    group.userData.mats.push(label.material);
    group.userData.baseOpacity = group.userData.mats.map((m) => m.opacity);
    applyHighlightTo(n, group);
    return group;
  }

  function applyLayoutForces() {
    const s = (n) => (n.kind === 'trial' ? 0.4 : n.kind === 'mesh' ? 0.3 : 0.2);
    if (state.spatial) {
      graph.d3Force('x', forceX((n) => anchor(n).x ?? 0).strength((n) => (anchor(n).x == null ? 0 : s(n))));
      graph.d3Force('y', forceY((n) => anchor(n).y ?? 0).strength((n) => (anchor(n).y == null ? 0 : s(n) * 0.6)));
      graph.d3Force('z', forceZ((n) => anchor(n).z ?? 0).strength((n) => (n.kind === 'mesh' ? 0.15 : 0.03)));
    } else {
      graph.d3Force('x', null);
      graph.d3Force('y', null);
      graph.d3Force('z', null);
    }
  }

  function isHl(l) {
    return state.hlLinks?.has(l.id);
  }

  function linkColor(l) {
    if (state.hlLinks && !state.hlLinks.has(l.id)) return 'rgba(80,100,130,0.08)';
    if (l.bridge) return '#5cf2c0';
    return LINK_COLORS[l.type] || '#6b7a90';
  }

  function applyHighlightTo(n, obj = n.__threeObj) {
    if (!obj?.userData?.mats) return;
    const on = !state.hlNodes || state.hlNodes.has(n.id);
    obj.userData.mats.forEach((m, i) => (m.opacity = on ? obj.userData.baseOpacity[i] : obj.userData.baseOpacity[i] * 0.12));
  }

  function refreshStyles() {
    data.nodes.forEach((n) => applyHighlightTo(n));
    graph.linkColor(graph.linkColor()).linkWidth(graph.linkWidth()).linkDirectionalParticles(graph.linkDirectionalParticles());
  }

  function tooltip(n) {
    const k = KINDS[visualKind(n)];
    const extra = n.data.mechanism || n.data.name || n.data.title || n.data.treeNumbers?.join(', ') || '';
    return `<b style="color:${k.color}">${k.label}</b><br>${escape(n.label)}${extra ? `<br><span style="color:#8fa3bf">${escape(String(extra).slice(0, 140))}</span>` : ''}`;
  }

  function linkTooltip(l) {
    const parts = [l.type.replace(/_/g, ' ').toLowerCase()];
    if (l.label) parts.push(l.label);
    if (l.mechanism) parts.push(l.mechanism);
    if (l.rationale) parts.push(l.rationale);
    if (l.provenance) parts.push(`via ${l.provenance}`);
    if (l.source?.label) parts.unshift(`${l.source.label} → ${l.target.label}`);
    return parts.map(escape).join('<br>');
  }

  return {
    graph,
    data,
    nodeById,
    clear() {
      data.nodes.length = 0;
      data.links.length = 0;
      nodeById.clear();
      linkById.clear();
      state.hlNodes = state.hlLinks = null;
      graph.graphData({ nodes: [], links: [] });
    },
    add({ nodes = [], links = [] }) {
      const now = performance.now();
      let fresh = 0;
      for (const n of nodes) {
        const ex = nodeById.get(n.id);
        if (ex) {
          Object.assign(ex, { ...n, data: { ...ex.data, ...n.data } });
          continue;
        }
        const node = { ...n, __born: now + fresh * 40 };
        fresh++;
        const neighborLink = links.find((l) => l.source === n.id || l.target === n.id);
        const otherId = neighborLink && (neighborLink.source === n.id ? neighborLink.target : neighborLink.source);
        const other = nodeById.get(otherId);
        const a = anchor(node);
        const jitter = () => (Math.random() - 0.5) * 30;
        if (other?.x != null) Object.assign(node, { x: other.x + jitter(), y: other.y + jitter(), z: other.z + jitter() });
        else if (a.x != null) Object.assign(node, { x: a.x + jitter(), y: (a.y ?? 0) + jitter(), z: jitter() });
        if (node.kind === 'trial') Object.assign(node, { fx: 0, fy: 120, fz: 0 });
        nodeById.set(n.id, node);
        data.nodes.push(node);
      }
      for (const l of links) {
        const ex = linkById.get(l.id);
        if (ex) {
          Object.assign(ex, { ...l, source: ex.source, target: ex.target });
          continue;
        }
        if (!nodeById.has(l.source) || !nodeById.has(l.target)) continue;
        const link = { ...l };
        linkById.set(l.id, link);
        data.links.push(link);
      }
      graph.graphData({ nodes: data.nodes, links: data.links });
    },
    highlight(nodeIds, linkIds) {
      state.hlNodes = nodeIds ? new Set(nodeIds) : null;
      state.hlLinks = linkIds ? new Set(linkIds) : nodeIds ? new Set() : null;
      refreshStyles();
    },
    neighbors(id) {
      return data.links
        .filter((l) => l.source.id === id || l.target.id === id)
        .map((l) => ({ link: l, node: l.source.id === id ? l.target : l.source, out: l.source.id === id }));
    },
    linksBetween(ids) {
      const set = new Set(ids);
      return data.links.filter((l) => set.has(l.source.id) && set.has(l.target.id)).map((l) => l.id);
    },
    focus(n) {
      const dist = 260;
      const r = 1 + dist / Math.hypot(n.x || 1, n.y || 1, n.z || 1);
      graph.cameraPosition({ x: n.x * r, y: n.y * r, z: n.z * r }, n, 1200);
    },
    fit(filter) {
      const box = graph.getGraphBbox(filter);
      if (!box) return;
      const [cx, cy, cz] = ['x', 'y', 'z'].map((k) => (box[k][0] + box[k][1]) / 2);
      const [w, h, d] = ['x', 'y', 'z'].map((k) => Math.max(box[k][1] - box[k][0], 60));
      const cam = graph.camera();
      const tanHalf = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
      // Side panels cover ~780px of the viewport; fit into the uncovered middle.
      const usable = Math.max(0.45, (window.innerWidth - 780) / window.innerWidth);
      const distW = (w * 0.5) / (tanHalf * cam.aspect * usable);
      const distH = (h * 0.5) / (tanHalf * 0.8);
      const dist = Math.max(distW, distH) + d * 0.2;
      graph.cameraPosition({ x: cx, y: cy + dist * 0.18, z: cz + dist }, { x: cx, y: cy, z: cz }, 1100);
    },
    setHidden(kind, hidden) {
      hidden ? state.hidden.add(kind) : state.hidden.delete(kind);
      graph.nodeVisibility(graph.nodeVisibility()).linkVisibility(graph.linkVisibility());
    },
    setSpatial(on) {
      state.spatial = on;
      frame.visible = on;
      applyLayoutForces();
      graph.d3ReheatSimulation();
    },
    setLabels(on) {
      state.labels = on;
      data.nodes.forEach((n) => n.__threeObj?.userData.label && (n.__threeObj.userData.label.visible = on));
    },
    setOrbit(on) {
      orbit = on;
      controls.autoRotate = on;
    },
  };
}

function starfield() {
  const g = new THREE.BufferGeometry();
  const N = 1800;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 1400 + Math.random() * 1800;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th)], i * 3);
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: 0x6f8cb3, size: 2, sizeAttenuation: true, transparent: true, opacity: 0.55 }));
}

function spatialFrame() {
  const group = new THREE.Group();
  const grid = new THREE.GridHelper(1400, 28, 0x1b2a40, 0x0e1828);
  grid.position.y = -240;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  group.add(grid);
  const lanes = [
    ['INTERVENTION SPACE', -330, '#38e1ff'],
    ['DISEASE SPACE', 330, '#ff4f9a'],
    ['MOLECULAR TARGETS', 0, '#ffb547'],
  ];
  for (const [text, x, color] of lanes) {
    const s = new SpriteText(text, 9, color);
    s.fontFace = 'Space Grotesk, sans-serif';
    s.fontWeight = '700';
    s.material.opacity = 0.22;
    s.material.transparent = true;
    s.material.depthWrite = false;
    s.position.set(x, x === 0 ? -210 : 200, 0);
    group.add(s);
  }
  return group;
}

function escape(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}
