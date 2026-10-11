import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

/** Loads the browser assets; `onProgress` receives the fraction of files loaded. */
export async function loadPrimeKG(onProgress) {
  const FILES = 7;
  let loaded = 0;
  const get = async (file, kind) => {
    const res = await fetch(`/primekg/${file}`);
    if (!res.ok || (kind !== 'text' && kind !== 'json' && res.headers.get('content-type')?.includes('text/html'))) {
      throw new Error('PrimeKG has not been built yet. Run `npm run primekg`, then reload.');
    }
    const body = await (kind === 'json' ? res.json() : kind === 'text' ? res.text() : res.arrayBuffer());
    onProgress?.(++loaded / FILES);
    return body;
  };
  const [meta, pos, types, degree, names, edges, edgeRels] = await Promise.all([
    get('meta.json', 'json'),
    get('positions.f32').then((b) => new Float32Array(b)),
    get('types.u8').then((b) => new Uint8Array(b)),
    get('degree.u32').then((b) => new Uint32Array(b)),
    get('names.txt', 'text').then((t) => t.split('\n')),
    get('edges.u32').then((b) => new Uint32Array(b)),
    get('edges-rel.u8').then((b) => new Uint8Array(b)),
  ]);
  return { meta, pos, types, degree, names, edges, edgeRels, N: types.length };
}

const NODE_VS = /* glsl */ `
  uniform float uTime, uSize, uPixelRatio, uDim, uReveal, uBase;
  uniform float uMask[16];
  attribute vec3 aColor;
  attribute float aType, aFire, aLock, aHi, aSize, aAlpha, aOrder;
  varying vec3 vColor;
  varying float vAlpha, vGlow;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float since = uTime - aFire;
    float act = since >= 0.0 ? exp(-since * 1.4) : 0.0;
    float hold = max(aLock, aHi);
    float glow = max(act, hold);
    float mask = uMask[int(aType + 0.5)];
    float reveal = smoothstep(aOrder * 2.0, aOrder * 2.0 + 0.4, uReveal);
    float base = uBase * mix(1.0, 0.3, uDim) * aAlpha;
    vAlpha = mask * reveal * min(1.0, base + glow);
    vGlow = glow;
    vColor = mix(aColor, vec3(1.0), act * 0.55);
    float size = uSize * aSize * (1.0 + act * 2.0 + hold * 0.9);
    // Slightly flatter than true perspective so close-up points do not balloon.
    gl_PointSize = min(80.0, mask * size * uPixelRatio * pow(420.0 / -mv.z, 0.8));
    gl_Position = projectionMatrix * mv;
  }`;

const NODE_FS = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha, vGlow;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float core = pow(1.0 - d * 2.0, 1.6);
    gl_FragColor = vec4(vColor * (0.55 + vGlow * 1.3), core * vAlpha);
  }`;

// Fibres fade with length (long edges between distant clusters would otherwise dominate) and
// with local density.
const FIBER_VS = /* glsl */ `
  uniform float uMask[16];
  uniform float uDim, uLength;
  attribute vec3 aColor;
  attribute float aType, aLen, aAlpha;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = mix(aColor, vec3(0.8), 0.45);
    float l = aLen / uLength;
    vAlpha = uMask[int(aType + 0.5)] * mix(1.0, 0.3, uDim) * aAlpha / (1.0 + l * l);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const FIBER_FS = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor, uOpacity * vAlpha); }`;

// A node's real edges, with pulses flowing outward from it.
const EGO_VS = /* glsl */ `
  attribute float aS, aPhase;
  attribute vec3 aColor;
  varying float vS, vPhase;
  varying vec3 vColor;
  void main() {
    vS = aS; vPhase = aPhase; vColor = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const EGO_FS = /* glsl */ `
  uniform float uTime, uOpacity;
  varying float vS, vPhase;
  varying vec3 vColor;
  void main() {
    float q = fract(uTime * 0.45 + vPhase - vS);
    float pulse = exp(-q * 10.0);
    gl_FragColor = vec4(vColor * (1.0 + pulse), uOpacity * (0.3 + 0.9 * pulse));
  }`;

// A signal is a comet travelling from A to B: bright head, fading tail, then the fibre fades out.
const SIGNAL_VS = /* glsl */ `
  attribute float aS, aT0, aDur, aStrength;
  attribute vec3 aColor;
  varying float vS, vT0, vDur, vStrength;
  varying vec3 vColor;
  void main() {
    vS = aS; vT0 = aT0; vDur = aDur; vStrength = aStrength; vColor = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const SIGNAL_FS = /* glsl */ `
  uniform float uTime, uLoop, uPeriod;
  varying float vS, vT0, vDur, vStrength;
  varying vec3 vColor;
  void main() {
    float p = (uTime - vT0) / vDur;
    if (p < 0.0) discard;
    float a;
    if (uLoop > 0.5) {
      float q = mod(p, uPeriod);
      float behind = q - vS;
      float head = behind >= 0.0 ? exp(-behind * 7.0) : 0.0;
      a = (0.22 + head * 1.1) * vStrength;
    } else {
      float behind = p - vS;
      if (behind < 0.0) discard;
      float trail = exp(-behind * 3.2);
      float head = exp(-behind * behind * 500.0);
      float fade = p > 1.0 ? exp(-(p - 1.0) * 1.1) : 1.0;
      a = (trail * 0.5 + head * 1.4) * fade * vStrength;
    }
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor * (1.0 + a), min(a, 1.0));
  }`;

const PULSE_VS = /* glsl */ `
  uniform float uTime, uPixelRatio, uLoop, uPeriod;
  attribute vec3 aB;
  attribute float aT0, aDur, aStrength;
  attribute vec3 aColor;
  varying vec3 vColor;
  varying float vA;
  void main() {
    float p = (uTime - aT0) / aDur;
    if (uLoop > 0.5) p = mod(p, uPeriod);
    float vis = (p >= 0.0 && p <= 1.0) ? 1.0 : 0.0;
    vec3 pos = mix(position, aB, clamp(p, 0.0, 1.0));
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    vColor = aColor;
    vA = vis * aStrength;
    gl_PointSize = vis * (5.0 + 5.0 * aStrength) * uPixelRatio * (420.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const PULSE_FS = /* glsl */ `
  varying vec3 vColor;
  varying float vA;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5 || vA <= 0.0) discard;
    float core = pow(1.0 - d * 2.0, 2.2);
    gl_FragColor = vec4(mix(vColor, vec3(1.0), 0.45) * 1.6, core * vA);
  }`;

// Reference frame around the layout: great circles on the sphere that holds 95% of nodes.
// Segments on the far side of the sphere (or, for axis lines, behind its centre) are dashed.
const WIRE_VS = /* glsl */ `
  uniform vec3 uCam;
  uniform float uR;
  attribute float aDist, aSurf;
  varying float vBack, vDist;
  void main() {
    vDist = aDist;
    vBack = step(dot(position, uCam), aSurf * uR * uR);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const WIRE_FS = /* glsl */ `
  uniform float uOpacity;
  varying float vBack, vDist;
  void main() {
    if (vBack > 0.5 && mod(vDist, 11.0) > 5.5) discard;
    gl_FragColor = vec4(vec3(0.8), uOpacity * (1.0 - 0.45 * vBack));
  }`;

const dynamicAttr = (geo, name, size, n) => {
  const a = new THREE.BufferAttribute(new Float32Array(n * size), size);
  a.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute(name, a);
  return a;
};

class SignalLayer {
  constructor(capacity, shared, loop = false) {
    this.cap = capacity;
    this.next = 0;
    this.count = 0;
    const lg = new THREE.BufferGeometry();
    this.l = {
      position: dynamicAttr(lg, 'position', 3, capacity * 2),
      aS: dynamicAttr(lg, 'aS', 1, capacity * 2),
      aT0: dynamicAttr(lg, 'aT0', 1, capacity * 2),
      aDur: dynamicAttr(lg, 'aDur', 1, capacity * 2),
      aStrength: dynamicAttr(lg, 'aStrength', 1, capacity * 2),
      aColor: dynamicAttr(lg, 'aColor', 3, capacity * 2),
    };
    this.l.aT0.array.fill(1e9);
    // Loop period in units of one hop, long enough for a 4-hop pathway plus a pause.
    this.period = 5.2;
    const uniforms = { uTime: shared.uTime, uPixelRatio: shared.uPixelRatio, uLoop: { value: loop ? 1 : 0 }, uPeriod: { value: this.period } };
    this.lines = new THREE.LineSegments(
      lg,
      new THREE.ShaderMaterial({ uniforms, vertexShader: SIGNAL_VS, fragmentShader: SIGNAL_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
    );
    const pg = new THREE.BufferGeometry();
    this.p = {
      position: dynamicAttr(pg, 'position', 3, capacity),
      aB: dynamicAttr(pg, 'aB', 3, capacity),
      aT0: dynamicAttr(pg, 'aT0', 1, capacity),
      aDur: dynamicAttr(pg, 'aDur', 1, capacity),
      aStrength: dynamicAttr(pg, 'aStrength', 1, capacity),
      aColor: dynamicAttr(pg, 'aColor', 3, capacity),
    };
    this.p.aT0.array.fill(1e9);
    this.pulses = new THREE.Points(
      pg,
      new THREE.ShaderMaterial({ uniforms, vertexShader: PULSE_VS, fragmentShader: PULSE_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
    );
    this.lines.frustumCulled = this.pulses.frustumCulled = false;
    this.dirty = false;
  }

  add(a, b, t0, dur, color, strength) {
    const k = this.next;
    this.next = (this.next + 1) % this.cap;
    this.count = Math.min(this.cap, this.count + 1);
    const L = this.l;
    L.position.array.set(a, k * 6);
    L.position.array.set(b, k * 6 + 3);
    L.aS.array[k * 2] = 0;
    L.aS.array[k * 2 + 1] = 1;
    for (const j of [0, 1]) {
      L.aT0.array[k * 2 + j] = t0;
      L.aDur.array[k * 2 + j] = dur;
      L.aStrength.array[k * 2 + j] = strength;
      L.aColor.array.set(color, (k * 2 + j) * 3);
    }
    const P = this.p;
    P.position.array.set(a, k * 3);
    P.aB.array.set(b, k * 3);
    P.aT0.array[k] = t0;
    P.aDur.array[k] = dur;
    P.aStrength.array[k] = strength;
    P.aColor.array.set(color, k * 3);
    this.dirty = true;
  }

  clear() {
    this.l.aT0.array.fill(1e9);
    this.p.aT0.array.fill(1e9);
    this.next = this.count = 0;
    this.dirty = true;
  }

  flush() {
    if (!this.dirty) return;
    for (const a of Object.values(this.l)) a.needsUpdate = true;
    for (const a of Object.values(this.p)) a.needsUpdate = true;
    this.dirty = false;
  }
}

const SIDE_COLORS = {
  intervention: new THREE.Color('#8c84ff'),
  disease: new THREE.Color('#ff8b3d'),
  A: new THREE.Color('#8c84ff'),
  B: new THREE.Color('#ff8b3d'),
  path: new THREE.Color('#ff4a2e'),
};

function wireGeometry(R) {
  const pos = [];
  const dist = [];
  const surf = [];
  const ring = (normal, segs = 192) => {
    const n = normal.clone().normalize();
    const u = new THREE.Vector3(Math.abs(n.x) < 0.9 ? 1 : 0, Math.abs(n.x) < 0.9 ? 0 : 1, 0).cross(n).normalize();
    const v = n.clone().cross(u);
    const at = (k) => u.clone().multiplyScalar(Math.cos((k / segs) * Math.PI * 2) * R).addScaledVector(v, Math.sin((k / segs) * Math.PI * 2) * R);
    let d = 0;
    for (let k = 0; k < segs; k++) {
      const a = at(k);
      const b = at(k + 1);
      pos.push(...a.toArray(), ...b.toArray());
      dist.push(d, (d += a.distanceTo(b)));
      surf.push(1, 1);
    }
  };
  const axis = (dir, len) => {
    const a = dir.clone().normalize().multiplyScalar(-len);
    const b = a.clone().negate();
    pos.push(...a.toArray(), ...b.toArray());
    dist.push(0, a.distanceTo(b));
    surf.push(0, 0);
  };
  const tilted = new THREE.Vector3(0, Math.cos(0.42), Math.sin(0.42));
  ring(new THREE.Vector3(0, 1, 0));
  ring(new THREE.Vector3(0, 0, 1));
  ring(new THREE.Vector3(Math.sin(1.05), 0, Math.cos(1.05)));
  ring(tilted);
  axis(new THREE.Vector3(0, 1, 0), R * 1.08);
  axis(tilted, R * 1.05);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('aDist', new THREE.Float32BufferAttribute(dist, 1));
  geo.setAttribute('aSurf', new THREE.Float32BufferAttribute(surf, 1));
  return geo;
}

// A coarse 3D grid over the layout. Points and fibres in crowded cells are drawn fainter, so
// clusters keep their texture under additive blending instead of saturating to white.
function densityGrid(pos, N, cells = 48) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < N * 3; k++) {
    const d = k % 3;
    if (pos[k] < lo[d]) lo[d] = pos[k];
    if (pos[k] > hi[d]) hi[d] = pos[k];
  }
  const axis = (v, d) => Math.min(cells - 1, Math.max(0, Math.floor(((v - lo[d]) / (hi[d] - lo[d] || 1)) * cells)));
  const cellAt = (x, y, z) => (axis(x, 0) * cells + axis(y, 1)) * cells + axis(z, 2);
  const cell = new Uint32Array(N);
  const counts = new Uint32Array(cells ** 3);
  for (let i = 0; i < N; i++) counts[(cell[i] = cellAt(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]))]++;
  return { cells, cell, counts, cellAt };
}

const fade = (ref, count, exp) => Math.min(1, Math.pow(ref / Math.max(1, count), exp));
const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const FIBER_OPACITY = 0.03;
const EGO_MAX = 4000;
const HINTS = 24;
const HINT_MIN = 1.45;

export function createBrain(container, kg, { onHover, onClick } = {}) {
  const { N, pos, types, degree, meta } = kg;
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  const pixelRatio = Math.min(window.devicePixelRatio, 1.75);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x000000, 1);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.className = 'labels-layer';
  container.appendChild(labelRenderer.domElement);

  const HOME = 1150;
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x000000, 0.0003);
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 8000);
  camera.position.set(0, 0.25 * HOME, 0.97 * HOME);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.35;
  controls.minDistance = 30;
  controls.maxDistance = 3000;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 0.38, 0.4, 0.42);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const shared = { uTime: { value: 0 }, uPixelRatio: { value: pixelRatio }, uDim: { value: 0 }, uMask: { value: new Array(16).fill(1) } };
  const typeColors = meta.types.map((t) => new THREE.Color(t.color));
  const egoColors = typeColors.map((c) => c.clone().lerp(new THREE.Color(1, 1, 1), 0.4));
  const grid = densityGrid(pos, N);

  // Neurons: every PrimeKG node, sized by its real number of connections.
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const colors = new Float32Array(N * 3);
  const size = new Float32Array(N);
  const alpha = new Float32Array(N);
  const order = new Float32Array(N);
  let maxLog = 1;
  for (let i = 0; i < N; i++) maxLog = Math.max(maxLog, Math.log10(1 + degree[i]));
  for (let i = 0; i < N; i++) {
    typeColors[types[i]].toArray(colors, i * 3);
    const l = Math.log10(1 + degree[i]) / maxLog;
    size[i] = 0.55 + 1.9 * l;
    alpha[i] = fade(6, grid.counts[grid.cell[i]], 0.6) * (0.55 + 0.45 * l);
    // The intro reveals hubs first, then their neighbourhoods fill in.
    order[i] = (1 - l) * 0.85 + Math.random() * 0.15;
  }
  geo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('aType', new THREE.BufferAttribute(Float32Array.from(types), 1));
  geo.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
  geo.setAttribute('aAlpha', new THREE.BufferAttribute(alpha, 1));
  geo.setAttribute('aOrder', new THREE.BufferAttribute(order, 1));
  const fire = new THREE.BufferAttribute(new Float32Array(N).fill(-1000), 1);
  const lock = new THREE.BufferAttribute(new Float32Array(N), 1);
  const hi = new THREE.BufferAttribute(new Float32Array(N), 1);
  geo.setAttribute('aFire', fire);
  geo.setAttribute('aLock', lock);
  geo.setAttribute('aHi', hi);
  const nodeUniforms = { ...shared, uSize: { value: 1.8 }, uReveal: { value: 0 }, uBase: { value: 1 } };
  const points = new THREE.Points(
    geo,
    new THREE.ShaderMaterial({ uniforms: nodeUniforms, vertexShader: NODE_VS, fragmentShader: NODE_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  scene.add(points);

  // Axons: a stratified sample of real PrimeKG edges.
  const M = kg.edges.length / 2;
  const fpos = new Float32Array(M * 6);
  const fcol = new Float32Array(M * 6);
  const ftype = new Float32Array(M * 2);
  const flen = new Float32Array(M * 2);
  const falpha = new Float32Array(M * 2);
  const fcell = new Uint32Array(M);
  const edgeCounts = new Uint32Array(grid.counts.length);
  for (let e = 0; e < M; e++) {
    const a = kg.edges[e * 2];
    const b = kg.edges[e * 2 + 1];
    fpos.set(pos.subarray(a * 3, a * 3 + 3), e * 6);
    fpos.set(pos.subarray(b * 3, b * 3 + 3), e * 6 + 3);
    typeColors[types[a]].toArray(fcol, e * 6);
    typeColors[types[b]].toArray(fcol, e * 6 + 3);
    ftype[e * 2] = types[a];
    ftype[e * 2 + 1] = types[b];
    const dx = pos[b * 3] - pos[a * 3], dy = pos[b * 3 + 1] - pos[a * 3 + 1], dz = pos[b * 3 + 2] - pos[a * 3 + 2];
    flen[e * 2] = flen[e * 2 + 1] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    edgeCounts[(fcell[e] = grid.cellAt(pos[a * 3] + dx / 2, pos[a * 3 + 1] + dy / 2, pos[a * 3 + 2] + dz / 2))]++;
  }
  for (let e = 0; e < M; e++) falpha[e * 2] = falpha[e * 2 + 1] = fade(4, edgeCounts[fcell[e]], 0.7);
  const fgeo = new THREE.BufferGeometry();
  fgeo.setAttribute('position', new THREE.BufferAttribute(fpos, 3));
  fgeo.setAttribute('aColor', new THREE.BufferAttribute(fcol, 3));
  fgeo.setAttribute('aType', new THREE.BufferAttribute(ftype, 1));
  fgeo.setAttribute('aLen', new THREE.BufferAttribute(flen, 1));
  fgeo.setAttribute('aAlpha', new THREE.BufferAttribute(falpha, 1));
  const fiberUniforms = { ...shared, uOpacity: { value: FIBER_OPACITY }, uLength: { value: 70 } };
  const fibers = new THREE.LineSegments(
    fgeo,
    new THREE.ShaderMaterial({ uniforms: fiberUniforms, vertexShader: FIBER_VS, fragmentShader: FIBER_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  scene.add(fibers);

  // Ego network: the real edges of the hovered or selected node.
  const egoGeo = new THREE.BufferGeometry();
  const ego = {
    position: dynamicAttr(egoGeo, 'position', 3, EGO_MAX * 2),
    aS: dynamicAttr(egoGeo, 'aS', 1, EGO_MAX * 2),
    aPhase: dynamicAttr(egoGeo, 'aPhase', 1, EGO_MAX * 2),
    aColor: dynamicAttr(egoGeo, 'aColor', 3, EGO_MAX * 2),
  };
  egoGeo.setDrawRange(0, 0);
  const egoUniforms = { uTime: shared.uTime, uOpacity: { value: 0.5 } };
  const egoLines = new THREE.LineSegments(
    egoGeo,
    new THREE.ShaderMaterial({ uniforms: egoUniforms, vertexShader: EGO_VS, fragmentShader: EGO_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  egoLines.frustumCulled = false;
  scene.add(egoLines);
  let egoNodes = [];

  const signals = new SignalLayer(24000, shared);
  const paths = new SignalLayer(1200, shared, true);
  scene.add(signals.lines, signals.pulses, paths.lines, paths.pulses);

  // Reference sphere, shown in the overview and faded out up close. Its outline is the sphere's
  // silhouette, recomputed for the camera every frame.
  const R = meta.layout?.radius ?? 420;
  const wireUniforms = { uCam: { value: new THREE.Vector3() }, uR: { value: R }, uOpacity: { value: 0 } };
  const wire = new THREE.LineSegments(
    wireGeometry(R),
    new THREE.ShaderMaterial({ uniforms: wireUniforms, vertexShader: WIRE_VS, fragmentShader: WIRE_FS, transparent: true, depthWrite: false }),
  );
  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(Array.from({ length: 192 }, (_, k) => new THREE.Vector3(Math.cos((k / 192) * Math.PI * 2), Math.sin((k / 192) * Math.PI * 2), 0))),
    new THREE.LineBasicMaterial({ color: 0xcfcfcf, transparent: true, opacity: 0, depthWrite: false, fog: false }),
  );
  wire.frustumCulled = outline.frustumCulled = false;
  scene.add(wire, outline);
  const axisLabels = [
    ['+Y', [0, 1.13, 0]],
    ['−Y', [0, -1.13, 0]],
    ['+X', [1.05, 0, 0]],
    ['−X', [-1.05, 0, 0]],
    ['+Z', [0, 0, 1.05]],
    ['−Z', [0, 0, -1.05]],
    ['O', [0.035, -0.04, 0]],
    [`R=${R}`, [0.74, 0, 0.74]],
  ].map(([text, [x, y, z]]) => {
    const el = document.createElement('div');
    el.className = 'axis-label';
    el.textContent = text;
    const o = new CSS2DObject(el);
    o.position.set(x * R, y * R, z * R);
    scene.add(o);
    return o;
  });

  const reticle = new CSS2DObject(Object.assign(document.createElement('div'), { className: 'reticle' }));
  reticle.visible = false;
  scene.add(reticle);

  // The page overlays panels on the canvas. A view offset centres the scene in the uncovered
  // stage, and the home distance fits the reference sphere inside it.
  const inset = { left: 0, right: 0, top: 0, bottom: 0 };
  let fitAngle = THREE.MathUtils.degToRad(20);
  let homeDist = HOME;
  /** Camera distance at which a ball of `radius` fills the uncovered stage. */
  const fitDist = (radius) => radius / Math.sin(fitAngle);
  function applyInset() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.setViewOffset(w, h, (inset.right - inset.left) / 2, (inset.bottom - inset.top) / 2, w, h);
    const stage = Math.min(w - inset.left - inset.right, h - inset.top - inset.bottom);
    const t = ((stage * 0.43) / (h / 2)) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    fitAngle = Math.atan(Math.max(0.05, t));
    homeDist = fitDist(R * 1.04);
  }

  // ---------- state ----------
  const state = { labels: true, fireDirty: false, lockDirty: false, cam: null, hover: -1, frame: 0 };
  const labelObjs = new Map();
  const timer = new THREE.Timer();
  timer.connect(document);
  const now = () => shared.uTime.value;
  const P = (i) => pos.subarray(i * 3, i * 3 + 3);

  function fireAt(i, t) {
    const cur = fire.array[i];
    if (cur < now() - 0.4 || t < cur) fire.array[i] = t;
    state.fireDirty = true;
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    applyInset();
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
    labelRenderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);

  // ---------- zoom-dependent labels ----------
  // Prominence is a node's degree percentile within its own type (anatomy terms carry thousands
  // of expression edges and would otherwise take every label). Candidates are the most prominent
  // nodes of each density cell; one is named when it is prominent and close to the camera.
  const prominence = new Float32Array(N);
  meta.types.forEach((_, k) => {
    const members = [];
    for (let i = 0; i < N; i++) if (types[i] === k) members.push(i);
    members.sort((a, b) => degree[a] - degree[b]);
    members.forEach((i, r) => (prominence[i] = members.length > 1 ? r / (members.length - 1) : 1));
  });
  const hintCandidates = (() => {
    const best = new Map();
    for (let i = 0; i < N; i++) {
      if (degree[i] < 5) continue;
      const c = grid.cell[i];
      const list = best.get(c);
      if (!list) best.set(c, [i]);
      else if (list.length < 2) list.push(i);
      else if (prominence[i] > prominence[list[1]]) list[1] = i;
      if (list?.length === 2 && prominence[list[1]] > prominence[list[0]]) list.reverse();
    }
    return Uint32Array.from([...best.values()].flat());
  })();
  const hintPool = Array.from({ length: HINTS }, () => {
    const el = document.createElement('div');
    el.className = 'nlabel hint';
    const o = new CSS2DObject(el);
    o.visible = false;
    o.userData = { rank: 4, node: -1 };
    scene.add(o);
    return o;
  });
  const hv = new THREE.Vector3();
  function updateHints() {
    const scored = [];
    if (state.labels) {
      for (let k = 0; k < hintCandidates.length; k++) {
        const i = hintCandidates[k];
        if (labelObjs.has(i) || shared.uMask.value[types[i]] < 0.5) continue;
        hv.fromArray(pos, i * 3);
        const s = 3.5 * prominence[i] - 2.6 * Math.log10(Math.max(1, hv.distanceTo(camera.position)) / 200);
        if (s < HINT_MIN) continue;
        hv.project(camera);
        if (hv.z > 1 || Math.abs(hv.x) > 0.92 || Math.abs(hv.y) > 0.92) continue;
        scored.push([s, i]);
      }
      scored.sort((a, b) => b[0] - a[0]);
    }
    // At most a quarter of the names go to any one type.
    const perType = new Map();
    const shown = scored.filter(([, i]) => {
      const n = perType.get(types[i]) || 0;
      perType.set(types[i], n + 1);
      return n < HINTS / 4;
    });
    hintPool.forEach((o, k) => {
      const i = shown[k]?.[1] ?? -1;
      o.visible = i >= 0;
      if (i < 0 || o.userData.node === i) return;
      o.userData.node = i;
      o.userData.w = 0;
      o.element.textContent = kg.names[i];
      o.element.style.setProperty('--c', meta.types[types[i]].color);
      o.position.fromArray(pos, i * 3);
    });
  }

  // ---------- picking ----------
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let pending = null;
  function pick(clientX, clientY) {
    const r = renderer.domElement.getBoundingClientRect();
    mouse.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(mouse, camera);
    raycaster.params.Points.threshold = Math.max(1, camera.position.distanceTo(controls.target) / 300);
    const mask = shared.uMask.value;
    // Among the points under the cursor, prefer the best-connected one near the front.
    let best = -1;
    let bestScore = -Infinity;
    const hits = raycaster.intersectObject(points);
    const front = hits[0]?.distance ?? 0;
    for (const hit of hits) {
      if (hit.distance > front * 1.15 + 20) break;
      if (mask[types[hit.index]] < 0.5) continue;
      const s = Math.log10(1 + degree[hit.index]) - hit.distanceToRay / raycaster.params.Points.threshold;
      if (s > bestScore) {
        bestScore = s;
        best = hit.index;
      }
    }
    return best;
  }
  renderer.domElement.addEventListener('pointermove', (e) => {
    pending = [e.clientX, e.clientY];
  });
  renderer.domElement.addEventListener('pointerleave', () => {
    pending = null;
    state.hover = -1;
    onHover?.(-1);
  });
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    onClick?.(pick(e.clientX, e.clientY));
  });
  controls.addEventListener('start', () => {
    controls.autoRotate = false;
    state.cam = null;
  });

  // ---------- loop ----------
  renderer.setAnimationLoop(() => {
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.1);
    shared.uTime.value += dt;
    nodeUniforms.uReveal.value = Math.min(3, nodeUniforms.uReveal.value + dt * 0.9);

    if (state.cam) {
      const c = state.cam;
      const k = Math.min(1, (performance.now() - c.start) / c.ms);
      const ease = 1 - Math.pow(1 - k, 3);
      controls.target.lerpVectors(c.fromT, c.toT, ease);
      camera.position.lerpVectors(c.fromP, c.toP, ease);
      if (k >= 1) state.cam = null;
    }
    controls.update();
    // Fewer fibres overlap per pixel up close, so each one is drawn stronger.
    const dist = camera.position.distanceTo(controls.target);
    const zoom = homeDist / Math.max(1, dist);
    fiberUniforms.uOpacity.value = FIBER_OPACITY * Math.min(6, Math.max(1, Math.pow(zoom, 0.9)));

    const camR = camera.position.length();
    const wireA = smooth(R * 1.5, R * 2.3, camR);
    wire.visible = outline.visible = wireA > 0.01;
    if (wire.visible) {
      wireUniforms.uOpacity.value = 0.3 * wireA;
      wireUniforms.uCam.value.copy(camera.position);
      outline.material.opacity = 0.45 * wireA;
      outline.position.copy(camera.position).multiplyScalar((R * R) / (camR * camR));
      outline.scale.setScalar(R * Math.sqrt(1 - (R * R) / (camR * camR)));
      outline.lookAt(camera.position);
    }
    for (const o of axisLabels) {
      o.visible = state.labels && wireA > 0.05;
      o.element.style.opacity = wireA;
    }

    if (pending) {
      const [x, y] = pending;
      pending = null;
      const i = pick(x, y);
      if (i !== state.hover) state.hover = i;
      onHover?.(i, x, y);
    }

    if (state.fireDirty) {
      fire.needsUpdate = true;
      state.fireDirty = false;
    }
    if (state.lockDirty) {
      lock.needsUpdate = true;
      state.lockDirty = false;
    }
    signals.flush();
    paths.flush();
    state.frame++;
    if (state.frame % 8 === 0) updateHints();
    composer.render();
    labelRenderer.render(scene, camera);
    if (state.labels && state.frame % 4 === 0) declutter();
  });

  // Named labels (seeds, focus, pathway nodes) stack vertically instead of overlapping; zoom
  // hints never move and simply hide when they would collide with anything placed before them.
  const LABEL_RANK = { seed: 0, focus: 1, path: 2, node: 3 };
  const proj = new THREE.Vector3();
  function declutter() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const placed = [];
    const objs = [...labelObjs.values(), ...hintPool.filter((o) => o.visible)].sort((a, b) => a.userData.rank - b.userData.rank);
    for (const o of objs) {
      const el = o.element;
      proj.copy(o.position).project(camera);
      if (proj.z > 1) continue;
      if (!o.userData.w) {
        o.userData.w = el.offsetWidth;
        o.userData.h = el.offsetHeight;
        if (!o.userData.w) continue;
      }
      const lw = o.userData.w + 6;
      const lh = o.userData.h + 3;
      const lift = lh / 2 + 7;
      const x = ((proj.x + 1) / 2) * w - lw / 2;
      const y = ((1 - proj.y) / 2) * h - lh / 2 - lift;
      let dy = null;
      for (const k of o.userData.rank === 4 ? [0] : [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]) {
        const top = y + k * lh;
        if (!placed.some((r) => x < r.x + r.w && x + lw > r.x && top < r.y + r.h && top + lh > r.y)) {
          dy = k * lh;
          break;
        }
      }
      if (dy === null) {
        el.style.opacity = '0';
        continue;
      }
      placed.push({ x, y: y + dy, w: lw, h: lh });
      el.style.opacity = '';
      el.style.marginTop = `${dy - lift}px`;
    }
  }

  function flyTo(target, distance, ms = 1400) {
    const dir = camera.position.clone().sub(controls.target).normalize();
    state.cam = {
      start: performance.now(),
      ms,
      fromT: controls.target.clone(),
      toT: target.clone(),
      fromP: camera.position.clone(),
      toP: target.clone().add(dir.multiplyScalar(distance)),
    };
  }

  function removeLabel(i) {
    const o = labelObjs.get(i);
    if (!o) return;
    scene.remove(o);
    o.element.remove();
    labelObjs.delete(i);
  }

  return {
    get time() {
      return now();
    },
    /** Fire neurons now, after `delay` seconds, or each at its own delay (array). */
    fire(indices, delay = 0) {
      indices.forEach((i, k) => fireAt(i, now() + (Array.isArray(delay) ? delay[k] : delay)));
    },
    /** Replay one hop of a search wavefront: signals depart and arrive at their cost-derived times. */
    wave({ side, hop, edges, nodes, times }) {
      const col = SIDE_COLORS[side].toArray();
      const t = now();
      const strength = hop === 1 ? 0.85 : 0.45;
      for (const [u, v, , tu, tv] of edges) signals.add(P(u), P(v), t + tu, Math.max(0.2, tv - tu), col, strength);
      nodes.forEach((v, k) => fireAt(v, t + times[k]));
    },
    /** Show node i's real edges to `nbrs`, highlighting the neighbours. */
    showEgo(i, nbrs) {
      for (const j of egoNodes) hi.array[j] = 0;
      const n = Math.min(EGO_MAX, nbrs.length);
      const c0 = egoColors[types[i]];
      for (let k = 0; k < n; k++) {
        const j = nbrs[k];
        ego.position.array.set(P(i), k * 6);
        ego.position.array.set(P(j), k * 6 + 3);
        ego.aS.array[k * 2] = 0;
        ego.aS.array[k * 2 + 1] = 1;
        ego.aPhase.array[k * 2] = ego.aPhase.array[k * 2 + 1] = Math.random();
        c0.toArray(ego.aColor.array, k * 6);
        egoColors[types[j]].toArray(ego.aColor.array, k * 6 + 3);
        hi.array[j] = 0.5;
      }
      hi.array[i] = 1;
      egoNodes = [i, ...nbrs.slice(0, n)];
      egoUniforms.uOpacity.value = Math.min(0.6, Math.max(0.12, 6 / Math.sqrt(n)));
      for (const a of Object.values(ego)) a.needsUpdate = true;
      egoGeo.setDrawRange(0, n * 2);
      hi.needsUpdate = true;
    },
    /** Draw the callout box around node i, or remove it (i < 0). */
    mark(i) {
      reticle.visible = i >= 0;
      if (i >= 0) reticle.position.fromArray(P(i));
    },
    clearEgo() {
      for (const j of egoNodes) hi.array[j] = 0;
      egoNodes = [];
      egoGeo.setDrawRange(0, 0);
      hi.needsUpdate = true;
    },
    lockPaths(list, selected = null) {
      paths.clear();
      lock.array.fill(0);
      const t = now();
      const hop = 0.55;
      const col = SIDE_COLORS.path.toArray();
      list.forEach((p, j) => {
        const on = !selected || p.id === selected;
        const s = selected ? (on ? 1.0 : 0.1) : Math.max(0.35, p.strength * 1.6);
        const offset = selected ? 0 : j * 0.37;
        // A selected pathway carries a continuous spike train rather than a single pulse per loop.
        const train = selected && on ? 4 : 1;
        p.nodes.forEach((n, k) => {
          lock.array[n] = Math.max(lock.array[n], on ? 1 : 0.2);
          if (!k) return;
          for (let r = 0; r < train; r++) {
            paths.add(P(p.nodes[k - 1]), P(n), t + offset + (k - 1) * hop + (r * paths.period * hop) / train, hop, col, s);
          }
        });
      });
      state.lockDirty = true;
    },
    clearPaths() {
      paths.clear();
      lock.array.fill(0);
      state.lockDirty = true;
    },
    clearSignals() {
      signals.clear();
      fire.array.fill(-1000);
      state.fireDirty = true;
    },
    setDim(on) {
      shared.uDim.value = on ? 1 : 0;
      labelRenderer.domElement.classList.toggle('dim', on);
    },
    setTypeVisible(t, on) {
      shared.uMask.value[t] = on ? 1 : 0;
    },
    setOrbit(on) {
      controls.autoRotate = on;
    },
    setLabels(on) {
      state.labels = on;
      for (const o of labelObjs.values()) o.visible = on;
      updateHints();
    },
    label(i, text, kind = 'node') {
      let o = labelObjs.get(i);
      if (!o) {
        const el = document.createElement('div');
        el.className = `nlabel ${kind}`;
        el.style.setProperty('--c', meta.types[types[i]].color);
        el.textContent = text;
        o = new CSS2DObject(el);
        o.position.fromArray(P(i));
        o.visible = state.labels;
        scene.add(o);
        labelObjs.set(i, o);
      } else {
        const rank = LABEL_RANK[kind.split(' ')[0]] ?? 3;
        if (rank > o.userData.rank) return o;
        o.element.className = `nlabel ${kind}`;
        o.userData.w = 0;
      }
      o.userData.rank = LABEL_RANK[kind.split(' ')[0]] ?? 3;
      return o;
    },
    /** Remove all labels, or only those of one kind (e.g. 'node'). */
    clearLabels(kind) {
      for (const [i, o] of [...labelObjs]) if (!kind || o.userData.rank === LABEL_RANK[kind]) removeLabel(i);
    },
    focus(indices, { distance, ms } = {}) {
      if (!indices.length) return;
      const box = new THREE.Box3();
      for (const i of indices) box.expandByPoint(new THREE.Vector3().fromArray(P(i)));
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      flyTo(center, distance ?? Math.max(160, fitDist(size * 0.55)), ms);
    },
    home(distance = homeDist) {
      flyTo(new THREE.Vector3(0, 0, 0), distance);
    },
    /** Pixels of the canvas covered by page chrome on each side; `place` also moves the camera home. */
    setInsets(px, { place = false } = {}) {
      Object.assign(inset, px);
      applyInset();
      camera.updateProjectionMatrix();
      if (place) camera.position.sub(controls.target).setLength(homeDist).add(controls.target);
    },
    position: P,
  };
}
