import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

export async function loadPrimeKG() {
  const get = async (file, kind) => {
    const res = await fetch(`/primekg/${file}`);
    if (!res.ok || (kind !== 'text' && kind !== 'json' && res.headers.get('content-type')?.includes('text/html'))) {
      throw new Error('PrimeKG has not been built yet. Run `npm run primekg`, then reload.');
    }
    return kind === 'json' ? res.json() : kind === 'text' ? res.text() : res.arrayBuffer();
  };
  const [meta, pos, types, names, edges, edgeRels] = await Promise.all([
    get('meta.json', 'json'),
    get('positions.f32').then((b) => new Float32Array(b)),
    get('types.u8').then((b) => new Uint8Array(b)),
    get('names.txt', 'text').then((t) => t.split('\n')),
    get('edges.u32').then((b) => new Uint32Array(b)),
    get('edges-rel.u8').then((b) => new Uint8Array(b)),
  ]);
  return { meta, pos, types, names, edges, edgeRels, N: types.length };
}

const NODE_VS = /* glsl */ `
  uniform float uTime, uSize, uPixelRatio, uDim, uReveal;
  uniform float uMask[16];
  uniform float uTypeAlpha[16];
  attribute vec3 aColor;
  attribute float aType, aFire, aLock, aSeed;
  varying vec3 vColor;
  varying float vAlpha, vGlow;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float since = uTime - aFire;
    float act = since >= 0.0 ? exp(-since * 1.4) : 0.0;
    float glow = max(act, aLock);
    float mask = uMask[int(aType + 0.5)];
    float reveal = smoothstep(aSeed * 2.0, aSeed * 2.0 + 0.5, uReveal);
    float twinkle = 0.85 + 0.15 * sin(uTime * 1.7 + aSeed * 60.0);
    float base = mix(0.3, 0.07, uDim) * twinkle * uTypeAlpha[int(aType + 0.5)];
    vAlpha = mask * reveal * min(1.0, base + glow);
    vGlow = glow;
    vColor = mix(aColor, vec3(1.0), act * 0.55);
    float size = uSize * (1.0 + glow * 2.6);
    gl_PointSize = min(72.0, mask * size * uPixelRatio * (420.0 / -mv.z));
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

const FIBER_VS = /* glsl */ `
  uniform float uMask[16];
  uniform float uDim;
  attribute vec3 aColor;
  attribute float aType;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vColor = aColor;
    vAlpha = uMask[int(aType + 0.5)] * mix(1.0, 0.35, uDim);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

const FIBER_FS = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor, uOpacity * vAlpha); }`;

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

class SignalLayer {
  constructor(capacity, shared, loop = false) {
    this.cap = capacity;
    this.next = 0;
    this.count = 0;
    const lg = new THREE.BufferGeometry();
    const attr = (geo, name, size, n) => {
      const a = new THREE.BufferAttribute(new Float32Array(n * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, a);
      return a;
    };
    this.l = {
      position: attr(lg, 'position', 3, capacity * 2),
      aS: attr(lg, 'aS', 1, capacity * 2),
      aT0: attr(lg, 'aT0', 1, capacity * 2),
      aDur: attr(lg, 'aDur', 1, capacity * 2),
      aStrength: attr(lg, 'aStrength', 1, capacity * 2),
      aColor: attr(lg, 'aColor', 3, capacity * 2),
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
      position: attr(pg, 'position', 3, capacity),
      aB: attr(pg, 'aB', 3, capacity),
      aT0: attr(pg, 'aT0', 1, capacity),
      aDur: attr(pg, 'aDur', 1, capacity),
      aStrength: attr(pg, 'aStrength', 1, capacity),
      aColor: attr(pg, 'aColor', 3, capacity),
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
  intervention: new THREE.Color('#38e1ff'),
  disease: new THREE.Color('#ff4f9a'),
  path: new THREE.Color('#5cf2c0'),
  focus: new THREE.Color('#ffffff'),
};

export function createBrain(container, kg, { onHover, onClick } = {}) {
  const { N, pos, types, meta } = kg;
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  const pixelRatio = Math.min(window.devicePixelRatio, 1.75);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x03050a, 1);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.className = 'labels-layer';
  container.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x03050a, 0.00035);
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, 8000);
  camera.position.set(0, 200, 900);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.35;
  controls.minDistance = 60;
  controls.maxDistance = 3000;

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 0.9, 0.5, 0.35);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const shared = { uTime: { value: 0 }, uPixelRatio: { value: pixelRatio }, uDim: { value: 0 }, uMask: { value: new Array(16).fill(1) } };
  const typeColors = meta.types.map((t) => new THREE.Color(t.color));

  // Neurons: every PrimeKG node.
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) typeColors[types[i]].toArray(colors, i * 3);
  geo.setAttribute('aColor', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('aType', new THREE.BufferAttribute(Float32Array.from(types), 1));
  const fire = new THREE.BufferAttribute(new Float32Array(N).fill(-1000), 1);
  const lock = new THREE.BufferAttribute(new Float32Array(N), 1);
  geo.setAttribute('aFire', fire);
  geo.setAttribute('aLock', lock);
  geo.setAttribute('aSeed', new THREE.BufferAttribute(Float32Array.from({ length: N }, Math.random), 1));
  // Dense types (genes, GO terms) would wash out under additive blending; scale them down.
  const typeAlpha = new Array(16).fill(1);
  meta.types.forEach((t, k) => (typeAlpha[k] = Math.min(1, Math.pow(3000 / t.count, 0.35))));
  const nodeUniforms = { ...shared, uSize: { value: 1.8 }, uReveal: { value: 0 }, uTypeAlpha: { value: typeAlpha } };
  const points = new THREE.Points(
    geo,
    new THREE.ShaderMaterial({ uniforms: nodeUniforms, vertexShader: NODE_VS, fragmentShader: NODE_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  scene.add(points);

  // Axons: a stratified sample of real PrimeKG edges as a faint fibre texture.
  const M = kg.edges.length / 2;
  const fpos = new Float32Array(M * 6);
  const fcol = new Float32Array(M * 6);
  const ftype = new Float32Array(M * 2);
  for (let e = 0; e < M; e++) {
    const a = kg.edges[e * 2];
    const b = kg.edges[e * 2 + 1];
    fpos.set(pos.subarray(a * 3, a * 3 + 3), e * 6);
    fpos.set(pos.subarray(b * 3, b * 3 + 3), e * 6 + 3);
    typeColors[types[a]].toArray(fcol, e * 6);
    typeColors[types[b]].toArray(fcol, e * 6 + 3);
    ftype[e * 2] = types[a];
    ftype[e * 2 + 1] = types[b];
  }
  const fgeo = new THREE.BufferGeometry();
  fgeo.setAttribute('position', new THREE.BufferAttribute(fpos, 3));
  fgeo.setAttribute('aColor', new THREE.BufferAttribute(fcol, 3));
  fgeo.setAttribute('aType', new THREE.BufferAttribute(ftype, 1));
  const fibers = new THREE.LineSegments(
    fgeo,
    new THREE.ShaderMaterial({ uniforms: { ...shared, uOpacity: { value: 0.0045 } }, vertexShader: FIBER_VS, fragmentShader: FIBER_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }),
  );
  scene.add(fibers);

  const signals = new SignalLayer(24000, shared);
  const paths = new SignalLayer(1200, shared, true);
  scene.add(signals.lines, signals.pulses, paths.lines, paths.pulses);

  // ---------- state ----------
  const state = { ambient: true, labels: true, fireDirty: false, lockDirty: false, cam: null, hover: -1, frame: 0 };
  const labelObjs = new Map();
  const clock = new THREE.Clock();
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
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
    labelRenderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);

  // ---------- picking ----------
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let pending = null;
  function pick(clientX, clientY) {
    const r = renderer.domElement.getBoundingClientRect();
    mouse.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(mouse, camera);
    raycaster.params.Points.threshold = Math.max(1.5, camera.position.distanceTo(controls.target) / 260);
    const mask = shared.uMask.value;
    for (const hit of raycaster.intersectObject(points)) {
      if (mask[types[hit.index]] > 0.5) return hit.index;
    }
    return -1;
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
  let lastAmbient = 0;
  const ambientColor = new THREE.Color();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1);
    shared.uTime.value += dt;
    const t = now();
    nodeUniforms.uReveal.value = Math.min(3, nodeUniforms.uReveal.value + dt * 0.9);

    if (state.ambient && t - lastAmbient > 0.05) {
      lastAmbient = t;
      const n = shared.uDim.value > 0.5 ? 1 : 3;
      for (let k = 0; k < n; k++) {
        const e = Math.floor(Math.random() * M);
        const a = kg.edges[e * 2];
        const b = kg.edges[e * 2 + 1];
        ambientColor.copy(typeColors[types[a]]);
        signals.add(P(a), P(b), t, 1.2 + Math.random(), ambientColor.toArray(), 0.28);
        fireAt(b, t + 1.4);
      }
    }

    if (state.cam) {
      const c = state.cam;
      const k = Math.min(1, (performance.now() - c.start) / c.ms);
      const ease = 1 - Math.pow(1 - k, 3);
      controls.target.lerpVectors(c.fromT, c.toT, ease);
      camera.position.lerpVectors(c.fromP, c.toP, ease);
      if (k >= 1) state.cam = null;
    }
    controls.update();

    if (pending) {
      const [x, y] = pending;
      pending = null;
      const i = pick(x, y);
      if (i !== state.hover) {
        state.hover = i;
        onHover?.(i, x, y);
      } else if (i >= 0) onHover?.(i, x, y);
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
    composer.render();
    labelRenderer.render(scene, camera);
    if (state.labels && labelObjs.size && ++state.frame % 6 === 0) declutter();
  });

  // Labels of nearby nodes (e.g. several drugs from one trial) stack vertically instead of overlapping.
  const LABEL_RANK = { seed: 0, focus: 1, path: 2, node: 3 };
  const proj = new THREE.Vector3();
  function declutter() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const placed = [];
    const objs = [...labelObjs.values()].sort((a, b) => a.userData.rank - b.userData.rank);
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
      for (const k of [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]) {
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

  return {
    get time() {
      return now();
    },
    /** Fire a set of neurons now (or at `delay` seconds). */
    fire(indices, delay = 0) {
      for (const i of indices) fireAt(i, now() + delay);
    },
    /** Send signals along edges [[u, v], ...], staggered so they read as a travelling wave. */
    propagate(edges, { side = 'focus', spread = 2.4, strength = 0.65 } = {}) {
      const col = SIDE_COLORS[side].toArray();
      const t = now();
      const step = edges.length > 1 ? spread / edges.length : 0;
      edges.forEach(([u, v], k) => {
        const t0 = t + k * step + Math.random() * 0.15;
        const dur = 0.7 + Math.random() * 0.5;
        signals.add(P(u), P(v), t0, dur, col, strength);
        fireAt(u, t0);
        fireAt(v, t0 + dur);
      });
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
    },
    setTypeVisible(t, on) {
      shared.uMask.value[t] = on ? 1 : 0;
    },
    setAmbient(on) {
      state.ambient = on;
    },
    setOrbit(on) {
      controls.autoRotate = on;
    },
    setLabels(on) {
      state.labels = on;
      for (const o of labelObjs.values()) o.visible = on;
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
    clearLabels() {
      for (const o of labelObjs.values()) {
        scene.remove(o);
        o.element.remove();
      }
      labelObjs.clear();
    },
    focus(indices, { distance, ms } = {}) {
      if (!indices.length) return;
      const box = new THREE.Box3();
      for (const i of indices) box.expandByPoint(new THREE.Vector3().fromArray(P(i)));
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      flyTo(center, distance ?? Math.max(220, size * 1.25), ms);
    },
    home() {
      flyTo(new THREE.Vector3(0, 0, 0), 920);
    },
    position: P,
  };
}
