// ForceAtlas2 (Jacomy, Venturini, Heymann & Bastian, PLoS ONE 2014) in three dimensions.
// Repulsion between all node pairs is approximated with a Barnes–Hut octree and computed in
// parallel worker threads; attraction acts along edges; FA2's adaptive speed (swinging vs.
// traction) controls convergence.
import os from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const shared = (T, n) => new T(new SharedArrayBuffer(n * T.BYTES_PER_ELEMENT));
const CHUNK = 512;
const MAX_DEPTH = 60;
// No node moves more than this fraction of the layout's RMS radius in one iteration, so a
// near-collision cannot fling a node out of the layout.
const MAX_STEP = 0.05;

class Octree {
  constructor(n, random) {
    this.random = random;
    this.alloc(Math.max(4096, n * 3));
  }

  alloc(cap) {
    this.cap = cap;
    this.child = shared(Int32Array, cap * 8);
    this.mass = shared(Float64Array, cap);
    this.mx = shared(Float64Array, cap);
    this.my = shared(Float64Array, cap);
    this.mz = shared(Float64Array, cap);
    this.half = shared(Float64Array, cap);
    this.cx = new Float64Array(cap);
    this.cy = new Float64Array(cap);
    this.cz = new Float64Array(cap);
  }

  get buffers() {
    const { child, mass, mx, my, mz, half } = this;
    return { child, mass, mx, my, mz, half };
  }

  /** Rebuilds the tree for the current positions; returns true if its storage was reallocated. */
  build(pos, mass, n) {
    let grown = false;
    while (!this.tryBuild(pos, mass, n)) {
      this.alloc(this.cap * 2);
      grown = true;
    }
    return grown;
  }

  // Child slots hold -1 (empty), a cell index (>= 0) or a body index encoded as -2 - body.
  tryBuild(pos, mass, n) {
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      if (z < z0) z0 = z;
      if (z > z1) z1 = z;
    }
    const { child, mass: cm, mx, my, mz, cx, cy, cz, half, cap, random } = this;
    let count = 1;
    child.fill(-1, 0, 8);
    cm[0] = mx[0] = my[0] = mz[0] = 0;
    cx[0] = (x0 + x1) / 2;
    cy[0] = (y0 + y1) / 2;
    cz[0] = (z0 + z1) / 2;
    half[0] = Math.max(x1 - x0, y1 - y0, z1 - z0) / 2 + 1e-9;

    const path = new Int32Array(MAX_DEPTH);
    for (let b = 0; b < n; b++) {
      let x = pos[b * 3], y = pos[b * 3 + 1], z = pos[b * 3 + 2];
      const m = mass[b];
      let c = 0;
      let depth = 0;
      for (;;) {
        path[depth++] = c;
        cm[c] += m;
        mx[c] += m * x;
        my[c] += m * y;
        mz[c] += m * z;
        const o = (x > cx[c] ? 1 : 0) | (y > cy[c] ? 2 : 0) | (z > cz[c] ? 4 : 0);
        const slot = c * 8 + o;
        const ch = child[slot];
        if (ch === -1) {
          child[slot] = -2 - b;
          break;
        }
        if (depth < MAX_DEPTH) {
          if (ch >= 0) {
            c = ch;
            continue;
          }
          const b2 = -2 - ch;
          const x2 = pos[b2 * 3], y2 = pos[b2 * 3 + 1], z2 = pos[b2 * 3 + 2];
          if (x2 !== x || y2 !== y || z2 !== z) {
            if (count === cap) return false;
            const h = half[c] / 2;
            const nc = count++;
            child.fill(-1, nc * 8, nc * 8 + 8);
            cx[nc] = cx[c] + (o & 1 ? h : -h);
            cy[nc] = cy[c] + (o & 2 ? h : -h);
            cz[nc] = cz[c] + (o & 4 ? h : -h);
            half[nc] = h;
            const m2 = mass[b2];
            cm[nc] = m2;
            mx[nc] = m2 * x2;
            my[nc] = m2 * y2;
            mz[nc] = m2 * z2;
            child[nc * 8 + ((x2 > cx[nc] ? 1 : 0) | (y2 > cy[nc] ? 2 : 0) | (z2 > cz[nc] ? 4 : 0))] = -2 - b2;
            child[slot] = nc;
            c = nc;
            continue;
          }
        }
        // Bodies closer than floating-point resolution can never be split: undo this descent,
        // nudge the body and insert it again.
        for (let k = 0; k < depth; k++) {
          const q = path[k];
          cm[q] -= m;
          mx[q] -= m * x;
          my[q] -= m * y;
          mz[q] -= m * z;
        }
        x = pos[b * 3] += (random() - 0.5) * 1e-6 * half[0];
        y = pos[b * 3 + 1] += (random() - 0.5) * 1e-6 * half[0];
        z = pos[b * 3 + 2] += (random() - 0.5) * 1e-6 * half[0];
        c = depth = 0;
      }
    }
    for (let c = 0; c < count; c++) {
      mx[c] /= cm[c];
      my[c] /= cm[c];
      mz[c] /= cm[c];
    }
    return true;
  }
}

// Barnes–Hut repulsion plus gravity for nodes [i0, i1); writes (not adds) into `force`.
function repulse(i0, i1, s, t, p, stack) {
  const { pos, mass, force } = s;
  const { child, mass: cm, mx, my, mz, half } = t;
  const { scaling, gravity, theta2 } = p;
  for (let i = i0; i < i1; i++) {
    const xi = pos[i * 3], yi = pos[i * 3 + 1], zi = pos[i * 3 + 2];
    const mi = mass[i];
    let fx = 0, fy = 0, fz = 0;
    let sp = 0;
    stack[sp++] = 0;
    while (sp) {
      const c = stack[--sp];
      const dx = xi - mx[c], dy = yi - my[c], dz = zi - mz[c];
      const d2 = dx * dx + dy * dy + dz * dz;
      const size = 2 * half[c];
      if (size * size < theta2 * d2) {
        const f = (scaling * mi * cm[c]) / d2;
        fx += dx * f;
        fy += dy * f;
        fz += dz * f;
        continue;
      }
      const base = c * 8;
      for (let o = 0; o < 8; o++) {
        const ch = child[base + o];
        if (ch === -1) continue;
        if (ch >= 0) {
          stack[sp++] = ch;
          continue;
        }
        const j = -2 - ch;
        if (j === i) continue;
        const ex = xi - pos[j * 3], ey = yi - pos[j * 3 + 1], ez = zi - pos[j * 3 + 2];
        const e2 = ex * ex + ey * ey + ez * ez;
        if (e2 > 0) {
          const f = (scaling * mi * mass[j]) / e2;
          fx += ex * f;
          fy += ey * f;
          fz += ez * f;
        }
      }
    }
    const d = Math.sqrt(xi * xi + yi * yi + zi * zi);
    if (d > 0) {
      const g = (gravity * mi) / d;
      fx -= xi * g;
      fy -= yi * g;
      fz -= zi * g;
    }
    force[i * 3] = fx;
    force[i * 3 + 1] = fy;
    force[i * 3 + 2] = fz;
  }
}

function runChunks(s, t, p, stack) {
  const chunks = Math.ceil(s.n / CHUNK);
  for (;;) {
    const c = Atomics.add(s.counter, 0, 1);
    if (c >= chunks) return;
    repulse(c * CHUNK, Math.min(s.n, (c + 1) * CHUNK), s, t, p, stack);
  }
}

if (!isMainThread && workerData?.fa2) {
  let s;
  let t;
  const stack = new Int32Array(8 * 256);
  parentPort.on('message', (msg) => {
    if (msg.type === 'init') s = msg.state;
    else if (msg.type === 'tree') t = msg.tree;
    else if (msg.type === 'step') {
      runChunks(s, t, msg.params, stack);
      parentPort.postMessage('done');
    }
  });
}

/**
 * Runs ForceAtlas2 and resolves with the final positions.
 * @param {Float64Array} init  xyz interleaved starting positions
 * @param {{src: Uint32Array, dst: Uint32Array, weight?: Float32Array}} edges  undirected, each pair once
 * @param {Float64Array} massIn  FA2 uses degree + 1
 */
export async function forceAtlas2(init, edges, massIn, opts = {}) {
  const { iterations = 600, scaling = 2, gravity = 1, theta = 1.2, linLog = false, threads = os.availableParallelism() - 1, random = Math.random, onIteration } = opts;
  const n = massIn.length;
  const pos = shared(Float64Array, n * 3);
  pos.set(init);
  const mass = shared(Float64Array, n);
  mass.set(massIn);
  const force = shared(Float64Array, n * 3);
  const counter = shared(Int32Array, 1);
  const state = { pos, mass, force, counter, n };
  const { src, dst, weight } = edges;
  const E = src.length;
  const tree = new Octree(n, random);
  const prev = new Float64Array(n * 3);
  const stack = new Int32Array(8 * 256);
  const params = { scaling, gravity, theta2: theta * theta };
  let speed = 1;
  let speedEfficiency = 1;
  const jitterTolerance = 1;

  const workers = Array.from({ length: Math.max(0, threads) }, () => new Worker(new URL(import.meta.url), { workerData: { fa2: true } }));
  for (const w of workers) w.postMessage({ type: 'init', state });
  const step = () =>
    Promise.all(
      workers.map(
        (w) =>
          new Promise((resolve) => {
            w.once('message', resolve);
            w.postMessage({ type: 'step', params });
          }),
      ),
    );

  try {
    for (let it = 0; it < iterations; it++) {
      const grown = tree.build(pos, mass, n);
      if (grown || it === 0) for (const w of workers) w.postMessage({ type: 'tree', tree: tree.buffers });
      Atomics.store(counter, 0, 0);
      const done = step();
      runChunks(state, tree.buffers, params, stack);
      await done;

      // Attraction along edges.
      for (let e = 0; e < E; e++) {
        const a = src[e], b = dst[e];
        const dx = pos[b * 3] - pos[a * 3], dy = pos[b * 3 + 1] - pos[a * 3 + 1], dz = pos[b * 3 + 2] - pos[a * 3 + 2];
        let f = weight ? weight[e] : 1;
        if (linLog) {
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          f = d > 0 ? (f * Math.log(1 + d)) / d : 0;
        }
        force[a * 3] += dx * f;
        force[a * 3 + 1] += dy * f;
        force[a * 3 + 2] += dz * f;
        force[b * 3] -= dx * f;
        force[b * 3 + 1] -= dy * f;
        force[b * 3 + 2] -= dz * f;
      }

      // Adaptive speed: lower it when nodes swing (force reverses), raise it when they glide.
      let swingingTotal = 0;
      let tractionTotal = 0;
      for (let i = 0; i < n; i++) {
        const k = i * 3;
        const sx = prev[k] - force[k], sy = prev[k + 1] - force[k + 1], sz = prev[k + 2] - force[k + 2];
        const tx = prev[k] + force[k], ty = prev[k + 1] + force[k + 1], tz = prev[k + 2] + force[k + 2];
        swingingTotal += mass[i] * Math.sqrt(sx * sx + sy * sy + sz * sz);
        tractionTotal += mass[i] * 0.5 * Math.sqrt(tx * tx + ty * ty + tz * tz);
      }
      const estimatedJT = 0.05 * Math.sqrt(n);
      let jt = jitterTolerance * Math.max(Math.sqrt(estimatedJT), Math.min(10, (estimatedJT * tractionTotal) / (n * n)));
      if (swingingTotal / tractionTotal > 2) {
        if (speedEfficiency > 0.05) speedEfficiency *= 0.5;
        jt = Math.max(jt, jitterTolerance);
      }
      const targetSpeed = (jt * speedEfficiency * tractionTotal) / swingingTotal;
      if (swingingTotal > jt * tractionTotal) {
        if (speedEfficiency > 0.05) speedEfficiency *= 0.7;
      } else if (speed < 1000) {
        speedEfficiency *= 1.3;
      }
      speed += Math.min(targetSpeed - speed, 0.5 * speed);

      let r2 = 0;
      for (let k = 0; k < n * 3; k++) r2 += pos[k] * pos[k];
      const maxStep = MAX_STEP * Math.sqrt(r2 / n);
      for (let i = 0; i < n; i++) {
        const k = i * 3;
        const sx = prev[k] - force[k], sy = prev[k + 1] - force[k + 1], sz = prev[k + 2] - force[k + 2];
        const swinging = mass[i] * Math.sqrt(sx * sx + sy * sy + sz * sz);
        const f = Math.sqrt(force[k] * force[k] + force[k + 1] * force[k + 1] + force[k + 2] * force[k + 2]);
        const factor = Math.min(speed / (1 + Math.sqrt(speed * swinging)), maxStep / (f || 1));
        pos[k] += force[k] * factor;
        pos[k + 1] += force[k + 1] * factor;
        pos[k + 2] += force[k + 2] * factor;
      }
      prev.set(force);
      onIteration?.(it, { speed, swinging: swingingTotal, traction: tractionTotal, pos });
    }
  } finally {
    await Promise.all(workers.map((w) => w.terminate()));
  }
  return Float64Array.from(pos);
}

// ---------- spectral starting positions ----------

function components(n, { offsets, nbrs }, useEdge) {
  const comp = new Int32Array(n).fill(-1);
  const stack = new Uint32Array(n);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] >= 0) continue;
    let top = 0;
    stack[top++] = s;
    comp[s] = count;
    while (top) {
      const u = stack[--top];
      for (let p = offsets[u]; p < offsets[u + 1]; p++) {
        if (!useEdge(p)) continue;
        const v = nbrs[p];
        if (comp[v] < 0) {
          comp[v] = count;
          stack[top++] = v;
        }
      }
    }
    count++;
  }
  return { comp, count };
}

/**
 * Diffusion coordinates: lazy random-walk smoothing with per-component centring and
 * Gram–Schmidt, i.e. orthogonal iteration towards the graph's low-frequency eigenvectors.
 * Gives FA2 a starting point where the global structure is already untangled.
 */
export function spectralInit(n, csr, useEdge, { iters = 40, random = Math.random } = {}) {
  const { offsets, nbrs } = csr;
  const { comp, count } = components(n, csr, useEdge);
  let X = new Float64Array(n * 3);
  let Y = new Float64Array(n * 3);
  for (let k = 0; k < X.length; k++) X[k] = random() * 2 - 1;
  const sums = new Float64Array(count * 3);
  const sizes = new Float64Array(count);
  for (let i = 0; i < n; i++) sizes[comp[i]]++;

  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < n; i++) {
      let s0 = 0, s1 = 0, s2 = 0, c = 0;
      for (let p = offsets[i]; p < offsets[i + 1]; p++) {
        if (!useEdge(p)) continue;
        const j = nbrs[p] * 3;
        s0 += X[j];
        s1 += X[j + 1];
        s2 += X[j + 2];
        c++;
      }
      const k = i * 3;
      if (c) {
        Y[k] = 0.5 * X[k] + (0.5 * s0) / c;
        Y[k + 1] = 0.5 * X[k + 1] + (0.5 * s1) / c;
        Y[k + 2] = 0.5 * X[k + 2] + (0.5 * s2) / c;
      } else {
        Y[k] = X[k];
        Y[k + 1] = X[k + 1];
        Y[k + 2] = X[k + 2];
      }
    }
    sums.fill(0);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) sums[comp[i] * 3 + d] += Y[i * 3 + d];
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) Y[i * 3 + d] -= sums[comp[i] * 3 + d] / sizes[comp[i]];
    for (let d = 0; d < 3; d++) {
      for (let q = 0; q < d; q++) {
        let dot = 0;
        for (let i = 0; i < n; i++) dot += Y[i * 3 + d] * Y[i * 3 + q];
        for (let i = 0; i < n; i++) Y[i * 3 + d] -= dot * Y[i * 3 + q];
      }
      let norm = 0;
      for (let i = 0; i < n; i++) norm += Y[i * 3 + d] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) Y[i * 3 + d] /= norm;
    }
    [X, Y] = [Y, X];
  }
  return X;
}
