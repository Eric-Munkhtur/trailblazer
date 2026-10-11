// Bottom telemetry strip: large counters and a small chart. Idle, it shows PrimeKG's totals and
// degree distribution (with the inspected node marked); during a search it counts the nodes
// reached as the replay unfolds and charts both wavefronts over replay time.
const $ = (s) => document.querySelector(s);
const SIDE_COLOR = { intervention: '#8c84ff', A: '#8c84ff', disease: '#ff8b3d', B: '#ff8b3d' };

const padded = (n, digits = 9) =>
  String(Math.max(0, Math.round(n)))
    .padStart(digits, '0')
    .replace(/\B(?=(\d{3})+(?!\d))/g, '.');
const short = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${+(n / 1e3).toFixed(1)}K` : `${Math.round(n)}`);
const pow2 = (b) => (b >= 10 ? `${2 ** (b - 10)}K` : `${2 ** b}`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/** Number of sorted values <= t. */
function countUpTo(sorted, t) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const reachedBy = (frames, t) => frames.reduce((a, f) => a + f.weight * countUpTo(f.times, t), 0);

function along(points, x) {
  for (let k = 1; k < points.length; k++) {
    const [x0, y0] = points[k - 1];
    const [x1, y1] = points[k];
    if (x <= x1) return y0 + ((y1 - y0) * Math.max(0, x - x0)) / (x1 - x0 || 1);
  }
  return points.at(-1)[1];
}

export function createTelemetry(kg, clock) {
  const bins = [];
  for (const d of kg.degree) {
    if (!d) continue;
    const b = Math.floor(Math.log2(d));
    bins[b] = (bins[b] || 0) + 1;
  }
  const hist = Array.from(bins, (c) => c || 0);

  let run = null;
  let marked = null;
  let view = null;
  let scaleX = null;

  const counter = (k, label, value) => {
    $(`#c${k}-l`).textContent = label;
    $(`#c${k}`).textContent = value;
  };

  function degreeView() {
    const points = hist.map((c, b) => [b + 0.5, Math.log10(1 + c)]);
    const yMax = Math.ceil(Math.max(...points.map((p) => p[1])));
    const xTicks = [];
    for (let b = 0; b <= hist.length; b += 2) xTicks.push([b, pow2(b)]);
    const yTicks = [];
    for (let e = 2; e < yMax; e += 2) yTicks.push([e, short(10 ** e)]);
    const x = marked && Math.log2(Math.max(1, marked.degree));
    return {
      label: 'Degree distribution · nodes per number of edges:',
      series: [{ points, color: '#cfcfcf' }],
      xMax: hist.length,
      yMax,
      xTicks,
      yTicks,
      marker: marked && { x, y: along(points, x), label: `${marked.name} · ${marked.degree.toLocaleString()}` },
    };
  }

  function runView() {
    const STEPS = 120;
    const series = Object.entries(run.sides).map(([side, frames]) => ({
      color: SIDE_COLOR[side] || '#cfcfcf',
      points: Array.from({ length: STEPS + 1 }, (_, k) => [(k / STEPS) * run.end, reachedBy(frames, (k / STEPS) * run.end)]),
    }));
    const top = Math.max(1, ...series.map((s) => s.points.at(-1)[1]));
    const step = run.end > 6 ? 2 : run.end > 2.5 ? 1 : 0.5;
    const xTicks = [];
    for (let t = 0; t <= run.end + 1e-6; t += step) xTicks.push([t, `${+t.toFixed(1)}`]);
    return {
      label: 'Nodes reached per wavefront · seconds of replay:',
      series,
      xMax: run.end,
      yMax: top * 1.1,
      xTicks,
      yTicks: [
        [top / 2, short(top / 2)],
        [top, short(top)],
      ],
      marker: run.meet != null && { x: run.meet, y: 0, label: 'Meet' },
      cursor: true,
    };
  }

  function draw() {
    const svg = $('#chart');
    const { width: W, height: H } = svg.getBoundingClientRect();
    if (!view || !W || !H) return;
    $('#chart-l').textContent = view.label;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const L = 34;
    const R = 8;
    const T = 3;
    const B = 13;
    const x = (v) => L + (v / view.xMax) * (W - L - R);
    const y = (v) => H - B - (v / view.yMax) * (H - B - T);
    scaleX = x;
    let out = `<rect class="box" x="${L}" y="${T}" width="${W - L - R}" height="${H - B - T}"/>`;
    for (const [v, label] of view.xTicks) out += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - B}"/><text x="${x(v)}" y="${H - 1}" text-anchor="middle">${label}</text>`;
    for (const [v, label] of view.yTicks) out += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${label}</text>`;
    for (const s of view.series) out += `<polyline class="line" stroke="${s.color}" points="${s.points.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')}"/>`;
    if (view.marker) {
      const mx = x(view.marker.x);
      const my = y(view.marker.y);
      const right = mx > W * 0.7;
      out += `<circle class="dot" cx="${mx}" cy="${my}" r="3"/><text class="mark" x="${right ? mx - 7 : mx + 7}" y="${Math.max(T + 9, my - 5)}" text-anchor="${right ? 'end' : 'start'}">${esc(view.marker.label)}</text>`;
    }
    if (view.cursor) out += `<line id="cursor" class="cursor" x1="0" x2="0" y1="${T}" y2="${H - B}" transform="translate(${L},0)"/>`;
    svg.innerHTML = out;
  }

  function tick() {
    if (!run) return;
    if (run.dirty) {
      run.dirty = false;
      view = runView();
      draw();
    }
    const t = clock() - run.start;
    const lit = Object.values(run.sides).reduce((a, frames) => a + reachedBy(frames, t), 0);
    counter(1, 'Nodes reached:', padded(t >= run.end ? run.explored : (run.explored * lit) / (run.total || 1)));
    $('#cursor')?.setAttribute('transform', `translate(${scaleX(Math.min(t, run.end))},0)`);
    if (t < run.end + 0.3) run.raf = requestAnimationFrame(tick);
  }

  return {
    idle() {
      if (run) cancelAnimationFrame(run.raf);
      run = null;
      counter(1, 'Entities:', padded(kg.N));
      counter(2, 'Relations:', padded(kg.meta.edges));
      $('#c3-box').classList.add('hidden');
      view = degreeView();
      draw();
    },
    /** Mark node i on the degree distribution (i < 0 clears it). */
    mark(i) {
      marked = i >= 0 ? { name: kg.names[i], degree: kg.degree[i] } : null;
      if (run) return;
      view = degreeView();
      draw();
    },
    search({ explored, ms, duration }) {
      if (run) cancelAnimationFrame(run.raf);
      run = { start: clock(), end: Math.max(0.1, duration), explored, sides: {}, total: 0, meet: null, dirty: true };
      counter(1, 'Nodes reached:', padded(0));
      counter(2, 'Search time:', `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`);
      $('#c3-box').classList.add('hidden');
      run.raf = requestAnimationFrame(tick);
    },
    signal({ side, reached, nodes, times }) {
      if (!run || !nodes.length) return;
      (run.sides[side] ||= []).push({ weight: reached / nodes.length, times: Float32Array.from(times).sort() });
      run.total += reached;
      run.dirty = true;
    },
    meet(times) {
      if (!run || !times?.length) return;
      run.meet = times.reduce((a, b) => Math.min(a, b), Infinity);
      run.dirty = true;
    },
    paths(n) {
      $('#c3').textContent = String(n).padStart(2, '0');
      $('#c3-box').classList.remove('hidden');
    },
    redraw: draw,
  };
}
