import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runAgent } from './agent.js';
import { llmConfig } from './llm.js';
import { primekg } from './primekg.js';
import { streamSearch } from './neural.js';
import { benchmarkFor, predictionHealth, predictTrial } from './prediction.js';

const app = express();
const PORT = Number(process.env.PORT) || 8790;
const runs = new Map();
const MAX_CACHED_RUNS = 3;
const MAX_RUN_BYTES = 2 * 1024 * 1024;
const RUN_TTL_MS = 15 * 60 * 1000;
const activeRequests = { graph: 0, prediction: 0 };
const visitors = new Map();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const kg = primekg();
app.disable('x-powered-by');
// The production API listens on loopback behind Caddy.
if (process.env.TRUST_LOCAL_PROXY === '1') app.set('trust proxy', 'loopback');

function limited(handler, group = 'graph') {
  return async (req, res, next) => {
    const now = Date.now();
    for (const [ip, record] of visitors) if (record.until <= now) visitors.delete(ip);
    const ip = req.ip;
    let record = visitors.get(ip);
    if (!record) {
      if (visitors.size >= 1000) return res.status(503).json({ error: 'Demo is busy. Please try again shortly.' });
      record = { count: 0, until: now + 60000 };
      visitors.set(ip, record);
    }
    const maxActive = group === 'prediction' ? 4 : 2;
    if (++record.count > 20 || activeRequests[group] >= maxActive) {
      res.setHeader('Retry-After', '30');
      return res.status(429).json({ error: 'Demo is busy. Please try again in 30 seconds.' });
    }
    activeRequests[group]++;
    try { await handler(req, res); } catch (error) { next(error); }
    finally { activeRequests[group]--; }
  };
}

app.get('/api/health', async (_req, res) => {
  const { provider, model, ready } = llmConfig();
  res.json({ ok: !!kg, llm: { provider, model, ready }, prediction: await predictionHealth(), primekg: kg ? { nodes: kg.N, edges: kg.E } : null });
});

app.get('/api/prediction/:nct', (req, res) => {
  const nct = req.params.nct.toUpperCase();
  if (!/^NCT\d{8}$/.test(nct)) return res.status(400).json({ error: 'Invalid NCT identifier' });
  res.json({ nct, benchmark: benchmarkFor(nct) });
});

app.post('/api/prediction/:nct', limited(async (req, res) => {
  const nct = req.params.nct.toUpperCase();
  if (!/^NCT\d{8}$/.test(nct)) return res.status(400).json({ error: 'Invalid NCT identifier' });
  res.setHeader('Cache-Control', 'no-store');
  res.json(await predictTrial(nct));
}, 'prediction'));

app.get('/api/primekg/search', (req, res) => {
  if (!kg) return res.status(503).json({ error: 'PrimeKG not built. Run `npm run primekg`.' });
  const types = req.query.types ? String(req.query.types).split(',') : undefined;
  res.json(kg.search(String(req.query.q || ''), { types, limit: Math.min(Number(req.query.limit) || 10, 50) }));
});

const isNode = (i) => Number.isInteger(i) && i >= 0 && i < kg.N;

app.get('/api/primekg/node/:i', (req, res) => {
  const i = Number(req.params.i);
  if (!kg) return res.status(503).json({ error: 'PrimeKG not built. Run `npm run primekg`.' });
  if (!isNode(i)) return res.status(404).json({ error: 'No such node' });
  res.json(kg.info(i));
});

app.get('/api/primekg/neighbors/:i', (req, res) => {
  const i = Number(req.params.i);
  if (!kg) return res.status(503).json({ error: 'PrimeKG not built. Run `npm run primekg`.' });
  if (!isNode(i)) return res.status(404).json({ error: 'No such node' });
  res.json(kg.neighbors(i, Math.min(Number(req.query.limit) || 1500, 4000)));
});

function openStream(req, res) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache, no-transform', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
  res.flushHeaders();
  const stream = { closed: false, send: (ev) => !stream.closed && res.write(`data: ${JSON.stringify(ev)}\n\n`) };
  const heartbeat = setInterval(() => { if (!stream.closed) res.write(': keep-alive\n\n'); }, 15000);
  heartbeat.unref();
  res.on('close', () => { stream.closed = true; clearInterval(heartbeat); });
  return stream;
}

// Graph-only neural search between any two nodes (no trial, no LLM).
app.get('/api/connect', limited(async (req, res) => {
  const a = Number(req.query.a);
  const b = Number(req.query.b);
  const { send } = openStream(req, res);
  if (!kg || !isNode(a) || !isNode(b) || a === b) {
    send({ type: 'error', message: kg ? 'Pick two different PrimeKG nodes to connect.' : 'PrimeKG not built. Run `npm run primekg`.' });
    return res.end();
  }
  const title = `${kg.nodes.names[a]} ⟷ ${kg.nodes.names[b]}`;
  send({ type: 'step', id: 'connect', title: 'Connect two nodes', status: 'running', detail: title });
  send({ type: 'seeds', seeds: [{ ...kg.node(a), side: 'A' }, { ...kg.node(b), side: 'B' }] });
  const s = await streamSearch(kg, [a], [b], send, (text) => send({ type: 'log', step: 'connect', text }), { sides: ['A', 'B'] });
  const kinds = new Set(s.paths.map((p) => p.kind)).size;
  send({ type: 'step', id: 'connect', title: 'Connect two nodes', status: 'done', detail: s.paths.length ? `${s.paths.length} pathways · ${kinds} kinds` : 'no pathway within four hops' });
  send({ type: 'done' });
  res.end();
}));

app.get('/api/assemble', limited(async (req, res) => {
  const nct = String(req.query.nct || '').trim().toUpperCase();
  const stream = openStream(req, res);
  const { send } = stream;

  if (!/^NCT\d{8}$/.test(nct)) {
    send({ type: 'error', message: 'Enter a valid NCT number, e.g. NCT02142738' });
    return res.end();
  }

  const entry = runs.get(nct);
  if (entry && entry.expires <= Date.now()) runs.delete(nct);
  const cached = req.query.fresh || !entry || entry.expires <= Date.now() ? null : entry.events;
  if (cached) {
    const start = Date.now();
    for (const { at, ev } of cached) {
      if (stream.closed) return;
      const wait = at - (Date.now() - start);
      if (wait > 0) await sleep(wait);
      send(ev);
    }
    return res.end();
  }

  const events = [];
  let eventBytes = 0;
  let cacheable = true;
  const t0 = Date.now();
  const emit = (ev) => {
    if (cacheable) {
      eventBytes += Buffer.byteLength(JSON.stringify(ev));
      if (eventBytes <= MAX_RUN_BYTES) events.push({ at: Date.now() - t0, ev });
      else { cacheable = false; events.length = 0; }
    }
    send(ev);
  };
  try {
    await runAgent(nct, emit);
    if (cacheable) {
      runs.delete(nct);
      while (runs.size >= MAX_CACHED_RUNS) runs.delete(runs.keys().next().value);
      runs.set(nct, { events, expires: Date.now() + RUN_TTL_MS });
    }
  } catch (e) {
    console.error(`[agent] ${nct}:`, e);
    emit({ type: 'error', message: e.message });
  }
  res.end();
}));

if (process.env.NODE_ENV === 'production') {
  const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');
  app.use(express.static(dist, { maxAge: '1h', setHeaders: (res, file) => {
    if (file.endsWith('index.html')) res.setHeader('Cache-Control', 'no-cache');
  } }));
}

app.listen(PORT, process.env.HOST || '127.0.0.1', () => {
  const { provider, model, ready } = llmConfig();
  console.log(`[verdia] API on http://localhost:${PORT}  ·  LLM: ${ready ? `${provider}/${model}` : 'NOT CONFIGURED (set a key in .env)'}  ·  PrimeKG: ${kg ? 'loaded' : 'NOT BUILT (npm run primekg)'}`);
});
