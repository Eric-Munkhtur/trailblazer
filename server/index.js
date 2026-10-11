import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runAgent } from './agent.js';
import { llmConfig } from './llm.js';
import { primekg } from './primekg.js';
import { streamSearch } from './neural.js';

const app = express();
const PORT = Number(process.env.PORT) || 8790;
const runs = new Map();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const kg = primekg();

app.get('/api/health', (_req, res) => {
  const { provider, model, ready } = llmConfig();
  res.json({ ok: true, llm: { provider, model, ready }, primekg: kg ? { nodes: kg.N, edges: kg.E } : null });
});

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
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  const stream = { closed: false, send: (ev) => !stream.closed && res.write(`data: ${JSON.stringify(ev)}\n\n`) };
  req.on('close', () => (stream.closed = true));
  return stream;
}

// Graph-only neural search between any two nodes (no trial, no LLM).
app.get('/api/connect', async (req, res) => {
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
});

app.get('/api/assemble', async (req, res) => {
  const nct = String(req.query.nct || '').trim().toUpperCase();
  const stream = openStream(req, res);
  const { send } = stream;

  if (!/^NCT\d{8}$/.test(nct)) {
    send({ type: 'error', message: 'Enter a valid NCT number, e.g. NCT02142738' });
    return res.end();
  }

  const cached = req.query.fresh ? null : runs.get(nct);
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
  const t0 = Date.now();
  const emit = (ev) => {
    events.push({ at: Date.now() - t0, ev });
    send(ev);
  };
  try {
    await runAgent(nct, emit);
    runs.set(nct, events);
  } catch (e) {
    console.error(`[agent] ${nct}:`, e);
    emit({ type: 'error', message: e.message });
  }
  res.end();
});

if (process.env.NODE_ENV === 'production') {
  const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'dist');
  app.use(express.static(dist));
}

app.listen(PORT, () => {
  const { provider, model, ready } = llmConfig();
  console.log(`[verdia] API on http://localhost:${PORT}  ·  LLM: ${ready ? `${provider}/${model}` : 'NOT CONFIGURED (set a key in .env)'}  ·  PrimeKG: ${kg ? 'loaded' : 'NOT BUILT (npm run primekg)'}`);
});
