import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runAgent } from './agent.js';
import { llmConfig } from './llm.js';
import { primekg } from './primekg.js';

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

app.get('/api/primekg/node/:i', (req, res) => {
  const i = Number(req.params.i);
  if (!kg) return res.status(503).json({ error: 'PrimeKG not built. Run `npm run primekg`.' });
  if (!Number.isInteger(i) || i < 0 || i >= kg.N) return res.status(404).json({ error: 'No such node' });
  res.json(kg.info(i));
});

app.get('/api/assemble', async (req, res) => {
  const nct = String(req.query.nct || '').trim().toUpperCase();
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  const send = (ev) => res.write(`data: ${JSON.stringify(ev)}\n\n`);
  let closed = false;
  req.on('close', () => (closed = true));

  if (!/^NCT\d{8}$/.test(nct)) {
    send({ type: 'error', message: 'Enter a valid NCT number, e.g. NCT02142738' });
    return res.end();
  }

  const cached = req.query.fresh ? null : runs.get(nct);
  if (cached) {
    const start = Date.now();
    for (const { at, ev } of cached) {
      if (closed) return;
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
    if (!closed) send(ev);
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
