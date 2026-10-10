import 'dotenv/config';
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runAgent } from './agent.js';
import { llmConfig } from './llm.js';

const app = express();
const PORT = Number(process.env.PORT) || 8787;
const runs = new Map();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

app.get('/api/health', (_req, res) => {
  const { provider, model, ready } = llmConfig();
  res.json({ ok: true, llm: { provider, model, ready } });
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
    for (const ev of cached) {
      if (closed) return;
      send(ev);
      if (ev.type === 'graph' || ev.type === 'step') await sleep(120);
    }
    return res.end();
  }

  const events = [];
  const emit = (ev) => {
    events.push(ev);
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
  console.log(`[verdia] API on http://localhost:${PORT}  ·  LLM: ${ready ? `${provider}/${model}` : 'NOT CONFIGURED (set a key in .env)'}`);
});
