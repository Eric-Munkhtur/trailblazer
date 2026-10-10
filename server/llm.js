const DEFAULT_MODELS = { openai: 'gpt-5-mini', anthropic: 'claude-sonnet-4-5', mock: 'mock' };

export function llmConfig() {
  const provider =
    process.env.LLM_PROVIDER ||
    (process.env.OPENAI_API_KEY ? 'openai' : process.env.ANTHROPIC_API_KEY ? 'anthropic' : null);
  return {
    provider,
    model: process.env.LLM_MODEL || DEFAULT_MODELS[provider] || null,
    ready: provider === 'mock' || (provider === 'openai' && !!process.env.OPENAI_API_KEY) || (provider === 'anthropic' && !!process.env.ANTHROPIC_API_KEY),
  };
}

function parseJSON(text) {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('LLM did not return JSON');
  return JSON.parse(text.slice(start, end + 1));
}

async function post(url, headers, body) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 120000);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`LLM API ${res.status}: ${text.slice(0, 300)}`);
    return JSON.parse(text);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ask the configured LLM for a JSON object. `mock` is used only when LLM_PROVIDER=mock.
 */
export async function askJSON({ system, user, mock }) {
  const { provider, model, ready } = llmConfig();
  if (!ready) {
    throw new Error('No LLM configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env and restart the server.');
  }

  if (provider === 'mock') return mock();

  if (provider === 'openai') {
    const base = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
    const data = await post(
      `${base}/chat/completions`,
      { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
      {
        model,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
      },
    );
    return parseJSON(data.choices?.[0]?.message?.content || '');
  }

  if (provider === 'anthropic') {
    const data = await post(
      'https://api.anthropic.com/v1/messages',
      { 'x-api-key': process.env.ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01' },
      {
        model,
        max_tokens: 4096,
        system: `${system}\nRespond with a single JSON object and nothing else.`,
        messages: [{ role: 'user', content: user }],
      },
    );
    return parseJSON((data.content || []).map((c) => c.text || '').join(''));
  }

  throw new Error(`Unknown LLM_PROVIDER "${provider}"`);
}
