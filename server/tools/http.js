const cache = new Map();
const TTL_MS = 15 * 60 * 1000;

export async function getJSON(url, { timeout = 20000, init = {}, cacheKey } = {}) {
  const key = cacheKey ?? (init.method && init.method !== 'GET' ? null : url);
  const cached = key && cache.get(key);
  if (cached && cached.expires > Date.now()) return cached.data;
  if (cached) cache.delete(key);

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(url, {
      ...init,
      signal: ctrl.signal,
      headers: { Accept: 'application/json', 'User-Agent': 'verdia/0.1 (knowledge-graph agent)', ...(init.headers || {}) },
    });
    if (!res.ok) {
      const err = new Error(`HTTP ${res.status} for ${url}`);
      err.status = res.status;
      throw err;
    }
    const data = await res.json();
    if (key && JSON.stringify(data).length < 512000) {
      while (cache.size >= 64) cache.delete(cache.keys().next().value);
      cache.set(key, { data, expires: Date.now() + TTL_MS });
    }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`Timed out after ${timeout}ms: ${url}`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}
