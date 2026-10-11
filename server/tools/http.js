const cache = new Map();

export async function getJSON(url, { timeout = 20000, init = {}, cacheKey } = {}) {
  const key = cacheKey ?? (init.method && init.method !== 'GET' ? null : url);
  if (key && cache.has(key)) return cache.get(key);

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
    if (key) cache.set(key, data);
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`Timed out after ${timeout}ms: ${url}`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}
