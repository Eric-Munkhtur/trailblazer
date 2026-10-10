export class GraphBuilder {
  constructor(emit) {
    this.emit = emit;
    this.nodes = new Map();
    this.links = new Map();
    this.pending = { nodes: [], links: [] };
  }

  node(id, props) {
    const existing = this.nodes.get(id);
    if (existing) {
      Object.assign(existing, props, { data: { ...existing.data, ...props.data } });
      if (props.kind && existing.kind !== props.kind) existing.roles = [...new Set([...(existing.roles || []), props.kind])];
      this.pending.nodes.push(existing);
      return existing;
    }
    const n = { id, ...props, data: props.data || {} };
    this.nodes.set(id, n);
    this.pending.nodes.push(n);
    return n;
  }

  has(id) {
    return this.nodes.has(id);
  }

  link(source, target, type, { source: provenance, ...rest } = {}) {
    if (!this.nodes.has(source) || !this.nodes.has(target) || source === target) return null;
    const props = provenance ? { ...rest, provenance } : rest;
    const id = `${source}|${type}|${target}`;
    const existing = this.links.get(id);
    if (existing) {
      Object.assign(existing, props);
      this.pending.links.push(existing);
      return existing;
    }
    const l = { id, source, target, type, ...props };
    this.links.set(id, l);
    this.pending.links.push(l);
    return l;
  }

  flush() {
    if (!this.pending.nodes.length && !this.pending.links.length) return;
    const dedupe = (arr) => [...new Map(arr.map((x) => [x.id, x])).values()];
    this.emit({ type: 'graph', nodes: dedupe(this.pending.nodes), links: dedupe(this.pending.links) });
    this.pending = { nodes: [], links: [] };
  }

  snapshot() {
    return { nodes: [...this.nodes.values()], links: [...this.links.values()] };
  }
}
