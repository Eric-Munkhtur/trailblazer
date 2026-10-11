const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const fixed = (n) => Number.isFinite(n) ? n.toFixed(3) : '—';
const pct = (n) => Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—';

function metrics(benchmark) {
  if (!benchmark) return '';
  return `<section class="prediction-section">
    <h3>How accurate is the model?</h3>
    <p class="prose">Measured on ${benchmark.nTest.toLocaleString()} historical ${esc(benchmark.dataset)} test trials. These numbers describe the benchmark, not the chance that this individual trial succeeds.</p>
    <div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>Model</th><th>Accuracy</th><th>ROC-AUC</th><th>AP</th></tr></thead><tbody>
      ${['attention', 'control', 'conditioned'].map((key) => {
        const m = benchmark.models[key];
        const label = { attention: 'Our attention', control: 'HINT control', conditioned: 'Our context' }[key];
        return `<tr><th>${label}</th><td>${pct(m.metrics.accuracy)}</td><td>${fixed(m.metrics.rocAuc)}</td><td>${fixed(m.metrics.averagePrecision)}</td></tr>`;
      }).join('')}
    </tbody></table></div>
    <p class="model-footnote">${esc(benchmark.conclusion)} Accuracy uses a 0.5 threshold. ROC-AUC measures ranking; AP summarizes precision and recall.</p>
  </section>`;
}

function historical(benchmark) {
  if (!benchmark?.historicalScores) return '';
  return `<details class="prediction-details"><summary>Saved historical scores for this trial</summary>
    <p class="prose">Recorded during evaluation on the frozen TOP inputs on ${esc(benchmark.recordedDate)}. Live scores use the current registry record and freshly prepared inputs, so they can differ.</p>
    ${['attention', 'control', 'conditioned'].map((key) => `<div class="model-row"><span>${esc(benchmark.models[key].name)}</span><b>${fixed(benchmark.historicalScores[key])}</b></div>`).join('')}
  </details>`;
}

export function createPredictionPanel(root, onStatus, onSearch) {
  const initial = root.innerHTML;
  let controller;
  let version = 0;
  let downloadURL;
  function nextExamples() {
    return `<div class="model-next"><p>Try a supported Phase II trial:</p>
      <button class="model-example" data-model-example="NCT01132482">NCT01132482 · Sildenafil / cystic fibrosis →</button>
      <button class="model-example" data-model-example="NCT01928927">NCT01928927 · TRAFIC / telmisartan →</button></div>`;
  }
  function bindExamples() {
    root.querySelectorAll('[data-model-example]').forEach((b) => b.addEventListener('click', () => onSearch?.(b.dataset.modelExample)));
  }
  function bindDownload(result) {
    if (downloadURL) URL.revokeObjectURL(downloadURL);
    downloadURL = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const link = root.querySelector('[data-download]');
    if (link) link.href = downloadURL;
  }

  function reset() {
    version++;
    controller?.abort();
    controller = null;
    if (downloadURL) URL.revokeObjectURL(downloadURL);
    downloadURL = null;
    root.innerHTML = initial;
    root.setAttribute('aria-busy', 'false');
  }

  async function load(nct) {
    reset();
    const current = version;
    controller = new AbortController();
    const { signal } = controller;
    root.setAttribute('aria-busy', 'true');
    root.innerHTML = `<div class="prediction-heading"><span class="model-badge">Preparing model inputs</span><h2>${esc(nct)}</h2></div>
      <div class="model-loading"><span class="model-pulse"></span><p>Resolving drugs and disease codes, encoding eligibility text, then running our trained model.</p></div>
      <p class="prose">The PrimeKG map builds alongside the prediction. The first score can take up to two minutes.</p><div id="benchmark-preview"></div>`;
    onStatus('running', 'Resolve inputs → BioBERT → trained HINT');
    let received = false;
    let previewBenchmark = null;
    fetch(`/api/prediction/${nct}`, { signal }).then((r) => r.ok ? r.json() : null).then((data) => {
      if (current !== version || received || !data) return;
      previewBenchmark = data.benchmark;
      const preview = root.querySelector('#benchmark-preview');
      if (preview) preview.innerHTML = metrics(data.benchmark) + historical(data.benchmark);
    }).catch(() => {});
    try {
      const response = await fetch(`/api/prediction/${nct}`, { method: 'POST', signal });
      const result = await response.json();
      if (current !== version) return;
      received = true;
      if (!response.ok) throw new Error(result.error || 'Prediction is temporarily unavailable.');
      root.setAttribute('aria-busy', 'false');
      render(result);
      const saved = result.benchmark?.historicalScores;
      onStatus(result.available || saved ? 'done' : 'unavailable', result.available ? `Our score ${fixed(result.scores.attention)} · CPU inference` : saved ? `Saved benchmark score ${fixed(saved.attention)} · live inputs unavailable` : result.reason);
    } catch (error) {
      if (current !== version || error.name === 'AbortError') return;
      received = true;
      root.setAttribute('aria-busy', 'false');
      render({ nct, available: false, reason: error.message, retryable: true, benchmark: previewBenchmark });
      onStatus('unavailable', error.message);
    }
  }

  function render(result) {
    const { nct, benchmark } = result;
    const saved = !result.available && benchmark?.historicalScores;
    const graphOnly = ['study_type', 'phase', 'intervention_type'].includes(result.reasonCode);
    const heading = `<div class="prediction-heading"><span class="model-badge ${saved ? 'historical' : ''}">${result.available ? (result.cached ? 'Recent model inference' : 'Live model inference') : saved ? 'Saved historical result' : graphOnly ? 'Graph exploration only' : 'Could not prepare model inputs'}</span>
      <h2>${esc(nct)}</h2>${result.trial ? `<p class="prose">${esc(result.trial.title)}</p><p class="model-study-type">${esc(result.trial.studyType || '')} ${esc(result.trial.phases?.join(' / ') || '· no drug-trial phase')}</p>` : ''}</div>`;
    if (saved) {
      root.innerHTML = `${heading}<div class="model-notice">This score was recorded on ${esc(benchmark.recordedDate)} using the frozen TOP test inputs. It is a historical benchmark result, not a fresh prediction of the current registry record.</div>
        <section class="model-score-card"><span class="score-label">Our model · recorded success score</span>
          <div class="model-score">${fixed(saved.attention)}<span>/ 1</span></div>
          <div class="model-row"><span>Recorded HINT control</span><b>${fixed(saved.control)}</b></div>
          <div class="model-row"><span>Recorded context attention</span><b>${fixed(saved.conditioned)}</b></div></section>
        <details class="prediction-details"><summary>Why a fresh score could not be calculated</summary><p class="prose">${esc(result.reason)}</p></details>
        <p class="prose">PrimeKG maps the current registry record. The recorded score used historical trial inputs. These are separate sources of evidence.</p>
        ${result.retryable ? '<button class="model-action" data-retry>Retry live prediction</button>' : ''}
        ${metrics(benchmark)}${nextExamples()}
        <a class="model-action" data-download download="verdia-${esc(nct)}-historical.json">Download historical result ↓</a>`;
      bindExamples();
      bindDownload({ ...result, mode: 'saved_benchmark', liveAvailable: false, scores: saved });
      root.querySelector('[data-retry]')?.addEventListener('click', () => load(nct));
      return;
    }
    if (!result.available) {
      root.innerHTML = `${heading}<div class="model-notice">${esc(result.reason || 'No model result is available.')}</div>
        <p class="prose">You can still inspect the trial’s drugs, diseases, and graph connections.</p>
        ${result.retryable ? '<button class="model-action" data-retry>Retry prediction</button>' : ''}
        ${nextExamples()}${metrics(benchmark)}${historical(benchmark)}`;
      root.querySelector('[data-retry]')?.addEventListener('click', () => load(nct));
      bindExamples();
      return;
    }
    const { scores, inputs } = result;
    const delta = (scores.attention - scores.control) * 100;
    const membership = { train: 'This trial was in the training set. Its score is not an independent evaluation.', valid: 'This trial was in the validation set used to select checkpoints.', test: 'This trial was in the held-out TOP test set. The live record may differ from its historical inputs.' }[result.benchmarkMembership];
    root.innerHTML = `${heading}
      <section class="model-score-card"><span class="score-label">Our model · success score</span>
        <div class="model-score">${fixed(scores.attention)}<span>/ 1</span></div>
        <div class="score-track" aria-hidden="true"><span style="width:${Math.max(0, Math.min(100, scores.attention * 100))}%"></span></div>
        <div class="score-scale"><span>Failure class</span><span>Success class</span></div>
        <div class="model-row"><span>Matched HINT control</span><b>${fixed(scores.control)}</b></div>
        <p class="model-footnote">Difference: ${delta >= 0 ? '+' : ''}${delta.toFixed(2)} percentage points. A higher score is not proof of a better prediction.</p>
      </section>
      <p class="prose">This is an experimental score for the benchmark’s overall trial-success label. It is not calibrated clinical odds or a separate safety/efficacy estimate.</p>
      ${membership ? `<div class="model-notice">${esc(membership)}</div>` : '<p class="model-footnote">This NCT ID is absent from our Phase II train, validation, and test files.</p>'}
      <section class="prediction-section"><h3>What the model used</h3>
        ${inputs.compounds.map((c) => `<div class="input-fact"><span>Drug structure</span><a href="${esc(c.source)}" target="_blank" rel="noopener">${esc(c.intervention)} · PubChem ${c.cid} ↗</a></div>`).join('')}
        ${inputs.diseases.map((d) => `<div class="input-fact"><span>Disease codes</span><b>${esc(d.condition)}</b><p>${esc(d.codes.join(', '))}</p></div>`).join('')}
        <div class="input-fact"><span>Eligibility</span><b>${inputs.inclusionLines} inclusion / ${inputs.exclusionLines} exclusion lines</b></div>
        ${inputs.sharedGroups ? '<p class="model-footnote">The upstream parser could not separate the two groups and used the full text for both, matching training.</p>' : ''}
        ${inputs.truncatedLines ? `<p class="model-footnote">${inputs.truncatedLines} long lines were capped at 512 tokens.</p>` : ''}
        <p class="model-footnote">Disease codes come from the training-input dictionary, including conservative name-format aliases. Registry outcomes, completion status, and published results are excluded from the model inputs.</p>
      </section>
      <details class="prediction-details"><summary>Eligibility lines emphasized by attention</summary>
        <p class="prose">These weights show how our pooling layer combined lines. They do not explain causality or prove why the trial will succeed.</p>
        ${['inclusion', 'exclusion'].map((group) => `<h3>${group}</h3>${result.attention[group].map((s) => `<div class="attention-line"><b>${pct(s.weight)}</b><p>${esc(s.text)}</p></div>`).join('')}`).join('')}
      </details>
      ${metrics(benchmark)}${historical(benchmark)}
      <section class="prediction-section"><h3>Graph + model</h3><p class="prose">PrimeKG maps the drug–disease relationships. HINT uses molecular graphs, disease codes, and eligibility text. The displayed PrimeKG paths do not generate this score.</p></section>
      <details class="prediction-details"><summary>Run provenance & limitations</summary>
        <p class="prose">${esc(result.generatedAt)} · ${result.durationSeconds}s on Azure CPU${result.cached ? ' · reused within the 15-minute cache window' : ''}.</p>
        <p class="prose">Sentence-attention checkpoint: seed ${benchmark.models.attention.seed}, selected epoch ${benchmark.models.attention.selectedEpoch}.</p>
        <code class="model-hash">${esc(benchmark.models.attention.checkpointSha256)}</code>
        <ul class="model-limitations">${result.limitations.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
        <a href="${esc(benchmark.upstreamUrl)}" target="_blank" rel="noopener">HINT source & noncommercial terms ↗</a>
      </details>
      <a class="model-action" data-download download="verdia-${esc(nct)}-prediction.json">Download this prediction record ↓</a>`;
    bindDownload(result);
  }
  return { load, reset };
}
