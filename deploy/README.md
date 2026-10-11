# Azure graph and HINT deployment

Host: Ubuntu 24.04, `azureuser@20.83.98.87`, resource group `VERDIA`.
VM: Standard_B2als_v2, 2 vCPU / 4 GB. No VM resize or paid LLM is configured.

This release serves the existing Three.js interface, Node API, compact PrimeKG
graph, and a private CPU HINT inference worker. The Prediction tab runs the
sentence-attention model and matched HINT control for supported Phase II trials.
Saved benchmark results are labeled separately. See `model_service/README.md`.

## Layout

- App: `/home/azureuser/verdia`
- Node 22: `/home/azureuser/.local/node/bin/node`
- Service: `verdia.service`, loopback port 8790, mock LLM
- Model service: `verdia-model.service`, loopback port 8791, two CPU threads
- HTTPS: Caddy, public ports 80/443, forwarding to the loopback API
- Logs: `journalctl -u verdia` and `journalctl -u caddy`

Build the frontend locally with `npm run build`. Transfer only `dist/`, `server/`,
`package.json`, `package-lock.json`, compact graph assets, and deployment files.
Run `npm ci --omit=dev --ignore-scripts` with the Node directory on PATH remotely.
Copy the service/Caddy configuration, enable the service, and reload Caddy.
Public DNS is the Azure public-IP DNS label; no purchased domain is required.

Never transfer `.env`, SSH keys, raw data downloads, or training caches as part of
this release. The remote `features.json` is deliberately `{}`: the public build
contains node names, identifiers, graph relationships and layout, without bundled
DrugBank/Mayo/other descriptive text. Original local datasets are left intact.

The API limits graph-search concurrency to two, prediction HTTP concurrency to
four, and requests to twenty per client per minute, replay cache to three entries of at most 2 MiB each, and fetched-record cache
to 64 records with a 15-minute lifetime. Stream heartbeats keep connections active.
The graph path display reports traversal cost, not an outcome probability.

The client graph runs on each visitor's device. The model worker uses about
857 MB at startup on this VM; no GPU or paid LLM is required for inference.
Install the pinned Python packages in the dedicated venv, provision the trusted
checkpoint assets and pinned BioBERT snapshot, then enable `verdia-model.service`.
Only the six named model files in `model_service/README.md` are transferred under
`data/predictions`; that directory is never served as public static files.

No automated test suite was added or run. Production build, service health,
external reachability, and browser inspection are recorded separately.

The graph starts off and only animates once a query maps seed nodes. Clearing
the query stops rendering and cancels the client requests. Examples now select
supported Phase II studies. Observational studies have an explicit graph-only
status; saved historical scores, where present, are labeled separately.
