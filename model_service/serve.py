"""Private CPU inference worker. Binds only to loopback; no outcome fields used.

Loads the trusted, hash-checked authors' object to preserve GRAM code ordering,
then strictly restores the two trained state dictionaries. No training occurs.
"""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(os.environ.get('MODEL_ASSETS', ROOT / 'data/predictions'))
sys.path[:0] = [str(ROOT / 'model_service/vendor'), str(ROOT / 'model_service')]

import torch
from rdkit import Chem
from transformers import BertModel, BertTokenizer
from HINT.model import HINTModel
from HINT.protocol_encode import split_protocol
from hint_attention import install_attention

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
EVIDENCE = json.loads((ASSETS / 'benchmark.json').read_text())
DISEASES = json.loads((ASSETS / 'disease-map.json').read_text())
MEMBERSHIP = json.loads((ASSETS / 'membership.json').read_text())
BIOBERT_REVISION = '924f12e0c3db7f156a765ad53fb6b11e7afedbc8'


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise RuntimeError(f'Checkpoint hash mismatch: {path.name}')


verify(ASSETS / 'authors_phase_II.ckpt', EVIDENCE['inputSha256']['phase_II.ckpt'])
# Only this reviewed local checkpoint is unpickled. Never accept uploaded models.
base = torch.load(ASSETS / 'authors_phase_II.ckpt', map_location='cpu', weights_only=False)
if not isinstance(base, HINTModel):
    raise RuntimeError('Unexpected checkpoint architecture')
MODELS = {}
for arm in ('control', 'attention'):
    path = ASSETS / f'{arm}_state.pt'
    verify(path, EVIDENCE['models'][arm]['checkpointSha256'])
    model = copy.deepcopy(base)
    if arm == 'attention':
        install_attention(model)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to('cpu')
    for module in model.modules():
        if hasattr(module, 'device'):
            module.device = torch.device('cpu')
    model.adj = model.adj.cpu()
    model.disease_encoder.padding_matrix = model.disease_encoder.padding_matrix.cpu()
    model.disease_encoder.mask_matrix = model.disease_encoder.mask_matrix.cpu()
    model.eval()
    MODELS[arm] = model
del base

TOKENIZER = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1', revision=BIOBERT_REVISION, local_files_only=True)
BERT = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1', revision=BIOBERT_REVISION, local_files_only=True).eval()
LOCK = threading.Lock()
STARTED = time.time()


def normal(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s.lower()).split())


def unavailable(reason, **details):
    return {'available': False, 'reason': reason, **details}


@torch.inference_mode()
def predict(payload):
    start = time.monotonic()
    nct = payload.get('nct', '')
    if not re.fullmatch(r'NCT\d{8}', nct):
        return unavailable('Invalid trial identifier.')
    if 'PHASE2' not in payload.get('phases', []):
        return unavailable('This checkpoint was trained for Phase II. This trial is outside its supported phase.')
    text = payload.get('eligibilityCriteria', '')
    if not isinstance(text, str) or not text.strip() or len(text) > 60000:
        return unavailable('A complete eligibility section of up to 60,000 characters is required.')
    conditions = payload.get('conditions', [])
    if not conditions or len(conditions) > 20:
        return unavailable('No supported condition list was supplied.')
    mapped, code_groups = [], []
    for name in conditions:
        # Exact training-dictionary mapping only: no invented or fuzzy ICD codes.
        codes = DISEASES.get(normal(name))
        if not codes:
            return unavailable(f'No unambiguous training-dictionary disease-code match for “{name}”. The graph can still be explored.')
        usable = [c for c in codes if c in MODELS['attention'].disease_encoder.code2index]
        if not usable:
            return unavailable(f'The disease codes for “{name}” are outside the model vocabulary.')
        mapped.append({'condition': name, 'codes': usable, 'omittedCodes': [c for c in codes if c not in usable],
                       'source': 'TOP Phase II training dictionary, including conservative name-format aliases'})
        code_groups.append(usable)
    compounds = payload.get('compounds', [])
    if not 1 <= len(compounds) <= 12:
        return unavailable('Between one and twelve resolved drug structures are required.')
    smiles = []
    for compound in compounds:
        smi = compound.get('smiles', '')
        if not isinstance(smi, str) or len(smi) > 6000:
            return unavailable('Drug structure exceeds the demo input limit.')
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not 1 <= mol.GetNumAtoms() <= 600:
            return unavailable('A drug structure could not be read within the model input limit.')
        if smi not in smiles:
            smiles.append(smi)

    parts = split_protocol(text)
    inclusion, exclusion = parts[0], parts[-1]
    sentences = list(dict.fromkeys(inclusion + exclusion))
    if not sentences or len(sentences) > 200:
        return unavailable('The demo supports eligibility sections with 1–200 nonempty lines; no lines were silently dropped.')
    vectors, truncated = {}, 0
    # Fresh embeddings for this record; never use the benchmark's bulk cache.
    encoded = []
    for sentence in sentences:
        ids = TOKENIZER.encode(sentence, add_special_tokens=True)
        if len(ids) > 512:
            truncated += 1
            ids = TOKENIZER.encode(sentence, add_special_tokens=True, truncation=True, max_length=512)
        encoded.append(ids)
    order = sorted(range(len(sentences)), key=lambda i: len(encoded[i]))
    for at in range(0, len(order), 4):
        chosen = order[at:at + 4]
        width = max(len(encoded[i]) for i in chosen)
        ids = torch.zeros(len(chosen), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(chosen):
            ids[row, :len(encoded[i])] = torch.tensor(encoded[i])
            mask[row, :len(encoded[i])] = 1
        hidden = BERT(input_ids=ids, attention_mask=mask, token_type_ids=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        for row, i in enumerate(chosen):
            vectors[sentences[i]] = pooled[row]
    features = [(torch.stack([vectors[s] for s in inclusion]), torch.stack([vectors[s] for s in exclusion]))]
    scores = {}
    for arm, model in MODELS.items():
        score = torch.sigmoid(model([smiles], [code_groups], features).reshape(-1))[0]
        if not torch.isfinite(score):
            raise RuntimeError('Non-finite model output')
        scores[arm] = float(score)
    _, attention = MODELS['attention'].protocol_encoder.forward_with_attention(features)
    highlights = {}
    for key, lines in [('inclusion', inclusion), ('exclusion', exclusion)]:
        ranked = sorted(zip(lines, attention[0][key].tolist()), key=lambda x: -x[1])[:4]
        highlights[key] = [{'text': s, 'weight': w} for s, w in ranked]
    return {
        'available': True, 'mode': 'live_inference', 'nct': nct, 'scores': scores,
        'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'durationSeconds': round(time.monotonic() - start, 2),
        'benchmarkMembership': MEMBERSHIP.get(nct),
        'inputs': {'compounds': compounds, 'diseases': mapped,
                   'eligibilitySha256': hashlib.sha256(text.encode()).hexdigest(),
                   'inclusionLines': len(inclusion), 'exclusionLines': len(exclusion),
                   'sharedGroups': len(parts) == 1, 'truncatedLines': truncated},
        'attention': highlights,
        'preprocessing': {'encoder': 'dmis-lab/biobert-base-cased-v1.1', 'revision': BIOBERT_REVISION,
                          'recipe': 'fp32; final layer; mean over real tokens including CLS/SEP; token_type_ids=1; upstream line parser'},
        'limitations': [
            'Experimental score for the TOP success label, not separate efficacy or safety predictions.',
            'The current registry record may include edits made after the historical trial finished.',
            'The live input-mapping pipeline has not been evaluated as a separate benchmark.',
            'PrimeKG paths are context and are not inputs to this checkpoint.',
            'Attention weights show pooling emphasis, not causal explanations.',
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, value):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path != '/health':
            return self.send_json(404, {'error': 'Not found'})
        self.send_json(200, {'ready': True, 'device': 'cpu', 'models': list(MODELS), 'busy': LOCK.locked()})

    def do_POST(self):
        if self.path != '/predict':
            return self.send_json(404, {'error': 'Not found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send_json(400, {'error': 'Invalid length'})
        if not 0 < size <= 131072:
            return self.send_json(413, {'error': 'Invalid input size'})
        if not LOCK.acquire(blocking=False):
            return self.send_json(429, {'error': 'The prediction worker is busy. Please retry shortly.'})
        try:
            payload = json.loads(self.rfile.read(size))
            self.send_json(200, predict(payload))
        except (ValueError, TypeError, KeyError):
            self.send_json(400, {'error': 'Invalid model input'})
        except Exception as error:
            print(f'Inference failed: {type(error).__name__}: {error}', flush=True)
            self.send_json(500, {'error': 'Model inference failed; no score was produced.'})
        finally:
            LOCK.release()


print('Verdia model worker ready on 127.0.0.1:8791 (CPU)', flush=True)
ThreadingHTTPServer(('127.0.0.1', 8791), Handler).serve_forever()
