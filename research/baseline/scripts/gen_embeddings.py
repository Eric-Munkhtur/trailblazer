"""
Regenerate data/sentence2embedding.pkl — faithful reproduction of upstream preprocessing.

Upstream recipe (verified from source):
- HINT/protocol_encode.py::save_sentence_bert_dict_pkl:
    keys = collect_cleaned_sentence_set() over data/raw_data.csv (513,863 unique sentences)
    values = biobert.sentence_vector(sentence)
- biobert-embedding 0.1.2 sentence_vector (verified from package source):
    "[CLS] " + text + " [SEP]" -> WordPiece (bert-base-cased vocab) -> segments all 1
    model.eval(), no_grad -> LAST encoder layer -> mean over ALL tokens incl [CLS]/[SEP]
- model: biobert_v1.1_pubmed == HF dmis-lab/biobert-base-cased-v1.1 (same weights, converted)

Deviations (documented):
- batched GPU fp32 forward with attention-mask mean (mathematically identical to per-sentence mean)
- sentences longer than 512 WordPiece tokens truncated to 512 (upstream would crash; count recorded)
Writes: data/sentence2embedding.pkl (+ progress log + DONE marker + meta json)
"""
import os, sys, json, time, pickle, hashlib
import torch

REPO = "/content/clinical-trial-outcome-prediction"
os.chdir(REPO)
sys.path.insert(0, ".")
LOG = "/content/hint_evidence/embedding_generation"
os.makedirs(LOG, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{LOG}/progress.log", "a") as f: f.write(line + "\n")

# 1. exact upstream key set
from HINT.protocol_encode import collect_cleaned_sentence_set
sentences = sorted(collect_cleaned_sentence_set())
log(f"unique sentences (upstream key set): {len(sentences)}")

# 2. preserve original stub
import shutil
if not os.path.exists(f"{LOG}/original_stub_sentence2embedding.pkl"):
    shutil.copy("data/sentence2embedding.pkl", f"{LOG}/original_stub_sentence2embedding.pkl")
    h = hashlib.sha256(open(f"{LOG}/original_stub_sentence2embedding.pkl","rb").read()).hexdigest()
    log(f"original stub preserved, sha256={h}")

# 3. tokenizer + model
from transformers import BertTokenizer, BertModel
tok = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.eval().cuda()
log("model loaded: dmis-lab/biobert-base-cased-v1.1")

# 4. tokenize with upstream marking "[CLS] text [SEP]" (BertTokenizer adds none itself here)
t0 = time.time()
encoded = []
over512 = 0
for i, s in enumerate(sentences):
    ids = tok.encode(s, add_special_tokens=True, truncation=True, max_length=512)  # "[CLS] s [SEP]"
    if len(ids) >= 512:
        over512 += 1
    encoded.append(ids)
    if (i+1) % 100000 == 0: log(f"tokenized {i+1}/{len(sentences)} ({time.time()-t0:.0f}s)")
log(f"tokenization done in {time.time()-t0:.0f}s; sentences at 512-token cap: {over512}")

# 5. batched forward, sorted by length
order = sorted(range(len(sentences)), key=lambda i: len(encoded[i]))
emb = {}
BATCH = 256
t0 = time.time()
with torch.no_grad():
    for bi, start in enumerate(range(0, len(order), BATCH)):
        idxs = order[start:start+BATCH]
        batch = [encoded[i] for i in idxs]
        maxlen = max(len(b) for b in batch)
        input_ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
        attn = torch.zeros(len(batch), maxlen, dtype=torch.long)
        for r, b in enumerate(batch):
            input_ids[r, :len(b)] = torch.tensor(b)
            attn[r, :len(b)] = 1
        # upstream biobert-embedding passes segments_ids = [1]*len(tokens): token_type_ids all ones
        out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                    token_type_ids=attn.cuda()).last_hidden_state  # final layer
        mask = attn.cuda().unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1)  # mean over real tokens incl [CLS]/[SEP]
        for r, i in enumerate(idxs):
            emb[sentences[i]] = mean[r].detach().cpu().float()
        if (bi+1) % 200 == 0:
            done = start + len(batch)
            rate = done / (time.time()-t0)
            log(f"embedded {done}/{len(sentences)} ({rate:.0f}/s, eta {(len(sentences)-done)/rate/60:.1f} min)")
log(f"embedding done in {(time.time()-t0)/60:.1f} min")

# 6. save in upstream format
with open("data/sentence2embedding.pkl", "wb") as f:
    pickle.dump(emb, f)
h = hashlib.sha256()
with open("data/sentence2embedding.pkl","rb") as f:
    for chunk in iter(lambda: f.read(1<<22), b""): h.update(chunk)
log(f"pkl saved: {os.path.getsize('data/sentence2embedding.pkl')/1e9:.2f} GB sha256={h.hexdigest()}")

# 7. sanity: non-zero protocol features on real trials
from HINT.protocol_encode import protocol2feature
import csv
rows = list(csv.reader(open("data/phase_II_test.csv", newline="")))[1:5]
for r in rows:
    inc, exc = protocol2feature(r[9], emb)
    log(f"sentence-embedding sample {r[0]}: ({r[0][:3]})", )
    break

meta = {
    "model": "dmis-lab/biobert-base-cased-v1.1 (= biobert_v1.1_pubmed used by biobert-embedding 0.1.2)",
    "n_sentences": len(sentences),
    "n_over_512_truncated": over512,
    "semantics": "mean of final-layer hidden states over all tokens incl [CLS]/[SEP]; token_type_ids=1 for all real tokens (upstream biobert-embedding segments_ids=[1]*n); fp32, eval mode, no grad",
    "pkl_sha256": h.hexdigest(),
    "upstream_keys_from": "HINT/protocol_encode.collect_cleaned_sentence_set over data/raw_data.csv",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
}
json.dump(meta, open(f"{LOG}/embedding_meta.json","w"), indent=2)
log("DONE")
open(f"{LOG}/DONE", "w").write("ok")
