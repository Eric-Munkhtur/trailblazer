"""
Verification: HF transformers implementation == pytorch-pretrained-bert 0.6.2 (biobert-embedding path)
on dmis-lab/biobert-base-cased-v1.1 weights, sentence_vector semantics (final layer, mean incl CLS/SEP).
"""
import os, sys, subprocess, json
r = subprocess.run("pip install -q pytorch-pretrained-bert==0.6.2 2>&1 | tail -1", shell=True, capture_output=True, text=True)
print("install:", (r.stdout + r.stderr).strip()[:200])

os.makedirs("/content/biobert_pkg", exist_ok=True)
from huggingface_hub import hf_hub_download
d = "/content/biobert_pkg"
for fn in ["config.json", "vocab.txt", "pytorch_model.bin"]:
    p = hf_hub_download("dmis-lab/biobert-base-cased-v1.1", fn)
    import shutil; shutil.copy(p, os.path.join(d, fn))
import shutil as sh
sh.copy(f"{d}/config.json", f"{d}/bert_config.json")
print("weights downloaded")

import torch
from pytorch_pretrained_bert import BertTokenizer as OldTok, BertModel as OldModel
from transformers import BertTokenizer as NewTok, BertModel as NewModel

old_tok, new_tok = OldTok.from_pretrained(d), NewTok.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
old_model, new_model = OldModel.from_pretrained(d), NewModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
old_model.eval(); new_model.eval()

sents = [
    "Patients must have histologically confirmed breast cancer.",
    "Age >= 18 years old.",
    " serum creatinine > 2.0 mg/dL",   # leading space on purpose (real data has these)
    "Pregnant or breastfeeding women are excluded.",
    "Signed written informed consent prior to any study-specific procedures.",
]
res = {"sentences": sents, "comparisons": []}
with torch.no_grad():
    for s in sents:
        # OLD package path (exact biobert-embedding sentence_vector)
        marked = "[CLS] " + s + " [SEP]"
        toks = old_tok.tokenize(marked)
        ids = old_tok.convert_tokens_to_ids(toks)
        enc, _ = old_model(torch.tensor([ids]), torch.tensor([[1]*len(ids)]))
        old_vec = torch.mean(enc[11][0], dim=0)
        # NEW path (my generator)
        ids2 = new_tok.encode(s, add_special_tokens=True)
        assert toks == new_tok.tokenize(marked), f"tokenization mismatch: {toks} vs {new_tok.tokenize(marked)}"
        out = new_model(input_ids=torch.tensor([ids2]), attention_mask=torch.tensor([[1]*len(ids2)]),
                        token_type_ids=torch.ones(1, len(ids2), dtype=torch.long)).last_hidden_state
        new_vec = torch.mean(out[0], dim=0)
        diff = (old_vec - new_vec).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(old_vec, new_vec, dim=0).item()
        res["comparisons"].append({"max_abs_diff": diff, "cosine": round(cos, 8), "n_tokens": len(ids)})
        print(f"n_tok={len(ids)} max_abs_diff={diff:.3e} cos={cos:.8f}")

ok = all(c["max_abs_diff"] < 1e-4 for c in res["comparisons"])
res["verdict"] = "EQUIVALENT (max_abs_diff < 1e-4 on all probes)" if ok else "CHECK FAILED"
print("VERDICT:", res["verdict"])
os.makedirs("/content/hint_evidence/embedding_generation", exist_ok=True)
json.dump(res, open("/content/hint_evidence/embedding_generation/equivalence_check.json", "w"), indent=2)
