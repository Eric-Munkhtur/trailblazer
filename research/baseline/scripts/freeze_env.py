"""Capture dependency versions, hardware info, upstream commit, and code-change patch."""
import os, subprocess, json, platform, sys
REPO = "/content/clinical-trial-outcome-prediction"
OUT = "/content/hint_evidence/environment"
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
info = {
    "python": sys.version, "platform": platform.platform(),
    "gpu": subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                          shell=True, capture_output=True, text=True).stdout.strip(),
    "torch": __import__("torch").__version__,
    "cuda_runtime": subprocess.run("nvcc --version 2>/dev/null | tail -1", shell=True, capture_output=True, text=True).stdout.strip(),
    "cpu_cores": os.cpu_count(),
    "upstream_commit": subprocess.run("git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip(),
    "upstream_remote": "https://github.com/futianfan/clinical-trial-outcome-prediction",
}
import rdkit, sklearn, numpy, transformers
info["rdkit"], info["sklearn"], info["numpy"], info["transformers"] = rdkit.__version__, sklearn.__version__, numpy.__version__, transformers.__version__
json.dump(info, open(f"{OUT}/environment.json", "w"), indent=2)
# full pip freeze
subprocess.run("pip freeze > /content/hint_evidence/environment/requirements_freeze.txt", shell=True)
# upstream code changes: diff tracked CODE files (exclude data artifacts)
subprocess.run("git status --short > /content/hint_evidence/environment/git_status.txt", shell=True)
subprocess.run("git diff -- '*.py' '*.md' '*.yml' '*.ipynb' > /content/hint_evidence/environment/upstream_code_changes.patch", shell=True)
print(json.dumps(info, indent=2))
print("patch size:", os.path.getsize(f"{OUT}/upstream_code_changes.patch"), "bytes (0 = no upstream code changes)")
print(open(f"{OUT}/git_status.txt").read())
