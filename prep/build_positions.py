"""Precompute 3D positions for ALL PrimeKG nodes — the 'galaxy' backdrop.

Nodes are clustered by type (drugs district, proteins district, …) so the
full 128k-node graph can be rendered as a static star-field without running
an 8M-link force simulation in the browser. Deterministic (seeded by name).
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")

TYPE_ORDER = ["drug", "disease", "gene/protein", "biological_process",
              "molecular_function", "cellular_component", "pathway",
              "anatomy", "effect/phenotype"]


def seed_point(name: str, radius: float) -> tuple[float, float, float]:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    # uniform-ish in a ball
    v = rng.normal(size=3)
    v = v / (np.linalg.norm(v) + 1e-9) * radius * rng.random() ** (1 / 3)
    return float(v[0]), float(v[1]), float(v[2])


def main() -> None:
    frames = []
    for i, t in enumerate(TYPE_ORDER):
        path = os.path.join(ART, f"nodes_{t.replace('/', '_')}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ang = 2 * math.pi * i / len(TYPE_ORDER)
        cx, cy, cz = 1500 * math.cos(ang), 400 * math.sin(2 * ang), 1500 * math.sin(ang)
        r = 80 + 3.4 * len(df) ** (1 / 3)
        pts = np.array([seed_point(n, r) for n in df["node_name"]])
        frames.append(pd.DataFrame({
            "name": df["node_name"], "type": t,
            "x": pts[:, 0] + cx, "y": pts[:, 1] + cy, "z": pts[:, 2] + cz,
        }))
        print(f"{t:22} {len(df):6} nodes  center=({cx:.0f},{cy:.0f},{cz:.0f}) r={r:.0f}")

    out = pd.concat(frames, ignore_index=True).round(0).astype({"x": int, "y": int, "z": int})
    out.to_csv(os.path.join(ART, "positions.csv.gz"), index=False, compression="gzip")
    print(f"\nwrote positions.csv.gz: {len(out):,} nodes")


if __name__ == "__main__":
    main()
