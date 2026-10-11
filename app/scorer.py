"""Transparent plausibility scorer.

A logistic model over graph-derived features (coefficients displayed in the UI),
optionally blended with structural-embedding similarity from prep/02_train_embeddings.ipynb.

Verdict bands: HIGH / MEDIUM / LOW / UNASSESSABLE. Contraindication edges drop
the score and raise a safety flag. Every feature is shown with its contribution.
"""
