"""LLM narrator — writes English, decides nothing.

Input: the structured facts the graph engine found (paths, edge types, score).
Constraint: "use ONLY these facts." Deterministic template fallback if the API
is unavailable, so the demo never depends on the network.
"""
