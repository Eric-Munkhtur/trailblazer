# Verdia HackRU technical report

**Eric Munkhtur · HackRU 2026 · Version 1.1 · October 11, 2026**

[Read or download the paper (PDF)](verdia.pdf) · [Editable LaTeX source](verdia.tex) · [Recorded experiments](../research/README.md)

The report explains HINT in plain language, shows the original architecture with Verdia’s changes annotated, and documents both fine-tuning attempts, benchmark results, hardware, runtime, and limitations.

Both variants were fine-tuned from the authors’ Phase II checkpoint. Neither demonstrated a reliable improvement over the matched fine-tuned control. The website’s PrimeKG exploration and HINT prediction are separate paths.

This is a HackRU technical report, not a peer-reviewed publication. The paper distinguishes completed Colab CPU runs, T4 embedding generation, MSI GPU fine-tuning, and Azure CPU serving.

## Build from source

`verdia.tex` is standalone: its architecture figure, annotations, tables, and references are embedded. Compile with pdfLaTeX twice for cross-references, or upload it to a compatible LaTeX editor. The original HINT figure is credited in the caption and references; see [third-party terms](../THIRD_PARTY.md).

The author’s working source remains open in the original Codex document editor. This directory contains its matching distribution copy and PDF.
