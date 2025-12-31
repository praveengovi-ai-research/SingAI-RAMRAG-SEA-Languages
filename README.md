# SingAI-RAMRAG: Risk-Aware Multilingual Retrieval-Augmented Generation for Customer Sales and Care in Southeast Asian Languages

Small and Medium Business (SMB) customer care represents a uniquely challenging domain for AI automation: it combines high-volume, cost-sensitive operations with strict policy constraints, multilingual user bases, and zero tolerance for credential disclosure or policy fabrication. Southeast Asia amplifies these challenges through extraordinary linguistic diversity—English, Malay, Indonesian, Tamil, and Mandarin, often code-switched within single interactions—and region-specific regulatory requirements. While Retrieval-Augmented Generation (RAG) can ground responses in knowledge bases, we find that retrieval-similarity thresholding alone effectively filters out-of-domain queries but fails to catch policy violations without unacceptable coverage loss.
We propose SingAI-RAMRAG, a two-stage risk-aware RAG controller combining: (i) a cosine-calibrated similarity gate for out-of-domain filtering, and (ii) Deterministic Symbolic Guardrails—explicit, auditable pattern-based rules for detecting credential disclosure, policy contradictions, and tool-required escalations. We deliberately choose deterministic symbolic methods over neural classifiers: in regulated SMB domains, compliance officers require inspectable decision logic, not black-box confidence scores. Evaluated on a 6,000-entry SMB knowledge base and 1,500 queries spanning six Southeast Asian language variants, our approach reduces policy violation allowance by 34 percentage points (from 94% to 60%) while improving the risk-coverage trade-off (AURC: 0.52 → 0.36). We observe significant multilingual retrieval variability (Singlish: 18.6% vs Malay: 4.0% recall), highlighting critical gaps for equitable regional deployment.

## The Problem

Standard RAG systems assume that if retrieved content is relevant, the response will be appropriate. In regulated customer care, this fails—semantic similarity doesn't prevent policy violations, credential leaks, or fabricated exceptions.

## Our Approach

SingAI-RAMRAG combines:

1. **Cosine-calibrated similarity gate** — filters out-of-domain queries
2. **Deterministic symbolic guardrails** — auditable pattern-based rules for detecting credential disclosure, policy contradictions, and escalation triggers

We use deterministic methods over neural classifiers because compliance officers need inspectable decision logic, not black-box scores.

## Results

- Policy violation allowance reduced by 34 percentage points (94% → 60%)
- Risk-coverage trade-off improved (AURC: 0.52 → 0.36)
- Tested on 6,000-entry knowledge base, 1,500 queries across six Southeast Asian language variants

Notable finding: significant multilingual retrieval variability (Singlish: 18.6% vs Malay: 4.0% recall).

## Architecture

Three-way routing: **Answer**, **Handoff**, or **Block**

Two-stage gating:
- Gate-A: Out-of-domain detection
- Gate-B: Policy violation detection

## Project Structure
```
SingAI_RAMRAG/
├── src/
│   ├── config/           # Configuration classes
│   ├── ingestion/        # Knowledge base indexing
│   ├── retrieval/        # Dense retriever
│   ├── guardrails/       # Symbolic guardrails (Gate-B)
│   └── pipeline.py       # Main pipeline
├── data/
├── notebooks/
└── tests/
```

## Datasets

We release two open datasets reflecting actual SMB support scenarios. See Huggingface datasets for details.

Huggingface datasets:
- praveengovi/multilingual-smb-sales-care-kb
- praveengovi/multilingual-smb-sales-care-eval-dataset

## Citation

___TO_BE_ADDED___

