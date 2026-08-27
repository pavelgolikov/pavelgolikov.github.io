---
layout: single
title: "arXiv Research Agent: Evaluation-First RAG"
permalink: /projects/arxiv-research-agent/
author_profile: true
excerpt: "A grounded RAG literature-review agent with hybrid vector retrieval, citation validation, reproducible LLM evals, and resumable LangGraph orchestration."
---

[[GitHub]](https://github.com/pavelgolikov/arxiv-research-agent) &#124; [[Evaluation Results]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/EVALS.md) &#124; [[Design Notes]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/DESIGN.md) &#124; [[Example Review]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/examples/example_review.md)

The **arXiv Research Agent** is an evaluation-first retrieval-augmented generation (RAG) system that turns a natural-language research question into a grounded Markdown literature review. It searches arXiv, screens candidates, downloads and parses selected papers, builds a persistent vector index, retrieves evidence for each analytical facet, validates every citation, and synthesizes the surviving claims with Gemini through LangChain.

## RAG & Vector Retrieval

The retrieval pipeline preserves PDF page provenance and assigns stable chunk IDs before storing embeddings in a persistent **Chroma vector database**. It supports four interchangeable retrieval strategies:

* dense semantic search with Gemini embeddings;
* sparse **BM25** keyword retrieval;
* hybrid dense+sparse retrieval using reciprocal-rank fusion; and
* hybrid retrieval followed by **cross-encoder reranking**.

Optional multi-query expansion generates paraphrases, fuses their candidate sets, and reranks once. Per-facet retrieval is scoped to a single paper so evidence cannot leak across sources.

## LLM Evals & Grounding

The project ships its evaluation artifacts rather than only claiming that the RAG pipeline works:

* a hand-labeled retrieval benchmark with **5 papers, 570 chunks, 50 questions, and 312 judged-relevant chunks**;
* a hand-labeled screening benchmark covering **7 research queries × 12 candidate papers**;
* four-way retrieval ablations using **MRR, nDCG@10, recall@5, and paired-bootstrap 95% confidence intervals**;
* relevance-threshold sweeps against the production selection rule;
* groundedness, citation-integrity, claim-support, and support-judge evaluations; and
* reproducible index rebuilding with a pool-coverage guard against evaluation drift.

The measured groundedness run retained **96.2% of proposed claims**, achieved **98.7% citation support integrity**, and passed **100% independent citation re-validation**. Against 70 human-graded citations, the support judge caught all 15 unsupported examples while falsely dropping 2 of 55 supported examples. The documented median end-to-end cost was **$0.054 per research question** across ten measured runs.

## Reliable Agent Orchestration

The LangGraph workflow uses `Send` map-reduce fan-out for concurrent screening and paper analysis, reducer-backed typed state, and deterministic sorting so output does not depend on concurrency. SQLite checkpointing powers `run`, `resume`, and `status`; completed branches are not repeated after interruption. Retry classification, exponential backoff, typed branch failures, and partial-report rendering allow useful results even when one paper or provider call fails.

The repository includes **163 tests** that require neither network access nor an API key, run in GitHub Actions on every push. The result is not only a demo of agentic RAG, but a measured and reproducible system for vector retrieval, grounded generation, and LLM evaluation.

## Stack

**LangGraph, LangChain, Gemini, Chroma, BM25, sentence-transformers, PyTorch, Pydantic, SQLite, PyMuPDF, pytest, GitHub Actions, Python 3.13.**
