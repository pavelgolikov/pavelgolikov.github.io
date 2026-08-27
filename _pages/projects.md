---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
redirect_from:
  - /portfolio/
---

## arXiv Research Agent

An evaluation-first retrieval-augmented generation (RAG) system for grounded literature review. Built with LangGraph, LangChain, Gemini, Chroma vector storage, BM25, hybrid retrieval, cross-encoder reranking, Pydantic, SQLite, and PyMuPDF. The project includes hand-labeled retrieval and screening datasets, reproducible evals, citation-grounding checks, resumable agent workflows, and 163 offline tests in CI.

[[Project Details]](/projects/arxiv-research-agent/) | [[GitHub]](https://github.com/pavelgolikov/arxiv-research-agent) | [[Evals]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/EVALS.md) | [[Example Review]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/examples/example_review.md)

## ArbiGraph

An open-source benchmark generator for testing whether tool-assisted LLM agents can retain, update, propagate, and discard typed computational state across arbitrarily scalable task graphs. Includes executable ground truth, controllable DAG topologies, exact evaluation, and released example datasets.

[[arXiv]](https://arxiv.org/abs/2607.20764) | [[GitHub]](https://github.com/pavelgolikov/ArbiGraph) | [[Hugging Face]](https://huggingface.co/datasets/PavelGolikov/arbigraph)
