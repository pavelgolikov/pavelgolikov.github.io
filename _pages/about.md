---
permalink: /
title: "Pavel Golikov"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

> 🚨 **I am currently transitioning to industry and actively seeking full-time roles as an AI Researcher / Scientist, with a focus on agentic AI, retrieval-augmented generation (RAG), LLM evaluation, robust machine reasoning, and AI alignment.**

I am an AI Researcher at the [University of Toronto](https://www.utoronto.ca/), working in [Prof. Gennady Pekhimenko's](https://www.cs.toronto.edu/~pekhMDL/) EcoSystem research group, and a member of the [Vector Institute](https://vectorinstitute.ai/). I build and evaluate reliable LLM systems, including agentic workflows, retrieval-augmented generation (RAG) pipelines, and dataset generators for context-management evaluation. My work spans hybrid vector search, grounded generation, and LLM evals through mechanistic interpretability and the fundamental limits of algorithmic reasoning.


My path to AI research is highly multidisciplinary, blending formal logic, low-level systems engineering, and security analysis. Before my graduate studies, I served as an Intelligence Operator in the Canadian Armed Forces, where I analyzed classified information. This experience ingrained the rigorous, threat-modeling mindset I now apply to AI security and adversarial testing.

Academically, I began with a BSc in Mathematics and Philosophy (formal logic). In my Master's research, I focused on distributed systems, building full-stack IoT streaming framework using C++, Python, AWS, and Apache Flink. Today, my background in computer systems allows me to approach ML not just mathematically, but with a rigorous engineering lens.

## Current Research & Open-Source Systems

* **ArbiGraph** - Built an open-source benchmark generator for evaluating context management in tool-assisted language agents. ArbiGraph composes executable natural-language math, GSM-style, and Python-tracing tasks into typed task graphs, making dependency length, distractors, branches, and scalar/list state controllable while preserving exact automatic verification. The framework supports user-defined DAG topologies, executable solvers, dynamic rejection sampling, graph visualization, and released datasets/results for reproducible evaluation. It also provides reinforcement learning with verifiable rewards (RLVR) environments with dataset-backed and on-demand episodes, hidden verifier state, exact binary rewards, and per-node diagnostics. In preliminary Qwen3.5-27B evaluations, isolated tasks remained high accuracy (94.5% math, 96.8% Python, 100.0% GSM), but dependent math accuracy fell to 75.5% on chains and 61.2% on multichains, exposing state-propagation failures hidden by single-task benchmarks. [[arXiv]](https://arxiv.org/abs/2607.20764) &#124; [[GitHub]](https://github.com/pavelgolikov/ArbiGraph) &#124; [[Hugging Face]](https://huggingface.co/datasets/PavelGolikov/arbigraph)

* **arXiv Research Agent** - Built an evaluation-first RAG literature-review agent with LangGraph, LangChain, Gemini, and persistent Chroma vector storage. The workflow searches arXiv, screens papers, performs page-preserving PDF parsing, and generates claim-level grounded reviews with page-anchored citations. Its retrieval layer supports dense embeddings, BM25 sparse search, hybrid reciprocal-rank fusion, cross-encoder reranking, and multi-query expansion. I evaluated the system on two hand-labeled datasets with MRR, nDCG, recall, groundedness, claim-support, threshold-sweep, and paired-bootstrap analyses; it also includes resumable SQLite checkpoints, typed partial failures, and 163 offline tests in CI. [[GitHub]](https://github.com/pavelgolikov/arxiv-research-agent) &#124; [[Project Details]](/projects/arxiv-research-agent/) &#124; [[Evals]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/EVALS.md)


## Selected Publications

* **ArbiGraph: Arbitrarily Scalable Verifiable Task Graphs for Evaluating Context Management** - Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, and Mark C. Jeffrey.
  *arXiv preprint arXiv:2607.20764*, 2026.
  [[arXiv]](https://arxiv.org/abs/2607.20764) &#124; [[GitHub]](https://github.com/pavelgolikov/ArbiGraph) &#124; [[Hugging Face]](https://huggingface.co/datasets/PavelGolikov/arbigraph)
  > *Brief:* Introduced ArbiGraph, a benchmark generator that turns context into typed computational state. It composes executable tasks into user-defined dataflow graphs, supports baseline/forgetting/chain/multichain evaluations, and tests whether agents can retain, update, propagate, and discard relevant context under exact verification. The public implementation also exposes these task graphs as RLVR environments with hidden verifier state, exact binary rewards, and per-node diagnostics.

* **Robust Reasoning Benchmark** - Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, and Mark C. Jeffrey.
  *arXiv preprint arXiv:2604.08571*, 2026. (Under review at NeurIPS 2026)
  [[arXiv]](https://arxiv.org/abs/2604.08571) &#124; [[DOI]](https://doi.org/10.48550/arXiv.2604.08571) &#124; [[Project Page]](https://github.com/pavelgolikov/Robust-Reasoning-Benchmark)
  > *Brief:* Introduced RRB to evaluate structural fragility in LLM reasoning. Used mechanistic interpretability to identify "Intra-Query Attention Dilution" in open-weights models and over-refusal in proprietary safety filters (Claude 4.6 Opus). Raised the open problem of the optimal granularity of reasoning - an important question for model reasoning and context management.

* **Fusing Adds and Shifts for Efficient Dot Products** - Pavel Golikov, Karthik Ganesan, Gennady Pekhimenko, and Mark C. Jeffrey.
  *IEEE Computer Architecture Letters*, 25(1), pp. 33-36, 2026.
  [[DOI]](https://doi.org/10.1109/LCA.2025.3637718)
  > *Brief:* Hardware architecture research proposing a novel algorithmic optimization for dot-product computations.

## Master's Thesis & Systems Infrastructure

* **Flexible IoT Streaming Engine Framework** (MSc Thesis, 2022)
  > *Brief:* Engineered a distributed data-streaming framework to automatically partition streaming compute queries between edge devices and the cloud. Built the full stack, including Arduino/C++/Python sensor programming (EMG/ECG data processing), socket networking, and cloud deployment using AWS and Apache Flink.
