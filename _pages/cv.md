---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

# Curriculum Vitae

[**📄 Download PDF CV**](/files/Pavel_Golikov_CV.pdf)

## Professional Profile
AI Researcher and agent engineer specializing in robust machine reasoning, retrieval-augmented generation (RAG), LLM evaluation, and alignment. Builds grounded agentic workflows, vector-retrieval systems, and dataset generators for context-management evaluation in LLM agents. Former Military Intelligence Operator who held a Top Secret clearance, bringing a threat-modeling mindset to AI security and model capability evaluation.

## Technical Skills
* **Languages:** Python, C++, Java, SQL, LaTeX
* **AI & Agents:** PyTorch, LangGraph, LangChain, RAG, Hugging Face, vLLM, Transformers, Gemini/OpenAI/Anthropic APIs
* **RAG & Evals:** Vector storage, embeddings, BM25, hybrid retrieval, cross-encoder reranking, MRR/nDCG/recall, groundedness, validation
* **Systems:** Linux, SQLite, GitHub Actions CI, AWS, Apache Flink, Distributed GPU Clusters

## AI Research & Agent Engineering
* **ArbiGraph** (2026 – Present)
  [[arXiv]](https://arxiv.org/abs/2607.20764) | [[GitHub]](https://github.com/pavelgolikov/ArbiGraph) | [[Hugging Face]](https://huggingface.co/datasets/PavelGolikov/arbigraph)
  * Built an open-source benchmark and dataset generator for evaluating context management in tool-assisted LLM agents across arbitrarily scalable, long-horizon task graphs.
  * Generated math, word-problem, and Python-tracing workflows with executable ground truth, enabling exact evaluation of state retention, updates, propagation, and stale-state reuse without manual labels.
  * Built reinforcement learning with verifiable rewards (RLVR) environments supporting dataset-backed and on-demand episodes, hidden verifier state, exact binary rewards, and per-node diagnostics.

* **arXiv Research Agent** (2026)
  [[GitHub]](https://github.com/pavelgolikov/arxiv-research-agent) | [[Evals]](https://github.com/pavelgolikov/arxiv-research-agent/blob/main/EVALS.md)
  * Built an evaluation-first **RAG** literature-review agent with **LangGraph, LangChain, Gemini, and Chroma vector storage**; orchestrates arXiv search, PDF parsing, hybrid retrieval, reranking, and citation-grounded synthesis.
  * Benchmarked dense, BM25, hybrid, and cross-encoder retrieval on 50 hand-labeled questions over 570 chunks using MRR, nDCG, recall, and paired-bootstrap confidence intervals; evaluated groundedness and claim support.
  * Engineered deterministic map-reduce fan-out, retry-aware partial results, and resumable **SQLite checkpointing**; shipped 163 offline tests in CI and measured a median end-to-end cost of $0.054 per run.

* **Robust Reasoning Benchmark (RRB)** (2026)
  [[arXiv]](https://arxiv.org/abs/2604.08571) | [[DOI]](https://doi.org/10.48550/arXiv.2604.08571) | [[GitHub]](https://github.com/pavelgolikov/Robust-Reasoning-Benchmark.git)
  * Designed a highly creative adversarial evaluation framework leveraging 13 deterministic textual perturbations to decouple an LLM's mechanical deciphering from its underlying mathematical logic.
  * Demonstrated that the attention drift occurs *within a single query's Chain-of-Thought*, empirically showing that intermediate reasoning steps pollute the dense attention mechanism.
  * Engineered custom mechanistic interpretability pipelines in **PyTorch** for layerwise attention-allocation analysis across token index boundaries, testing models ranging from 7B to 30B parameters.

* **Fusing Adds and Shifts for Efficient Dot Products** (2026)
  [[IEEE CAL]](https://ieeexplore.ieee.org/abstract/document/11269714) | [[GitHub]](https://github.com/mcj-group/fased-verilog.git)
  *Hardware ML Research — Toronto, ON*
  * Proposed and validated a novel algorithmic optimization for dot-product computations, demonstrating a strong foundational understanding of hardware-level ML primitives and efficiency.

## Engineering & Professional Experience
* **Graduate Researcher (PhD, on leave) — ML Systems & Agent Evaluation** (2022 – Present)
  *University of Toronto, EcoSystem Research Group*
  * Conducted research spanning ML systems and efficient computation, with later work focused on robust machine reasoning, agent evaluation, context management, and AI alignment, in the EcoSystem Research Group and as a member of the Vector Institute.
  * Developed open-source evaluation frameworks, agentic RAG systems, and mechanistic-interpretability analyses, resulting in first-author research on context management and reasoning robustness.

* **Graduate Researcher (MSc) — Distributed Systems** (2020 – 2022)
  *University of Toronto, EcoSystem Research Group*
  * Engineered a flexible IoT distributed data-streaming framework from scratch, designed to automatically partition computational streaming queries between edge devices and cloud instances.
  * Built the full software stack: programmed Arduino/C++ sensors for real-time biological data collection (EMG/ECG), developed custom socket networking protocols, and deployed cloud infrastructure using AWS and Apache Flink.

* **Intelligence Operator** (2013 – 2018)
  *Canadian Armed Forces*
  * Formerly held a Top Secret security clearance while conducting rigorous analysis of classified information streams to produce actionable intelligence reports for command elements.
  * Developed a strong adversarial threat-modeling mindset, emphasizing operational security, rigorous data validation, and the identification of logical vulnerabilities in complex, multi-agent scenarios.

* **Mathematics Teacher** (2012 – 2013 & 2018 – 2019)
  *Blyth Academy*
  * Taught foundational mathematics to students in Grades 10, 11, and 12, developing the ability to distill and communicate complex quantitative concepts.

## Education
* **PhD in Computer Science (on leave as of September 2026)**, University of Toronto, 2022 – Present
* **Master of Science (MSc) in Computer Science**, University of Toronto, 2020 – 2022
* **Bachelor of Science (BSc) in Mathematics and Philosophy (Formal Logic)**, University of Toronto, Graduated 2011
