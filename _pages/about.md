---
permalink: /
title: "Pavel Golikov"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

> 🚨 **I am currently transitioning to industry and actively seeking full-time roles as an AI Researcher / Scientist, with a focus on robust machine reasoning and AI alignment.**

I am an AI Researcher at the [University of Toronto](https://www.utoronto.ca/), working in [Prof. Gennady Pekhimenko's](https://www.cs.toronto.edu/~pekhMDL/) EcoSystem research group, and a member of the [Vector Institute](https://vectorinstitute.ai/). My research focuses on understanding the fundamental limits of algorithmic reasoning in Large Language Models. I use adversarial evaluations and mechanistic interpretability to diagnose structural failures in frontier models—such as context pollution and intra-query attention dilution.

My path to AI research is highly multidisciplinary, blending formal logic, low-level systems engineering, and security analysis. Before my graduate studies, I served as an Intelligence Operator in the Canadian Armed Forces, where I held a Top Secret clearance and analyzed classified information. This experience ingrained the rigorous, threat-modeling mindset I now apply to AI security and adversarial testing.

Academically, I began with a BSc in Mathematics and Philosophy (formal logic). In my Master's research, I focused on distributed systems, building full-stack IoT streaming framework using C++, Python, AWS, and Apache Flink. Today, my background in computer systems allows me to approach ML not just mathematically, but with a rigorous engineering lens.

## Current Research

* **Context Management Benchmark:** Investigating foundation models' ability to maintain working memory and logical coherence over arbitrarily long, customizable mathematical and coding reasoning chains. This work explores the boundaries of attention mechanisms in agentic and multi-step tasks.

## Selected Publications

* **Robust Reasoning Benchmark**
  **Pavel Golikov**, Evgenii Opryshko, Gennady Pekhimenko, and Mark C. Jeffrey.
  *arXiv preprint arXiv:2604.08571*, 2026. (Under review at NeurIPS 2026)
  [[arXiv]](https://arxiv.org/abs/2604.08571) | [[DOI]](https://doi.org/10.48550/arXiv.2604.08571) | [[Project Page]](https://github.com/pavelgolikov/Robust-Reasoning-Benchmark)
  > *Brief:* Introduced RRB to evaluate structural fragility in LLM reasoning. Used mechanistic interpretability to identify "Intra-Query Attention Dilution" in open-weights models and exposed critical over-refusal vulnerabilities in proprietary safety filters (Claude 4.6 Opus). Raised the open problem of the optimal granularity of reasoning - a critical question for AI safety, model alignment, and context management.

* **Fusing Adds and Shifts for Efficient Dot Products**
  **Pavel Golikov**, Karthik Ganesan, Gennady Pekhimenko, and Mark C. Jeffrey.
  *IEEE Computer Architecture Letters*, 25(1), pp. 33-36, 2025.
  [[DOI]](https://doi.org/10.1109/LCA.2025.3637718)
  > *Brief:* Hardware architecture research proposing a novel algorithmic optimization for dot-product computations.

## Master's Thesis & Systems Infrastructure

* **Flexible IoT Streaming Engine Framework** (MSc Thesis, 2022)
  > *Brief:* Engineered a distributed data-streaming framework to automatically partition streaming compute queries between edge devices and the cloud. Built the full stack, including Arduino/C++/Python sensor programming (EMG/ECG data processing), socket networking, and cloud deployment using AWS and Apache Flink.