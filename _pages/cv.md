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

[**📄 Download 1-Page PDF CV**](/files/Pavel_Golikov_CV.pdf)

## Professional Profile
AI Researcher specializing in machine reasoning and alignment. Blends formal logic with rigorous low-level systems engineering to architect custom agentic frameworks and highly creative adversarial evaluations. Former Military Intelligence Operator with a Top Secret clearance, bringing a rigorous, threat-modeling mindset to AI security and model capability evaluations.

## Technical Skills
* **Languages:** Python, C++, Java, SQL, LaTeX
* **ML & AI:** PyTorch, vLLM, HuggingFace, Transformers, Google/Anthropic/OpenAI APIs, Agent Harness
* **Systems & Infrastructure:** Linux/Ubuntu Server, AWS, Apache Flink, Distributed GPU Clusters
* **Core Competencies:** Large Language Models (LLMs), Machine Reasoning, AI Alignment, Context Management, Mechanistic Interpretability, Adversarial Evaluations, Threat Modeling, Distributed Systems

## First-Author AI Research
* **ArbiGraph** (2026 – Present)
  [[GitHub]](https://github.com/pavelgolikov/ArbiGraph)
  * Built **ArbiGraph**, a Python evaluation framework for testing whether LLM agents can follow long, multi-step workflows without losing, mixing up, or reusing stale intermediate state.
  * Converted math, word-problem, and Python-tracing tasks into automatically generated workflows with executable ground truth, enabling exact grading without manual labels.
  * Added controls for workflow length, branching, irrelevant distractors, and value types, letting researchers reproduce agent failure modes and scale difficulty without hand-crafting prompts.
  * Implemented the agent evaluation harness around a calculator tool, including answer extraction, tool-call validation, continuation handling, and repair prompts for incomplete or malformed runs.

* **Robust Reasoning Benchmark (RRB)** (2026)
  [[arXiv]](https://arxiv.org/abs/2604.08571) | [[DOI]](https://doi.org/10.48550/arXiv.2604.08571) | [[GitHub]](https://github.com/pavelgolikov/Robust-Reasoning-Benchmark.git)
  * Designed a highly creative adversarial evaluation framework leveraging 13 deterministic textual perturbations to decouple an LLM's mechanical deciphering from its underlying mathematical logic.
  * Demonstrated that the attention drift occurs *within a single query's Chain-of-Thought*, empirically showing that intermediate reasoning steps pollute the dense attention mechanism, identifying the optimal **granularity of reasoning** as a critical open research problem.
  * Engineered custom mechanistic interpretability pipelines in **PyTorch** to extract and analyze causal attention probability matrices across token index boundaries, testing models ranging from 7B to 30B parameters.

* **Fusing Adds and Shifts for Efficient Dot Products** (2026)
  [[IEEE CAL]](https://ieeexplore.ieee.org/abstract/document/11269714) | [[GitHub]](https://github.com/mcj-group/fased-verilog.git)
  * Proposed and validated a novel algorithmic optimization for dot-product computations, demonstrating a strong foundational understanding of hardware-level ML primitives and efficiency.

## Engineering & Professional Experience
* **Systems & Infrastructure Engineering (MSc Thesis)** (2020 – 2022)
  *University of Toronto*
  * Engineered a flexible IoT distributed data-streaming framework from scratch, designed to automatically partition computational streaming queries between edge devices and cloud instances.
  * Built the full software stack: programmed Arduino/C++ sensors for real-time biological data collection (EMG/ECG), developed custom socket networking protocols, and deployed cloud infrastructure using AWS and Apache Flink.

* **Intelligence Operator** (2013 – 2018)
  *Canadian Armed Forces*
  * Held a Top Secret security clearance, conducting rigorous analysis of classified information streams to produce actionable intelligence reports for command elements.
  * Developed a strong adversarial threat-modeling mindset, emphasizing operational security, rigorous data validation, and the identification of logical vulnerabilities in complex, multi-agent scenarios.

* **Mathematics Teacher** (2012 – 2013; 2018 – 2019)
  *Blyth Academy*
  * Taught foundational mathematics to students in Grades 10, 11, and 12, developing the ability to distill and communicate complex quantitative concepts.

## Co-Authored Systems Research
* **GPUPool: A Holistic Approach to Fine-Grained GPU Sharing in the Cloud**
  *PACT 2022* | Co-authored with Xiaodan Serina Tan, Nandita Vijaykumar, Gennady Pekhimenko.
* **Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training**
  *USENIX ATC 2021* | Co-authored with Geoffrey X. Yu, Yubo Gao, Gennady Pekhimenko.

## Education
* **PhD in Computer Science (Paused to transition to industry)**, University of Toronto, 2022 – Present
* **Master of Science (MSc) in Computer Science**, University of Toronto, 2020 – 2022
* **Bachelor of Science (BSc) in Mathematics and Philosophy (Formal Logic)**, University of Toronto, Graduated 2011
