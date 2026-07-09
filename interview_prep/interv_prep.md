Given your background, I would **not** study this like a beginner ML curriculum. I’d organize prep around the things interviewers expect you to **explain, derive, implement, debug, and design**.

The best order is:

> **ML/math fundamentals → deep learning → Transformers → LLM training/post-training → inference/evaluation/RAG → agentic systems → research/system-design practice**

Agents should come **after** LLMs, RAG, inference, and evaluation, because practical agentic systems are mostly composed from those pieces.

---

# 0. Interview prep mindset

For every topic, prepare at 4 levels:

1. **Conceptual** — explain it clearly.
2. **Mathematical** — derive core losses/objectives.
3. **Implementation** — code a minimal version in PyTorch/NumPy.
4. **Systems/research judgment** — discuss trade-offs, ablations, failure modes, and evaluation.

This is especially important for research engineer / research scientist / applied scientist roles.

---

# 1. Core ML, math, statistics, and optimization

## 1.1 Probability and statistics

Cover:

- Random variables, expectation, variance, covariance.
- Common distributions: Bernoulli, categorical, Gaussian, exponential family.
- Bayes’ rule, conditional independence.
- Cross-entropy, KL divergence, entropy.
- Maximum likelihood estimation, MAP estimation.
- Confidence intervals, hypothesis testing, p-values.
- Bootstrap, permutation tests.
- Bias, variance, irreducible error.
- Calibration and uncertainty.

Be able to answer:

- Why does cross-entropy correspond to maximum likelihood?
- What is KL divergence measuring?
- Difference between aleatoric and epistemic uncertainty.
- How would you test whether model A is significantly better than model B?

## 1.2 Linear algebra and calculus

Cover:

- Matrix multiplication shapes.
- Eigendecomposition, SVD, PCA.
- Norms, dot products, cosine similarity.
- Jacobians, Hessians, chain rule.
- Matrix calculus for neural networks.
- Low-rank approximations.

Be able to derive:

- Gradient of linear regression.
- Gradient of logistic regression.
- Softmax + cross-entropy gradient.
- Backpropagation through a 2-layer MLP.
- Why low-rank adaptation can reduce trainable parameters.

## 1.3 Optimization

Cover:

- Gradient descent, SGD, mini-batch SGD.
- Momentum, RMSProp, Adam, AdamW.
- Learning-rate schedules, warmup, cosine decay.
- Weight decay vs L2 regularization.
- Gradient clipping.
- Convex vs non-convex optimization.
- Saddle points, local minima, sharp vs flat minima.
- Batch size effects.
- Loss landscapes and training instability.

Be able to explain:

- Why AdamW decouples weight decay.
- Why warmup helps Transformer training.
- Why larger batch sizes may require LR scaling.
- Why training loss can decrease while validation loss worsens.

---

# 2. Classical machine learning fundamentals

Even for LLM roles, interviewers still ask these to test judgment.

Cover:

## 2.1 Supervised learning

- Linear regression.
- Logistic regression.
- Naive Bayes.
- k-nearest neighbors.
- SVMs.
- Decision trees.
- Random forests.
- Gradient boosting / XGBoost / LightGBM.
- Ensembles.
- Regularization: L1, L2, elastic net.

## 2.2 Unsupervised and representation learning

- k-means.
- Gaussian mixture models.
- PCA.
- t-SNE / UMAP at a conceptual level.
- Autoencoders.
- Contrastive learning basics.

## 2.3 Evaluation and data methodology

Cover:

- Train/validation/test splits.
- Cross-validation.
- Data leakage.
- Class imbalance.
- ROC-AUC vs PR-AUC.
- Precision, recall, F1.
- Calibration curves.
- Confusion matrices.
- Offline vs online metrics.
- Distribution shift.
- Dataset contamination.

Be ready for applied questions like:

- “Our model has high AUC but bad user experience. Why?”
- “How would you evaluate a fraud detection model?”
- “When is accuracy a bad metric?”
- “How do you debug a model that performs well offline but poorly in production?”

---

# 3. Deep learning fundamentals

This is the bridge from classical ML to LLMs.

## 3.1 Neural network basics

Cover:

- MLPs.
- Activations: ReLU, GELU, SiLU/SwiGLU.
- Initialization: Xavier, He, scaled init.
- Forward and backward propagation.
- Autograd.
- Loss functions.
- Dropout.
- Residual connections.
- BatchNorm, LayerNorm, RMSNorm.

Be able to explain:

- Why residual connections help optimization.
- Why normalization stabilizes training.
- Why GELU/SwiGLU are common in modern Transformers.
- Difference between pre-norm and post-norm architectures.

## 3.2 Training mechanics

Cover:

- Optimizers.
- Mixed precision.
- Gradient checkpointing.
- Gradient accumulation.
- Distributed data parallelism.
- Numerical stability.
- Exploding/vanishing gradients.
- Overfitting and underfitting.
- Hyperparameter tuning.

## 3.3 Sequence modeling before Transformers

You do not need to go extremely deep, but know:

- RNNs.
- LSTMs/GRUs.
- Seq2seq.
- Attention before Transformers.
- Encoder-decoder architectures.

This helps you explain *why* Transformers became dominant.

---

# 4. Transformer architecture

This is a core interview area.

The canonical Transformer replaced recurrence/convolution-heavy sequence modeling with attention-based sequence modeling, with scaled dot-product attention as the core operation. ([doi.org](https://doi.org/10.48550/arXiv.1706.03762?utm_source=openai))

You should cover:

## 4.1 Tokenization and embeddings

- Word-level, character-level, subword tokenization.
- BPE, WordPiece, unigram tokenization.
- Vocabulary size trade-offs.
- Token embeddings.
- Positional embeddings.
- Learned vs sinusoidal position embeddings.
- RoPE, ALiBi, relative positional encodings.

## 4.2 Attention

Know this cold:

\\[
\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
\\]

Cover:

- Queries, keys, values.
- Scaled dot-product attention.
- Self-attention vs cross-attention.
- Multi-head attention.
- Causal masking.
- Padding masks.
- Attention complexity: \\(O(n^2)\\).
- KV cache.
- Multi-query attention.
- Grouped-query attention.
- Long-context attention variants.

Be able to answer:

- Why divide by \\(\\sqrt{d_k}\\)?
- What is the shape of Q, K, V?
- Why use multiple heads?
- How does causal attention differ from bidirectional attention?
- Why is attention expensive for long contexts?

## 4.3 Transformer blocks

Cover:

- Residual connections.
- LayerNorm / RMSNorm.
- Feed-forward networks.
- GELU, SwiGLU.
- Dropout.
- Pre-norm vs post-norm.
- Encoder-only, decoder-only, encoder-decoder models.
- BERT-style vs GPT-style vs T5-style architectures.
- Mixture-of-Experts at a conceptual level.

Be able to implement:

- Single-head attention.
- Multi-head attention.
- Causal mask.
- Transformer block.
- Tiny decoder-only language model.

---

# 5. LLM pretraining

Now move from architecture to training.

Large language model interviews often test whether you understand the **full pretraining lifecycle**, not just the Transformer block.

## 5.1 Objectives

Cover:

- Causal language modeling.
- Masked language modeling.
- Denoising objectives.
- Next-token prediction.
- Perplexity.
- Teacher forcing.
- Sequence packing.
- Loss masking.

## 5.2 Data pipeline

Cover:

- Web-scale corpus construction.
- Deduplication.
- Filtering.
- Quality scoring.
- Toxicity filtering.
- Language balancing.
- Domain mixing.
- Tokenizer training.
- Dataset contamination.
- Train/eval leakage.
- Copyright, privacy, and PII issues.

## 5.3 Scaling laws and compute

Study scaling laws carefully. Kaplan et al. reported empirical power-law relationships between loss and model size, dataset size, and compute; Chinchilla-style compute-optimal training emphasized that token budget and model size must be balanced, not just scaling parameters. ([arxiv.org](https://arxiv.org/abs/2001.08361?utm_source=openai))

Cover:

- Parameter count.
- Token count.
- FLOPs estimation.
- Compute-optimal training.
- Overtraining smaller models.
- Data quality vs data quantity.
- Scaling curves.
- Emergent capabilities — discuss carefully, without overclaiming.

Also know GPT-3-style few-shot/in-context learning as a historical milestone in LLM behavior. ([arxiv.org](https://arxiv.org/abs/2005.14165?utm_source=openai))

## 5.4 Distributed training

For research engineer roles especially, cover:

- Data parallelism.
- Tensor parallelism.
- Pipeline parallelism.
- FSDP / ZeRO-style sharding.
- Activation checkpointing.
- Mixed precision: FP16, BF16, FP8 conceptually.
- Optimizer state memory.
- Communication overhead.
- Gradient accumulation.
- Checkpointing and fault tolerance.
- Debugging divergence.

Be ready for questions like:

- “Estimate memory needed to train a 7B model.”
- “Why did training diverge after 20k steps?”
- “How would you scale from 8 GPUs to 512 GPUs?”
- “What causes GPU underutilization?”

---

# 6. Post-training and alignment

After pretraining, study how models become useful assistants.

## 6.1 Supervised fine-tuning

Cover:

- Instruction tuning.
- Chat templates.
- Multi-turn conversation data.
- Formatting and masking.
- Data quality.
- Synthetic instruction data.
- Distillation from stronger models.

## 6.2 RLHF and preference optimization

Know the classic RLHF pipeline:

1. Pretrained model.
2. Supervised fine-tuning.
3. Preference data collection.
4. Reward model training.
5. Policy optimization, often with PPO-style RL.
6. KL regularization against the reference model.

InstructGPT is the canonical paper to know here; it showed that human-feedback fine-tuning could make language models better follow user intent. ([arxiv.org](https://arxiv.org/abs/2203.02155?utm_source=openai))

Then study direct preference methods:

- DPO.
- IPO.
- KTO.
- ORPO conceptually.
- Reward modeling vs reward-free preference optimization.
- KL constraints.
- Preference pair construction.
- Reward hacking.
- Over-optimization.
- Length bias.

DPO is especially important because it reframes preference optimization so the policy can be optimized directly from preference data without separately training an explicit reward model in the same way as RLHF. ([arxiv.org](https://arxiv.org/abs/2305.18290?utm_source=openai))

Be able to explain:

- SFT vs RLHF vs DPO.
- Why RLHF can be unstable.
- Why reward models can be gamed.
- How you would evaluate an aligned model.
- How you would debug a model that becomes overly verbose, overly cautious, or reward-hacky.

---

# 7. Adaptation, fine-tuning, and model efficiency

This is very common for applied scientist and research engineer roles.

Cover:

## 7.1 Fine-tuning approaches

- Full fine-tuning.
- Partial fine-tuning.
- Linear probing.
- Adapters.
- Prefix tuning.
- Prompt tuning.
- LoRA.
- QLoRA.
- Continued pretraining.
- Domain adaptation.
- Catastrophic forgetting.

LoRA freezes the pretrained weights and injects trainable low-rank matrices, while QLoRA combines low-rank adaptation with quantized base models to reduce memory requirements for fine-tuning. ([arxiv.org](https://arxiv.org/abs/2106.09685?utm_source=openai))

## 7.2 Compression and efficiency

Cover:

- Quantization.
- Post-training quantization.
- Quantization-aware training.
- Distillation.
- Pruning.
- Sparsity.
- Speculative decoding.
- Smaller specialist models vs larger generalist models.

Be ready for:

- “When would you fine-tune instead of using RAG?”
- “When is LoRA insufficient?”
- “How do you prevent catastrophic forgetting?”
- “How do you evaluate whether fine-tuning helped?”

---

# 8. LLM inference and serving

This is essential for research engineer and applied scientist interviews.

Cover:

## 8.1 Decoding

- Greedy decoding.
- Beam search.
- Temperature.
- Top-k sampling.
- Top-p / nucleus sampling.
- Repetition penalties.
- Length penalties.
- Constrained decoding.
- Structured outputs.
- JSON/function-call generation.
- Logprobs.

Be able to explain:

- Why lower temperature gives more deterministic outputs.
- Why beam search is often not ideal for open-ended generation.
- How top-p differs from top-k.
- How to get valid JSON reliably.

## 8.2 Serving systems

Cover:

- Prefill vs decode phases.
- KV cache.
- Continuous batching.
- Dynamic batching.
- Paged attention conceptually.
- Tensor parallel inference.
- Model parallel serving.
- Latency vs throughput.
- Cost per token.
- Cold starts.
- Rate limits.
- Caching.
- Streaming.
- Monitoring.

Be ready for:

- “How would you serve a 70B model under latency constraints?”
- “How do you reduce cost without hurting quality?”
- “Why is decode slower than prefill?”
- “What metrics do you monitor in production?”

---

# 9. LLM evaluation

Do this before agents. Most agent failures are evaluation failures.

Cover:

## 9.1 General LLM eval

- Perplexity.
- Task-specific benchmarks.
- Human evaluation.
- Pairwise preference evaluation.
- LLM-as-judge.
- Reference-based vs reference-free evaluation.
- Factuality.
- Hallucination.
- Robustness.
- Calibration.
- Bias and toxicity.
- Safety evaluation.
- Regression test suites.
- Dataset contamination.

## 9.2 Applied evaluation

Cover:

- Offline evals.
- Online A/B tests.
- Inter-rater agreement.
- Confidence intervals.
- Power analysis.
- Error taxonomies.
- Slice-based analysis.
- Latency/cost/quality trade-offs.
- Guardrail effectiveness.
- Longitudinal monitoring.

Be able to design evals for:

- A customer support bot.
- A code assistant.
- A medical summarization tool.
- A RAG system.
- An autonomous web agent.
- A fine-tuned domain model.

---

# 10. Retrieval-Augmented Generation

Study RAG before agents because most practical agents need retrieval.

The original RAG formulation combined retrieval with generation for knowledge-intensive NLP tasks, and the general idea remains central to grounded LLM systems. ([arxiv.org](https://arxiv.org/abs/2005.11401?utm_source=openai))

Cover:

## 10.1 Retrieval

- Dense embeddings.
- Bi-encoders.
- Cross-encoders.
- Hybrid search.
- BM25.
- Vector databases.
- Approximate nearest neighbors.
- HNSW conceptually.
- Chunking.
- Metadata filtering.
- Query rewriting.
- Multi-query retrieval.
- Reranking.
- Context compression.

## 10.2 Generation with retrieval

- Context stuffing.
- Citation generation.
- Grounded answering.
- Faithfulness.
- Source attribution.
- Retrieval failure modes.
- Stale knowledge.
- Permission-aware retrieval.
- Private data handling.

## 10.3 RAG evaluation

Cover:

- Retrieval precision/recall.
- Hit rate.
- MRR/NDCG.
- Answer correctness.
- Faithfulness.
- Citation accuracy.
- Context relevance.
- End-to-end task success.

Be ready for:

- “How do you debug hallucinations in a RAG system?”
- “How do you choose chunk size?”
- “How do you evaluate the retriever separately from the generator?”
- “When would you fine-tune instead of using RAG?”

---

# 11. Agentic AI systems

Now study agents.

Think of an agent as:

> **LLM + state + tools + environment + control loop + memory + evaluation + safety constraints**

## 11.1 Agent fundamentals

Cover:

- Agent loop: observe → reason/plan → act → observe → update.
- Tools/actions.
- Environment state.
- Short-term memory.
- Long-term memory.
- Planning.
- Reflection.
- Task decomposition.
- Human-in-the-loop control.
- Autonomous vs semi-autonomous agents.
- Workflows vs agents.

A good interview answer should distinguish between:

- A simple LLM call.
- A deterministic workflow.
- A tool-using assistant.
- A planner-executor system.
- A fully autonomous multi-step agent.

## 11.2 Reasoning and planning patterns

Study:

- Chain-of-thought conceptually.
- ReAct.
- Plan-and-execute.
- Tree of Thoughts.
- Self-consistency.
- Reflection.
- Critic/reviewer models.
- Verifier models.
- Search over reasoning paths.

ReAct is important because it interleaves model reasoning with task-specific actions, allowing the model to update its plan based on observations from tools or environments. ([arxiv.org](https://arxiv.org/abs/2210.03629?utm_source=openai))

Tree of Thoughts generalizes simple step-by-step reasoning by exploring multiple possible intermediate reasoning paths. ([arxiv.org](https://arxiv.org/abs/2305.10601?utm_source=openai))

Reflexion is worth knowing as an example of agents using feedback, including verbal feedback, to improve future attempts. ([arxiv.org](https://arxiv.org/abs/2303.11366?utm_source=openai))

## 11.3 Tool use

Cover:

- Function calling.
- Tool schemas.
- Argument validation.
- Tool selection.
- Tool routing.
- Tool result parsing.
- Retrying failed calls.
- Rate limits.
- Idempotency.
- Sandboxed execution.
- Browser tools.
- Code execution.
- SQL/database tools.
- Search tools.
- API tools.

Toolformer is a useful paper here because it studies language models learning when and how to call APIs. ([arxiv.org](https://arxiv.org/abs/2302.04761?utm_source=openai))

Be ready for:

- “How do you prevent an agent from calling the wrong tool?”
- “How do you recover from tool failure?”
- “How do you validate tool arguments?”
- “How do you handle irreversible actions?”

## 11.4 Memory

Cover:

- Conversation memory.
- Scratchpads.
- Episodic memory.
- Semantic memory.
- Procedural memory.
- Vector memory.
- Summarization memory.
- Memory retrieval.
- Memory eviction.
- Memory consistency.
- Privacy and retention.

Key interview point:

> Most production systems do not need vague “infinite memory.” They need well-scoped state, retrieval, permissions, and evaluation.

## 11.5 Agent architectures

Cover:

- Single-agent loops.
- Planner-executor.
- Router-specialist systems.
- Critic/reviewer agents.
- Multi-agent debate.
- Hierarchical agents.
- State machines.
- DAG workflows.
- Graph-based orchestration.
- Event-driven agents.

Be able to compare:

- Deterministic workflow vs autonomous agent.
- Single agent vs multi-agent.
- Planning upfront vs reactive planning.
- Tool-use agent vs RAG pipeline.
- Agent with memory vs stateless assistant.

## 11.6 Agent evaluation

Agent evaluation should focus on task success, not just response quality.

Cover:

- Task success rate.
- Number of steps.
- Cost.
- Latency.
- Tool-call accuracy.
- Recovery from errors.
- Human intervention rate.
- Safety violations.
- Regression tests.
- Simulated environments.
- Long-horizon task evaluation.

Know agent benchmarks conceptually:

- **WebArena** for web-based autonomous agent tasks.
- **SWE-bench** for resolving real GitHub issues.
- **GAIA** for general assistant-style tasks.
- **AgentBench** for evaluating LLMs as agents across environments. ([arxiv.org](https://arxiv.org/abs/2307.13854?utm_source=openai))

## 11.7 Agent safety and security

This is increasingly important.

Cover:

- Prompt injection.
- Tool injection.
- Data exfiltration.
- Permission boundaries.
- Least-privilege tools.
- Sandboxing.
- Secrets management.
- Audit logs.
- Human approval for irreversible actions.
- Guardrails.
- Policy enforcement.
- Jailbreak resistance.
- Monitoring and incident response.

Be ready for:

- “Your agent can send emails. How do you prevent disasters?”
- “How do you stop prompt injection from retrieved documents?”
- “How do you safely let an agent execute code?”
- “How do you evaluate whether an agent is safe enough to deploy?”

---

# 12. Coding and implementation topics

Run this in parallel with the theory.

For ML/AI research roles, you should be able to implement:

## Core ML

- Linear regression.
- Logistic regression.
- k-means.
- PCA.
- Decision tree or simple GBDT conceptually.
- Metrics: AUC, F1, precision/recall.

## Deep learning

- MLP training loop.
- Backprop manually for a small network.
- BatchNorm or LayerNorm.
- Dropout.
- Adam optimizer.
- Gradient clipping.

## Transformers/LLMs

- Tokenizer usage.
- Causal self-attention.
- Multi-head attention.
- Transformer decoder block.
- Tiny GPT-style model.
- Sampling: temperature, top-k, top-p.
- Beam search.
- KV cache conceptually or minimally.
- LoRA layer.
- Simple RAG pipeline.
- Basic agent loop with tools.

## Systems

- PyTorch profiling.
- Dataloader bottlenecks.
- GPU memory debugging.
- Mixed precision.
- Distributed training basics.
- Model serving API.
- Batch inference.

---

# 13. Role-specific weighting

## Research Scientist

Emphasize:

- Mathematical depth.
- Paper reading and critique.
- Novel research ideas.
- Ablation design.
- Experimental methodology.
- Theory behind objectives.
- Failure analysis.
- Research presentations.

Prepare to discuss:

- Your research contributions.
- Why your method worked.
- What baselines you used.
- What ablations mattered.
- What failed.
- What you would do next.

## Research Engineer

Emphasize:

- PyTorch excellence.
- Training/debugging large models.
- Distributed systems.
- Inference optimization.
- Reproducibility.
- Clean experimentation.
- Profiling and memory efficiency.
- Production-quality research code.

Prepare to answer:

- “How would you train this at scale?”
- “How would you make this faster?”
- “Why is GPU utilization low?”
- “How do you debug NaNs?”
- “How do you structure an experiment repo?”

## Applied Scientist

Emphasize:

- Problem formulation.
- Metrics.
- Business/product constraints.
- Experimentation.
- A/B testing.
- Causal reasoning.
- Model evaluation.
- Deployment trade-offs.
- User impact.

Prepare to answer:

- “How do you know the model improved the product?”
- “What metric would you optimize?”
- “How do you handle noisy labels?”
- “When would you use RAG vs fine-tuning?”
- “How do you launch safely?”

---

# Recommended study order summary

Use this order:

1. **Math/stat/optimization refresh**
2. **Classical ML and evaluation**
3. **Deep learning foundations**
4. **Transformers**
5. **LLM pretraining**
6. **Post-training: SFT, RLHF, DPO**
7. **Fine-tuning, LoRA/QLoRA, compression**
8. **Inference and serving**
9. **LLM evaluation**
10. **RAG**
11. **Agentic systems**
12. **Safety/security for agents**
13. **ML system design + research discussion practice**
14. **Mock interviews and project deep dives**

If you want the highest-leverage version: prioritize **Transformers, LLM training/post-training, evaluation, RAG, inference systems, and agents**. Classical ML/math should be refreshed enough that you can derive and explain fundamentals confidently.
