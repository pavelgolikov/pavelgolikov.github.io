# Study packet: model-training fundamentals and loss-curve diagnosis

This is optimized for a **research pairing interview**, where the strongest answer is rarely just “that’s overfitting.” A better response is:

> “Assuming the curves use comparable losses and evaluation settings, overfitting is the leading hypothesis. I would verify that with these checks, then run the cheapest discriminating experiment.”

Use this loop:

1. **Observe** the evidence.
2. **Generate** several plausible hypotheses.
3. **Choose a test** that separates those hypotheses.
4. **Intervene** only after identifying the likely mechanism.
5. **Verify** that the intervention changed the expected quantity.

---

## 1. The core training mechanism

A conventional training step is:

$$
x,y
\;\xrightarrow{\text{model}}\;
z=f_\theta(x)
\;\xrightarrow{\text{loss}}\;
L
\;\xrightarrow{\text{backprop}}\;
\nabla_\theta L
\;\xrightarrow{\text{optimizer}}\;
\theta'
$$

In PyTorch, that normally corresponds to:

```python
optimizer.zero_grad()
logits = model(inputs)
loss = loss_fn(logits, targets)
loss.backward()
optimizer.step()
```

Autograd records the operations connecting parameters to the loss, then applies the chain rule backward through that graph. The optimizer updates only the parameters it was explicitly given. ([docs.pytorch.org](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html?utm_source=openai))

A failure can therefore come from any stage:

1. Data or targets are wrong.
2. The model forward pass is wrong.
3. The loss or masking is wrong.
4. The loss is disconnected from the graph.
5. Gradients are zero, absent, or nonfinite.
6. The optimizer does not update the intended parameters.
7. Evaluation or logging is misleading.

That list is a useful diagnostic stack.

---

## 2. Cross-entropy intuition

For a classification problem, a model produces logits $z_1,\ldots,z_C$. Softmax converts them to probabilities:

$$
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}
$$

If the correct class is $y$, cross-entropy is:

$$
\ell=-\log p_y
$$

Equivalently, in a numerically stable implementation:

$$
\ell=\operatorname{logsumexp}(z)-z_y
$$

Some useful values:

| Probability on correct answer | Loss |
|---:|---:|
| $0.9$ | $0.105$ |
| $0.5$ | $0.693$ |
| $0.1$ | $2.303$ |
| $0.01$ | $4.605$ |

**Intuition:** loss is not simply “right or wrong.” It measures how much probability the model assigned to the correct answer. Confident mistakes are heavily penalized.

An especially useful result is:

$$
\frac{\partial \ell}{\partial z_j}
=
p_j-\mathbf 1[j=y]
$$

- For the correct class, this is $p_y-1<0$, so gradient descent raises its logit.
- For incorrect classes, it is $p_j>0$, so gradient descent lowers their logits.
- Incorrect classes receiving more probability are pushed down more strongly.

PyTorch’s `CrossEntropyLoss` expects logits and combines the stable log-softmax and negative-log-likelihood operations. You generally should not manually apply softmax and then take a logarithm. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?utm_source=openai))

---

## 3. The causal-language-model objective

For tokens $x_1,\ldots,x_T$, a causal LM minimizes:

$$
L(\theta)=
-\frac{1}{\sum_t m_t}
\sum_t m_t
\log p_\theta(x_t\mid x_{<t})
$$

Here $m_t\in\{0,1\}$ is a **loss mask** identifying which tokens contribute to the objective.

Three masks must be conceptually separated:

1. **Causal mask:** prevents attending to future tokens.
2. **Attention/padding mask:** prevents attention to padding.
3. **Loss mask:** determines which target tokens contribute to loss.

For supervised fine-tuning, you might allow the model to attend to the prompt while computing loss only on the response tokens.

Implementation convention also matters:

- A custom model may require you to use `logits[:, :-1]` against `labels[:, 1:]`.
- Hugging Face causal-LM classes such as `GPT2LMHeadModel` shift labels internally, so passing `labels=input_ids` is appropriate. Shifting manually as well would create a **double-shift bug**.
- Labels set to `-100` are conventionally ignored by the loss. ([huggingface.co](https://huggingface.co/docs/transformers/model_doc/gpt2?utm_source=openai))

For a causal LM, perplexity is:

$$
\operatorname{PPL}=e^L
$$

Thus a loss of $2$ corresponds to perplexity $e^2\approx7.39$. Perplexity comparisons depend on tokenization and evaluation context, so they are not automatically comparable across tokenizers or evaluation procedures. ([huggingface.co](https://huggingface.co/docs/transformers/perplexity?utm_source=openai))

### Useful baseline

For uniform predictions over a vocabulary of size $V$:

$$
L=\log V
$$

For $V=50{,}000$:

$$
\log(50{,}000)\approx10.82
$$

A from-scratch model stuck around this value may still be producing nearly uniform predictions. A pretrained model should normally begin substantially below this baseline.

---

## 4. Learning rate intuition

For plain gradient descent:

$$
g_t=\nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

where $\eta$ is the learning rate.

A local approximation explains why both very small and very large learning rates are bad:

$$
L(\theta-\eta g)
\approx
L(\theta)
-\eta\lVert g\rVert^2
+
\frac{\eta^2}{2}g^\top Hg
$$

- The negative term is the expected downhill improvement.
- The positive quadratic term captures curvature.
- If $\eta$ is too small, movement is negligible.
- If $\eta$ is too large, the curvature term can dominate, causing overshoot, oscillation, spikes, or divergence.

Adam adds momentum and coordinate-wise scaling, but the basic diagnostic principle remains: inspect both the **gradient** and the **actual parameter update**. ([deeplearningbook.org](https://www.deeplearningbook.org/contents/numerical.html?utm_source=openai))

Minibatch gradients are noisy estimates of the full-dataset gradient. Under standard sampling assumptions, the standard error decreases roughly as $1/\sqrt B$, where $B$ is batch size. This is why smaller batches generally produce noisier curves, though noise can also reveal data or optimization problems. ([deeplearningbook.org](https://www.deeplearningbook.org/contents/optimization.html))

---

# 5. The highest-value debugging sequence

Before memorizing individual curve shapes, memorize this sequence.

## Step 1: Confirm that the curves are comparable

Ask:

- Are train and validation using the same underlying loss?
- Is one a sum and the other a mean?
- Are they averaged per sequence, per batch, or per valid token?
- Does training loss include a regularization term?
- Are different label masks used?
- Is train loss measured throughout the epoch while validation is measured using the final model?
- Is validation using `model.eval()`?
- Is one curve heavily smoothed?
- Are they evaluating the same checkpoint?

For variable-length LM batches, averaging batch means can differ from:

$$
\frac{\text{total negative log-likelihood}}
     {\text{total valid tokens}}
$$

The latter is normally the cleanest corpus-level comparison.

## Step 2: Inspect examples end to end

For several samples, display:

- Raw input
- Tokenized input
- Labels
- Shifted input-label pairs
- Attention mask
- Loss mask
- Number of valid target tokens
- Decoded prompt and response

This catches an enormous fraction of “optimizer problems.”

## Step 3: Try to overfit one small, trustworthy batch

Use deterministic data and disable augmentation. A sufficiently capable model should usually drive the training loss very low on one small batch.

- **Cannot fit one batch:** suspect an implementation, objective, gradient, optimizer, or numerical problem.
- **Can fit one batch but not the dataset:** suspect capacity, regularization, data difficulty/noise, or optimization at scale.

## Step 4: Inspect the gradient and the update separately

Ask:

1. Does the loss have a `grad_fn`?
2. Does each intended parameter have `requires_grad=True`?
3. Is it in the optimizer?
4. Is its gradient `None`, zero, finite, or enormous?
5. Does it actually change after `optimizer.step()`?

## Step 5: Locate the first nonfinite value

Check, in order:

$$
\text{inputs}
\rightarrow
\text{activations/logits}
\rightarrow
\text{loss}
\rightarrow
\text{gradients}
\rightarrow
\text{updated parameters}
$$

The first nonfinite value is usually much more informative than the eventual NaN loss.

## Step 6: Re-evaluate the training set in evaluation mode

Compute:

- Online training loss
- Training-set loss in `eval()` mode after the epoch
- Validation loss in `eval()` mode

This is especially useful when validation loss is unexpectedly lower than training loss.

---

# 6. Loss-curve diagnosis

## Case 1: Training loss decreases, validation loss increases

### Leading interpretation

If the losses are comparable, the model is increasingly fitting properties specific to the training set rather than properties that generalize. The widening difference

$$
L_{\text{val}}-L_{\text{train}}
$$

is the generalization gap. The classic curve is validation loss initially falling and then rising while training loss continues falling. ([developers.google.com](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting?utm_source=openai))

### What to check first

1. **Split integrity**
   - Exact or near duplicates
   - Chunks from the same document in both splits
   - Same users or entities crossing splits
   - Random split when a time- or group-based split is required

2. **Leakage**
   - Preprocessing fitted on all data
   - Target information appearing in features or prompts
   - Validation data used during data selection
   - Repeated tuning on the same validation set

3. **Distribution mismatch**
   - Different class frequencies
   - Different lengths, domains, sources, or time periods

Data leakage generally biases validation performance optimistically, rather than directly causing the classic rising-validation curve. Nevertheless, the split must be audited before trusting any generalization conclusion. ([scikit-learn.org](https://scikit-learn.org/stable/common_pitfalls.html?highlight=linearregression))

### Appropriate interventions

- Restore the checkpoint with minimum validation loss.
- Early stop.
- Add or improve training data.
- Deduplicate or improve the split.
- Increase weight decay or dropout.
- Reduce model capacity.
- Reduce the number of training epochs.
- Use better task-relevant augmentation.
- Reduce label noise.
- For fine-tuning, consider a smaller learning rate or fewer trainable parameters.

### Interview-ready answer

> “Assuming the two losses use the same definition, this is most consistent with overfitting. Before changing regularization, I would audit split contamination and train-validation distribution differences. If the split is sound, I would select the checkpoint at minimum validation loss and test stronger regularization, less capacity, or more data.”

---

## Case 2: Both losses are high and flat

First distinguish:

- **High and flat:** likely a problem.
- **Low and flat:** possibly normal convergence or an irreducible limit.

### Main hypothesis categories

#### A. The model is not actually being optimized

- Learning rate is zero or extremely small.
- Optimizer is missing parameters.
- Parameters are frozen.
- Gradient scaler repeatedly skips steps.
- The loss is disconnected.
- `optimizer.step()` is missing.
- A newly replaced output head was added after optimizer construction.

#### B. The objective or labels are broken

- Labels have the wrong shape or type.
- Labels are shifted incorrectly.
- Padding or prompt masks remove all useful targets.
- Input-label pairs are misaligned.
- Labels are effectively random.
- Classification labels map to the wrong class IDs.

#### C. Genuine underfitting

- Insufficient model capacity
- Excessive regularization
- Features or context are insufficient
- Task is noisy or ambiguous
- Model architecture cannot represent the required mapping

### Best discriminating experiment

**Overfit one small batch.**

If it cannot, do not begin by increasing model capacity. First prove that the model, loss, gradients, and optimizer form a functioning training loop.

### Useful clue for LMs

If a from-scratch LM with vocabulary size 50,000 stays almost exactly around $10.82$, inspect whether it is still approximately uniform and whether the output head receives gradients.

### Interview-ready answer

> “Both losses being high and flat could mean underfitting, but it could equally be a broken optimization path. My first test would be to overfit a single clean batch. If that fails, I would inspect label alignment, masks, gradient flow, optimizer membership, learning rate, and actual parameter deltas.”

---

## Case 3: Loss is noisy or spikes

Some minibatch noise is expected. Diagnose the **structure** of the noise.

### Shape clues

- **Small random fluctuations:** normal batch stochasticity.
- **Spikes at the same place every epoch:** data ordering, a particular shard, or scheduler event.
- **Spike associated with long batches:** reduction or token-count issue.
- **Spike that immediately recovers:** hard/corrupt batch or transient numerical event.
- **Spike that permanently damages training:** unstable parameter update.
- **Increasing spike amplitude:** likely divergence.
- **Only happens in FP16:** mixed-precision range issue.

### Likely causes

- Learning rate too high
- Missing or inadequate warmup
- Small effective batch size
- Outlier or corrupt examples
- Poor data shuffling
- Exploding gradients
- Gradient accumulation without correct scaling
- Mixed-precision overflow
- Sudden learning-rate scheduler change
- Variable-length loss logged as a sum
- Distributed reduction or logging bug

Google’s loss-curve exercises specifically recommend checking bad examples, reducing learning rate, and testing on a tiny trustworthy subset when loss oscillates or jumps. ([developers.google.com](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves))

### Discriminating tests

Log these together:

```text
step
batch/shard ID
loss
learning rate
valid token count
max sequence length
gradient norm
update norm
AMP scale
```

Then:

1. Reload a checkpoint from before the spike.
2. Replay the suspicious batch.
3. Replay a normal batch.
4. Lower the learning rate by 10×.
5. Run the same step in FP32.
6. Inspect the raw examples.

If the same batch reliably spikes, suspect the data or its preprocessing. If the spike depends on the preceding optimization trajectory, suspect instability.

### Interview-ready answer

> “I would first determine whether the noise is ordinary minibatch variance or a structured failure. I’d correlate spikes with the batch, sequence length, learning rate, gradient norm, and AMP scale. Replaying the offending batch from a fixed checkpoint is a useful way to separate a data issue from an optimizer-state issue.”

---

## Case 4: Validation loss is lower than training loss

This is not automatically an error.

### Benign explanations

1. **Dropout is active during training but disabled during evaluation.**
2. **Training augmentation or noise makes training examples harder.**
3. **The displayed training loss includes explicit regularization.**
4. **Training loss is averaged throughout the epoch, while validation uses the better model at the end of the epoch.**
5. **The validation subset is simply easier or cleaner.**

PyTorch dropout randomly removes activations during training and becomes an identity operation in evaluation. Also, `model.eval()` and `torch.no_grad()` solve different problems: `eval()` changes behaviors such as dropout, while `no_grad()` disables graph recording. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.dropout.Dropout.html?utm_source=openai))

The “training loss averaged throughout the epoch versus validation at epoch end” effect is a documented reason validation loss can be lower. ([keras.io](https://keras.io/getting_started/faq/?utm_source=openai))

### Best test

At the end of an epoch:

1. Set `model.eval()`.
2. Run the validation code on the training dataset.
3. Use exactly the same loss, mask, and reduction.

Call this `train_eval_loss`.

Interpretation:

- `online_train_loss > train_eval_loss ≈ val_loss`  
  Likely dropout, augmentation, or timing.

- `train_eval_loss > val_loss`  
  Validation may genuinely be easier—or possibly contaminated.

- Large unexplained differences  
  Inspect loss reductions, masks, preprocessing, and duplicates.

### Interview-ready answer

> “Validation below online training loss can be completely legitimate because training may use dropout, augmentation, or earlier model states. I would calculate training-set loss again in evaluation mode using the exact validation pipeline. If validation remains much easier, I would then inspect the split and loss normalization.”

---

## Case 5: Training loss is flat

Think of the chain:

$$
\text{loss}
\rightarrow
\text{graph}
\rightarrow
\text{gradient}
\rightarrow
\text{optimizer}
\rightarrow
\text{parameter update}
$$

### Common implementation failures

- `loss.requires_grad == False`
- `loss.grad_fn is None`
- `.detach()` was applied inside the computation.
- A tensor was converted to NumPy and back.
- A new loss was constructed from `loss.item()`.
- Forward pass accidentally ran inside `torch.no_grad()`.
- `requires_grad=False`
- Parameters are not registered as `nn.Parameter`.
- Parameters are not in the optimizer.
- Output head was replaced after creating the optimizer.
- Learning rate is zero.
- `optimizer.step()` is absent.
- All labels are ignored.
- The wrong loss target is used.
- An `argmax` or other nondifferentiable operation occurs before the loss.
- Logging is reading a stale value.

PyTorch records a `grad_fn` only when the operation participates in the autograd graph, while `.detach()` explicitly removes the autograd relationship. Optimizers update only their registered parameter collections. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/autograd?utm_source=openai))

### Interpret gradient states carefully

- **`grad is None`:** parameter was not connected to the loss or not used.
- **Gradient exactly zero:** connected, but current computation produced no signal.
- **Gradient nonzero, update zero:** optimizer/LR/scaler problem.
- **Update nonzero, logged loss flat:** logging, scale, insufficient progress, or hard optimization.

### Interview-ready answer

> “For a flat training curve, I’d verify the mechanics before tuning hyperparameters. I’d check that the loss has a graph, intended parameters require gradients and belong to the optimizer, gradients are finite and nonzero, and a parameter actually changes after the optimizer step.”

---

## Case 6: Loss becomes NaN

Do not treat “NaN” as a single diagnosis. Find the first nonfinite value.

### Common causes

- NaN or infinity in the data
- Learning rate too high
- Exploding activations or gradients
- FP16 overflow or underflow
- `log(0)`
- Division by zero
- Exponentiating large values
- Manual softmax followed by log
- Zero valid-token denominator
- All labels ignored in a mean reduction
- Attention row in which every position is masked
- Invalid custom normalization
- Parameters already corrupted by an earlier step

Mixed precision has a narrower representable range, and gradient scaling exists partly to address representability problems. PyTorch recommends disabling autocast or running suspicious regions in FP32 when diagnosing nonfinite values. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/amp.html?highlight=module&utm_source=openai))

### Debugging protocol

1. Check that inputs and labels are finite.
2. Check logits before computing loss.
3. Check that loss is finite.
4. Backpropagate.
5. If using AMP, unscale gradients.
6. Check every gradient for finiteness.
7. Check the gradient norm.
8. Step the optimizer.
9. Check parameters again.

Use anomaly detection during debugging:

```python
with torch.autograd.detect_anomaly():
    loss.backward()
```

It can identify the forward operation associated with a failing backward operation, although it is too slow for normal training. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/autograd?utm_source=openai))

For AMP and clipping:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)

grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True,
)

scaler.step(optimizer)
scaler.update()
```

Gradients should be unscaled before inspection or clipping; the scaler can skip the optimizer step when gradients contain infinities or NaNs. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/notes/amp_examples.html?utm_source=openai))

A subtle rule: **mask before performing an invalid operation**, not only afterward. For example, computing a division by zero and masking the resulting infinity may still produce NaN gradients because the invalid operation remains in the graph. ([docs.pytorch.org](https://docs.pytorch.org/docs/main/notes/autograd.html?utm_source=openai))

### Interview-ready answer

> “I would locate the first nonfinite tensor rather than guessing from the final NaN. I’d check inputs, logits, loss, unscaled gradients, and parameters in that order. Then I’d retry the failing batch in FP32 and at a lower learning rate, while checking masks and any manual log, exp, or division operations.”

---

## Case 7: Loss decreases but the evaluation metric does not improve

The loss is usually a differentiable **surrogate objective**. The actual metric may be discrete, thresholded, sequence-level, or weighted differently. Training minimizes the surrogate in the hope that it improves the desired performance measure; that implication is not guaranteed. ([deeplearningbook.org](https://www.deeplearningbook.org/contents/optimization.html))

### Simple classification example

Suppose an example is already classified correctly:

- Correct-class probability moves from $0.55$ to $0.90$.
- Accuracy remains 1.
- Cross-entropy improves substantially.

Or suppose:

- Correct class moves from $0.20$ to $0.35$.
- Incorrect top class moves from $0.21$ to $0.36$.
- Loss improves.
- Accuracy remains 0.

A metric can therefore remain flat while probabilities improve.

### LLM-specific causes

- Loss is dominated by easy prompt tokens.
- Prompt tokens should have been masked.
- Common tokens dominate rare task-critical tokens.
- Token NLL improves but exact match requires every token to be correct.
- Teacher-forced token prediction improves, but autoregressive generation does not.
- Generation uses the wrong prompt template.
- Stop-token or maximum-length settings are wrong.
- Sampling makes evaluation noisy.
- Tokenizer or normalization differs between training and evaluation.
- Evaluation uses the wrong checkpoint.
- The metric implementation is flawed.

For illustration, if ten answer tokens were independently correct with probability $0.95$, the probability of the whole sequence being exact would be:

$$
0.95^{10}\approx0.60
$$

Thus strong token-level performance can coexist with much lower sequence-level exact match.

### Other common causes

- Class imbalance
- Wrong classification threshold
- Label-ID mapping bug
- Spurious correlations
- Train/evaluation distribution shift
- Metric averaging method hides minority-class failure

### Discriminating tests

1. Compute the metric on the training set.
2. Compare token-level accuracy with free-running generation.
3. Report metrics per class, domain, length, and source.
4. Inspect a confusion matrix.
5. Validate the metric on hand-written examples.
6. Compare loss on only the tokens/classes the metric cares about.
7. Inspect model probabilities, not just final predictions.
8. Verify generation and decoding settings.

### Interview-ready answer

> “A decreasing surrogate loss does not guarantee movement in a discrete metric. I would check whether the objective weights the behavior the metric measures. For an LM, I’d inspect prompt-versus-response loss, token accuracy versus free-running generation, and evaluation decoding settings before changing the optimizer.”

---

# 7. Minimal PyTorch diagnostic toolkit

## Gradient and optimizer audit

Run this after `loss.backward()` and before `optimizer.step()`:

```python
import torch

def audit_gradients(model, optimizer, loss):
    assert loss.requires_grad, "Loss is detached."
    assert loss.grad_fn is not None, "No autograd graph reaches the loss."

    optimizer_param_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }

    rows = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        grad = p.grad

        rows.append({
            "name": name,
            "in_optimizer": id(p) in optimizer_param_ids,
            "grad_is_none": grad is None,
            "grad_norm": (
                None if grad is None
                else float(grad.detach().float().norm())
            ),
            "grad_is_finite": (
                None if grad is None
                else bool(torch.isfinite(grad).all())
            ),
        })

    return rows
```

Interpret it as follows:

| Observation | Likely direction |
|---|---|
| Not in optimizer | Optimizer construction problem |
| Gradient is `None` | Disconnected, unused, or frozen |
| Gradient is zero | No signal, saturation, mask, or wrong target |
| Gradient nonfinite | Numerical issue |
| Healthy gradient but no update | LR, scaler, or optimizer issue |

To inspect an update, clone only a few selected parameters rather than the entire LLM:

```python
name, probe = next(
    (n, p)
    for n, p in model.named_parameters()
    if p.requires_grad
)

before = probe.detach().clone()

optimizer.step()

delta_norm = (probe.detach() - before).float().norm()
weight_norm = probe.detach().float().norm()

print({
    "parameter": name,
    "delta_norm": float(delta_norm),
    "relative_update": float(delta_norm / (weight_norm + 1e-12)),
})
```

There is no universal “correct” update ratio, but zero or astronomically large values are immediately informative.

---

## Causal-LM label audit

```python
labels = input_ids.clone()

# Padding does not contribute to loss.
labels[attention_mask == 0] = -100

# For response-only supervised fine-tuning:
labels[~response_token_mask] = -100

valid_tokens = (labels != -100).sum()
assert valid_tokens > 0, "Batch has no supervised tokens."

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,  # Common HF causal-LM classes shift internally.
)

loss = outputs.loss
assert torch.isfinite(loss)
```

For a custom implementation that does **not** shift internally:

```python
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

loss = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

Shift exactly once.

---

# 8. LLM training checklist

Before trusting an LM curve, check:

- [ ] Input and target are shifted exactly once.
- [ ] Future tokens are blocked by the causal mask.
- [ ] Padding is excluded from loss.
- [ ] Every batch has at least one valid target token.
- [ ] Prompt tokens are included or excluded intentionally.
- [ ] Loss is normalized by valid target-token count.
- [ ] EOS and document boundaries are handled intentionally.
- [ ] Packed examples cannot attend across invalid boundaries.
- [ ] Train and validation use the same tokenizer and chat template.
- [ ] Validation uses the intended context window.
- [ ] Perplexity uses the same tokenization and aggregation procedure.
- [ ] Generation evaluation uses the correct prompt, stop tokens, and maximum length.
- [ ] Train/validation splitting is done by document, user, or group when appropriate.

Hugging Face’s causal-LM and perplexity documentation is particularly useful for understanding label shifting, ignored labels, token-level NLL, and context-window effects. ([huggingface.co](https://huggingface.co/docs/transformers/main_classes/data_collator?utm_source=openai))

---

# 9. Interview flashcards

### What is the first test when a model does not learn?

Try to overfit a single clean batch.

### What is the difference between a `None` gradient and a zero gradient?

- `None`: no gradient was computed for that parameter.
- Zero: the parameter participated, but the derivative happened to be zero.

### Why are `model.eval()` and `torch.no_grad()` not interchangeable?

- `eval()` changes module behavior such as dropout.
- `no_grad()` disables graph recording.

### What if gradients are nonzero but parameters do not change?

Check optimizer membership, learning rate, AMP step skipping, and whether the inspected parameters belong to that optimizer.

### What if a spike occurs at the same step every epoch?

Inspect the corresponding batch, data shard, ordering, and scheduler event.

### What if a spike occurs only after many normal updates?

Suspect optimizer-state instability, accumulated large weights, scheduler behavior, or progressive numerical corruption.

### Why can validation loss be lower than training loss?

Dropout, augmentation, explicit regularization, end-of-epoch timing, or an easier validation set.

### Why can loss improve while accuracy is flat?

Loss is probability-sensitive; accuracy only changes when the argmax changes.

### Why can LM loss improve while exact match is flat?

Token-level NLL may improve on easy or prompt tokens while critical answer tokens or autoregressive generation remain wrong.

### What is the most useful response to NaN?

Find the first nonfinite input, activation, loss, gradient, or parameter.

---

# 10. Recommended practical exercises

## Exercise A: Break the graph

Train a small model normally, then introduce:

```python
hidden = hidden.detach()
```

Observe:

- Which layers receive `grad=None`?
- Can the output head still learn?
- What does the loss curve look like?

## Exercise B: Omit parameters from the optimizer

Construct the optimizer using only the backbone, not the output head. Then reverse it.

Inspect gradient norms and parameter deltas.

## Exercise C: Learning-rate sweep

Run the same small problem with:

```text
1e-6
1e-4
1e-2
1
```

Record loss, gradient norm, and update norm.

## Exercise D: Validation below training

Train with dropout. Compare:

1. Online train loss
2. Train loss recomputed in `eval()` mode
3. Validation loss

## Exercise E: Masking error

For a tiny causal LM:

1. Train with the correct shift.
2. Double shift the labels.
3. Include padding in the loss.
4. Mask every label.
5. Let prompt tokens dominate response tokens.

Explain every resulting curve.

## Exercise F: Loss/metric mismatch

Construct an imbalanced classifier where accuracy remains high by predicting the majority class. Track:

- Cross-entropy
- Accuracy
- Minority recall
- Confusion matrix

---

# 11. A focused four-hour study plan

## Hour 1: Rebuild the mechanism

- Write the PyTorch training loop from memory.
- Derive or explain $p-\text{one-hot}(y)$.
- Explain learning rate using the curvature approximation.
- Explain `grad=None` versus zero.

## Hour 2: Debugging mechanics

- Implement the one-batch overfit test.
- Add the gradient/optimizer audit.
- Introduce `.detach()`, a zero LR, and missing optimizer parameters.
- Verify that you can identify each from evidence.

## Hour 3: LLM-specific mechanics

- Review causal shifting.
- Review attention mask versus loss mask.
- Calculate valid-token-normalized loss.
- Explain perplexity.
- Practice identifying prompt-loss domination.

## Hour 4: Verbal curve diagnosis

For every curve, practice saying:

1. Leading hypothesis
2. Alternative hypotheses
3. First discriminating test
4. Likely intervention
5. Evidence that would confirm success

---

# 12. Authoritative online reading

Read these in approximately this order:

1. **PyTorch: Optimizing Model Parameters** — the standard forward/loss/backward/step loop. ([docs.pytorch.org](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html?utm_source=openai))
2. **PyTorch: Autograd Mechanics** — `requires_grad`, graph construction, evaluation mode, and gradient disabling. ([docs.pytorch.org](https://docs.pytorch.org/docs/main/notes/autograd.html?utm_source=openai))
3. **Google ML Crash Course: Interpreting Loss Curves** — visual exercises on oscillation, spikes, overfitting, and bad data. ([developers.google.com](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves))
4. **Deep Learning, Chapter 4** — numerical computation and gradient-descent intuition. ([deeplearningbook.org](https://www.deeplearningbook.org/contents/numerical.html?utm_source=openai))
5. **Deep Learning, Chapters 7–8** — regularization and optimization. ([deeplearningbook.org](https://www.deeplearningbook.org/contents/regularization.html?utm_source=openai))
6. **PyTorch AMP documentation and examples** — overflow, gradient scaling, unscaling, and clipping. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/amp.html?highlight=module&utm_source=openai))
7. **Hugging Face: Causal Language Modeling** — next-token training workflow. ([huggingface.co](https://huggingface.co/docs/transformers/v4.36.1/tasks/language_modeling?utm_source=openai))
8. **Hugging Face: GPT-2 model documentation** — internal label shifting and ignored labels. ([huggingface.co](https://huggingface.co/docs/transformers/model_doc/gpt2?utm_source=openai))
9. **Hugging Face: Perplexity of Fixed-Length Models** — the relationship between NLL, perplexity, tokenization, and context windows. ([huggingface.co](https://huggingface.co/docs/transformers/perplexity?utm_source=openai))
10. **Scikit-learn: Common Pitfalls** — especially preprocessing and data leakage. ([scikit-learn.org](https://scikit-learn.org/stable/common_pitfalls.html?highlight=linearregression))

---

# 13. Using AI agents effectively during the pairing

Use agents to gather evidence in parallel, but retain ownership of the hypothesis tree.

### Agent A: Training-loop audit

> Inspect this training loop for disconnected losses, missing optimizer parameters, incorrect zeroing/stepping, frozen parameters, and scheduler mistakes. Return line-level evidence and the smallest test for each hypothesis.

### Agent B: Data and label audit

> Trace one example from raw data through tokenization to input IDs, labels, attention masks, and loss masks. Verify causal shifting, padding exclusion, response masking, and valid-token counts.

### Agent C: Evaluation audit

> Compare train and validation code for differences in model mode, loss definition, reduction, checkpoint, tokenizer, preprocessing, and generation settings.

### Agent D: Numerical audit

> Identify operations that can create NaN or infinity. Propose checks that locate the first nonfinite tensor and a minimal FP32 reproduction.

The researcher-friendly behavior is to summarize their output as:

> “The agents found four possibilities. Two are inconsistent with the observed nonzero gradients. The fastest remaining test is to compare parameter deltas before and after the optimizer step.”

That demonstrates scientific reasoning rather than simply outsourcing the answer.