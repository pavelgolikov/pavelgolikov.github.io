You’re right. The earlier version mixed levels of abstraction:

- It gave a high-level training loop.
- Then zoomed into **two pieces of the loss/objective stage**: cross-entropy and causal language modeling.
- Then jumped to **one piece of the optimizer stage**: learning rate.
- It skipped equally important stages such as target construction, the forward pass, backpropagation, optimizer mechanics, minibatch sampling, and validation.

That was not a coherent presentation.

The original three details were chosen because they answer three important questions:

1. **Causal-LM objective:** What prediction task are we asking the model to perform?
2. **Cross-entropy:** How do we numerically score its predictions?
3. **Learning rate:** How large a parameter change do we make from that score?

But those are not the only things you need to understand. They belong inside a larger process.

---

# The complete map

An end-to-end training system has eight connected stages:

```text
1. Define the task and split the data
             ↓
2. Construct inputs, targets, and masks
             ↓
3. Run the model forward to produce logits
             ↓
4. Convert logits and targets into a scalar objective
             ↓
5. Backpropagate: compute gradients for every parameter
             ↓
6. Use the optimizer to update the parameters
             ↓
7. Repeat over minibatches while logging training behavior
             ↓
8. Periodically validate on held-out data
```

Here is where the earlier topics fit:

```text
Causal-LM objective ── stages 1, 2, 3, and 4
Cross-entropy      ── stage 4
Learning rate      ── stage 6
```

The missing subjects were:

```text
Data and label construction ── stages 1 and 2
Forward computation         ── stage 3
Gradient computation        ── stage 5
Optimizer mechanics         ── stage 6
Minibatch stochasticity     ── stage 7
Validation/generalization   ── stage 8
Numerical precision         ── cuts across stages 3–6
```

We will now walk through the whole process in order.

---

# 1. Define what the model is supposed to learn

Training starts before any computation occurs. We must first define a learning problem.

In general, we have:

- A model with parameters $\theta$
- Inputs $x$
- Desired targets $y$
- A function that measures how bad the predictions are

The abstract goal is:

$$
\theta^*
=
\arg\min_\theta
\mathbb{E}_{(x,y)\sim \text{real data}}
\left[
\ell(f_\theta(x),y)
\right]
$$

This says:

> Find model parameters that produce low loss on examples from the real-world data distribution.

We do not actually possess the entire real-world distribution. We have a finite dataset, so we approximate the expectation with an average:

$$
L_{\text{train}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^N
\ell(f_\theta(x_i),y_i)
$$

This is the **empirical training loss**.

## For a causal language model

A causal LM is given earlier tokens and asked to predict the next token:

$$
p_\theta(x_t\mid x_1,\ldots,x_{t-1})
$$

For the sequence:

```text
The cat sat down
```

the conceptual training examples are:

```text
Context             Target
--------------------------------
<BOS>               The
<BOS> The           cat
<BOS> The cat       sat
<BOS> The cat sat   down
```

The model is not directly trained to “write good text.” It is trained to assign high probability to the correct next token in these examples.

That is what the **causal-language-model objective** defines: the prediction problem itself.

## Train, validation, and test splits

The dataset is normally divided into:

- **Training data:** directly used to update parameters
- **Validation data:** used to measure generalization and select checkpoints/hyperparameters
- **Test data:** reserved for final evaluation

The optimizer sees only training examples. Validation tells us whether improvements on training data transfer to unseen examples.

This distinction will later explain:

```text
training loss ↓, validation loss ↑
```

The training procedure is succeeding at its direct objective, but the resulting behavior is not generalizing.

---

# 2. Construct inputs, targets, and masks

The raw data must now be converted into tensors.

For an LM, this usually includes:

```text
input_ids
attention_mask
labels
loss mask
```

These have different roles.

## Tokenization

Suppose tokenization produces:

```text
[<BOS>, The, cat, sat, down, <EOS>]
```

Conceptually:

- `<BOS>` should predict `The`
- `The` should predict `cat`
- `cat` should predict `sat`
- `sat` should predict `down`
- `down` should predict `<EOS>`

In a custom implementation, we can express this as:

```text
Model outputs used:  positions 0 through T-2
Targets used:        positions 1 through T-1
```

Or:

```python
prediction_logits = logits[:, :-1, :]
target_tokens = input_ids[:, 1:]
```

This is the **one-token shift**.

Some libraries perform that shift internally. The important invariant is:

> The representation at position $t$ should be scored against the token at position $t+1$, exactly once.

Common bugs include:

- No shift
- Shifting in the wrong direction
- Shifting twice
- Inputs and labels having inconsistent truncation
- Labels belonging to a different example

These bugs can produce flat or misleading loss curves even if the optimizer works perfectly.

---

## Three different masks

For an LM, “the mask” can refer to three separate mechanisms.

### A. Causal attention mask

This operates **inside the model’s forward pass**.

At the position containing `cat`, the model may attend to:

```text
<BOS> The cat
```

but not:

```text
sat down <EOS>
```

Otherwise it could simply look at the answer it is supposed to predict.

The causal mask therefore defines what information the model is allowed to use.

### B. Padding attention mask

Sequences in a batch usually have different lengths, so shorter ones are padded:

```text
Sequence A: The cat sat <EOS>
Sequence B: Hello <EOS> <PAD> <PAD>
```

The padding attention mask tells the model not to treat `<PAD>` tokens as meaningful context.

This also affects the forward pass.

### C. Loss mask

The loss mask controls which token predictions are actually scored.

Consider supervised fine-tuning:

```text
User: What is the capital of France?
Assistant: Paris.
```

The full sequence may be provided as context. But we might want to train only on:

```text
Paris.
```

The model may attend to the user prompt, but prompt positions do not contribute to the loss.

Conceptually:

```text
Tokens:     [User prompt ........] [Paris] [.]
Loss mask:  [0 0 0 0 0 0 0 0 0] [  1  ] [1]
```

This distinction is crucial:

- **Attention mask:** what information can be seen?
- **Loss mask:** which predictions are graded?

If the loss mask accidentally selects only prompt tokens, the loss can decrease while response quality remains unchanged. If it selects no tokens, the loss may become undefined, zero, or NaN depending on the implementation.

At the end of Stage 2, we therefore have a batch containing valid inputs and a precise specification of the predictions we want to reward.

---

# 3. Run the forward pass

Now we apply the current model to the input batch.

Let:

- $B$ be batch size
- $T$ be sequence length
- $V$ be vocabulary size

The model returns logits:

$$
Z=f_\theta(X)
$$

with shape:

$$
[B,T,V]
$$

For every example and token position, there is one score for every vocabulary token.

For example, the logits at the position containing `The` might conceptually be:

```text
cat      2.4
dog      1.8
sat      0.3
car     -0.2
...
```

These are not probabilities yet. They are unrestricted scores.

## What happens inside the model?

For a Transformer LM, the forward pass approximately does:

```text
token IDs
   ↓
token embeddings + position information
   ↓
causal self-attention and MLP layers
   ↓
hidden state at every token position
   ↓
language-model output head
   ↓
vocabulary logits at every position
```

All the trainable matrices in:

- Embeddings
- Attention projections
- MLP layers
- Normalization layers
- Output head

are part of $\theta$.

At this stage, no learning has happened. The model has only made predictions using its current parameters.

## Why a causal LM can train all positions simultaneously

Conceptually, next-token prediction looks sequential:

```text
Predict The
then predict cat
then predict sat
...
```

But during training, the whole ground-truth sequence is available. The causal attention mask ensures that position $t$ cannot see future positions.

This allows all next-token predictions to be computed in one parallel forward pass.

This is sometimes called **teacher forcing**:

- During training, each position receives the real previous tokens.
- During generation, the model receives its own previously generated tokens.

This difference is one reason token-level loss can improve without an equivalent improvement in free-running generation.

---

# 4. Convert the predictions into one scalar objective

We now have:

- Logits from the model
- Correct next-token targets
- A mask specifying which targets matter

But backpropagation requires a scalar quantity representing how bad the batch was.

This is where **cross-entropy** enters.

## From logits to probabilities

At a particular position, suppose the logits are:

$$
z_1,\ldots,z_V
$$

Softmax conceptually turns these into probabilities:

$$
p_j
=
\frac{e^{z_j}}{\sum_k e^{z_k}}
$$

The probabilities are positive and sum to one.

If the correct next token is $y$, the per-token cross-entropy is:

$$
\ell=-\log p_y
$$

This answers:

> How much probability did the model assign to the correct next token?

Examples:

| Probability on correct token | Loss |
|---:|---:|
| $0.90$ | $0.105$ |
| $0.50$ | $0.693$ |
| $0.10$ | $2.303$ |
| $0.01$ | $4.605$ |

A confident wrong prediction is strongly penalized.

## Why cross-entropy is useful for learning

Accuracy only says whether the largest-probability token was correct. Cross-entropy provides a more detailed signal.

Suppose the correct token is `cat`.

### Before

```text
P(cat) = 0.10
P(dog) = 0.60
```

### After

```text
P(cat) = 0.30
P(dog) = 0.40
```

The model is still wrong because `dog` remains the top prediction. Accuracy has not changed.

But cross-entropy improves:

$$
-\log 0.10=2.303
$$

$$
-\log 0.30=1.204
$$

This gives the optimizer useful information before the prediction becomes correct.

---

## The output-layer error signal

A particularly useful result is:

$$
\frac{\partial \ell}{\partial z_j}
=
p_j-\mathbf{1}[j=y]
$$

For the correct token:

$$
\frac{\partial \ell}{\partial z_y}=p_y-1
$$

This is negative, so gradient descent will try to increase the correct token’s logit.

For every incorrect token:

$$
\frac{\partial \ell}{\partial z_j}=p_j
$$

This is positive, so gradient descent will try to decrease its logit.

Thus cross-entropy creates an error signal that says:

```text
Raise the correct token's score.
Lower the incorrect token scores.
Lower especially those incorrect tokens receiving too much probability.
```

This error signal will be propagated backward through the model in Stage 5.

---

## Combining token losses

The model makes many predictions in one batch. We first obtain a loss for every valid token:

$$
\ell_{b,t}
=
-\log p_\theta(y_{b,t}\mid x_{b,\le t})
$$

Then apply the loss mask $m_{b,t}$:

$$
L_{\text{data}}
=
\frac{
\sum_{b,t}m_{b,t}\ell_{b,t}
}{
\sum_{b,t}m_{b,t}
}
$$

This is the average negative log-likelihood over valid target tokens.

The denominator matters. We normally want to divide by the number of valid targets, not by:

- The padded sequence length
- The number of batches
- A fixed maximum length
- The number of examples regardless of their token counts

Otherwise train and validation losses may not be comparable.

## Optional regularization

The training objective may also include a regularization term:

$$
J(\theta)
=
L_{\text{data}}(\theta)
+
\lambda R(\theta)
$$

For example:

$$
R(\theta)=\frac{1}{2}\lVert\theta\rVert^2
$$

The scalar sent into backpropagation is then $J$, not merely the data loss.

However, regularization can enter training in different places:

- An explicit penalty may be added to the scalar objective.
- Dropout modifies the forward pass.
- AdamW weight decay modifies the optimizer update.

These differences matter when interpreting training versus validation loss.

At the end of Stage 4, the entire batch has been summarized into one scalar:

```python
loss
```

That scalar tells us how badly the current model performed on this batch.

---

# 5. Backpropagate the loss

The loss tells us that the model is wrong, but not yet how each individual parameter contributed to the error.

Backpropagation computes:

$$
g_i
=
\frac{\partial L}{\partial \theta_i}
$$

for every trainable parameter $\theta_i$.

Collectively:

$$
g=\nabla_\theta L
$$

The gradient answers:

> If I increase this parameter slightly, in which direction and by approximately how much will the loss change?

## How the error travels backward

At the logits, cross-entropy produces:

$$
p-\operatorname{onehot}(y)
$$

Backpropagation then applies the chain rule:

```text
loss
  ↓
logits
  ↓
LM output head
  ↓
final Transformer layer
  ↓
earlier Transformer layers
  ↓
embeddings
```

For a simplified chain:

$$
\theta
\rightarrow h
\rightarrow z
\rightarrow L
$$

the chain rule says:

$$
\frac{\partial L}{\partial \theta}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial h}
\frac{\partial h}{\partial \theta}
$$

Cross-entropy supplies the first error signal:

$$
\frac{\partial L}{\partial z}
$$

The model architecture determines how that signal propagates to every earlier parameter.

## What `loss.backward()` does

In PyTorch:

```python
loss.backward()
```

computes and stores gradients in:

```python
parameter.grad
```

It does **not** update parameters.

This distinction is essential:

```text
backward()         computes gradients
optimizer.step()   changes parameters
```

## Why graph connectivity matters

The framework can calculate a gradient only if it recorded a differentiable path from a parameter to the loss.

The path can be broken by:

- `.detach()`
- Converting a tensor to NumPy
- Constructing a new tensor from `loss.item()`
- Running the forward pass under `no_grad`
- Using nondifferentiable decisions such as `argmax` before the loss
- Setting `requires_grad=False`

If a parameter does not contribute to the loss, its gradient is often `None`.

If it contributes but the derivative is zero, its gradient is a tensor containing zeros.

Those mean different things:

```text
grad is None
→ No gradient path was computed.

grad is zero
→ A path exists, but the local derivative produced no signal.
```

## Why gradients are zeroed

By default, PyTorch accumulates gradients:

```python
parameter.grad = old_gradient + new_gradient
```

Therefore each ordinary step begins with:

```python
optimizer.zero_grad()
```

Otherwise gradients from unrelated batches would unintentionally add together.

Intentional gradient accumulation is possible, but it changes when and how the optimizer update occurs.

At the end of Stage 5, we have not yet learned anything. We have only computed a proposed direction for changing every parameter.

---

# 6. Convert gradients into a parameter update

Now the optimizer uses the gradients to modify the parameters.

For plain stochastic gradient descent:

$$
\theta_{k+1}
=
\theta_k-\eta g_k
$$

where:

- $\theta_k$ is the current parameter vector
- $g_k$ is the gradient from the current batch
- $\eta$ is the learning rate

This is where the earlier **learning-rate explanation** belongs.

## What the gradient and learning rate do separately

The gradient specifies a direction and relative magnitudes:

```text
Change parameter A strongly downward.
Change parameter B slightly upward.
Do not change parameter C.
```

The learning rate controls the overall update scale:

```text
gradient = proposed direction
learning rate = how far to move
```

If the learning rate is too small:

- Parameter updates are tiny.
- Training loss may appear flat.
- Progress may require an impractical number of steps.

If it is too large:

- Updates overshoot useful regions.
- Loss oscillates or spikes.
- Activations or parameters may explode.
- Training may produce infinities or NaNs.

---

## Adam and AdamW

LLMs are generally not trained with plain SGD. Adam-like optimizers maintain state across updates.

Simplifying slightly, Adam computes:

$$
m_k
=
\beta_1m_{k-1}
+
(1-\beta_1)g_k
$$

$$
v_k
=
\beta_2v_{k-1}
+
(1-\beta_2)g_k^2
$$

The parameter update is approximately:

$$
\Delta\theta_k
=
-\eta_k
\frac{\hat m_k}
{\sqrt{\hat v_k}+\epsilon}
$$

Intuitively:

- $m$ is a moving average of gradient direction.
- $v$ tracks recent squared gradient magnitude.
- Dividing by $\sqrt v$ gives different effective scaling to different parameters.
- $\eta$ still scales the overall update.
- $\epsilon$ prevents division by zero.

AdamW also applies weight decay separately from the gradient-based update.

Because Adam has persistent state, an update depends not only on the current batch but also on previous gradients. This is why the same batch can behave differently depending on the optimizer state that precedes it.

## Gradient clipping

Between backward and the optimizer step, we may limit the gradient norm:

```text
loss.backward()
      ↓
inspect/unscale gradients
      ↓
clip gradients
      ↓
optimizer.step()
```

Clipping prevents a single extremely large gradient from causing an unbounded update.

It is a guardrail, not necessarily a cure. If clipping activates constantly, the underlying causes still need investigation.

## Learning-rate scheduling

The learning rate may change over time:

```text
warmup → peak learning rate → gradual decay
```

Warmup makes the earliest updates smaller while optimizer statistics and model activations stabilize.

A scheduler therefore acts at Stage 6 by changing $\eta_k$ before or after each update, depending on the implementation.

At the end of Stage 6, the model parameters have actually changed:

$$
\theta_k\rightarrow\theta_{k+1}
$$

The next forward pass will therefore produce slightly different logits.

---

# 7. Repeat over minibatches

The ideal objective averages over the entire training dataset:

$$
L_{\text{train}}
=
\frac{1}{N}
\sum_{i=1}^N \ell_i
$$

Computing that full gradient before every update would be expensive. Instead, we sample a minibatch $B_k$:

$$
L_{B_k}
=
\frac{1}{|B_k|}
\sum_{i\in B_k}\ell_i
$$

and use:

$$
g_k=\nabla_\theta L_{B_k}
$$

as an estimate of the full-dataset gradient.

This creates the feedback loop:

```text
Current parameters
       ↓
Predictions on one minibatch
       ↓
Batch loss
       ↓
Batch gradients
       ↓
Parameter update
       ↓
New parameters
       ↓
Predictions on the next minibatch
```

Training is this loop repeated thousands or millions of times.

## Why training loss is noisy

Different batches contain different:

- Examples
- Sequence lengths
- Domains
- Difficulty levels
- Numbers of valid target tokens
- Outliers

Therefore batch losses and gradients fluctuate.

A larger batch generally gives a more stable estimate of the dataset-average direction. A smaller batch produces noisier updates.

Noise itself is not automatically bad. The diagnostic question is whether it is:

- Small and random
- Associated with particular examples
- Periodic
- Growing over time
- Accompanied by large gradient or update norms
- Causing permanent damage

## Epochs and steps

A **step** usually means one optimizer update.

An **epoch** means processing roughly the entire training dataset once.

If there are 10,000 batches and one optimizer update per batch:

```text
1 epoch = 10,000 steps
```

With gradient accumulation over four batches:

```text
4 forward/backward passes = 1 optimizer step
```

This distinction matters when configuring:

- Learning-rate schedules
- Logging
- Evaluation frequency
- Checkpoint frequency

## What the displayed training loss means

The training loss displayed for an epoch is often an average of losses observed while the model was changing:

```text
Batch 1: evaluated with θ₁
Batch 2: evaluated with θ₂
Batch 3: evaluated with θ₃
...
```

It is not necessarily the loss of the final model on the training set.

That fact will become important in Stage 8.

---

# 8. Validate the current model

Periodically, training pauses and the current parameter state is evaluated on held-out data.

Conceptually:

```text
Freeze current parameters
         ↓
Run forward passes on validation data
         ↓
Compute validation loss and metrics
         ↓
Do not call backward()
         ↓
Do not update parameters
```

Validation answers a different question from training.

- **Training loss:** Is the optimization procedure fitting the data it receives?
- **Validation loss:** Does the learned behavior transfer to unseen examples?

## Evaluation mode

During validation, the model is placed in evaluation mode:

```python
model.eval()
```

This changes operations such as dropout from training behavior to inference behavior.

Gradient recording is also disabled:

```python
with torch.no_grad():
    ...
```

These are separate:

- `model.eval()` changes model behavior.
- `torch.no_grad()` avoids building a backward graph.

## Loss versus evaluation metric

Validation may compute both:

- The same cross-entropy loss used for training
- A task-specific metric such as accuracy, F1, exact match, pass rate, or reward

The optimizer receives feedback from the training loss, not necessarily from the metric.

The connection is:

```text
training loss
    ↓
backpropagation
    ↓
parameter updates

evaluation metric
    ↓
reported to researcher
    ✕
normally no direct gradient
```

This explains how loss can decrease while a metric remains flat.

For example, cross-entropy rewards moving the correct-token probability from $0.1$ to $0.4$, even if another token still has probability $0.41$. Exact-match accuracy does not improve until the prediction actually changes.

## Generalization

If training and validation losses both decrease:

```text
The model is learning patterns that transfer.
```

If training decreases but validation increases:

```text
The optimization loop is fitting training data,
but the resulting behavior is becoming less transferable.
```

That is the central meaning of overfitting.

If both remain high and flat:

```text
Either the earlier training stages are broken,
or the model/objective cannot fit the task.
```

Validation closes the loop by telling us whether successful optimization also produced useful learning.

---

# The complete loop in code

The following is a custom causal-LM loop written to expose every stage explicitly:

```python
import torch
import torch.nn.functional as F

model.train()

for batch in train_loader:
    # ---------------------------------------------------------
    # Stage 2: Constructed batch
    # ---------------------------------------------------------
    input_ids = batch["input_ids"]          # [B, T]
    attention_mask = batch["attention_mask"] # [B, T]
    loss_mask = batch["loss_mask"]          # [B, T]

    # Remove gradients left from the previous update.
    optimizer.zero_grad(set_to_none=True)

    # ---------------------------------------------------------
    # Stage 3: Forward pass
    # ---------------------------------------------------------
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits                 # [B, T, V]

    # ---------------------------------------------------------
    # Stage 4: Next-token objective and cross-entropy
    # ---------------------------------------------------------
    # Position t predicts token t+1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().float()

    batch_size, prediction_length, vocab_size = shift_logits.shape

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_targets.view(-1),
        reduction="none",
    ).view(batch_size, prediction_length)

    valid_token_count = shift_mask.sum()
    assert valid_token_count > 0

    loss = (
        per_token_loss * shift_mask
    ).sum() / valid_token_count

    # ---------------------------------------------------------
    # Stage 5: Backpropagation
    # ---------------------------------------------------------
    loss.backward()

    # Optional protection between backward and update.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    # ---------------------------------------------------------
    # Stage 6: Parameter update
    # ---------------------------------------------------------
    optimizer.step()
    scheduler.step()

    # ---------------------------------------------------------
    # Stage 7: Logging
    # ---------------------------------------------------------
    log({
        "train_loss": loss.item(),
        "learning_rate": scheduler.get_last_lr()[0],
    })
```

Validation uses the same target and loss construction, but removes Stages 5 and 6:

```python
model.eval()

total_nll = 0.0
total_valid_tokens = 0

with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_mask = loss_mask[:, 1:].float()

        B, T, V = shift_logits.shape

        per_token_loss = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_targets.reshape(-1),
            reduction="none",
        ).reshape(B, T)

        total_nll += (
            per_token_loss * shift_mask
        ).sum().item()

        total_valid_tokens += shift_mask.sum().item()

validation_loss = total_nll / total_valid_tokens

model.train()
```

The important difference is:

```text
Training:
forward → loss → backward → update

Validation:
forward → loss/metrics
```

---

# Where additional training mechanisms plug in

Real training adds several mechanisms to this base loop. They are not independent topics; each occupies a precise location.

| Mechanism | Location in process | What it changes |
|---|---|---|
| Data augmentation | Stage 2 | Makes training examples different or harder |
| Causal mask | Stage 3 | Prevents access to future tokens |
| Dropout | Stage 3 | Randomly changes the training-time forward pass |
| Cross-entropy | Stage 4 | Turns logits and labels into error |
| Loss masking | Stage 4 | Selects which token errors matter |
| Explicit regularization | Stage 4 | Adds a penalty to the scalar objective |
| Backpropagation | Stage 5 | Converts scalar error into parameter gradients |
| Gradient accumulation | Stages 5–6 | Combines gradients across multiple batches |
| Mixed precision | Stages 3–6 | Changes numerical representation and scaling |
| Gradient clipping | Between 5 and 6 | Limits gradient magnitude before updating |
| Adam/AdamW | Stage 6 | Transforms gradients into updates |
| Learning rate | Stage 6 | Controls overall update scale |
| Weight decay | Stage 6 | Shrinks parameters during optimization |
| Warmup/scheduler | Stage 6 | Changes learning rate over time |
| Batch sampling | Stage 7 | Determines the stochastic gradient estimate |
| Validation loss | Stage 8 | Measures held-out objective performance |
| Evaluation metric | Stage 8 | Measures task behavior, possibly differently from loss |
| Early stopping | After Stage 8 | Selects when to stop updating |
| Checkpointing | Between iterations | Saves particular parameter/optimizer states |

This is the map you should use whenever a mechanism is introduced: ask **where in the loop it acts** and **what quantity it changes**.

---

# Mapping loss-curve problems back to the process

The curve patterns now correspond to failures in particular stages.

| Observation | First stages to inspect |
|---|---|
| Training loss is flat | 2: labels/masks; 4: loss; 5: graph/gradients; 6: optimizer |
| Both train and validation are high | 1–6, or insufficient model capacity |
| Training is noisy or spikes | 2: batch data; 5: gradients; 6: LR/update; numerical precision |
| Loss becomes NaN | Find first nonfinite value across stages 2–6 |
| Validation is lower than training | 3: dropout/mode; 4: loss definition; 7–8: timing and aggregation |
| Train decreases, validation increases | Training loop works; inspect generalization, data split, and regularization |
| Loss decreases, metric stays flat | Stage 4 objective does not fully align with Stage 8 metric |

This is why loss curves cannot be diagnosed purely from their shape. The shape is evidence about where in this process the failure might be.

---

# What you should know in detail

No, cross-entropy, the causal objective, and learning rate are not the only three detailed topics.

For loss-curve diagnosis, the coherent core is:

## Highest priority

1. **Data, labels, and masks**
   - Next-token shifting
   - Padding
   - Prompt versus response loss
   - Split integrity

2. **Forward-pass outputs**
   - Logits
   - Train versus evaluation mode
   - Causal attention

3. **Loss construction**
   - Cross-entropy
   - Reduction and normalization
   - Regularization
   - Loss versus metric

4. **Backpropagation**
   - Chain rule intuition
   - Graph connectivity
   - `grad=None` versus zero gradients
   - Gradient accumulation

5. **Optimizer update**
   - SGD intuition
   - Adam/AdamW intuition
   - Learning rate
   - Gradient clipping
   - Schedulers and warmup

6. **Minibatch training**
   - Why loss is noisy
   - Batch size
   - Steps versus epochs
   - Batch composition

7. **Validation and generalization**
   - Train versus validation loss
   - Evaluation mode
   - Overfitting and underfitting
   - Metric mismatch

8. **Numerical stability**
   - NaN and infinity
   - Mixed precision
   - Exploding gradients
   - Stable loss implementations

## A separate, adjacent subject

The internal Transformer architecture lives mostly inside Stage 3:

- Attention
- MLP blocks
- Residual connections
- Layer normalization
- Embeddings
- Output head

You need some understanding of those components to diagnose layer-specific gradient or numerical problems, but they are a distinct layer of detail. You do not need to derive every Transformer operation before understanding the training loop itself.

The central mental model to retain is:

> The data construction defines the question.  
> The forward pass produces an answer.  
> The loss scores the answer.  
> Backpropagation assigns responsibility for the error.  
> The optimizer changes the responsible parameters.  
> Repetition fits the training data.  
> Validation determines whether that learning generalizes.