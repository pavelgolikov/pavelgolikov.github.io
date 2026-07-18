# Wednesday Study Program: Hidden States and Linear Probes

## First, a date clarification

The Wednesday block in the original schedule referred to **Wednesday, July 15, 2026**. Today is **Saturday, July 18**, and your interview is Monday, July 20, so treat this as today’s main technical preparation block.

- **Core program:** approximately 2 hours 30 minutes
- **Full program with bug hunt and timed simulation:** approximately 3 hours 15 minutes
- If time is limited, complete Exercises 1–5 first.

---

# 1. What Wednesday is about

Tuesday’s program taught you how to evaluate a language model’s output.

Wednesday’s program moves one level deeper:

> Instead of examining only the model’s final token probabilities, you will extract its internal hidden representations and test whether those representations contain information about a property of the input.

When `output_hidden_states=True`, Hugging Face returns:

- one tensor for the initial embedding output
- one tensor for the output of each Transformer layer
- each tensor has shape `[batch, sequence, hidden_size]` ([huggingface.co](https://huggingface.co/docs/transformers/model_doc/gpt2?utm_source=openai))

You will then train a **linear probe**:

$$
\text{probe}(h)=Wh+b
$$

The language model remains frozen. Only the small linear classifier is trained.

## The central research question

Here is the precise question you will investigate:

> **After a frozen language model reads a sentence, does its hidden representation contain enough linearly accessible information for a simple classifier to determine whether the sentence’s main noun is an animal or a vehicle—even when the classifier is tested on nouns it never saw during training?**

For example:

- Training noun: `cat`
- Test noun: `elephant`
- Both are animals, but the probe never sees `elephant` during training.

This is a useful toy experiment because it practices:

- extracting hidden states
- choosing a layer
- pooling token representations
- freezing a model
- training a linear classifier
- preventing train/test leakage
- adding chance and shuffled-label baselines
- distinguishing **decodability** from **causality**

## What a successful probe would mean

If the probe performs well on unseen nouns, you may conclude:

> “Animal-versus-vehicle category information is linearly decodable from this model representation under this particular dataset, split, layer, and pooling method.”

You may **not** conclude:

- that the model consciously represents animals and vehicles
- that the model uses this information when generating text
- that the selected hidden dimensions causally control behavior
- that the result generalizes to all concepts or datasets

A probe measures accessibility to a classifier, not causal use by the model.

---

# 2. Should you use your coding agent?

**Yes.** Use the same agent you intend to use during the interview.

Your coding agent should:

- write the hidden-state extraction function
- implement pooling
- implement the probe training loop
- run the experiment
- help diagnose shape or device errors

You should personally decide and verify:

- what property is being predicted
- which examples belong in train, validation, and test
- which layer is selected
- how token representations are pooled
- whether there is data leakage
- what baseline is appropriate
- what the result does and does not establish

Use the same sequence as Tuesday:

1. You state the experimental plan.
2. Ask the agent to review the plan without coding.
3. Correct its plan.
4. Ask it for the smallest executable implementation.
5. Inspect the critical lines.
6. Run it.
7. Ask for a targeted audit.
8. Explain the result yourself.

---

# 3. Set up the Wednesday directory — 10 minutes

Reuse Tuesday’s environment rather than installing everything again.

```bash
cd ae-interview-prep
source .venv/bin/activate

mkdir wednesday_probe
cd wednesday_probe
```

On Windows:

```powershell
cd ae-interview-prep
.venv\Scripts\Activate.ps1

mkdir wednesday_probe
cd wednesday_probe
```

Create:

```text
wednesday_probe/
├── hidden_probe.py
└── notes.md
```

Run Tuesday’s smoke test once:

```bash
python ../smoke.py
```

You do not need to reinstall or redownload the model if it is already cached.

---

# Exercise 1: Inspect all hidden states

**Time: 20 minutes**

## Plain-English question

> When the model processes a padded batch of sentences, what hidden-state tensors does it return, and what does each tensor index represent?

## Why this question matters

A common implementation mistake is to write something like:

```python
hidden = outputs.hidden_states[5]
```

without knowing whether index 5 refers to:

- the fifth Transformer layer
- the sixth Transformer layer
- the embedding output
- the final layer

Before using a representation, you need to inspect the complete structure.

## Use these four sentences

```python
texts = [
    "The cat slept.",
    "The old truck stopped beside the bridge.",
    "Yesterday the horse wandered through the quiet field.",
    "Several people watched the airplane disappear behind the clouds.",
]
```

They intentionally have different lengths so padding occurs.

## Agent planning prompt

```text
Do not write code yet.

We want to inspect every hidden-state tensor returned by
distilbert/distilgpt2 for a padded batch of four variable-length
sentences.

Explain:
- the expected input tensor shapes,
- what outputs.hidden_states contains,
- what hidden_states[0] represents,
- what hidden_states[-1] represents,
- why padding and attention masks matter,
- and the assertions we should make.

Keep the explanation under 15 bullet points.
```

Correct the agent if it fails to mention that the tuple includes the embedding output.

## Implementation prompt

```text
Create hidden_probe.py using torch and transformers.

Requirements:
- Set torch.manual_seed(0).
- Load distilbert/distilgpt2 with AutoTokenizer and
  AutoModelForCausalLM.
- Set the pad token to the EOS token if necessary.
- Use right padding.
- Tokenize the four supplied sentences with padding=True,
  truncation=True, max_length=64, and return_tensors="pt".
- Select CUDA, then MPS, then CPU.
- Move the model and all tensors to the selected device.
- Freeze every language-model parameter.
- Put the model in eval mode.
- Run it under torch.inference_mode().
- Pass output_hidden_states=True and use_cache=False.
- Print:
  - input_ids shape,
  - attention_mask shape,
  - model.config.n_layer,
  - the number of returned hidden-state tensors,
  - the shape of every hidden-state tensor,
  - logits shape.
- Assert that the number of hidden-state tensors is
  model.config.n_layer + 1.
- Do not train anything yet.
```

`use_cache=False` is appropriate because this is representation extraction, not incremental text generation; the cache is unnecessary for this exercise. The hidden-state tuple should contain the embedding output followed by one tensor for each model layer. ([huggingface.co](https://huggingface.co/docs/transformers/model_doc/gpt2?utm_source=openai))

## Pass criteria

You should observe:

```text
input_ids:       [4, T]
attention_mask:  [4, T]
hidden_states[0]:  [4, T, H]
hidden_states[1]:  [4, T, H]
...
hidden_states[-1]: [4, T, H]
logits:          [4, T, V]
```

The exact values of `T`, `H`, and `V` should be printed by your script rather than hard-coded.

## Questions to answer aloud

### What is `hidden_states[0]`?

The model’s initial embedding-level representation, before the sequence has passed through all Transformer blocks.

### What is `hidden_states[-1]`?

The representation returned after the model’s final Transformer layer.

### Why does every layer have the same `[B, T, H]` shape?

Each layer produces one hidden vector of dimension `H` for every token in every sequence.

### Why use `model.eval()` and `torch.inference_mode()`?

- `model.eval()` disables training behavior such as dropout.
- `torch.inference_mode()` prevents autograd graph construction.
- The base model is frozen, and no gradient through it is needed while extracting representations.

---

# Exercise 2: Convert token representations into one vector per sentence

**Time: 25 minutes**

## Plain-English question

> The model gives us one hidden vector for every token. How do we create a single representation for the entire sentence without accidentally using padding?

The hidden state has shape:

```text
[B, T, H]
```

The linear probe needs:

```text
[B, H]
```

You will implement and compare two pooling methods.

---

## Method A: Final non-padding token

For a right-padded causal language model, the final real token has seen every previous token in the sentence. Its hidden vector can therefore serve as a sequence summary.

Do **not** use:

```python
hidden[:, -1, :]
```

For shorter sequences, position `-1` may be padding.

Instead:

```python
def final_nonpadding_pool(hidden, attention_mask):
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(
        hidden.size(0),
        device=hidden.device,
    )
    return hidden[batch_indices, last_indices]
```

This implementation assumes:

- right padding
- at least one real token per example

---

## Method B: Masked mean pooling

This averages all real token representations while excluding padding:

```python
def masked_mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1)

    return summed / counts
```

Padding makes variable-length inputs rectangular, but padding positions should not contribute to pooling. ([huggingface.co](https://huggingface.co/docs/transformers/v4.21.0/pad_truncation?utm_source=openai))

## Agent prompt

```text
Extend hidden_probe.py with two functions:

1. final_nonpadding_pool(hidden, attention_mask)
2. masked_mean_pool(hidden, attention_mask)

For every returned hidden-state layer:
- apply both pooling methods,
- assert that the result has shape [batch_size, hidden_size],
- print the layer number and pooled shapes.

Also print:
- the attention-mask token count for each example,
- the final non-padding token index for each example.

Add an assertion that no final index is less than zero or greater
than the sequence length minus one.
```

## Padding-invariance test

Run the shortest sentence in two settings:

```python
short_text = "The cat slept."
long_text = (
    "Several people watched the large vehicle travel slowly "
    "through the crowded city center."
)
```

First:

```python
representation_alone = extract(["The cat slept."])
```

Then:

```python
representations_batched = extract([
    "The cat slept.",
    long_text,
])
```

Compare the short sentence’s representation in both cases.

```python
assert torch.allclose(
    representation_alone[0],
    representations_batched[0],
    atol=1e-4,
    rtol=1e-4,
)
```

Do this separately for:

- final non-padding pooling
- masked mean pooling

## Deliberate mistake

Temporarily use:

```python
wrong_pool = hidden[:, -1, :]
```

Compare it with the correct final-token representation for the shortest sentence.

Explain:

> “The final tensor position is the final position in the padded batch, not necessarily the final real token in each sequence.”

---

# Exercise 3: Construct a controlled probe dataset

**Time: 25 minutes**

## Plain-English research question

> Can a linear classifier trained on frozen model representations distinguish sentences about animals from sentences about vehicles when the test sentences contain nouns that never appeared during probe training?

## Why use a synthetic dataset?

The goal today is not to obtain a publishable scientific result. The goal is to practice:

- controlled dataset construction
- balanced labels
- explicit train/validation/test separation
- protection against lexical leakage
- clear experimental interpretation

A synthetic dataset lets you understand exactly how each example was created.

---

## Vocabulary split

### Probe-training nouns

```python
TRAIN_ANIMALS = [
    "cat",
    "dog",
    "horse",
    "rabbit",
]

TRAIN_VEHICLES = [
    "car",
    "truck",
    "bus",
    "van",
]
```

### Held-out test nouns

```python
TEST_ANIMALS = [
    "tiger",
    "elephant",
    "dolphin",
    "eagle",
]

TEST_VEHICLES = [
    "bicycle",
    "airplane",
    "boat",
    "motorcycle",
]
```

No held-out noun should appear in the probe’s training or validation data.

---

## Sentence templates

Use the same templates for both categories. Otherwise, the probe could classify templates rather than concepts.

```python
TEMPLATES = [
    "Yesterday, I noticed the {noun} near the old bridge.",
    "The photograph clearly showed the {noun} in the center.",
    "During the conversation, someone mentioned the {noun} twice.",
    "We stopped for a moment to examine the {noun} closely.",
    "The story began when the {noun} appeared unexpectedly.",
    "From a distance, I could still recognize the {noun}.",
    "The guide answered several questions about the {noun}.",
    "Everyone in the room was discussing the {noun}.",
    "By the end of the day, we had found the {noun}.",
    "The report included one unusual detail about the {noun}.",
]
```

## Exact split design

### Training split

Use:

- all 8 training nouns
- templates 0 through 7

Size:

```text
8 nouns × 8 templates = 64 examples
```

### Validation split

Use:

- the same 8 training nouns
- templates 8 and 9

Size:

```text
8 nouns × 2 templates = 16 examples
```

This evaluates whether the probe works on new sentence templates while seeing familiar nouns.

### Test split

Use:

- all 8 held-out test nouns
- all 10 templates

Size:

```text
8 nouns × 10 templates = 80 examples
```

This is the most important split because it evaluates category generalization to unseen nouns.

## Labels

Use:

```text
animal  = 0
vehicle = 1
```

Create columns:

```text
text
label
noun
category
```

You can construct each split using `Dataset.from_dict`, which creates a Hugging Face dataset from lists stored in a dictionary. ([huggingface.co](https://huggingface.co/docs/datasets/main/package_reference/main_classes?utm_source=openai))

## Agent prompt

```text
Extend hidden_probe.py to construct three Hugging Face Dataset
objects: train, validation, and test.

Use the exact noun lists, templates, labels, and split rules supplied.

Each dataset must contain:
- text,
- label,
- noun,
- category.

Shuffle each split with seed 0.

Print:
- number of examples in each split,
- label count in each split,
- unique nouns in each split,
- five example rows.

Add assertions that:
- train has 64 examples,
- validation has 16 examples,
- test has 80 examples,
- every split is balanced,
- no test noun appears in train or validation,
- every label is either 0 or 1.
```

## Additional confound check

Tokenize every sentence and report:

```text
mean token count for animals
mean token count for vehicles
```

Do this separately for train, validation, and test.

If one category is substantially longer, the probe could partially use length rather than semantic category.

You do not necessarily have to eliminate the difference today, but you must report it.

---

# Exercise 4: Extract representations and train a linear probe

**Time: 40 minutes**

## Plain-English research question

> Using the frozen model’s final-layer, final-non-padding-token representation, can a two-class linear classifier predict animal versus vehicle on held-out nouns?

Start with exactly one representation choice:

- **Layer:** final hidden layer
- **Pooling:** final non-padding token

Do not compare every layer yet. First make one end-to-end experiment work.

---

## Step 1: Freeze the language model

```python
for parameter in model.parameters():
    parameter.requires_grad_(False)

model.eval()
```

Verify:

```python
assert all(
    not parameter.requires_grad
    for parameter in model.parameters()
)
```

## Step 2: Extract representations once

Create:

```python
def extract_representations(
    dataset,
    layer_index,
    pooling,
    batch_size=16,
):
    ...
```

The function should return:

```text
features: [N, H]
labels:   [N]
```

Use:

- `output_hidden_states=True`
- `use_cache=False`
- `torch.inference_mode()`
- batches of 16
- `.float().cpu()` before storing the pooled features

Expected shapes:

```text
train features:      [64, H]
validation features: [16, H]
test features:       [80, H]
```

## Step 3: Standardize using training statistics only

Calculate:

```python
train_mean = train_features.mean(dim=0, keepdim=True)

train_std = train_features.std(
    dim=0,
    keepdim=True,
    unbiased=False,
).clamp_min(1e-5)
```

Apply those same values to all splits:

```python
train_x = (train_features - train_mean) / train_std
val_x = (val_features - train_mean) / train_std
test_x = (test_features - train_mean) / train_std
```

Do **not** calculate normalization statistics using validation or test features. That would incorporate evaluation-distribution information into preprocessing.

## Step 4: Train the probe

Use:

```python
probe = torch.nn.Linear(hidden_size, 2)
```

`nn.Linear` applies an affine transformation to the final feature dimension, so `[N, H]` becomes `[N, 2]`. ([docs.pytorch.org](https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html?utm_source=openai))

Use:

```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    probe.parameters(),
    lr=1e-2,
    weight_decay=1e-3,
)
```

Cross-entropy should receive:

```text
probe logits: [N, 2]
labels:       [N]
```

with integer labels `0` or `1`. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?utm_source=openai))

Train for at most 200 epochs. Select the best probe state using **validation accuracy**, with validation loss as a tie-breaker.

Do not use test accuracy to select the best epoch.

---

## Full agent prompt

```text
Extend hidden_probe.py into an end-to-end linear-probe experiment.

Representation extraction:
- Freeze all language-model parameters.
- Use model.eval().
- Extract hidden states under torch.inference_mode().
- Use the final hidden-state layer.
- Pool using the final non-padding token.
- Process examples in batches of 16.
- Return CPU float32 feature tensors and torch.long labels.
- Assert feature shapes for train, validation, and test.

Preprocessing:
- Calculate feature mean and standard deviation using only the
  training features.
- Apply those training statistics to train, validation, and test.
- Clamp standard deviations to at least 1e-5.

Probe:
- Use torch.nn.Linear(hidden_size, 2).
- Print the number of trainable probe parameters.
- Use CrossEntropyLoss.
- Use AdamW with lr=1e-2 and weight_decay=1e-3.
- Train for at most 200 full-batch epochs.
- Set a reproducible seed before initializing the probe.
- Calculate train and validation accuracy every epoch.
- Save the state with the best validation accuracy, using lower
  validation loss as a tie-breaker.
- Restore the selected state and evaluate test accuracy exactly once.

Verification:
- Assert all language-model parameters have requires_grad=False.
- Assert all language-model parameter gradients remain None.
- Assert at least one probe parameter receives a gradient.
- Report train, validation, and test sample counts and accuracies.
- Report the selected epoch.
- Print a 2x2 test confusion matrix using only torch.
- Do not use sklearn, numpy, pandas, Trainer, or additional packages.
```

## Parameter-count exercise

Before running, predict the number of probe parameters.

For hidden dimension `H` and two output classes:

$$
\text{parameters}=2H+2
$$

The weight has shape:

```text
[2, H]
```

The bias has shape:

```text
[2]
```

Add:

```python
expected_parameters = 2 * hidden_size + 2

actual_parameters = sum(
    parameter.numel()
    for parameter in probe.parameters()
)

assert actual_parameters == expected_parameters
```

---

## How to interpret the result

### If training accuracy is high but test accuracy is near chance

Possible explanations include:

- the probe memorized training nouns
- the representation does not organize unseen category words linearly
- the validation split was too easy
- the training dataset was too small
- the selected pooling method was weak
- the optimizer or learning rate was poor

Do not conclude that the model contains no category information.

### If held-out test accuracy is clearly above chance

You may say:

> “A linear classifier trained on final-layer representations generalized to held-out category words, providing evidence that animal-versus-vehicle information is linearly decodable from this representation.”

You should add:

> “This does not establish that the model uses this representation causally.”

---

# Exercise 5: Add baselines and a shuffled-label control

**Time: 25 minutes**

## Question A: Is the probe better than a trivial classifier?

Because the test data is balanced, a majority-class classifier should achieve:

```text
50% accuracy
```

Still calculate it from the actual labels rather than hard-coding it:

```python
count_0 = test_labels.eq(0).sum()
count_1 = test_labels.eq(1).sum()

majority_accuracy = (
    torch.maximum(count_0, count_1)
    / len(test_labels)
)
```

Print:

- class counts
- majority baseline
- probe accuracy
- improvement over the majority baseline

---

## Question B: Could the pipeline appear successful even when labels contain no meaningful relationship to the representations?

This is what the shuffled-label control tests.

### Correct procedure

1. Keep representations unchanged.
2. Randomly permute the **training labels**.
3. Train a new linear probe.
4. Evaluate against the real test labels.
5. Do not shuffle the test labels.
6. Do not choose the shuffled probe’s epoch based on true test accuracy.

Use the epoch count selected by the real probe:

```text
If the real probe selected epoch 47,
train every shuffled-label probe for exactly 48 epochs.
```

Repeat with five shuffle seeds:

```text
0, 1, 2, 3, 4
```

Report:

```text
real probe test accuracy
majority baseline
mean shuffled-label test accuracy
standard deviation of shuffled-label test accuracy
```

## Agent prompt

```text
Add two controls to hidden_probe.py.

Control 1: majority baseline
- Calculate it from the real test-label counts.
- Do not hard-code 0.5.

Control 2: shuffled training labels
- For seeds 0 through 4, create a fresh random permutation of only
  the training labels.
- Initialize a fresh linear probe for every run.
- Train for exactly the number of epochs selected by the real probe.
- Do not select epochs using the test set.
- Evaluate each shuffled probe against the real test labels.
- Report every shuffled accuracy, their mean, and their standard
  deviation using torch.

Also report shuffled-label training accuracy, but explain that high
training accuracy does not invalidate the control because a
high-dimensional probe may memorize random labels. The important
quantity is held-out test accuracy.
```

## Interpretation

If the real probe significantly outperforms both:

- the majority baseline
- the shuffled-label controls

then the result is less likely to be explained by a broken training or evaluation pipeline.

If shuffled-label test accuracy is consistently high, investigate:

- train/test overlap
- accidental label leakage in the text
- reuse of test labels during selection
- normalization using all splits
- examples or features being misaligned with labels

---

# Exercise 6: Compare layers and pooling methods

**Time: 25 minutes**

## Plain-English research question

> At which stage of the model is animal-versus-vehicle category information most linearly accessible, and does the result depend on how token representations are combined?

You will compare:

### Layers

1. Embedding output:
   ```python
   layer_index = 0
   ```

2. Middle layer:
   ```python
   layer_index = len(hidden_states) // 2
   ```

3. Final layer:
   ```python
   layer_index = len(hidden_states) - 1
   ```

### Pooling methods

1. Final non-padding token
2. Masked mean

That gives six candidate configurations.

## Important evaluation rule

Use **validation accuracy** to compare the six candidates.

Do not look at test accuracy while choosing:

- layer
- pooling method
- learning rate
- epoch

Then evaluate the chosen configuration on test once.

## Agent prompt

```text
Extend the experiment to compare six representation choices:

Layers:
- embedding output,
- middle hidden-state layer,
- final hidden-state layer.

Pooling:
- final non-padding token,
- masked mean.

For every candidate:
- extract train and validation representations,
- standardize using that candidate's training representations only,
- initialize a fresh linear probe with the same seed,
- use the same optimizer settings,
- select the candidate's epoch using validation data,
- report train and validation accuracy.

Create a table containing:
- layer name,
- numerical layer index,
- pooling method,
- train accuracy,
- validation accuracy,
- selected epoch.

Select the candidate with the highest validation accuracy.
Only after selection, extract or evaluate its test representations
and report final test accuracy.

Do not select the representation using test results.
```

## What to consider when interpreting the layer comparison

### If the embedding layer performs well

The probe may be exploiting category information already present in token embeddings. The result may rely heavily on noun identity.

### If middle or final layers perform better

Contextual processing may have made the category more linearly accessible.

### If the middle layer beats the final layer

That is not necessarily an error. Information can become:

- more linearly accessible
- less linearly accessible
- distributed differently

as it moves through the model.

### If masked mean beats final-token pooling

The useful information may be distributed across several token positions rather than concentrated in the final token.

None of these comparisons proves that a layer causally uses the representation.

---

# Exercise 7: Fifteen-minute bug hunt

## Context

Suppose your coding agent produced the following implementation.

Its intended goal is:

- keep the language model frozen
- extract fair sequence representations
- train only a linear probe
- report a valid held-out test result

Find at least eight problems.

```python
model.train()

outputs = model(
    **batch,
    output_hidden_states=True,
)

hidden = outputs.hidden_states[-1]
features = hidden[:, -1, :]

all_features = torch.cat([
    train_features,
    val_features,
    test_features,
])

mean = all_features.mean(dim=0)
std = all_features.std(dim=0)

train_features = (train_features - mean) / std
val_features = (val_features - mean) / std
test_features = (test_features - mean) / std

probe = torch.nn.Linear(
    hidden.shape[-1],
    2,
)

optimizer = torch.optim.AdamW(
    list(model.parameters()) +
    list(probe.parameters()),
)

for epoch in range(200):
    logits = probe(train_features)
    loss = loss_fn(logits, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    test_accuracy = accuracy(
        probe(test_features),
        test_labels,
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_state = probe.state_dict()
```

## Answer key

1. `model.train()` enables training behavior such as dropout.
2. The model parameters were not explicitly frozen.
3. Hidden-state extraction is not inside `inference_mode()` or `no_grad()`.
4. `hidden[:, -1, :]` may select padding.
5. Features were not explicitly detached from the model graph.
6. Normalization statistics use validation and test features.
7. The standard deviation is not protected against zero values.
8. The optimizer includes the language-model parameters.
9. Test accuracy is calculated every epoch.
10. Test performance is used to select the best probe state.
11. There is no validation-based model selection.
12. There is no majority baseline.
13. There is no shuffled-label control.
14. No random seed is set.
15. There is no verification that labels and features remain aligned.
16. There is no verification of train/test noun separation.
17. `state_dict()` should be deep-copied or its tensors cloned when saving the best state.

After making your list, ask:

```text
Review this probe implementation for:
- model freezing,
- padding,
- graph retention,
- preprocessing leakage,
- optimizer scope,
- test-set leakage,
- reproducibility,
- and missing baselines.

For every issue, identify the exact line, explain the failure mode,
and give the smallest correction. Do not rewrite the complete program.
```

---

# Optional Exercise 8: Thirty-minute transfer simulation

This tests whether you can apply the same machinery to a new question rather than merely repeating the animal/vehicle exercise.

## Interview-style task statement

> A collaborator suspects that the middle-layer representation of a frozen language model contains linearly decodable information about whether a passage includes numerical content. Using WikiText validation examples, construct a balanced dataset in which the positive class contains at least one numerical digit and the negative class contains no digits. Train a linear probe on middle-layer, final-non-padding-token representations. Evaluate it against a majority baseline and a shuffled-training-label control. Report potential sequence-length and topic confounds.

## More explicit translation

Your label function is:

```python
label = int(
    any(character.isdigit() for character in text)
)
```

You need:

- 80 digit-containing examples
- 80 examples without digits
- no empty examples
- a reproducible train/validation/test split
- balanced classes in each split
- middle-layer hidden states
- final non-padding pooling
- a linear probe
- majority and shuffled-label baselines
- mean token count by class

## Suggested time allocation

- **0–3 minutes:** Restate the question and identify confounds.
- **3–6:** Define split, representation, metric, and baselines.
- **6–18:** Ask the agent for the minimal implementation.
- **18–23:** Run and debug.
- **23–27:** Audit masking, split, and label alignment.
- **27–30:** Explain result and limitations.

## Good final interpretation

> “The probe performed above the majority and shuffled-label baselines, so numerical-content information is linearly decodable from the selected middle-layer representation. However, this is an easy lexical property because digits are directly visible in the input. Numerical passages may also differ in length or topic, so the result does not establish an abstract representation of numerical reasoning.”

---

# Questions you should answer aloud afterward

## 1. What does a linear probe test?

Whether a property is linearly decodable from a selected representation under a particular dataset and split.

## 2. Why freeze the base model?

So the experiment measures information already present in its representations rather than teaching the base model the classification task.

## 3. Why extract representations once?

It is faster, prevents accidental base-model updates, and separates representation extraction from probe training.

## 4. Why use a held-out noun split?

A random example split could put sentences containing `cat` in both train and test. The probe might memorize individual noun identities instead of learning a category boundary that generalizes.

## 5. Why not use `hidden[:, -1, :]`?

For padded batches, `-1` may identify a padding position rather than the final real token.

## 6. What is the difference between final-token and mean pooling?

- Final-token pooling uses one token representation that has seen the earlier causal context.
- Mean pooling averages all real token representations.
- They impose different assumptions about where sequence information is represented.

## 7. Why is high training accuracy insufficient?

A high-dimensional linear probe can memorize a small training set. Held-out validation and test performance are what matter.

## 8. Why add a shuffled-label baseline?

It checks whether the pipeline can appear successful even when training labels have no real relationship with the representations.

## 9. Why calculate normalization statistics using only training data?

Using validation or test statistics incorporates evaluation-distribution information into the training pipeline.

## 10. Why choose the layer using validation rather than test?

The layer is a hyperparameter. Selecting it on test data would overfit your experiment to the test set.

## 11. Does strong probe accuracy prove causal use?

No. It shows decodability, not that the model depends on that information for its behavior.

## 12. Does chance probe accuracy prove the information is absent?

No. The information could be:

- nonlinearly encoded
- present at another layer
- distributed across token positions
- obscured by poor pooling
- inaccessible to the probe with the available data

---

# Useful interview narration

Practice saying these sentences naturally:

> “This experiment tests linear decodability, not causality.”

> “I am splitting by noun rather than randomly by sentence to reduce lexical memorization.”

> “The language model is frozen; only the linear probe receives gradients.”

> “I compute the final real token index from the attention mask rather than assuming the final tensor position is valid.”

> “I use training statistics only when normalizing the representations.”

> “I select the layer and pooling method on validation data, then evaluate the test set once.”

> “The shuffled-label control is intended to detect leakage or a broken evaluation pipeline.”

> “Poor probe performance would not prove the representation is absent; it may not be linearly accessible under this pooling method.”

---

# Definition of done

You have completed the Wednesday block when you can:

- extract every hidden-state layer
- explain the extra embedding entry in the hidden-state tuple
- convert `[B, T, H]` into `[B, H]` using two pooling methods
- correctly handle padding
- create noun-disjoint train and test sets
- train only an `nn.Linear` probe
- prove that the base-model parameters have no gradients
- report train, validation, and test accuracy separately
- calculate a majority baseline
- implement a shuffled-label control
- select layer and pooling on validation rather than test
- explain why probe accuracy demonstrates decodability, not causality
- complete the central experiment in approximately 30–40 minutes with your coding agent assisting rather than making the research decisions for you