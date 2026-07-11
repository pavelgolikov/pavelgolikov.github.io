# Mathematical Breakdown of Transformer Masks

This document details exactly how and when the three primary masks are applied in causal language modeling.

## 1. Causal Mask

**When:** During the forward pass, inside every self-attention layer.  
**What it is applied to:** The raw, unnormalized attention scores (dot products of Queries and Keys) before the `softmax` is applied.

**The Math:**  
Let $S \in \mathbb{R}^{T \times T}$ be the matrix of unnormalized attention scores for a sequence of length $T$, where $S_{i,j} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$.  

The causal mask $M^{\text{causal}} \in \mathbb{R}^{T \times T}$ is defined as a matrix of zeros and negative infinities:

$$
M^{\text{causal}}_{i,j} = \begin{cases} 
0 & \text{if } i \geq j \\ 
-\infty & \text{if } i < j 
\end{cases}
$$

This mask is added to $S$:

$$
S'_{i,j} = S_{i,j} + M^{\text{causal}}_{i,j}
$$

Then, the attention weights $A$ are computed by applying the `softmax` over the columns (dimension $j$):

$$
A_{i,j} = \text{softmax}(S'_i)_j = \frac{\exp(S'_{i,j})}{\sum_{k=1}^T \exp(S'_{i,k})}
$$

Because $\exp(-\infty) = 0$, the attention weight $A_{i,j}$ becomes exactly $0$ for any future token ($j > i$). 

---

## 2. Attention/Padding Mask

**When:** During the forward pass, simultaneously with the causal mask.  
**What it is applied to:** The pre-softmax attention scores ($S$).

**The Math:**  
Suppose a sequence has $L$ valid tokens but is padded to length $T$ ($T > L$) to fit into a batch. 

The padding mask $M^{\text{pad}} \in \mathbb{R}^T$ is defined as:

$$
m^{\text{pad}}_j = \begin{cases} 
0 & \text{if } j \leq L \text{ (valid token)} \\ 
-\infty & \text{if } j > L \text{ (padding token)} 
\end{cases}
$$

We broadcast this to a $T \times T$ matrix $M^{\text{pad}}$ where every row is identical ($M^{\text{pad}}_{i,j} = m^{\text{pad}}_j$). 

In practice, both the causal mask and the padding mask are added together before the `softmax`:

$$
S'_{i,j} = S_{i,j} + M^{\text{causal}}_{i,j} + M^{\text{pad}}_{i,j}
$$

$$
A_{i,j} = \text{softmax}(S'_i)_j
$$

This ensures that $\exp(S'_{i,j}) = 0$ whenever token $j$ is a padding token, preventing the model from mixing padding into its internal representations.

---

## 3. Loss Mask

**When:** During the backward pass computation (technically at the very end of the forward pass, when computing the final objective).  
**What it is applied to:** The individual token-level cross-entropy loss values.

**The Math:**  
Let $z_t \in \mathbb{R}^V$ be the model's predicted logits at sequence position $t$, and let $y_t$ be the true target token class.  

The unmasked cross-entropy loss at position $t$ is computed in a few logical steps:

**Step 1: Convert raw scores to a probability**
The model outputs raw, unnormalized scores called **logits** ($z_t$) for every possible word in the vocabulary $V$. 
To turn these raw scores into valid percentages (probabilities), we apply the **Softmax** function. The probability the model assigns to the *correct* target token $y_t$ is:

$$
P(\text{correct token}) = \frac{\exp((z_t)_{y_t})}{\sum_{k=1}^V \exp((z_t)_k)}
$$

*(Why exponents? The $\exp()$ function turns any raw logit into a positive number. Dividing by the sum of all exponents ensures all the probabilities add up to exactly 1.0).*

**Step 2: Calculate the Loss**
The cross-entropy loss for a single token simply penalizes the model based on the probability it assigned to the *correct* answer. The formula you can easily understand is:

$$
\ell_t = -\log( P(\text{correct token}) )
$$

**Step 3: The Combined Formula**
If we substitute the math from Step 1 into the formula from Step 2, we get the full "gibberish" combined expression that you originally saw:

$$
\ell_t = -\log \left( \frac{\exp((z_t)_{y_t})}{\sum_{k=1}^V \exp((z_t)_k)} \right)
$$

The loss mask $M^{\text{loss}} \in \{0, 1\}^T$ is a binary vector where $M^{\text{loss}}_t = 1$ if we want to train on predicting token $y_t$, and $0$ if we want to ignore it (e.g., if token $y_t$ is a padding token or belongs to the prompt).

The total masked loss $\mathcal{L}$ for the sequence is the weighted average of the token losses:

$$
\mathcal{L} = \frac{\sum_{t=1}^T M^{\text{loss}}_t \cdot \ell_t}{\sum_{t=1}^T M^{\text{loss}}_t}
$$

By multiplying $\ell_t$ by $0$, that specific prediction contributes nothing to $\mathcal{L}$. Consequently, the gradient $\frac{\partial \mathcal{L}}{\partial z_t} = 0$, meaning the optimizer will not update any parameters to improve the prediction at position $t$.
