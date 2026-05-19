# Softmax

> **TL;DR:** Softmax converts a vector of arbitrary real numbers (logits) into a valid probability distribution --- all values in $(0, 1)$ and summing to $1$. It appears everywhere: attention weights, classification heads, contrastive losses.

---

## What is it?

Softmax is an activation function that takes a vector $x \in \mathbb{R}^n$ and produces a vector of the same size where each element is positive and the elements sum to one. It is the standard way to turn raw model outputs (logits) into probabilities.

It is a "soft" version of argmax: instead of picking the single largest element, it assigns the most probability mass to the largest element while still giving some mass to the others.

---

## How it works

![[basics_softmax.png]]

[🔗 Open interactive Softmax Visualizer](../../interactive/softmax.html)

### Formula

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

### Numerical stability trick

Naively computing $e^{x_i}$ can overflow when $x_i$ is large. The standard fix: subtract the maximum value before exponentiating. This does not change the result mathematically:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

### Temperature scaling

A temperature parameter $\tau$ controls the "sharpness" of the distribution:

$$\text{softmax}(x_i / \tau)$$

- **Low $\tau$ (e.g., 0.1):** Distribution becomes peaky --- almost all probability on the largest logit. Approaches argmax as $\tau \to 0$.
- **High $\tau$ (e.g., 10):** Distribution becomes uniform --- all options roughly equally likely.
- **$\tau = 1$:** Standard softmax.

Temperature is used in generation sampling, knowledge distillation, and contrastive losses (e.g., CLIP uses a learned temperature).

### Pseudocode

```python
def softmax(x, temperature=1.0):
    # x: vector of logits [n]
    x = x / temperature            # temperature scaling
    x = x - max(x)                 # numerical stability
    exp_x = exp(x)                 # element-wise exponential
    return exp_x / sum(exp_x)      # normalize to probabilities
```

```python
# In attention (simplified)
def attention_weights(Q, K, d_k):
    scores = Q @ K.T / sqrt(d_k)   # raw attention scores
    weights = softmax(scores)       # convert to probabilities
    return weights
```

---

## Why it matters

- **Probabilistic interpretation**: Softmax gives model outputs a clean probabilistic meaning, enabling cross-entropy loss, sampling, and calibration.
- **Differentiable**: Unlike argmax, softmax is smooth and differentiable, so gradients flow through it during training.
- **Attention mechanism**: In [[Self-Attention]], softmax converts raw dot-product scores into attention weights that sum to 1, determining how much each token attends to every other token.
- **Temperature as a control knob**: Temperature scaling is a simple but powerful way to control the exploration-exploitation tradeoff in generation without retraining.

---

## Used in

Virtually all neural network papers:
- **Attention mechanisms** --- softmax over query-key dot products ([[Self-Attention]], [[Flash Attention]])
- **Classification outputs** --- softmax over class logits (ResNet, BERT, ViT, etc.)
- **Contrastive learning** --- softmax over similarity scores ([[Contrastive Learning]], CLIP, SimCLR)
- **Autoregressive generation** --- softmax over next-token logits ([[Autoregressive Generation]])
- **Knowledge distillation** --- softmax with high temperature to produce soft targets (Hinton et al., 2015)

---

**See also:** [[Self-Attention]], [[Contrastive Learning]], [[Autoregressive Generation]]
