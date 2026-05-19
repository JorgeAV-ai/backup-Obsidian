---
tags:
  - basics
  - optimization
---

**TL;DR:** Adam combines momentum and adaptive learning rates with bias correction. AdamW fixes weight decay. StableAdamW adds update clipping. Default go-to optimizer for deep learning.

## What is it?

**Adaptive Moment Estimation (Adam)** is an optimizer introduced by Kingma & Ba (2014) that maintains per-parameter adaptive learning rates by tracking exponential moving averages of both the gradient (first moment, like momentum) and the squared gradient (second moment, like RMSProp). Bias correction compensates for the zero-initialization of these estimates.

**AdamW** (Loshchilov & Hutter, 2017) decouples weight decay from the gradient-based update. In vanilla Adam, L2 regularization interacts poorly with the adaptive learning rate: the per-parameter scaling effectively applies different regularization strengths to different parameters. AdamW fixes this by applying weight decay directly to the weights, outside the adaptive step.

**StableAdamW** (used in ModernBERT) adds Adafactor-style update clipping, bounding the RMS of the parameter update to prevent training instabilities at scale.

## How it works

![[basics_adam.png]]

[🔗 Open interactive Adam Optimizer Visualizer](../interactive/adam.html)

### Default hyperparameters

| Param | Default | Meaning |
|---|---|---|
| $\beta_1$ | 0.9 | Decay rate for first moment (momentum) |
| $\beta_2$ | 0.999 | Decay rate for second moment (variance) |
| $\epsilon$ | $10^{-8}$ | Numerical stability constant |
| $\alpha$ | — | Learning rate (set by scheduler) |
| $\lambda$ | — | Weight decay coefficient (AdamW only) |

### Formulas

**First moment (mean of gradients):**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment (mean of squared gradients):**
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Pseudocode: SGD -> Adam -> AdamW progression

```
# --- SGD with Momentum ---
for each parameter θ:
    g = compute_gradient(θ)
    v = β * v + g                    # momentum
    θ = θ - α * v

# --- Adam ---
for each parameter θ:
    g = compute_gradient(θ)
    m = β1 * m + (1 - β1) * g       # first moment
    v = β2 * v + (1 - β2) * g²      # second moment
    m_hat = m / (1 - β1^t)          # bias correction
    v_hat = v / (1 - β2^t)          # bias correction
    θ = θ - α * m_hat / (√v_hat + ε)

# --- AdamW (decoupled weight decay) ---
for each parameter θ:
    g = compute_gradient(θ)
    m = β1 * m + (1 - β1) * g
    v = β2 * v + (1 - β2) * g²
    m_hat = m / (1 - β1^t)
    v_hat = v / (1 - β2^t)
    θ = θ - α * m_hat / (√v_hat + ε)
    θ = θ * (1 - λ)                 # weight decay applied SEPARATELY
```

The key difference: in Adam with L2 regularization, the gradient `g` already contains the regularization term, so it gets divided by `√v_hat` — meaning parameters with large adaptive rates get less regularization. AdamW applies `(1 - λ)` directly to the weights, keeping regularization uniform.

### StableAdamW

StableAdamW clips the update by its RMS:

```
update = m_hat / (√v_hat + ε)
rms = RMS(update)
if rms > clip_threshold:
    update = update * (clip_threshold / rms)
θ = θ - α * update
θ = θ * (1 - λ)
```

This prevents catastrophic training spikes, especially in long training runs (e.g., ModernBERT training over 2T tokens).

## Why it matters

- **Adam** converges faster than SGD on most tasks because it adapts per-parameter learning rates
- **AdamW** is the standard optimizer for transformers — correct weight decay prevents overfitting without interfering with the adaptive mechanism
- **StableAdamW** enables stable long training runs at scale without loss spikes

## Used in

- Virtually all transformer-based papers in this vault: ModernBERT, FasterViT, Mixtral, etc.
- AdamW is the default in PyTorch's `torch.optim.AdamW` and HuggingFace Trainer

---

See also: [[Backpropagation]], [[Learning Rate Scheduling]]
