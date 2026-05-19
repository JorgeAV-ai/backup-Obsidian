# Layer Normalization

> **TL;DR:** Layer Normalization (LayerNorm) normalizes activations across the feature dimension for each individual sample, stabilizing training and enabling deeper networks. Unlike BatchNorm, it has no dependency on batch size and is the standard in Transformers.

---

## What is it?

Layer Normalization re-centers and re-scales the activations of a single sample by computing mean and variance across the feature (hidden) dimension. It was introduced by Ba, Kiros & Hinton (2016) as an alternative to Batch Normalization that works identically during training and inference.

### LayerNorm vs [[BatchNorm]]

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch size | Yes | No |
| Needs running stats at inference | Yes | No |
| Standard in | CNNs | Transformers |
| Behavior at train vs test | Different | Identical |

---

## How it works

![[basics_layer_norm.png]]

[🔗 Open interactive Layer Norm Visualizer](../interactive/layer_norm.html)

### Formula

For an input vector $x \in \mathbb{R}^d$ (one sample, across all features):

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ --- mean across features
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ --- variance across features
- $\gamma, \beta \in \mathbb{R}^d$ --- learned scale and shift (affine parameters)
- $\epsilon$ --- small constant for numerical stability (e.g., $10^{-5}$)

### Pre-Norm vs Post-Norm

The original Transformer (Vaswani et al., 2017) used **Post-Norm**: normalize after the residual addition. Most modern architectures use **Pre-Norm**: normalize before the sublayer. Pre-Norm is more stable during training and allows training deeper models without careful learning rate warmup.

### Pseudocode

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    # x: [batch, seq_len, d_model]
    mu    = mean(x, axis=-1, keepdims=True)    # mean across features
    sigma = std(x, axis=-1, keepdims=True)     # std across features
    x_hat = (x - mu) / (sigma + eps)           # normalize
    return gamma * x_hat + beta                # scale and shift
```

```python
# Post-Norm Transformer block (original Transformer)
def post_norm_block(x):
    # Attention sublayer
    attn_out = self_attention(x)
    x = layer_norm(x + attn_out)     # norm AFTER residual

    # FFN sublayer
    ffn_out = feed_forward(x)
    x = layer_norm(x + ffn_out)      # norm AFTER residual
    return x
```

```python
# Pre-Norm Transformer block (GPT-2, LLaMA, modern standard)
def pre_norm_block(x):
    # Attention sublayer
    attn_out = self_attention(layer_norm(x))   # norm BEFORE sublayer
    x = x + attn_out                           # residual

    # FFN sublayer
    ffn_out = feed_forward(layer_norm(x))      # norm BEFORE sublayer
    x = x + ffn_out                            # residual
    return x
```

---

## Why it matters

- **Training stability**: Normalizing activations prevents them from growing or shrinking uncontrollably across layers, making optimization much smoother.
- **Batch-size independent**: Unlike BatchNorm, LayerNorm works with any batch size (including batch size 1), which is essential for autoregressive generation and variable-length sequences.
- **Enables depth**: Without normalization, training very deep Transformers (dozens or hundreds of layers) is extremely difficult. LayerNorm + [[Residual Connections]] together make depth feasible.
- **Pre-Norm is now standard**: The shift from post-norm to pre-norm was a key practical insight that improved training stability across GPT-2, GPT-3, LLaMA, and most modern LLMs.

---

## Used in

All Transformer-based papers:
- **Original Transformer** (Vaswani et al., 2017) --- post-norm
- **GPT-2, GPT-3** (Radford et al., 2019; Brown et al., 2020) --- pre-norm
- **BERT** (Devlin et al., 2019) --- post-norm
- **LLaMA** (Touvron et al., 2023) --- pre-norm with RMSNorm variant
- **Vision Transformers (ViT)** (Dosovitskiy et al., 2020) --- pre-norm

---

**See also:** [[BatchNorm]], [[Residual Connections]], [[Transformer]]
