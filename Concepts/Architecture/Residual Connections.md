# Residual Connections

> **TL;DR:** Residual (skip) connections add the input of a block directly to its output: $y = F(x) + x$. This lets gradients flow unimpeded through deep networks and means each layer only needs to learn the *residual* --- the difference from identity --- rather than the full transformation.

---

## What is it?

A residual connection (also called a skip connection or shortcut connection) bypasses a block of computation by adding the block's input directly to its output. Introduced in **ResNet** (He et al., 2015), this simple idea solved the degradation problem in very deep networks and was later adopted as a core component of every Transformer.

### Core intuition

> "The network only needs to learn the DIFFERENCE from identity, not the full transformation."

Without a skip connection, a layer must learn the entire desired mapping $H(x)$. With one, it only needs to learn the residual $F(x) = H(x) - x$, and the output becomes $F(x) + x$. If the optimal transformation is close to identity (common in deep networks), $F(x) \approx 0$ is much easier to learn than $H(x) \approx x$.

---

## How it works

![[basics_residual.png]]

[🔗 Open interactive Residual Connections Demo](../../interactive/residual.html)

### Formula

$$y = F(x) + x$$

where $F$ is the sublayer (self-attention, feed-forward network, convolution block, etc.) and $x$ is the input to that sublayer.

### Why it solves vanishing gradients

During backpropagation, the gradient of the loss with respect to $x$ is:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F(x)}{\partial x} + I\right)$$

The $+I$ (identity) term means the gradient always has a direct path back through the skip connection, regardless of what $F$ does. Even if $\frac{\partial F}{\partial x}$ vanishes, the gradient still flows through the identity branch.

### Pseudocode

```python
# Basic residual block
def residual_block(x):
    return F(x) + x
```

```python
# Post-Norm residual (original Transformer, Vaswani et al., 2017)
def post_norm_residual(x):
    out = sublayer(x)          # e.g., self-attention or FFN
    out = x + out              # residual addition
    out = layer_norm(out)      # normalize AFTER addition
    return out
```

```python
# Pre-Norm residual (GPT-2, LLaMA, modern standard)
def pre_norm_residual(x):
    out = layer_norm(x)        # normalize BEFORE sublayer
    out = sublayer(out)        # e.g., self-attention or FFN
    out = x + out              # residual addition
    return out
```

```python
# Full Transformer layer with both sublayers (pre-norm)
def transformer_layer(x):
    # Attention block
    x = x + self_attention(layer_norm(x))
    # Feed-forward block
    x = x + feed_forward(layer_norm(x))
    return x
```

---

## Why it matters

- **Enables depth**: Without residual connections, networks deeper than ~20 layers suffered from degradation (not just vanishing gradients, but actually higher training error). ResNet showed that 152-layer networks could train successfully. Modern LLMs go to 80+ Transformer layers.
- **Gradient highway**: The skip connection provides an unimpeded path for gradients during backpropagation, making optimization practical for deep architectures.
- **Easier optimization landscape**: The loss surface becomes smoother when residual connections are present, leading to faster convergence.
- **Composability**: Each residual block can be seen as making a small, incremental refinement to the representation. This "iterative refinement" view is central to how Transformers process information.

### Historical note

He et al. (2015) won the ImageNet competition with a 152-layer ResNet. Before this, going beyond ~20 layers actually *hurt* performance. The key insight was not just about gradients but about the optimization landscape: it is easier to learn a small perturbation $F(x)$ than a full mapping $H(x)$.

---

## Used in

All deep network papers, especially:
- **ResNet** (He et al., 2015) --- introduced skip connections for CNNs
- **Original Transformer** (Vaswani et al., 2017) --- residual around attention and FFN sublayers
- **GPT / GPT-2 / GPT-3** --- residual connections in every layer
- **BERT, ViT, LLaMA, PaLM, Gemini** --- all use residual connections
- **U-Net** (Ronneberger et al., 2015) --- skip connections between encoder and decoder (slightly different: concatenation instead of addition)

---

**See also:** [[Layer Normalization]], [[Transformer]]
