GeGLU is a gated activation function that multiplies a GELU-activated projection with a linear projection of the input, producing a more expressive feed-forward layer than standard GELU alone.

## What is it?

GeGLU (Gated GELU Linear Unit) is a variant of the Gated Linear Unit (GLU) family of activations, introduced by Shazeer (2020) in the paper "GLU Variants Improve Transformer." In a standard transformer FFN, the input is projected up to a larger hidden dimension, passed through an activation function, and projected back down. GeGLU replaces the single activation step with a *gating* mechanism: it splits the upward projection into two parallel linear transformations --- one passed through [[GELU]], the other kept linear --- and multiplies them element-wise. The gated branch controls how much information flows through.

The idea behind gating is that one branch learns *what* information to represent (the value branch) while the other learns *whether* to let it through (the gate branch). This multiplicative interaction gives the network a richer, more flexible nonlinearity than applying a single activation function. Empirically, gated variants consistently outperform their ungated counterparts across language modeling benchmarks, with GeGLU and SwiGLU being the most popular choices in modern architectures.

A practical trade-off: because GeGLU uses two separate projection matrices for the upward pass (instead of one), matching the same parameter count as a standard FFN requires reducing the intermediate hidden dimension by a factor of $2/3$. Despite the smaller hidden size, gated variants still outperform standard FFNs at the same parameter budget.

## How it works

![[basics_geglu.png]]

**Formula.** Given input $x \in \mathbb{R}^{d}$:

$$\text{GeGLU}(x) = \text{GELU}(x W_1 + b_1) \odot (x W_2 + b_2)$$

where:
- $W_1, W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$ are the gate and value projection matrices
- $\odot$ denotes element-wise (Hadamard) multiplication
- $\text{GELU}(z) = z \cdot \Phi(z)$, where $\Phi$ is the standard normal CDF

**Full GeGLU FFN block:**

$$\text{FFN}_{\text{GeGLU}}(x) = \bigl[\text{GELU}(x W_1) \odot (x W_2)\bigr]\, W_3$$

where $W_3 \in \mathbb{R}^{d_{\text{ff}} \times d}$ is the down-projection.

**The GLU family.** The general form of a GLU variant is:

$$\text{GLU}_\sigma(x) = \sigma(x W_1) \odot (x W_2)$$

| Variant | Activation $\sigma$ | Notes |
|---|---|---|
| GLU (Dauphin et al., 2017) | Sigmoid | Original formulation |
| ReGLU | ReLU | Simple but effective |
| **GeGLU** | GELU | Smooth gating, widely used |
| SwiGLU | Swish / SiLU | Used in LLaMA, PaLM |

**Standard FFN vs GeGLU FFN:**

| | Standard FFN | GeGLU FFN |
|---|---|---|
| Up-projection | 1 matrix: $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ | 2 matrices: $W_1, W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$ |
| Activation | $\text{GELU}(x W_{\text{up}})$ | $\text{GELU}(x W_1) \odot (x W_2)$ |
| Down-projection | $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ | $W_3 \in \mathbb{R}^{d_{\text{ff}} \times d}$ |
| Total weight matrices | 2 | 3 |
| Typical $d_{\text{ff}}$ | $4d$ | $\frac{8}{3}d$ (to match param count) |

**Pseudocode for a GeGLU FFN block:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Gate and value projections (can be fused into one matmul)
        self.W_gate  = nn.Linear(d_model, d_ff, bias=False)
        self.W_value = nn.Linear(d_model, d_ff, bias=False)
        # Down-projection
        self.W_out   = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.gelu(self.W_gate(x))   # GELU-activated gate
        value = self.W_value(x)          # Linear value branch
        hidden = gate * value            # Element-wise gating
        return self.W_out(hidden)        # Project back to d_model

# Typical usage: d_model=768, d_ff=2048 (≈ 8/3 * 768)
# ffn = GeGLU_FFN(d_model=768, d_ff=2048)
```

## Why it matters

Gated activations like GeGLU provide a strictly more expressive nonlinearity than standard activations at the same parameter budget. The gating mechanism allows the network to learn input-dependent feature selection inside the FFN, which translates to better perplexity and downstream task performance. GeGLU and its sibling SwiGLU have become the default FFN activation in modern transformer architectures, replacing the standard ReLU or GELU FFN used in earlier models like BERT and GPT-2.

## Used in

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]
