GELU is a smooth, non-monotonic activation function that multiplies the input by the probability of it being positive under a standard normal distribution, and has become the default activation in Transformer architectures.

## What is it?

The Gaussian Error Linear Unit (GELU), introduced by Hendrycks and Gimpel in 2016, takes an elegantly probabilistic approach to activation functions. The core intuition: instead of a hard threshold (like ReLU's "zero if negative, pass if positive"), GELU asks "what is the probability that this input is greater than other inputs drawn from a standard normal?" and scales the input by that probability. Inputs that are large and positive get passed through almost unchanged (probability near 1). Inputs that are large and negative get squashed to near zero (probability near 0). Inputs near zero get a smooth, probabilistic blend.

Concretely, GELU multiplies the input $x$ by $\Phi(x)$, the CDF of the standard normal distribution. This makes GELU smooth everywhere (infinitely differentiable), unlike ReLU which has a sharp kink at zero. It is also **non-monotonic**: GELU has a slight dip for negative values before approaching zero. Around $x \approx -0.17$, the function reaches a minimum of approximately $-0.17 \cdot \Phi(-0.17) \approx -0.048$. This non-monotonicity means GELU can output small negative values, which is thought to help with optimization by providing richer gradient information in the negative regime.

Why does GELU dominate in Transformers? The smoothness is key. Transformers process sequences where attention weights create continuous, differentiable flow of information. A smooth activation function aligns better with this continuous optimization landscape. Empirically, GELU consistently outperforms ReLU in Transformer-based models (BERT, GPT, ViT), and it has become the de facto standard. The improvement is modest but consistent -- typically a few tenths of a percent on benchmarks, but it compounds across the many feed-forward layers in a deep Transformer.

## How it works

![[basics_gelu.png]]

[🔗 Open interactive Activations Explorer](../../interactive/activations.html)

### Exact formula

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of the standard normal distribution:

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

So the full expanded form is:

$$\text{GELU}(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

### Derivative

$$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the PDF of the standard normal. Note: the gradient is always well-defined and smooth. For large positive $x$, the gradient approaches 1 (like ReLU). For large negative $x$, it approaches 0 (like ReLU). But around zero, it transitions smoothly.

### Tanh approximation

Computing the exact $\text{erf}$ can be expensive on some hardware. The tanh approximation is very accurate and widely used:

$$\text{GELU}(x) \approx 0.5\,x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715\,x^3\right)\right)\right)$$

This approximation has a maximum error of about $0.004$ across all $x$.

### Sigmoid approximation

An even simpler approximation uses the sigmoid function $\sigma$:

$$\text{GELU}(x) \approx x \cdot \sigma(1.702\,x)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$. This is faster to compute and the error is small (max $\approx 0.02$), though less accurate than the tanh version. The constant $1.702$ was found by fitting.

### Pseudocode for all three variants

```python
import math

# ============================================================
# Variant 1: Exact GELU (using erf)
# ============================================================
def gelu_exact(x):
    """
    Exact GELU using the error function.
    Most accurate, but erf() may be slow on some hardware.
    """
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# Variant 2: Tanh approximation
# ============================================================
def gelu_tanh(x):
    """
    GELU approximated with tanh.
    Used in GPT-2, BERT, and most Transformer implementations.
    Max error ~0.004 vs exact.
    """
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    ))


# ============================================================
# Variant 3: Sigmoid approximation
# ============================================================
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def gelu_sigmoid(x):
    """
    GELU approximated with sigmoid.
    Fastest, slightly less accurate (max error ~0.02).
    """
    return x * sigmoid(1.702 * x)
```

```python
# Vectorized PyTorch implementations
import torch
import torch.nn.functional as F

# Built-in exact GELU (uses erf internally)
y = F.gelu(x)

# Tanh approximation (explicit)
def gelu_tanh_torch(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

# Sigmoid approximation (explicit)
def gelu_sigmoid_torch(x):
    return x * torch.sigmoid(1.702 * x)

# PyTorch >= 1.12 also supports:
y = F.gelu(x, approximate='tanh')   # tanh approximation
y = F.gelu(x, approximate='none')   # exact (default)
```

### Behavior summary table

| $x$ | $\Phi(x)$ | $\text{GELU}(x)$ | Gradient |
|---|---|---|---|
| $-3$ | $0.0013$ | $-0.004$ | $\approx 0.0$ |
| $-1$ | $0.1587$ | $-0.159$ | $-0.083$ |
| $0$ | $0.5$ | $0.0$ | $0.5$ |
| $1$ | $0.8413$ | $0.841$ | $1.083$ |
| $3$ | $0.9987$ | $2.996$ | $\approx 1.0$ |

Note that GELU(0) = 0 and the gradient at 0 is 0.5 (contrast with ReLU's undefined gradient at 0).

## Why it matters

GELU was introduced to bridge the gap between two extremes:
- **ReLU**: hard thresholding. Simple, fast, but the discontinuous derivative at zero and the dead neuron problem limit performance in architectures with many layers and residual connections.
- **Smooth alternatives** (ELU, Softplus): smooth but lack the stochastic/probabilistic motivation.

GELU's probabilistic framing -- "weight the input by how likely it is to be positive" -- gives it a natural interpretation as a **smooth, stochastic gate**. In the limit, as the input distribution becomes very peaked, GELU converges to ReLU. So ReLU is a special case of GELU.

What came before GELU in Transformers? The original Transformer (Vaswani et al., 2017) used ReLU. BERT (Devlin et al., 2018) switched to GELU and it stuck. GPT-2, GPT-3, ViT, and essentially all modern Transformers use GELU (or its close relative SwiGLU / GeGLU in newer models like LLaMA and PaLM).

The key properties that make GELU dominant in Transformers:
1. **Smoothness**: better optimization landscape for attention-based architectures
2. **Non-zero gradient for negative inputs**: unlike ReLU, slightly negative inputs still contribute gradient signal
3. **Non-monotonicity**: the slight dip for negative values provides richer gradient information
4. **Empirical superiority**: consistent improvements over ReLU on language and vision Transformer benchmarks

## Used in

- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
