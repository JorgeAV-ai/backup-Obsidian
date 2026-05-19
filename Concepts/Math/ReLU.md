ReLU is the default activation function in deep learning: it outputs zero for negative inputs and passes positive inputs unchanged, solving the vanishing gradient problem that plagued sigmoid/tanh networks.

## What is it?

The Rectified Linear Unit (ReLU) is almost absurdly simple: if the input is positive, pass it through; if it's negative, output zero. That's it. Yet this trivial function was one of the most important practical breakthroughs in deep learning. Before ReLU became standard (popularized around 2010-2012, notably by Nair & Hinton and the AlexNet paper), networks used sigmoid or tanh activations. These saturating functions crush large inputs into a flat region where the gradient is nearly zero, making it extremely difficult for gradients to propagate through many layers -- the infamous **vanishing gradient problem**.

ReLU sidesteps this entirely. For any positive input, the gradient is exactly 1. Gradients flow backward through ReLU layers without shrinking, which is why deep networks suddenly became trainable when the community switched to ReLU. It is also computationally trivial -- just a comparison and a branch, no exponentials or divisions.

The tradeoff is the **dead neuron problem**. If a neuron's pre-activation becomes negative for all inputs in the training set (e.g., due to a large negative bias update), the gradient through that neuron is permanently zero. It will never recover. In practice, this means a fraction of neurons in a ReLU network can "die" during training, contributing nothing. This is usually manageable (and partly why we over-parameterize), but it motivated variants like Leaky ReLU and parametric ReLU.

## How it works

![[basics_relu.png]]

[🔗 Open interactive Activations Explorer](../interactive/activations.html)

### Formula

$$\text{ReLU}(x) = \max(0, x)$$

Equivalently:

$$\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Derivative

$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

In practice, the subgradient at $x = 0$ is set to 0 (or sometimes 1 -- it doesn't matter in floating point since $x = 0$ exactly is a measure-zero event).

### Pseudocode

```python
def relu_forward(x):
    """Forward pass for ReLU activation."""
    return max(0, x)  # element-wise

def relu_backward(x, grad_output):
    """Backward pass for ReLU activation.

    x: the original input (cached from forward pass)
    grad_output: gradient flowing from the layer above
    """
    grad_input = grad_output.clone()
    grad_input[x <= 0] = 0  # kill gradient where input was non-positive
    return grad_input
```

```python
# Efficient vectorized implementation (NumPy)
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x, grad_output):
    return grad_output * (x > 0).astype(float)
```

### Properties at a glance

- **Non-saturating** for positive inputs (gradient = 1, always)
- **Sparse activation**: roughly 50% of neurons output zero for random inputs, giving a form of natural sparsity
- **Scale-equivariant**: $\text{ReLU}(\alpha x) = \alpha \cdot \text{ReLU}(x)$ for $\alpha > 0$
- **Not zero-centered**: outputs are always $\geq 0$, which can cause zig-zagging gradient updates (one reason BatchNorm helps)

## Why it matters

ReLU's introduction (or more precisely, its adoption as the default) was a watershed moment. Here is what came before and after:

**Before ReLU -- Sigmoid and Tanh era:**
- $\sigma(x) = \frac{1}{1+e^{-x}}$ outputs in $(0, 1)$. Gradient: $\sigma(x)(1 - \sigma(x))$, max value 0.25 at $x=0$. Multiplying many of these together during backprop through 10+ layers and gradients vanish exponentially.
- $\tanh(x)$ outputs in $(-1, 1)$. Better than sigmoid (zero-centered, max gradient 1.0), but still saturates for large $|x|$.

**ReLU solved vanishing gradients** but introduced dead neurons. This led to a family of variants:

| Activation | Formula | Advantage over ReLU |
|---|---|---|
| **Leaky ReLU** | $\max(0.01x, x)$ | Small gradient for $x < 0$ prevents dead neurons |
| **Parametric ReLU (PReLU)** | $\max(\alpha x, x)$, $\alpha$ learned | Learns the optimal negative slope |
| **ELU** | $x$ if $x>0$, $\alpha(e^x - 1)$ if $x \leq 0$ | Smooth, pushes mean activation toward zero |
| **GELU** | $x \cdot \Phi(x)$ | Smooth, probabilistic gating, dominant in Transformers |

Despite these variants, plain ReLU remains the default for CNNs and many architectures due to its simplicity, speed, and the fact that it just works. The variants shine in specific contexts: GELU in Transformers, ELU when you need smoother gradients, Leaky ReLU when dead neurons are a measurable problem.

## Used in

- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
