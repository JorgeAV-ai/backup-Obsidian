# Dropout

> **TL;DR:** Randomly zero out neurons during training to prevent overfitting. At inference, use the full network. Simple, effective, and still used everywhere.

---

## What is it?

Dropout is a regularization technique introduced by Srivastava et al. (2014) that prevents neural networks from overfitting by randomly "dropping" (zeroing out) a fraction of neurons during each training step. The key insight is that this forces the network to not rely on any single neuron, building redundancy across the network.

During training, each neuron is independently set to zero with probability $p$ (the dropout rate). The surviving activations are scaled up by $\frac{1}{1-p}$ so the expected value of each activation remains the same. This is called **inverted dropout** and is the standard implementation.

At inference time, dropout is turned off entirely. The full network is used as-is, with no scaling needed (because inverted dropout already handled that during training).

---

## How it works

![[basics_dropout.png]]

[🔗 Open interactive Dropout Visualizer](../interactive/dropout.html)

### Formula

During training:

$$y = \frac{x \cdot m}{1 - p}$$

where $m \sim \text{Bernoulli}(1 - p)$ is a binary mask (each element is 0 with probability $p$, 1 with probability $1-p$), and the division by $(1-p)$ is the inverted dropout scaling.

During inference:

$$y = x$$

No mask, no scaling. The full network is used directly.

### Why the scaling?

Without scaling, a neuron that receives input from 1000 neurons during training would on average see only $1000 \times (1-p)$ active inputs. At inference, all 1000 are active, so the magnitude would be different. Inverted dropout compensates during training so that inference requires no changes.

### Typical dropout rates

| Context | Typical $p$ | Rationale |
|---|---|---|
| Transformers (attention, FFN) | 0.1 | Large models already regularized by data scale |
| Older CNNs (AlexNet, VGG) | 0.5 | Smaller datasets, heavier overfitting |
| After embeddings | 0.1--0.3 | Regularize input representations |
| Small datasets | 0.3--0.5 | More regularization needed |

### Pseudocode: dropout forward pass

```python
def dropout_forward(x, p=0.5, training=True):
    """
    x: input activations, any shape
    p: probability of dropping each element
    training: whether the model is in training mode
    """
    if not training:
        return x                          # inference: no dropout

    # Generate binary mask: 1 with prob (1-p), 0 with prob p
    mask = random_bernoulli(shape=x.shape, prob=1 - p)

    # Apply mask and scale to maintain expected value
    return (x * mask) / (1 - p)
```

```python
# Variant: storing mask for backward pass
def dropout_forward_with_mask(x, p=0.5, training=True):
    if not training:
        return x, None

    mask = (random_uniform(x.shape) > p).float()
    output = (x * mask) / (1 - p)
    return output, mask     # mask needed for backward pass

def dropout_backward(grad_output, mask, p):
    # Gradient flows only through non-dropped neurons
    return (grad_output * mask) / (1 - p)
```

---

## Why it matters

- **Prevents overfitting**: The most direct effect. By randomly disabling neurons, the network cannot memorize training examples through fragile co-adaptations between specific neurons.
- **Ensemble interpretation**: Training with dropout is approximately equivalent to training an ensemble of $2^n$ sub-networks (where $n$ is the number of neurons) and averaging their predictions at inference. Each training step trains a different sub-network.
- **Forces redundancy**: No single neuron can become a critical bottleneck, because it might be dropped at any time. The network learns distributed representations where information is spread across many neurons.
- **Dead simple to implement**: One line in PyTorch (`nn.Dropout(p=0.1)`), no hyperparameter tuning beyond choosing $p$.
- **Complements other regularizers**: Dropout works alongside weight decay, data augmentation, and [[BatchNorm]] (though BatchNorm's own regularization effect sometimes reduces the need for heavy dropout).

### Variants

- **DropConnect**: Drop individual weights instead of entire neurons.
- **Spatial Dropout**: Drop entire feature map channels (useful in CNNs).
- **DropPath / Stochastic Depth**: Drop entire residual blocks. Common in modern vision architectures.

---

## Used in

- Most training pipelines: virtually every modern neural network uses some form of dropout
- Transformers: applied after attention weights and after FFN layers
- CNNs: applied in fully connected classifier heads
- [[BatchNorm]] partially replaces dropout's regularization effect in some architectures

---

**See also:** [[BatchNorm]]
