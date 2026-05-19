The algorithm that lets neural networks learn: compute a loss, propagate gradients backward through the computational graph using the chain rule, and update every weight in the direction that reduces the loss.

## What is it?

Backpropagation (Rumelhart, Hinton & Williams, 1986) is the method used to compute the gradient of a loss function with respect to every learnable parameter in a neural network. It is not a learning algorithm by itself -- it is the *gradient computation* step that makes gradient-based optimizers (SGD, [[Adam]], etc.) possible.

The core idea is the **chain rule of calculus** applied systematically to a computational graph. During the **forward pass**, each operation (matrix multiply, activation function, loss computation) stores its inputs and intermediate values. During the **backward pass**, gradients flow from the loss backward through each operation, accumulating via the chain rule:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$

where $L$ is the loss, $y$ is the output of the operation that depends on weight $w$, and $\frac{\partial L}{\partial y}$ is the gradient that has already been propagated from layers above.

For a deeper network with layers $f_1, f_2, \dots, f_n$, the gradient at layer $k$ is:

$$\frac{\partial L}{\partial w_k} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_{k+1}}{\partial a_k} \cdot \frac{\partial a_k}{\partial w_k}$$

This product of Jacobians is what makes backpropagation both powerful and fragile: the gradient must flow through every layer, and if those intermediate derivatives are consistently less than 1 or greater than 1, gradients vanish or explode.

## How it works

![[basics_backprop.png]]

[🔗 Open interactive Backpropagation Demo](../../interactive/backprop.html)

### Computational graph view

A neural network is a directed acyclic graph (DAG) of operations. The forward pass evaluates the graph from inputs to loss. The backward pass traverses it in reverse topological order, applying the chain rule at each node.

```
Input x ──► [Linear W1] ──► [ReLU] ──► [Linear W2] ──► [MSE Loss] ──► L
               │                            │                │
          (store x)                    (store a1)       (store a2, target)
```

Forward: compute and store activations at each node.
Backward: starting from $\frac{\partial L}{\partial L} = 1$, propagate gradients to every node.

### Concrete example: one neuron, one step

Suppose we have a single weight $w = 0.5$, input $x = 2$, target $t = 3$, and MSE loss.

**Forward pass:**

$$y = w \cdot x = 0.5 \times 2 = 1.0$$
$$L = \frac{1}{2}(y - t)^2 = \frac{1}{2}(1.0 - 3)^2 = 2.0$$

**Backward pass (chain rule):**

$$\frac{\partial L}{\partial y} = y - t = 1.0 - 3.0 = -2.0$$
$$\frac{\partial y}{\partial w} = x = 2.0$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} = (-2.0) \times 2.0 = -4.0$$

**Gradient descent update** (learning rate $\eta = 0.1$):

$$w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w} = 0.5 - 0.1 \times (-4.0) = 0.9$$

The weight moved from 0.5 toward 1.5 (the ideal value $t/x = 3/2$), reducing the loss.

### Pseudocode: forward and backward for a 2-layer network

```python
import numpy as np

# --- Forward pass ---
def forward(x, W1, b1, W2, b2, target):
    # Layer 1
    z1 = W1 @ x + b1              # pre-activation
    a1 = np.maximum(0, z1)        # ReLU activation

    # Layer 2 (output)
    z2 = W2 @ a1 + b2
    y = z2                         # linear output (regression)

    # Loss (MSE)
    loss = 0.5 * np.sum((y - target) ** 2)

    # Cache for backward pass
    cache = (x, z1, a1, W1, W2, y, target)
    return loss, cache

# --- Backward pass ---
def backward(cache):
    x, z1, a1, W1, W2, y, target = cache

    # Gradient of loss w.r.t. output
    dL_dy = y - target                         # (output_dim,)

    # Layer 2 gradients
    dL_dW2 = dL_dy[:, None] @ a1[None, :]     # outer product
    dL_db2 = dL_dy
    dL_da1 = W2.T @ dL_dy                     # propagate to layer 1

    # ReLU backward: pass gradient only where z1 > 0
    dL_dz1 = dL_da1 * (z1 > 0).astype(float)

    # Layer 1 gradients
    dL_dW1 = dL_dz1[:, None] @ x[None, :]
    dL_db1 = dL_dz1

    return dL_dW1, dL_db1, dL_dW2, dL_db2
```

### Pseudocode: gradient descent update step

```python
def gradient_descent_step(params, grads, lr=0.01):
    """
    params: list of parameter arrays [W1, b1, W2, b2]
    grads:  list of gradient arrays  [dW1, db1, dW2, db2]
    lr:     learning rate
    """
    for i in range(len(params)):
        params[i] = params[i] - lr * grads[i]
    return params

# Training loop
for epoch in range(num_epochs):
    loss, cache = forward(x, W1, b1, W2, b2, target)
    dW1, db1, dW2, db2 = backward(cache)
    W1, b1, W2, b2 = gradient_descent_step(
        [W1, b1, W2, b2],
        [dW1, db1, dW2, db2],
        lr=0.01
    )
```

### Common problems

**Vanishing gradients.** When the chain of partial derivatives $\frac{\partial a_{k+1}}{\partial a_k}$ is consistently less than 1 (e.g., sigmoid saturating near 0 or 1), the gradient shrinks exponentially as it flows backward through many layers. Deep networks (50+ layers) become untrainable because early layers receive near-zero gradients. [[Residual Connections]] solve this by adding a skip path ($y = f(x) + x$) whose gradient is $1 + \frac{\partial f}{\partial x}$, ensuring the gradient is at least 1 regardless of what $f$ does.

**Exploding gradients.** When the intermediate Jacobians have eigenvalues greater than 1, gradients grow exponentially. This manifests as NaN losses or wild parameter updates. The standard fix is **gradient clipping**: before the update step, scale the gradient vector so its norm does not exceed a threshold.

```python
def clip_gradient_norm(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = [g * scale for g in grads]
    return grads
```

**Memory cost.** The forward pass must store all intermediate activations for the backward pass. For very large models this becomes prohibitive. [[Gradient Checkpointing]] trades compute for memory: discard some activations during the forward pass and recompute them during the backward pass.

## Why it matters

Backpropagation is the engine behind all modern deep learning. Every model in this vault -- from [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)|ModernBERT]] to large language models -- is trained by computing gradients with backpropagation and updating weights with an optimizer like [[Adam]]. Understanding how gradients flow (and fail to flow) through a network is essential for diagnosing training failures, choosing architectures, and understanding why techniques like [[Residual Connections]] and [[Gradient Checkpointing]] exist.

## Used in

- Training of every neural network model in this vault
- [[Adam]] (uses backpropagation gradients as input to its adaptive update rule)
- [[Residual Connections]] (designed to improve gradient flow during backpropagation)
- [[Gradient Checkpointing]] (trades compute for memory during backpropagation)
