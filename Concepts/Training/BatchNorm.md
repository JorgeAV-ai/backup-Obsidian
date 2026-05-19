Batch Normalization normalizes activations per mini-batch during training, reducing internal covariate shift and enabling faster, more stable training with higher learning rates.

## What is it?

Batch Normalization (BatchNorm) is a technique introduced by Ioffe and Szegedy in 2015 that addresses a fundamental headache in deep network training: **internal covariate shift**. The idea is simple in hindsight. As parameters in earlier layers update during backpropagation, the distribution of inputs to every subsequent layer shifts. Each layer is constantly chasing a moving target. This makes training slow, fragile, and sensitive to initialization and learning rate choices.

BatchNorm tackles this by normalizing the inputs to each layer across the current mini-batch. For each feature (channel), it computes the mean and variance of that feature over all examples in the batch, then shifts and scales the normalized result using two learnable parameters: $\gamma$ (scale) and $\beta$ (shift). These parameters let the network recover the identity transform if that turns out to be optimal, so normalization never reduces the model's representational power.

A critical subtlety: **training and inference behave differently**. During training, statistics are computed per mini-batch. During inference, you don't have a batch -- you might be processing a single image. So during training, BatchNorm maintains exponential moving averages of the mean and variance (often called "running mean" and "running variance"). At inference time, these stored running statistics are used instead of batch statistics. This is why you must call `model.eval()` in PyTorch before inference -- it switches BatchNorm (and Dropout) to evaluation mode.

## How it works

![[basics_batchnorm.png]]

[🔗 Open interactive BatchNorm Visualizer](../../interactive/batchnorm.html)

### Forward pass formulas

Given a mini-batch $B = \{x_1, x_2, \dots, x_m\}$ of size $m$:

**Step 1: Compute mini-batch mean**

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2: Compute mini-batch variance**

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Step 3: Normalize**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\epsilon$ is a small constant (e.g., $10^{-5}$) for numerical stability.

**Step 4: Scale and shift**

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ and $\beta$ are learnable parameters initialized to 1 and 0 respectively.

**Running statistics update (during training):**

$$\mu_{\text{running}} \leftarrow (1 - \alpha)\,\mu_{\text{running}} + \alpha\,\mu_B$$

$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha)\,\sigma^2_{\text{running}} + \alpha\,\sigma_B^2$$

where $\alpha$ is the momentum (default 0.1 in PyTorch).

### Pseudocode

```python
def batchnorm_forward(x, gamma, beta, running_mean, running_var,
                      training=True, momentum=0.1, eps=1e-5):
    """
    x: input tensor of shape (N, C, H, W) for conv or (N, C) for linear
       Normalization is computed per-channel (over N, H, W or just N).
    gamma: learnable scale, shape (C,)
    beta: learnable shift, shape (C,)
    running_mean: exponential moving average of mean, shape (C,)
    running_var: exponential moving average of variance, shape (C,)
    """
    if training:
        # Compute batch statistics over all dims except C
        mu_B = x.mean(dim=(0, 2, 3), keepdim=True)      # shape: (1, C, 1, 1)
        var_B = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

        # Normalize
        x_hat = (x - mu_B) / sqrt(var_B + eps)

        # Update running statistics (detached from gradient computation)
        running_mean = (1 - momentum) * running_mean + momentum * mu_B.squeeze()
        running_var  = (1 - momentum) * running_var  + momentum * var_B.squeeze()
    else:
        # Inference: use stored running statistics
        x_hat = (x - running_mean.view(1, -1, 1, 1)) / sqrt(running_var.view(1, -1, 1, 1) + eps)

    # Scale and shift
    y = gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

    return y, running_mean, running_var
```

### Backward pass (sketch)

During backprop, gradients flow through the normalization. The key insight is that because $\mu_B$ and $\sigma_B^2$ depend on every element of the batch, the gradient for each $x_i$ depends on the entire batch. This couples the gradients across examples in the batch, which is part of why BatchNorm has a regularization effect similar to (but distinct from) dropout.

## Why it matters

Before BatchNorm, training deep networks required careful initialization (Xavier, He) and conservative learning rates. BatchNorm largely solved this:

- **Faster convergence**: You can use much higher learning rates without diverging because the normalization keeps activations in a well-behaved range.
- **Reduces sensitivity to initialization**: The normalization absorbs bad scaling from initialization.
- **Regularization effect**: Because statistics are computed per mini-batch, there is noise in the normalization. This acts as a mild regularizer, sometimes reducing the need for Dropout.
- **Enables deeper networks**: Pre-BatchNorm, very deep networks were essentially untrainable without careful tricks. BatchNorm made going deeper routine.

### Comparison with other normalizations

| Method | Normalizes over | Depends on batch size | Typical use case |
|---|---|---|---|
| **BatchNorm** | (N, H, W) per channel | Yes | CNNs with large batches |
| **LayerNorm** | (C, H, W) per sample | No | Transformers, RNNs |
| **InstanceNorm** | (H, W) per sample per channel | No | Style transfer |
| **GroupNorm** | Groups of channels per sample | No | CNNs with small batches |

**BatchNorm** struggles with small batch sizes (noisy statistics) and sequence models (variable lengths). **LayerNorm** normalizes across features for each individual sample, making it batch-size-independent -- this is why Transformers almost universally use LayerNorm. **InstanceNorm** normalizes per-sample, per-channel, which strips style information -- hence its dominance in style transfer. **GroupNorm** is a compromise: it splits channels into groups and normalizes within each group per sample, giving batch-size independence while still normalizing over spatial dimensions.

## Used in

- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
