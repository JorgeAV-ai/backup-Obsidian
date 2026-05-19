Efficient Channel Attention: a lightweight channel attention mechanism that replaces the two fully-connected layers in SE-Net with a single 1D convolution, achieving similar accuracy with far fewer parameters.

## What is it?

Channel attention mechanisms learn to re-weight feature map channels so the network can emphasize informative channels and suppress less useful ones. Squeeze-and-Excitation (SE-Net, Hu et al., 2018) pioneered this idea with a *squeeze* step (global average pooling) followed by an *excitation* step (two FC layers with a bottleneck). The bottleneck reduction ratio introduces a hyperparameter and forces cross-channel interactions through a compressed representation, which can hurt performance.

Efficient Channel Attention (ECA, Wang et al., 2020) simplifies this by observing that each channel only needs to interact with its *local neighbors*, not all channels at once. After global average pooling, ECA applies a **1D convolution** with an adaptive kernel size $k$ along the channel dimension, followed by a sigmoid. This captures local cross-channel interactions without any dimensionality reduction.

The key insight is that the appropriate range of channel interaction (the kernel size $k$) should scale with the number of channels. ECA derives $k$ automatically from the channel count $C$ using a simple formula, eliminating the need to tune a reduction ratio.

## How it works

![[basics_eca.png]]

### Pipeline

$$\text{Input } X \in \mathbb{R}^{H \times W \times C} \xrightarrow{\text{GAP}} z \in \mathbb{R}^{C} \xrightarrow{\text{1D Conv}(k)} s \in \mathbb{R}^{C} \xrightarrow{\sigma} w \in \mathbb{R}^{C} \xrightarrow{\text{scale}} \hat{X}$$

1. **Global Average Pooling:** $z_c = \frac{1}{HW}\sum_{h,w} X_{h,w,c}$
2. **1D Convolution (kernel $k$, circular padding):** each channel attends to its $k$ nearest neighbors in the channel dimension.
3. **Sigmoid activation:** $w_c = \sigma(s_c)$
4. **Channel-wise rescaling:** $\hat{X}_{h,w,c} = w_c \cdot X_{h,w,c}$

### Adaptive kernel size

The kernel size $k$ is determined by the number of channels $C$:

$$k = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{\text{odd}}$$

where $\gamma = 2$, $b = 1$, and $|\cdot|_{\text{odd}}$ rounds to the nearest odd integer. This means:
- $C = 64 \Rightarrow k = 3$
- $C = 256 \Rightarrow k = 5$
- $C = 512 \Rightarrow k = 5$

More channels get a wider kernel, capturing longer-range channel dependencies.

### Comparison with SE-Net

| | SE-Net | ECA |
|---|---|---|
| Mechanism | GAP -> FC -> ReLU -> FC -> Sigmoid | GAP -> 1D Conv -> Sigmoid |
| Parameters | $\frac{2C^2}{r}$ (reduction ratio $r$) | $k$ (kernel size, typically 3-7) |
| Cross-channel | Global (all-to-all via FC) | Local (k-nearest neighbors) |
| Hyperparameter | Reduction ratio $r$ | None (k derived from $C$) |

For $C = 512, r = 16$: SE adds $2 \times 512^2 / 16 = 32{,}768$ params per block. ECA adds just $k \approx 5$ params per block.

### Pseudocode

```python
import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 == 1 else k + 1  # ensure odd

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        z = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        z = z.unsqueeze(1)                              # (B, 1, C)
        s = self.conv(z)                                # (B, 1, C)
        w = self.sigmoid(s).squeeze(1)                  # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)        # (B, C, H, W)
```

## Why it matters

Channel attention is a cheap way to boost CNN (and hybrid Transformer) performance, but even SE-Net's overhead can add up in lightweight networks. ECA achieves comparable or better accuracy than SE-Net with orders-of-magnitude fewer parameters per attention block, making it practical to insert into every layer of a network without meaningful cost. Its parameter-free kernel size formula also removes a hyperparameter search burden.

## Used in

- [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]
