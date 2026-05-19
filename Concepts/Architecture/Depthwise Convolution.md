A convolution variant where each input channel is convolved with its own dedicated filter, eliminating cross-channel mixing and drastically reducing parameters.

## What is it?

In a standard convolution, every output channel is produced by a filter that spans *all* input channels. This means a single filter has shape $k \times k \times d_{in}$, and you need $d_{out}$ of them, giving $d_{out} \times d_{in} \times k^2$ parameters. Most of those parameters exist just to mix information across channels.

Depthwise convolution takes a different approach: it assigns one $k \times k$ filter to each input channel independently. Channel $i$ of the input is convolved only with filter $i$, producing channel $i$ of the output. No information flows between channels at this stage. The total parameter count drops to $d \times k^2$, where $d$ is the number of channels.

By itself, depthwise convolution is too restrictive because it cannot learn cross-channel relationships. The practical solution, introduced in MobileNet (Howard et al., 2017), is **depthwise separable convolution**: a depthwise convolution followed by a **pointwise** ($1 \times 1$) convolution that mixes channels. The pointwise step has $d_{in} \times d_{out}$ parameters, and the combined cost is still far lower than a standard convolution.

## How it works

![[basics_depthwise_conv.png]]

### Standard convolution (for comparison)

For input $X \in \mathbb{R}^{H \times W \times d_{in}}$, filters $W \in \mathbb{R}^{k \times k \times d_{in} \times d_{out}}$:

$$Y_{h,w,j} = \sum_{c=1}^{d_{in}} \sum_{m,n} W_{m,n,c,j} \cdot X_{h+m,\, w+n,\, c}$$

**Parameters:** $d_{out} \times d_{in} \times k^2$
**FLOPs per spatial position:** $d_{out} \times d_{in} \times k^2$

### Depthwise convolution

Each channel has its own $k \times k$ filter $W^{dw} \in \mathbb{R}^{k \times k \times d}$:

$$Y_{h,w,c}^{dw} = \sum_{m,n} W^{dw}_{m,n,c} \cdot X_{h+m,\, w+n,\, c}$$

**Parameters:** $d \times k^2$
**FLOPs per spatial position:** $d \times k^2$

### Depthwise separable convolution

1. **Depthwise step:** apply the depthwise conv above.
2. **Pointwise step:** apply a $1 \times 1$ convolution to mix channels:

$$Y_{h,w,j} = \sum_{c=1}^{d} W^{pw}_{c,j} \cdot Y^{dw}_{h,w,c}$$

**Total parameters:** $d \times k^2 + d \times d_{out}$

### Pseudocode

```python
# --- Standard convolution ---
for j in range(d_out):
    for h, w in spatial_positions:
        out[h, w, j] = 0
        for c in range(d_in):
            for m, n in kernel_window:
                out[h, w, j] += W[m, n, c, j] * X[h+m, w+n, c]

# --- Depthwise convolution ---
for c in range(d):
    for h, w in spatial_positions:
        out[h, w, c] = 0
        for m, n in kernel_window:
            out[h, w, c] += W_dw[m, n, c] * X[h+m, w+n, c]
# Note: no summation over channels — each channel is independent
```

### Parameter and FLOP comparison

For $d_{in} = d_{out} = d$, kernel size $k$, spatial size $H \times W$:

| | Parameters | FLOPs |
|---|---|---|
| Standard conv | $d^2 k^2$ | $H W d^2 k^2$ |
| Depthwise conv | $d k^2$ | $H W d k^2$ |
| Depthwise separable | $d k^2 + d^2$ | $H W d (k^2 + d)$ |

The reduction factor of depthwise separable over standard is approximately:

$$\frac{1}{d} + \frac{1}{k^2}$$

For $d = 256, k = 3$ this is roughly $\frac{1}{256} + \frac{1}{9} \approx 8\text{-}9\times$ fewer parameters.

## Why it matters

Standard convolutions are the main computational bottleneck in CNNs. Depthwise separable convolutions achieve comparable accuracy at a fraction of the cost, making it possible to deploy deep networks on mobile devices and edge hardware. They are the backbone of lightweight architectures like MobileNet, EfficientNet, and increasingly appear as efficient spatial-mixing components inside vision transformers.

## Used in

- [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]
