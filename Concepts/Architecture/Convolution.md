# Convolution

> **TL;DR:** The fundamental operation of CNNs: slide a small filter (kernel) over the input, computing dot products at each position to produce a feature map. Simple idea, enormous impact.

---

## What is it?

A convolution is a mathematical operation that combines two functions to produce a third. In deep learning, it means taking a small learnable filter and sliding it across the input (an image, a spectrogram, a sequence), computing element-wise multiplications and summing the results at every position. The output is called a **feature map**.

The key insight is **weight sharing**: the same filter is applied everywhere, so the network learns to detect a feature (an edge, a texture, a motif) regardless of where it appears in the input. This is what gives CNNs their translation equivariance and makes them dramatically more parameter-efficient than fully connected layers for grid-structured data.

Each convolutional layer typically has multiple filters, each producing its own feature map. Early layers learn low-level features (edges, corners), deeper layers compose these into high-level features (eyes, wheels, sentences).

---

## How it works

![[basics_convolution.png]]

[🔗 Open interactive Convolution Explorer](../interactive/convolution.html)

### Core formula (2D discrete convolution)

$$(f * g)(i,j) = \sum_m \sum_n f(m,n) \cdot g(i-m, j-n)$$

In practice, deep learning frameworks implement **cross-correlation** (no kernel flip), but everyone still calls it convolution:

$$\text{output}(i,j) = \sum_m \sum_n \text{input}(i+m, j+n) \cdot \text{kernel}(m,n)$$

### Parameters

| Parameter | What it controls |
|---|---|
| **Kernel size** ($K$) | Spatial extent of the filter (e.g., $3 \times 3$, $5 \times 5$) |
| **Stride** ($S$) | Step size between filter applications (stride 2 halves spatial dims) |
| **Padding** ($P$) | Zeros added around input border ("same" padding preserves spatial size) |
| **Dilation** ($D$) | Spacing between kernel elements (dilated/atrous convolution widens receptive field without more params) |

### Output size formula

For an input of spatial size $N$, kernel size $K$, padding $P$, stride $S$, and dilation $D$:

$$\text{output size} = \left\lfloor\frac{N + 2P - D(K - 1) - 1}{S}\right\rfloor + 1$$

For the common case of no dilation ($D=1$), this simplifies to:

$$\left\lfloor\frac{N + 2P - K}{S}\right\rfloor + 1$$

### 1D, 2D, and 3D convolutions

| Variant | Input shape | Use case |
|---|---|---|
| **Conv1D** | $(B, C, L)$ | Audio, time series, text (character-level) |
| **Conv2D** | $(B, C, H, W)$ | Images, spectrograms |
| **Conv3D** | $(B, C, D, H, W)$ | Video, volumetric data (medical scans) |

### Pseudocode: 2D convolution forward pass (naive nested loops)

```python
def conv2d_naive(input, kernel, stride=1, padding=0):
    """
    input:  (C_in, H, W)     -- single example, C_in input channels
    kernel: (C_out, C_in, K, K) -- C_out filters
    output: (C_out, H_out, W_out)
    """
    # Pad the input
    input_padded = zero_pad(input, padding)
    C_in, H, W = input_padded.shape[0], input_padded.shape[1], input_padded.shape[2]
    C_out, _, K, _ = kernel.shape
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1

    output = zeros(C_out, H_out, W_out)

    for co in range(C_out):           # each output channel (filter)
        for i in range(H_out):        # each output row
            for j in range(W_out):    # each output column
                h_start = i * stride
                w_start = j * stride
                patch = input_padded[:, h_start:h_start+K, w_start:w_start+K]
                output[co, i, j] = sum(patch * kernel[co])  # dot product

    return output
```

### Pseudocode: im2col trick (how frameworks actually do it)

```python
def conv2d_im2col(input, kernel, stride=1, padding=0):
    """
    Reshape patches into columns, then do a single matrix multiply.
    Trades memory for speed -- exploits optimized GEMM routines.
    """
    input_padded = zero_pad(input, padding)
    C_out, C_in, K, _ = kernel.shape

    # Extract all patches and stack them as columns
    # Each column is a flattened (C_in * K * K) patch
    col = im2col(input_padded, K, stride)   # shape: (C_in*K*K, H_out*W_out)

    # Reshape kernel to (C_out, C_in*K*K)
    kernel_matrix = kernel.reshape(C_out, -1)

    # Single big matrix multiplication
    output = kernel_matrix @ col             # shape: (C_out, H_out*W_out)

    # Reshape back to spatial dims
    return output.reshape(C_out, H_out, W_out)
```

The im2col approach converts convolution into a matrix multiplication, which GPUs are extremely good at. The trade-off is higher memory usage (patches overlap, so data is duplicated in the column matrix).

---

## Why it matters

- **Parameter efficiency**: A $3 \times 3$ kernel with 64 filters on 64-channel input has only $64 \times 64 \times 3 \times 3 = 36{,}864$ parameters, regardless of spatial resolution. A fully connected layer on a $224 \times 224$ image would need billions.
- **Translation equivariance**: The same filter applied everywhere means a feature detected at one location can be recognized at any other location.
- **Hierarchical feature learning**: Stacking convolutions with pooling builds up receptive fields, letting the network compose low-level features into high-level representations.
- **Foundation of CNNs**: From LeNet (1998) to ResNet to ConvNeXt, convolutions have been the workhorse of computer vision for decades.
- **Still relevant in the ViT era**: Many modern architectures (FasterViT, ConvNeXt, EfficientNet) use convolutions alongside or instead of attention, especially for efficiency in early layers.

---

## Used in

- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]

---

**See also:** [[Depthwise Convolution]], [[BatchNorm]]
