Reduce model precision from high-bitwidth floating point (FP32) to lower representations (FP16, INT8, INT4), shrinking model size and memory footprint by 2-8x while preserving most of the original accuracy.

## What is it?

Neural network weights and activations are typically stored in 32-bit floating point (FP32), but most of this precision is unnecessary for inference. Quantization maps these high-precision values to a lower-precision representation using a scale and optional zero-point offset. The savings are direct: a model in INT4 uses 8x less memory than the same model in FP32.

This matters enormously in practice. **LLaMA 70B** in FP32 requires ~280 GB of memory — far beyond any single GPU. In INT4, the same model fits in ~35 GB, within reach of a single consumer GPU (e.g., RTX 4090 with 24 GB VRAM can run a 4-bit 70B model with some offloading).

There are two main approaches:
- **Post-Training Quantization (PTQ):** Quantize a pretrained model without further training. Fast and easy, but can degrade quality for aggressive quantization (INT4 and below). Methods like GPTQ and AWQ use calibration data to minimize quantization error.
- **Quantization-Aware Training (QAT):** Simulate quantization during training so the model learns to be robust to reduced precision. Higher quality but requires a full training run.

## How it works

![[basics_quantization.png]]

[🔗 Open interactive Quantization Explorer](../../interactive/quantization.html)

### The quantization formula

**Linear (affine) quantization** maps a floating-point value $x$ to an integer:

$$x_{\text{int}} = \text{round}\!\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

**Dequantization** reverses this:

$$\hat{x} = \text{scale} \times (x_{\text{int}} - \text{zero\_point})$$

Where:
- $\text{scale} = \frac{x_{\max} - x_{\min}}{2^b - 1}$ (for $b$-bit quantization)
- $\text{zero\_point} = \text{round}\!\left(-\frac{x_{\min}}{\text{scale}}\right)$

### Pseudocode

```python
def quantize(x, num_bits=8):
    """Quantize a floating-point tensor to num_bits integer."""
    x_min, x_max = x.min(), x.max()
    qmin, qmax = 0, 2**num_bits - 1

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(-x_min / scale)
    zero_point = clamp(zero_point, qmin, qmax)

    x_int = round(x / scale) + zero_point
    x_int = clamp(x_int, qmin, qmax)
    return x_int, scale, zero_point


def dequantize(x_int, scale, zero_point):
    """Recover approximate floating-point values."""
    return scale * (x_int - zero_point)
```

The error introduced is bounded by $\pm \frac{\text{scale}}{2}$ per element. Smaller scale (narrower range or more bits) means less error.

### Precision comparison

| Format | Bits | Bytes per param | Memory for 70B model | Notes |
|---|---|---|---|---|
| FP32 | 32 | 4.0 | 280 GB | Full precision, training default |
| FP16 | 16 | 2.0 | 140 GB | Half precision, common for inference |
| BF16 | 16 | 2.0 | 140 GB | Same size as FP16, larger dynamic range |
| INT8 | 8 | 1.0 | 70 GB | Good quality, 4x compression |
| INT4 | 4 | 0.5 | 35 GB | Aggressive but practical with good methods |

### GGUF format

**GGUF** (GPT-Generated Unified Format) is the standard file format for quantized models used by **llama.cpp** and its ecosystem. It stores quantized weights along with metadata (architecture, tokenizer, quantization scheme) in a single file. Common quantization levels in GGUF:

| Quant type | Bits per weight | Quality | Use case |
|---|---|---|---|
| Q2_K | ~2.5 | Poor | Extreme compression, testing only |
| Q4_K_M | ~4.5 | Good | Best balance of size and quality |
| Q5_K_M | ~5.5 | Very good | Slight quality bump over Q4 |
| Q6_K | ~6.5 | Excellent | Near-FP16 quality |
| Q8_0 | 8 | Near lossless | Maximum quality quantized |

### Group quantization

Rather than quantizing an entire tensor with a single scale and zero-point, **group quantization** divides the tensor into groups of $g$ elements (e.g., $g = 128$) and computes separate quantization parameters per group. This captures local variation in the weight distribution and significantly reduces quantization error, especially at low bit-widths. GPTQ, AWQ, and most GGUF quant types use group quantization.

## Why it matters

Quantization is what makes large language models accessible on consumer hardware. Without it, running a 70B model requires multiple A100 GPUs; with INT4 quantization, the same model runs on a single GPU or even on a MacBook with sufficient RAM. The quality loss from modern quantization methods (GPTQ, AWQ, GGUF Q4_K_M) is surprisingly small — typically less than 1% degradation on benchmarks compared to FP16. This has been a key enabler of the local LLM movement and edge deployment.

## Used in

- [[LoRA]]
- [[Mixed Precision (FP16 BF16)]]
