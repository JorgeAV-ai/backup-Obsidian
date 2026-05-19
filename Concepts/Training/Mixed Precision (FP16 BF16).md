---
tags:
  - basics
  - efficiency
---

**TL;DR:** Train with 16-bit floats (FP16 or BF16) for ~2x speed and ~half memory, while keeping master weights in FP32 for accuracy. BF16 is preferred for training due to its wider exponent range.

## What is it?

**Mixed precision training** uses lower-precision floating-point formats (16-bit) for the bulk of computation while maintaining a full-precision (FP32) copy of the weights for accumulation. This exploits hardware tensor cores that operate much faster on 16-bit data.

### FP16 vs BF16

| Format | Exponent bits | Mantissa bits | Range | Precision |
|---|---|---|---|---|
| FP32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | High |
| FP16 | 5 | 10 | $\pm 65504$ | Higher than BF16 |
| BF16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | Lower than FP16 |

- **FP16** has more mantissa bits (10 vs 7), so it's more precise for small differences, but its limited range ($\pm 65504$) causes overflow/underflow issues during training
- **BF16** has the same exponent range as FP32 (8 bits), so it rarely overflows — making it more robust for training gradients that can span a huge range
- **BF16 is preferred for training** because gradient magnitudes vary wildly and the exponent range matters more than mantissa precision

## How it works

![[basics_mixed_precision.png]]

[🔗 Open interactive Mixed Precision Demo](../../interactive/mixed_precision.html)

### The core idea

1. **Master weights** are stored in FP32 (never lost)
2. **Forward pass** runs in FP16/BF16 (fast, uses tensor cores)
3. **Loss scaling** (FP16 only): multiply loss by a scale factor before backward to push small gradients out of the underflow zone
4. **Backward pass** runs in FP16/BF16
5. **Unscale gradients** (FP16 only): divide gradients by the scale factor
6. **Weight update** happens in FP32: cast gradients to FP32, apply optimizer step to master weights

### Loss scaling (FP16)

FP16 can only represent values down to $\sim 6 \times 10^{-8}$. Many gradients during training are smaller than this and become zero (underflow). Loss scaling multiplies the loss by a large factor (e.g., 1024 or dynamically chosen) before the backward pass, shifting the gradient distribution into representable range.

BF16 has the same exponent range as FP32, so it rarely needs loss scaling.

### Pseudocode

```
# Mixed precision training loop

# Initialize
master_weights = model.parameters()           # FP32
optimizer = AdamW(master_weights)
scaler = GradScaler()                         # for FP16; not needed for BF16

for batch in dataloader:
    optimizer.zero_grad()

    # --- Forward in FP16/BF16 ---
    with autocast(dtype=float16):             # or bfloat16
        logits = model(batch.input)           # computed in FP16
        loss = criterion(logits, batch.target)

    # --- Loss scaling + Backward (FP16 path) ---
    scaled_loss = scaler.scale(loss)          # loss * scale_factor
    scaled_loss.backward()                    # gradients in FP16

    # --- Unscale + Clip + Step in FP32 ---
    scaler.unscale_(optimizer)                # grads /= scale_factor, cast to FP32
    clip_grad_norm_(master_weights, max_norm) # gradient clipping
    scaler.step(optimizer)                    # optimizer step on FP32 master weights
    scaler.update()                           # adjust scale factor dynamically

    # --- BF16 path (simpler, no scaler needed) ---
    # with autocast(dtype=bfloat16):
    #     logits = model(batch.input)
    #     loss = criterion(logits, batch.target)
    # loss.backward()
    # optimizer.step()
```

### Memory breakdown

For a model with $P$ parameters:
- FP32 only: $4P$ bytes (weights) + $8P$ bytes (Adam states) = $12P$ bytes
- Mixed precision: $4P$ (FP32 master) + $2P$ (FP16 working copy) + $8P$ (Adam in FP32) = $14P$ bytes for weights/optimizer, but **activations** (the main memory consumer) are halved

The real savings come from activations during forward/backward pass being in 16-bit, which for large batch sizes dominates memory.

## Why it matters

- **~2x throughput** on GPUs with tensor cores (A100, H100, etc.)
- **~Half the activation memory**, enabling larger batches or longer sequences
- **Minimal accuracy loss** when done correctly (master weights preserve precision)
- BF16 on modern hardware (Ampere+) makes it nearly free — just wrap in `autocast`

## Used in

- Standard practice in all modern model training (pretraining and fine-tuning)
- BF16 is the default for LLM training on A100/H100 GPUs
- FP16 with loss scaling is still used on older GPUs (V100, T4)

---

See also: [[Quantization]], [[Backpropagation]]
