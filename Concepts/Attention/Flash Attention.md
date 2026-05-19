An IO-aware, exact attention algorithm that avoids materializing the full $N \times N$ attention matrix in GPU high-bandwidth memory, reducing memory from $O(N^2)$ to $O(N)$ and achieving 2-4x wall-clock speedup on long sequences.

## What is it?

Standard self-attention computes $\text{softmax}(QK^T / \sqrt{d})V$ by first materializing the full $N \times N$ score matrix in GPU HBM (high-bandwidth memory). For long sequences this matrix dominates both memory usage and runtime, not because of arithmetic cost, but because of *memory IO*: reading and writing large intermediate matrices to slow HBM is the true bottleneck.

Flash Attention (Dao et al., 2022) restructures the attention computation to work in **tiles** that fit in the GPU's fast on-chip SRAM. It never writes the full $N \times N$ matrix to HBM. Instead, it streams blocks of $Q$, $K$, $V$ into SRAM, computes partial attention within each tile, and uses an **online softmax** trick to combine partial results on the fly. The final output is mathematically identical to standard attention -- this is *not* an approximation.

The key insight is that the softmax normalization constant can be computed incrementally. As new blocks of keys are processed, the running maximum and sum-of-exponentials are updated, and previously computed partial outputs are rescaled. This is an extension of the classical online softmax algorithm (Milakov & Gimelshein, 2018) to the full attention + value-weighted sum.

## How it works

![[basics_flash_attention.png]]

[🔗 Open interactive Flash Attention Visualizer](../interactive/flash_attention.html)

### GPU memory hierarchy

| Memory | Size | Bandwidth | Latency |
|---|---|---|---|
| SRAM (on-chip) | ~20 MB (A100) | ~19 TB/s | Very low |
| HBM (off-chip) | 40-80 GB (A100) | ~2 TB/s | High |

Standard attention writes $O(N^2)$ data to HBM. Flash Attention keeps all intermediate results in SRAM, writing only the final $O(N \cdot d)$ output to HBM.

### Memory comparison

| | Standard Attention | Flash Attention |
|---|---|---|
| HBM memory for intermediates | $O(N^2)$ | $O(N)$ |
| Total FLOPs | $O(N^2 d)$ | $O(N^2 d)$ (same) |
| HBM reads/writes | $O(N^2 + Nd)$ | $O(N^2 d^2 / M)$* |

*Where $M$ is SRAM size. Since $M \gg d^2$ typically, total IO is much less than standard attention.

### The tiling strategy with online softmax

The core idea:

1. Divide $Q$ into blocks of $B_r$ rows and $K, V$ into blocks of $B_c$ rows.
2. For each $Q$-block, iterate over all $K$-blocks, computing partial scores in SRAM.
3. Maintain running statistics $m_i$ (row-wise max) and $\ell_i$ (row-wise sum of exponentials) for correct softmax normalization.
4. Rescale previously accumulated output when the running max changes.

### Pseudocode (simplified tiling loop)

```python
# Q, K, V in HBM, each (N, d)
# Block sizes: B_r (query block), B_c (key block)
# Output O in HBM, (N, d)

# Divide Q into T_r = ceil(N / B_r) blocks
# Divide K, V into T_c = ceil(N / B_c) blocks

for i in range(T_r):                     # for each query block
    Q_i = load_from_HBM(Q[i])            # (B_r, d) -> SRAM
    O_i = zeros(B_r, d)                  # accumulator in SRAM
    m_i = -inf * ones(B_r)               # running row max
    l_i = zeros(B_r)                     # running row sum-of-exp

    for j in range(T_c):                 # for each key block
        K_j = load_from_HBM(K[j])        # (B_c, d) -> SRAM
        V_j = load_from_HBM(V[j])        # (B_c, d) -> SRAM

        # Compute block scores in SRAM
        S_ij = Q_i @ K_j.T / sqrt(d)     # (B_r, B_c)

        # Online softmax update
        m_new = max(m_i, rowmax(S_ij))    # new running max
        P_ij = exp(S_ij - m_new)          # stable exp

        # Rescale old accumulator for the new max
        scale = exp(m_i - m_new)
        O_i = O_i * scale[:, None] + P_ij @ V_j
        l_i = l_i * scale + rowsum(P_ij)
        m_i = m_new

    # Final normalization
    O_i = O_i / l_i[:, None]
    write_to_HBM(O[i], O_i)
```

Notice that the $N \times N$ matrix `S` is never fully materialized. Only one $(B_r \times B_c)$ tile exists at a time, entirely in SRAM.

### Why it is NOT an approximation

Flash Attention computes the exact same result as standard attention. The online softmax rescaling ensures numerical equivalence. The only difference is the *order* of operations: instead of computing the full score matrix first and then normalizing, it incrementally normalizes as it processes each key block. The associativity of the exponential function makes this mathematically exact (up to floating point rounding, which is comparable to standard attention).

### Wall-clock performance

- **2-4x speedup** on sequences of length 1K-16K on A100 GPUs.
- **5-20x memory reduction** for the attention computation.
- Enables training with much longer context lengths without running out of memory.
- Flash Attention-2 further improves throughput by optimizing the parallelism and work partitioning across GPU thread blocks.

## Why it matters

Attention's $O(N^2)$ memory cost was the main practical barrier to long-context models. Flash Attention removes this barrier without sacrificing exactness or requiring architectural changes. It is now the default attention implementation in most major frameworks (PyTorch 2.0+, HuggingFace, etc.) and has been essential for training and deploying models with context lengths of 8K, 32K, 128K and beyond.

## Used in

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]
