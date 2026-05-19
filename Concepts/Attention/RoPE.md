Rotary Positional Embedding encodes position by rotating query and key vectors in pair-wise dimensions, so that their dot product naturally depends only on relative position.

## What is it?

Rotary Positional Embedding (RoPE), introduced by Su et al. (2021), is a method for injecting positional information into transformer attention. Unlike absolute positional embeddings (as in BERT), which add a learned vector to the token representation, RoPE *multiplies* query and key vectors by a rotation matrix whose angle is determined by the token's position. This means position is encoded in the geometry of the vectors rather than as an additive bias.

The core insight is elegant: if you rotate query vector $q$ at position $m$ and key vector $k$ at position $n$ by position-dependent angles, their inner product $\langle R_m q,\, R_n k \rangle$ ends up depending only on the relative distance $m - n$. This gives the model translation-invariant attention scores without any explicit relative-position lookup table, and it works with standard dot-product attention unchanged.

RoPE operates on pairs of dimensions. For each consecutive pair $(2i, 2i{+}1)$ it applies a 2D rotation with a frequency that decreases with dimension index. Low-indexed pairs rotate fast (capturing fine-grained, nearby positions) while high-indexed pairs rotate slowly (capturing long-range order). This multi-frequency scheme is reminiscent of sinusoidal positional encodings but is applied multiplicatively and directly inside attention.

## How it works

![[basics_rope.png]]

[🔗 Open interactive RoPE Explorer](../interactive/rope.html)

**Frequency schedule.** Given embedding dimension $d$ and a base frequency parameter $\theta_{\text{base}}$ (typically 10000):

$$\theta_i = \theta_{\text{base}}^{\;-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1$$

**Rotation matrix.** For a token at position $m$, the rotation applied to dimension pair $(2i, 2i{+}1)$ is:

$$R_m^{(i)} = \begin{pmatrix} \cos(m\,\theta_i) & -\sin(m\,\theta_i) \\ \sin(m\,\theta_i) & \phantom{-}\cos(m\,\theta_i) \end{pmatrix}$$

The full rotation matrix $R_m$ is block-diagonal, with one $2 \times 2$ block per dimension pair.

**Relative-position property.** Because rotations compose as $R_m^\top R_n = R_{n-m}$:

$$\langle R_m\, q,\; R_n\, k \rangle = q^\top R_m^\top R_n\, k = q^\top R_{n-m}\, k$$

The dot product depends only on $n - m$, the relative distance.

**Pseudocode for applying RoPE to Q and K:**

```python
import torch

def precompute_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute cos/sin tables for all positions."""
    i = torch.arange(0, dim, 2).float()          # [dim/2]
    theta = base ** (-i / dim)                     # [dim/2]
    positions = torch.arange(max_seq_len).float()  # [T]
    angles = positions[:, None] * theta[None, :]   # [T, dim/2]
    cos_table = angles.cos()                       # [T, dim/2]
    sin_table = angles.sin()                       # [T, dim/2]
    return cos_table, sin_table

def apply_rope(x, cos_table, sin_table):
    """
    x: (batch, seq_len, num_heads, dim)
    cos_table, sin_table: (seq_len, dim/2)
    """
    T = x.shape[1]
    cos = cos_table[:T][None, :, None, :]  # (1, T, 1, dim/2)
    sin = sin_table[:T][None, :, None, :]

    # Split into even/odd dimension pairs
    x_even = x[..., 0::2]   # (..., dim/2)
    x_odd  = x[..., 1::2]   # (..., dim/2)

    # Apply 2D rotation to each pair
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out

# Usage inside attention:
# Q = apply_rope(Q, cos_table, sin_table)
# K = apply_rope(K, cos_table, sin_table)
# attn = softmax(Q @ K^T / sqrt(d))
```

**Why base $\theta$ matters for context extension.** Each dimension pair completes a full $2\pi$ rotation every $2\pi / \theta_i$ positions. High-frequency pairs (small $i$, large $\theta_i$) wrap around quickly; low-frequency pairs (large $i$, small $\theta_i$) change very slowly. If the model is trained with $\theta_{\text{base}} = 10\,000$ up to 4096 tokens, the slowest-rotating pair may never complete a full cycle. Increasing $\theta_{\text{base}}$ (e.g., to 500\,000 or 1\,000\,000) slows all rotations, meaning no dimension pair "wraps around" within a longer context window. This is the principle behind methods like CodeLlama's extended context and YaRN: scale the base so that the frequency spectrum remains well-conditioned at longer sequence lengths.

## Comparison with other positional encoding schemes

| Method | Mechanism | Relative position? | Extrapolation |
|---|---|---|---|
| Absolute (BERT) | Learned vector added to embeddings | No (implicit only) | Poor beyond training length |
| Sinusoidal (Transformer) | Fixed sin/cos added to embeddings | Theoretically, but weak in practice | Limited |
| ALiBi | Linear bias subtracted from attention logits | Yes (explicit linear penalty) | Good, but assumes recency bias |
| **RoPE** | Rotation of Q, K vectors | Yes (via rotation composition) | Extensible with base scaling |

ALiBi is simpler (no learned parameters, no extra compute) but bakes in a strong monotonic distance decay. RoPE is more flexible: the model can learn arbitrary relative-position patterns through the Q/K projections, and context length can be extended post-training by adjusting $\theta_{\text{base}}$.

## Why it matters

RoPE solves the problem of encoding relative position in transformers without requiring explicit relative-position bias tables (which scale quadratically in memory) or sacrificing the efficiency of standard dot-product attention. It is parameter-free, easy to implement, and --- crucially --- allows context length extension after training by simply rescaling the frequency base. This has made it the de facto positional encoding for modern large language models (LLaMA, Mistral, Qwen, etc.) and increasingly for encoder models as well.

## Used in

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]
