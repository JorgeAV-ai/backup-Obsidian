Unpadding removes wasted computation on padding tokens by concatenating all real tokens into a flat sequence and tracking boundaries with cumulative sequence lengths.

## What is it?

In standard batched transformer inference and training, all sequences in a batch are padded to the same length (typically the longest sequence in the batch). This means the model performs full self-attention and FFN computations on padding tokens that carry no information and whose outputs are immediately discarded. For batches with high length variance, the wasted compute can be substantial --- a batch where most sequences are 50 tokens but one is 512 tokens means 90%+ of the computation is on padding.

Unpadding (also called "variable-length attention" or "sequence packing") eliminates this waste. Instead of maintaining a 2D tensor of shape $(B, T_{\max})$, all real (non-padding) tokens are concatenated into a single flat 1D sequence of shape $(N_{\text{total}},)$, where $N_{\text{total}} = \sum_i L_i$ is the total number of real tokens across all sequences. A small auxiliary array called `cu_seqlens` (cumulative sequence lengths) tracks where each sequence starts and ends within this flat tensor. The attention mechanism uses `cu_seqlens` to ensure that tokens only attend within their own sequence, never across sequence boundaries.

This technique is especially effective when combined with Flash Attention, which natively supports variable-length inputs through its `flash_attn_varlen_func` API. The combination delivers both the memory efficiency of Flash Attention and the compute savings of skipping padding, yielding 10--20% throughput improvements on typical workloads and even larger gains on batches with high length variance.

## How it works

![[basics_unpadding.png]]

**Step 1: Remove padding and build `cu_seqlens`.**

Given a padded batch of shape $(B, T_{\max})$ with an attention mask indicating real tokens:

```python
import torch

def unpad_batch(input_ids, attention_mask):
    """
    input_ids:      (B, T_max) - padded token IDs
    attention_mask:  (B, T_max) - 1 for real tokens, 0 for padding

    Returns:
        flat_ids:    (N_total,) - concatenated real tokens
        cu_seqlens:  (B+1,)    - cumulative sequence length boundaries
        max_seqlen:  int       - longest sequence in batch
        indices:     (N_total,) - original positions (for re-padding)
    """
    # Compute actual lengths
    lengths = attention_mask.sum(dim=1)          # (B,)
    max_seqlen = lengths.max().item()

    # Build cumulative sequence lengths (starts at 0)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = lengths.cumsum(dim=0)

    # Flatten: keep only real tokens
    indices = attention_mask.bool().flatten().nonzero(as_tuple=False).squeeze()
    flat_ids = input_ids.flatten()[indices]

    return flat_ids, cu_seqlens, max_seqlen, indices
```

**Example:**

```
Batch (padded):
  Seq 0: [The, cat, sat, PAD, PAD]   length = 3
  Seq 1: [Hello, world, PAD, PAD, PAD]  length = 2
  Seq 2: [A, B, C, D, E]             length = 5

After unpadding:
  flat_ids:   [The, cat, sat, Hello, world, A, B, C, D, E]
  cu_seqlens: [0, 3, 5, 10]
  max_seqlen: 5
  N_total:    10  (vs. 15 with padding)
```

**Step 2: Run through the transformer.**

The flat token sequence is passed through embedding, then through each transformer layer. Attention uses the variable-length API:

```python
from flash_attn import flash_attn_varlen_func

def unpadded_attention(q, k, v, cu_seqlens, max_seqlen):
    """
    q, k, v: (N_total, num_heads, head_dim)
    cu_seqlens: (B+1,) int32
    max_seqlen: int

    Returns: (N_total, num_heads, head_dim)
    """
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,  # bidirectional for encoders
    )
```

Flash Attention internally uses `cu_seqlens` to apply attention only within each sequence's boundaries, preventing cross-sequence attention without needing an explicit mask tensor.

**Step 3: Reconstruct padded output (if needed).**

For tasks that require padded output (e.g., token classification with a fixed-size output tensor):

```python
def repad_batch(flat_hidden, indices, batch_size, max_seqlen, hidden_dim):
    """
    flat_hidden: (N_total, hidden_dim)
    indices:     (N_total,) - positions in the original padded tensor

    Returns: (B, T_max, hidden_dim) with zeros for padding positions
    """
    padded = torch.zeros(batch_size * max_seqlen, hidden_dim,
                         device=flat_hidden.device, dtype=flat_hidden.dtype)
    padded[indices] = flat_hidden
    return padded.view(batch_size, max_seqlen, hidden_dim)
```

**Full pipeline summary:**

```
Padded batch (B, T_max)
        |
   [unpad_batch]  -->  flat tokens (N_total,) + cu_seqlens (B+1,)
        |
   [embedding]    -->  (N_total, d_model)
        |
   [transformer layers with flash_attn_varlen_func]
        |
   [repad_batch]  -->  (B, T_max, d_model)  (optional, only if needed)
        |
   [task head]
```

## Why it matters

Unpadding solves the problem of wasted computation on padding tokens, which is one of the most straightforward sources of inefficiency in batched transformers. The savings are proportional to the padding ratio: in batches with high length variance (common in real-world data), throughput improvements of 10--20% are typical, and gains can exceed 30% in extreme cases. Beyond raw throughput, unpadding also reduces memory usage since the intermediate activation tensors are proportional to $N_{\text{total}}$ rather than $B \times T_{\max}$. Combined with Flash Attention, it has become a standard optimization in efficient transformer implementations.

## Used in

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]
