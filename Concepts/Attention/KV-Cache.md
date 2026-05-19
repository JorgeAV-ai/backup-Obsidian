During autoregressive generation, cache the Key and Value tensors from previous tokens so that each new token only computes its own attention query against the stored keys and values, avoiding the full recomputation that would otherwise grow quadratically with sequence length.

## What is it?

In autoregressive generation, a language model produces tokens one at a time. At each step, the self-attention mechanism needs Key and Value projections for *every* token generated so far. Without caching, generating token $t$ requires recomputing $K$ and $V$ for all $t$ tokens from scratch — meaning the work to generate a full sequence of length $N$ scales as $O(N^2)$ in attention computations.

The **KV-Cache** stores the Key and Value matrices from all previous steps. When generating a new token, only the Query for the new token is computed, and it attends over the cached Keys and Values. The new token's own $K$ and $V$ are appended to the cache for future steps. This reduces per-step attention from $O(t \cdot d)$ recomputation down to $O(d)$ for the new projections, plus the $O(t \cdot d)$ dot products that are unavoidable.

## How it works

![[basics_kv_cache.png]]

[🔗 Open interactive KV-Cache Visualizer](../../interactive/kv_cache.html)

### Without KV-Cache (naive generation)

At every decoding step $t$, the model recomputes attention over the full sequence:

```
for t in range(1, max_tokens):
    tokens = [tok_1, tok_2, ..., tok_t]

    # Recompute ALL projections from scratch
    Q = W_q @ embed(tokens)     # (t, d)
    K = W_k @ embed(tokens)     # (t, d)
    V = W_v @ embed(tokens)     # (t, d)

    scores = Q @ K.T / sqrt(d)  # (t, t)
    out = softmax(scores) @ V   # (t, d)
    next_token = decode(out[-1])
```

This recomputes $K$ and $V$ for tokens 1 through $t-1$ at *every* step, even though those values never change (each layer is causal, so earlier tokens' representations are fixed).

### With KV-Cache

```
K_cache = []   # will hold (N, d) across all steps
V_cache = []

for t in range(1, max_tokens):
    x_t = embed(tok_t)          # embedding of just the new token

    # Compute projections for the NEW token only
    q_t = W_q @ x_t             # (1, d)
    k_t = W_k @ x_t             # (1, d)
    v_t = W_v @ x_t             # (1, d)

    # Append to cache
    K_cache.append(k_t)         # cache is now (t, d)
    V_cache.append(v_t)

    # Attend: new query against ALL cached keys/values
    scores = q_t @ K_cache.T / sqrt(d)   # (1, t)
    out = softmax(scores) @ V_cache      # (1, d)

    next_token = decode(out)
```

Only one row of $Q$, $K$, $V$ is computed per step. The cached $K$ and $V$ from all previous steps are reused directly.

### Memory footprint

The cache stores $K$ and $V$ for every layer and every generated token:

$$\text{Cache size} = 2 \times L \times N \times d \times \text{bytes\_per\_param}$$

where $L$ = number of layers, $N$ = sequence length, $d$ = hidden dimension.

**Example — LLaMA 2 70B (FP16):**

| Parameter | Value |
|---|---|
| Layers ($L$) | 80 |
| Hidden dim ($d$) | 8192 |
| Sequence length ($N$) | 4096 |
| Bytes per param (FP16) | 2 |

$$\text{Cache} = 2 \times 80 \times 4096 \times 8192 \times 2 \approx 10.7 \text{ GB}$$

For a batch of 32 sequences, that is ~340 GB just for the KV-Cache — often exceeding the model weights themselves.

### The fundamental trade-off

| | Without cache | With cache |
|---|---|---|
| Compute per step | $O(t \cdot d)$ redundant projections | $O(d)$ new projections only |
| Memory | $O(d)$ (no state) | $O(L \cdot t \cdot d)$ (grows linearly) |
| Total compute for $N$ tokens | $O(N^2 \cdot d)$ | $O(N \cdot d)$ for projections |

Speed is dramatically better with the cache, but memory grows linearly with every generated token. For very long sequences or large batches, the cache becomes the dominant memory consumer, which motivates techniques like grouped-query attention (GQA), multi-query attention (MQA), and KV-cache quantization.

## Why it matters

KV-Caching is not optional in practice — it is the standard mechanism that makes autoregressive generation feasible at interactive speeds. Without it, generating a 1000-token response would recompute attention over all previous tokens at every step, making inference roughly 500x slower (on average). Every modern inference engine (vLLM, TensorRT-LLM, llama.cpp, etc.) implements KV-Caching as a core component. The memory pressure it creates is one of the main constraints on batch size and context length in production LLM serving.

## Used in

- [[Autoregressive Generation]]
- [[Self-Attention]]
- [[Quantization]]
