The mechanism by which each element in a sequence computes a weighted combination of all other elements, letting the model dynamically decide what context is relevant at every position. The core building block of all transformers.

## What is it?

Self-attention (also called Scaled Dot-Product Attention) is the operation that gives transformers their power. The central idea is simple: **each word looks at every other word to decide what is important for understanding meaning**. Unlike recurrent networks, which process tokens one by one and compress history into a fixed-size hidden state, self-attention lets every token directly interact with every other token in a single step.

The mechanism was introduced as part of the Transformer architecture (Vaswani et al., 2017, "Attention Is All You Need") and comes in two flavors that almost always appear together:

1. **Scaled Dot-Product Attention** -- the atomic operation that computes attention weights and applies them.
2. **Multi-Head Attention** -- running multiple independent attention operations in parallel on different learned projections, then combining the results. This lets the model attend to information from different representation subspaces at different positions simultaneously.

### The library analogy

Think of attention as a library lookup:

- **Query (Q)**: "What am I looking for?" -- each token formulates a question about what information it needs.
- **Key (K)**: "What does each book offer?" -- every token advertises a summary of what it contains.
- **Value (V)**: "The actual content of each book" -- the real information to be retrieved.

The attention score between a query and a key is their dot product: if the query and key point in similar directions, the score is high, meaning "this book is relevant to my question." The scores are normalized via softmax so they form a probability distribution, and the output is the weighted sum of values according to those probabilities.

## How it works

![[basics_self_attention.png]]

[🔗 Open interactive Self-Attention Explorer](../../interactive/self_attention.html)

### The core formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{N \times d_k}$ -- queries (one per token)
- $K \in \mathbb{R}^{N \times d_k}$ -- keys (one per token)
- $V \in \mathbb{R}^{N \times d_v}$ -- values (one per token)
- $d_k$ -- dimension of keys/queries
- $\sqrt{d_k}$ -- scaling factor to prevent dot products from growing large and pushing [[Softmax]] into regions with vanishing gradients

The matrices $Q$, $K$, $V$ are obtained by linearly projecting the input $X \in \mathbb{R}^{N \times d_{model}}$:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$ and $W^V \in \mathbb{R}^{d_{model} \times d_v}$ are learned parameter matrices.

### Multi-Head Attention

Instead of performing a single attention function with $d_{model}$-dimensional keys, values, and queries, multi-head attention projects them $h$ times with different learned projections, performs attention in parallel, and concatenates:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V)$$

Where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$, and typically $d_k = d_v = d_{model} / h$.

Each head can learn to attend to different types of relationships: one head might track syntactic dependencies, another semantic similarity, another coreference.

### Causal (masked) attention

In autoregressive models (decoders), each token must only attend to tokens at earlier positions. This is enforced by adding a mask to the attention scores before softmax:

$$\text{CausalAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is an upper-triangular matrix of $-\infty$ values:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

The $-\infty$ entries become zero after softmax, effectively preventing the model from "seeing the future."

### Pseudocode

#### (1) Single-head scaled dot-product attention

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (B, N, d_k)   -- queries
        K: (B, N, d_k)   -- keys
        V: (B, N, d_v)   -- values
        mask: optional (B, N, N) or (1, N, N) -- attention mask
    Returns:
        output: (B, N, d_v) -- weighted combination of values
        attn_weights: (B, N, N) -- attention probabilities
    """
    d_k = Q.shape[-1]

    # Compute raw attention scores
    scores = Q @ K.transpose(-2, -1)   # (B, N, N)
    scores = scores / sqrt(d_k)        # scale to prevent large magnitudes

    # Apply mask (e.g., causal mask or padding mask)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Normalize to probabilities
    attn_weights = softmax(scores, dim=-1)  # (B, N, N)

    # Weighted sum of values
    output = attn_weights @ V               # (B, N, d_v)
    return output, attn_weights
```

#### (2) Multi-head attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One projection per head (implemented as a single large matrix)
        self.W_q = Linear(d_model, d_model)  # projects to (num_heads * d_k)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)  # output projection

    def forward(self, X, mask=None):
        """
        Args:
            X: (B, N, d_model) -- input sequence
            mask: optional attention mask
        Returns:
            output: (B, N, d_model)
        """
        B, N, d_model = X.shape
        h = self.num_heads
        d_k = self.d_k

        # Step 1: Project input into Q, K, V
        Q = self.W_q(X)  # (B, N, d_model)
        K = self.W_k(X)
        V = self.W_v(X)

        # Step 2: Reshape into multiple heads
        # (B, N, d_model) -> (B, N, h, d_k) -> (B, h, N, d_k)
        Q = Q.reshape(B, N, h, d_k).transpose(1, 2)
        K = K.reshape(B, N, h, d_k).transpose(1, 2)
        V = V.reshape(B, N, h, d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention (per head, in parallel)
        # scores: (B, h, N, N)
        scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = softmax(scores, dim=-1)  # (B, h, N, N)
        attn_output = attn_weights @ V           # (B, h, N, d_k)

        # Step 4: Concatenate heads
        # (B, h, N, d_k) -> (B, N, h, d_k) -> (B, N, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, d_model)

        # Step 5: Final linear projection
        output = self.W_o(attn_output)  # (B, N, d_model)
        return output
```

#### (3) Causal (masked) self-attention

```python
class CausalSelfAttention(MultiHeadAttention):
    """Self-attention with a causal mask for autoregressive models."""

    def forward(self, X):
        B, N, d_model = X.shape

        # Build causal mask: lower-triangular matrix of ones
        # Position i can attend to positions 0..i (inclusive)
        causal_mask = torch.tril(torch.ones(N, N, device=X.device))
        # Shape: (1, 1, N, N) for broadcasting over (B, h, N, N)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Reuse multi-head attention with the causal mask
        return super().forward(X, mask=causal_mask)
```

### Step-by-step walkthrough

For a sequence of 4 tokens with $d_k = 3$:

```
Input X:  ["The", "cat", "sat", "down"]

1. Project:    Q = X @ W_q     K = X @ W_k     V = X @ W_v

2. Scores:     S = Q @ K^T     (4x4 matrix)
               S[i][j] = how much token i should attend to token j

3. Scale:      S = S / sqrt(3)

4. (Optional)  Apply causal mask to zero out future positions

5. Softmax:    A = softmax(S, dim=-1)    (each row sums to 1)

6. Output:     O = A @ V
               Each row of O is a weighted blend of all value vectors,
               weighted by how "relevant" each token is to the query token
```

### Complexity analysis

| Operation | Time | Memory |
|---|---|---|
| $QK^T$ computation | $O(N^2 \cdot d_k)$ | $O(N^2)$ for the score matrix |
| Softmax | $O(N^2)$ | $O(N^2)$ |
| Score $\times V$ | $O(N^2 \cdot d_v)$ | $O(N \cdot d_v)$ |
| **Total** | $O(N^2 \cdot d)$ | $O(N^2 + N \cdot d)$ |

The $O(N^2)$ term is the fundamental bottleneck. For a sequence of length 4096 with 32 attention heads, the score matrix alone requires $4096^2 \times 32 \times 4$ bytes $\approx$ 2 GB in float32. This quadratic scaling is why long-context transformers require techniques like [[Flash Attention]] to remain practical. It is also why there is active research into sub-quadratic attention mechanisms (linear attention, sparse attention, state-space models).

## Why it matters

Self-attention is the operation that replaced recurrence and convolution as the primary sequence modeling mechanism. Its advantages:

1. **Global receptive field in one layer**: every token can attend to every other token, regardless of distance. RNNs need $O(N)$ steps to propagate information across a sequence; attention does it in $O(1)$ layers.
2. **Parallelizable**: unlike RNNs, all positions are computed simultaneously, making full use of GPU parallelism during training.
3. **Interpretable**: attention weights can be inspected to understand which tokens the model considers relevant (though this should be done with care).
4. **Flexible**: the same mechanism supports bidirectional context (encoders), causal generation (decoders), and cross-modal conditioning ([[Cross-Attention]]).

The scaling factor $1/\sqrt{d_k}$ is a small but critical detail. Without it, when $d_k$ is large, dot products grow in magnitude, causing softmax to saturate (output nearly one-hot), which kills gradient flow and makes training unstable.

## Used in

Virtually every transformer-based paper in this vault, including but not limited to:

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]]
- [[OCR-free Document Understanding Transformer (Donut)]]

Related concepts: [[Flash Attention]], [[RoPE]], [[Cross-Attention]], [[Softmax]]
