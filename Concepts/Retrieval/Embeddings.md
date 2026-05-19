# Embeddings

> **TL;DR:** Embeddings convert discrete tokens (words, subwords, pixels) into dense, continuous vectors that capture semantic meaning. They are the entry point of every neural model that processes discrete data.

---

## What is it?

An embedding is a learned mapping from a finite vocabulary of discrete symbols to a continuous vector space. Instead of representing a word as a sparse one-hot vector of size $V$ (vocabulary size), we look up a dense vector of size $d$ (embedding dimension) from a trainable matrix.

The classic intuition: in a well-trained embedding space, **"king" - "man" + "woman" $\approx$ "queen"**. This shows that embeddings encode semantic relationships as geometric directions --- gender, tense, plurality, and more all correspond to consistent offsets in the vector space.

### Token embeddings vs Positional embeddings

- **Token embeddings** encode *what* the token is (its identity/meaning).
- **Positional embeddings** encode *where* the token sits in the sequence (its position).

In a Transformer, both are combined (usually added) before the first layer:

$$h_0 = W_E[\text{token}_i] + P[\text{pos}_i]$$

where $P$ can be learned (GPT-2) or computed via sinusoidal functions (original Transformer) or rotary methods ([[RoPE]]).

---

## How it works

![[basics_embeddings.png]]

[🔗 Open interactive Embeddings Explorer](../../interactive/embeddings.html)

### Formula

Given a vocabulary of size $V$ and embedding dimension $d$:

$$e_i = W_E[i] \quad \text{where } W_E \in \mathbb{R}^{V \times d}$$

This is just a table lookup --- row $i$ of the matrix $W_E$ is the embedding for token $i$.

### From one-hot to dense vector

Mathematically, if $x$ is a one-hot vector:

$$e = W_E^\top x$$

But in practice, we never multiply by a one-hot vector. We just index into the matrix directly (much faster).

### Pseudocode

```python
class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = Matrix(vocab_size, embed_dim)   # W_E
        self.pos_embed   = Matrix(max_seq_len, embed_dim)  # P
        self.scale = sqrt(embed_dim)  # scaling factor (used in original Transformer)

    def forward(self, token_ids):
        # token_ids: [batch_size, seq_len] of integers in [0, vocab_size)
        tok_emb = self.token_embed[token_ids]   # lookup, shape [B, S, d]
        pos_emb = self.pos_embed[:seq_len]      # positional, shape [S, d]

        # Scale token embeddings (as in Vaswani et al., 2017)
        x = tok_emb * self.scale + pos_emb
        return x  # [B, S, d]
```

The scaling by $\sqrt{d}$ prevents the token embeddings from being too small relative to the positional embeddings when $d$ is large.

---

## Why it matters

- **Dimensionality reduction**: A vocabulary of 50,000 tokens would require 50,000-dimensional one-hot vectors. Embeddings compress this to a few hundred dimensions while preserving (and learning) semantic structure.
- **Semantic similarity**: Similar words end up close together in the embedding space, enabling generalization.
- **Foundation of all sequence models**: Every Transformer, RNN, or CNN operating on text starts with an embedding layer. The quality of the embedding directly affects everything downstream.
- **Transfer learning**: Pre-trained embeddings (Word2Vec, GloVe, or the embedding layer of a pre-trained LLM) carry knowledge that transfers to new tasks.

---

## Used in

Virtually all papers that process discrete tokens:
- **Original Transformer** (Vaswani et al., 2017) --- token + sinusoidal positional embeddings
- **GPT / GPT-2 / GPT-3** --- learned token + learned positional embeddings
- **BERT** --- token + position + segment embeddings
- **Vision Transformers (ViT)** --- patch embeddings (linear projection of image patches) + positional embeddings
- **CLIP, DALL-E** --- embeddings for both text tokens and image patches

---

**See also:** [[Tokenization (BPE)]], [[RoPE]], [[Self-Attention]]
