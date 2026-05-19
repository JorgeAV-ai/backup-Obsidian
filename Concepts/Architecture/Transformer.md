The dominant neural network architecture for sequence modeling, replacing recurrence with stacked layers of [[Self-Attention]] and feed-forward networks. The foundation of virtually all modern NLP and increasingly all of deep learning.

## What is it?

The Transformer was introduced in "Attention Is All You Need" (Vaswani et al., 2017) and has since become the default architecture for language models, vision models, speech models, and multi-modal systems. The key contribution was showing that a model built entirely from attention mechanisms and position-wise feed-forward networks -- with no recurrence or convolution -- could outperform RNN-based models on sequence-to-sequence tasks while being far more parallelizable.

The original Transformer is an **encoder-decoder** architecture, but in practice three variants dominate:

| Variant | Structure | Masking | Typical use | Examples |
|---|---|---|---|---|
| **Encoder-only** | Stack of encoder layers | Bidirectional (no mask) | Classification, retrieval, representation learning | BERT, [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)\|ModernBERT]] |
| **Decoder-only** | Stack of decoder layers | Causal mask | Text generation, language modeling | GPT, LLaMA, Mistral |
| **Encoder-Decoder** | Encoder + decoder with cross-attention | Encoder: bidirectional; Decoder: causal + cross-attn | Translation, document understanding, summarization | Original Transformer, T5, [[OCR-free Document Understanding Transformer (Donut)\|Donut]] |

## How it works

![[basics_transformer.png]]

[🔗 Open interactive Transformer Visualizer](../interactive/transformer.html)

### Full architecture overview

```
Input tokens
     |
     v
[Token Embedding] + [Positional Encoding]
     |
     v
 +-----------------------------------------+
 |         Encoder (x N layers)            |
 |                                         |
 |  +-----------------------------------+  |
 |  | Multi-Head Self-Attention         |  |
 |  +-----------------------------------+  |
 |                  |                      |
 |            Add & LayerNorm              |
 |                  |                      |
 |  +-----------------------------------+  |
 |  | Feed-Forward Network (FFN)        |  |
 |  +-----------------------------------+  |
 |                  |                      |
 |            Add & LayerNorm              |
 +-----------------------------------------+
     |
     | encoder output
     v
 +-----------------------------------------+
 |         Decoder (x N layers)            |
 |                                         |
 |  +-----------------------------------+  |
 |  | Masked Multi-Head Self-Attention  |  |
 |  +-----------------------------------+  |
 |                  |                      |
 |            Add & LayerNorm              |
 |                  |                      |
 |  +-----------------------------------+  |
 |  | Multi-Head Cross-Attention        |  |
 |  |   Q: from decoder                 |  |
 |  |   K, V: from encoder output       |  |
 |  +-----------------------------------+  |
 |                  |                      |
 |            Add & LayerNorm              |
 |                  |                      |
 |  +-----------------------------------+  |
 |  | Feed-Forward Network (FFN)        |  |
 |  +-----------------------------------+  |
 |                  |                      |
 |            Add & LayerNorm              |
 +-----------------------------------------+
     |
     v
[Linear + Softmax] -> output probabilities
```

### Why each component exists

| Component | Purpose |
|---|---|
| **Token Embedding** | Maps discrete token IDs to dense vectors in $\mathbb{R}^{d_{model}}$ |
| **Positional Encoding** | Injects position information (since attention is permutation-invariant). Original paper used fixed sinusoidal encodings; modern models use [[RoPE]] or learned embeddings |
| **Self-Attention** | Lets each token aggregate information from all other tokens. The core mechanism for capturing dependencies regardless of distance |
| **Residual Connections** | $\text{output} = x + \text{sublayer}(x)$. Enables gradient flow through deep stacks. Without them, gradients vanish in networks with 12+ layers. See [[Residual Connections]] |
| **[[Layer Normalization]]** | Normalizes activations to zero mean and unit variance per token. Stabilizes training by reducing internal covariate shift |
| **Feed-Forward Network (FFN)** | Two-layer MLP applied independently to each token: $\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$. This is where per-token non-linear transformation happens. Attention mixes information across tokens; FFN transforms each token's representation |
| **Causal Mask** | Prevents decoder tokens from attending to future positions, preserving the autoregressive property |
| **Cross-Attention** | Lets the decoder query the encoder output. See [[Cross-Attention]] |

### Pseudocode

#### (1) Transformer Encoder Layer

```python
class TransformerEncoderLayer:
    """One layer of the Transformer encoder."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # Sub-layer 1: Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        # Sub-layer 2: Position-wise feed-forward network
        self.ffn = Sequential(
            Linear(d_model, d_ff),    # expand: d_model -> d_ff (typically 4x)
            GELU(),                    # non-linearity (original used ReLU)
            Linear(d_ff, d_model),    # project back: d_ff -> d_model
        )
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (B, N, d_model) -- input sequence
            padding_mask: (B, 1, 1, N) -- mask for padding tokens
        Returns:
            out: (B, N, d_model)
        """
        # Sub-layer 1: Self-attention + residual + norm
        # Post-norm variant (original Transformer):
        attn_out = self.self_attn(x, x, x, mask=padding_mask)  # Q=K=V=x
        x = self.norm1(x + self.dropout1(attn_out))

        # Sub-layer 2: FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x
```

#### (2) Transformer Decoder Layer (with causal mask + cross-attention)

```python
class TransformerDecoderLayer:
    """One layer of the Transformer decoder."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # Sub-layer 1: Masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        # Sub-layer 2: Multi-head cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

        # Sub-layer 3: Position-wise feed-forward network
        self.ffn = Sequential(
            Linear(d_model, d_ff),
            GELU(),
            Linear(d_ff, d_model),
        )
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)

    def forward(self, x, encoder_output, causal_mask, encoder_mask=None):
        """
        Args:
            x: (B, T_dec, d_model) -- decoder input
            encoder_output: (B, T_enc, d_model) -- encoder output
            causal_mask: (1, 1, T_dec, T_dec) -- lower-triangular mask
            encoder_mask: (B, 1, 1, T_enc) -- padding mask for encoder
        Returns:
            out: (B, T_dec, d_model)
        """
        # Sub-layer 1: Masked self-attention (causal)
        # Decoder tokens can only attend to previous decoder tokens
        self_attn_out = self.self_attn(x, x, x, mask=causal_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Sub-layer 2: Cross-attention
        # Q from decoder, K and V from encoder
        cross_attn_out = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Sub-layer 3: Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x
```

#### (3) Full Transformer forward pass

```python
class Transformer:
    """Full encoder-decoder Transformer."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_encoder_layers, num_decoder_layers, max_seq_len):
        # Embeddings
        self.token_embed = Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        # Or: self.pos_encoding = RoPE(d_model, max_seq_len)

        self.embed_scale = sqrt(d_model)  # scale embeddings (Vaswani et al.)

        # Encoder stack
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_encoder_layers)
        ]

        # Decoder stack
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_decoder_layers)
        ]

        # Output head
        self.output_proj = Linear(d_model, vocab_size)

    def encode(self, src_tokens, src_mask):
        """
        Args:
            src_tokens: (B, T_src) -- source token IDs
            src_mask: (B, 1, 1, T_src) -- padding mask
        Returns:
            encoder_output: (B, T_src, d_model)
        """
        # Step 1: Token embedding + positional encoding
        x = self.token_embed(src_tokens) * self.embed_scale
        x = x + self.pos_encoding(x)

        # Step 2: Pass through N encoder layers
        for layer in self.encoder_layers:
            x = layer(x, padding_mask=src_mask)

        return x  # (B, T_src, d_model)

    def decode(self, tgt_tokens, encoder_output, src_mask):
        """
        Args:
            tgt_tokens: (B, T_tgt) -- target token IDs (shifted right)
            encoder_output: (B, T_src, d_model)
            src_mask: (B, 1, 1, T_src) -- encoder padding mask
        Returns:
            logits: (B, T_tgt, vocab_size)
        """
        T_tgt = tgt_tokens.shape[1]

        # Step 1: Token embedding + positional encoding
        x = self.token_embed(tgt_tokens) * self.embed_scale
        x = x + self.pos_encoding(x)

        # Step 2: Build causal mask
        # Lower-triangular: position i can attend to positions 0..i
        causal_mask = torch.tril(torch.ones(T_tgt, T_tgt))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Step 3: Pass through N decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output=encoder_output,
                causal_mask=causal_mask,
                encoder_mask=src_mask,
            )

        # Step 4: Project to vocabulary
        logits = self.output_proj(x)  # (B, T_tgt, vocab_size)
        return logits

    def forward(self, src_tokens, tgt_tokens, src_mask=None):
        """Full forward pass: encode source, decode target."""
        encoder_output = self.encode(src_tokens, src_mask)
        logits = self.decode(tgt_tokens, encoder_output, src_mask)
        return logits

    # At inference time, tokens are generated one at a time:
    def generate(self, src_tokens, src_mask, max_len, eos_token):
        """Autoregressive greedy decoding."""
        encoder_output = self.encode(src_tokens, src_mask)

        # Start with <BOS> token
        generated = [BOS_TOKEN]

        for step in range(max_len):
            tgt = tensor(generated).unsqueeze(0)       # (1, t)
            logits = self.decode(tgt, encoder_output, src_mask)
            next_token = logits[0, -1, :].argmax()     # greedy
            generated.append(next_token)
            if next_token == eos_token:
                break

        return generated
```

### Pre-Norm vs. Post-Norm

The original Transformer uses **post-norm**: $\text{LayerNorm}(x + \text{sublayer}(x))$. Most modern implementations use **pre-norm**: $x + \text{sublayer}(\text{LayerNorm}(x))$. Pre-norm is easier to train (more stable gradients at initialization) and generally does not require careful learning rate warmup. The pseudocode above shows post-norm to match the original paper; switching to pre-norm simply means moving the LayerNorm call before the sublayer.

### Decoder-only simplification

For decoder-only models (GPT, LLaMA), the architecture is simpler -- there is no encoder and no cross-attention. Each layer is just:

```
x -> LayerNorm -> Causal Self-Attention -> + residual
  -> LayerNorm -> FFN                    -> + residual
```

This is the architecture used by the majority of modern large language models.

### Key hyperparameters (original Transformer)

| Parameter | Base model | Big model |
|---|---|---|
| $d_{model}$ | 512 | 1024 |
| $h$ (heads) | 8 | 16 |
| $d_k = d_v = d_{model}/h$ | 64 | 64 |
| $d_{ff}$ | 2048 | 4096 |
| $N$ (layers) | 6 | 6 |
| Parameters | 65M | 213M |

Modern LLMs scale these dramatically (LLaMA-70B: $d_{model}=8192$, 64 heads, 80 layers).

## Why it matters

The Transformer solved three fundamental limitations of recurrent architectures:

1. **Parallelism**: RNNs process tokens sequentially -- token $t$ depends on the hidden state from token $t-1$. This makes training slow because you cannot parallelize across time steps. Transformers compute all positions simultaneously, fully utilizing modern GPU hardware.

2. **Long-range dependencies**: In RNNs, information from early tokens must survive through many sequential processing steps to reach later tokens, and gradients must flow back through all those steps. In practice, this means RNNs struggle with dependencies spanning more than ~100 tokens. In a Transformer, any two tokens interact directly through attention in a single layer, regardless of their distance.

3. **Gradient flow**: The combination of residual connections and layer normalization creates a "gradient highway" through the network. Each sublayer adds its contribution to the residual stream rather than replacing it. This is why transformers can be stacked to 100+ layers while RNNs rarely exceeded 2-4 layers.

The Transformer's scalability has driven the "scaling laws" era: performance improves predictably with more parameters, more data, and more compute. This property, more than any architectural detail, is why transformers dominate modern AI.

## Used in

Every paper in this vault is either a Transformer model or builds on top of the Transformer architecture, including:

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)]] (encoder-only)
- [[OCR-free Document Understanding Transformer (Donut)]] (encoder-decoder)

Related concepts: [[Self-Attention]], [[Layer Normalization]], [[Residual Connections]], [[Embeddings]], [[RoPE]]
