The attention mechanism that connects an encoder and a decoder by letting the decoder query information from the encoder's output, enabling conditional generation.

## What is it?

Cross-attention is a variant of the attention mechanism where the queries come from one sequence (typically the decoder) and the keys and values come from a different sequence (typically the encoder). This is the fundamental mechanism that allows a decoder to "look at" and selectively extract information from an encoded input while generating its output token by token.

The concept originates from the original Transformer architecture (Vaswani et al., 2017), where it appears in every decoder layer as the "encoder-decoder attention" sublayer. In a machine translation model, for instance, the decoder generating French tokens uses cross-attention to attend to the English source sentence encoded by the encoder. At each generation step, the decoder decides which parts of the input are most relevant to producing the next output token.

The key difference from self-attention is the source of Q, K, and V. In self-attention, all three are derived from the same sequence — every token attends to every other token in the same sequence. In cross-attention, Q comes from the decoder (the sequence being generated) while K and V come from the encoder (the conditioning input). This asymmetry is what enables one modality or sequence to condition on another: a text decoder can attend to image features, a language model can attend to a document encoding, or an audio model can attend to text prompts.

## How it works

![[basics_cross_attention.png]]

### The Cross-Attention Formula

Given decoder hidden states $H_{dec} \in \mathbb{R}^{T_{dec} \times d_{model}}$ and encoder output $H_{enc} \in \mathbb{R}^{T_{enc} \times d_{model}}$:

$$Q = H_{dec} \cdot W^Q, \quad K = H_{enc} \cdot W^K, \quad V = H_{enc} \cdot W^V$$

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q_{dec} \, K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

Where:
- $W^Q \in \mathbb{R}^{d_{model} \times d_k}$ projects decoder states into queries
- $W^K, W^V \in \mathbb{R}^{d_{model} \times d_k}$ project encoder states into keys and values
- $d_k$ is the dimension per attention head
- The attention matrix has shape $(T_{dec} \times T_{enc})$ — each decoder position attends over all encoder positions

### Multi-Head Cross-Attention

In practice, cross-attention is performed with multiple heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{CrossAttn}(Q W_i^Q, \; K W_i^K, \; V W_i^V)$$

### Self-Attention vs. Cross-Attention

| Property          | Self-Attention              | Cross-Attention                  |
|-------------------|-----------------------------|----------------------------------|
| Q source          | Same sequence               | Decoder sequence                 |
| K, V source       | Same sequence               | Encoder sequence                 |
| Attention matrix  | $(T \times T)$              | $(T_{dec} \times T_{enc})$       |
| Purpose           | Contextualize within a seq  | Bridge two different sequences   |
| Causal masking    | Often (in decoders)         | Never (decoder sees full input)  |

### Where Cross-Attention Sits in a Decoder Layer

A standard Transformer decoder layer has three sublayers:

1. **Masked self-attention** — decoder tokens attend to previous decoder tokens (causal)
2. **Cross-attention** — decoder tokens attend to encoder output
3. **Feed-forward network** — position-wise MLP

Each sublayer is wrapped in residual connections and layer normalization.

### Pseudocode: Cross-Attention Layer

```python
class CrossAttentionLayer:
    """
    Cross-attention sublayer within a Transformer decoder block.
    Q from decoder, K and V from encoder.
    """
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Separate projections: Q from decoder, K/V from encoder
        self.W_q = Linear(d_model, d_model)  # projects decoder states
        self.W_k = Linear(d_model, d_model)  # projects encoder states
        self.W_v = Linear(d_model, d_model)  # projects encoder states
        self.W_o = Linear(d_model, d_model)  # output projection
        self.layer_norm = LayerNorm(d_model)

    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        Args:
            decoder_hidden: (B, T_dec, d_model) — current decoder representations
            encoder_output: (B, T_enc, d_model) — final encoder representations
            encoder_mask:   (B, 1, 1, T_enc) — mask for padding in encoder
        Returns:
            out: (B, T_dec, d_model) — decoder states enriched with encoder info
        """
        residual = decoder_hidden
        x = self.layer_norm(decoder_hidden)

        B, T_dec, d_model = x.shape
        _, T_enc, _ = encoder_output.shape
        h = self.num_heads
        d_k = self.d_k

        # Project Q from decoder, K and V from encoder
        Q = self.W_q(x).reshape(B, T_dec, h, d_k).transpose(1, 2)
        #   -> (B, h, T_dec, d_k)
        K = self.W_k(encoder_output).reshape(B, T_enc, h, d_k).transpose(1, 2)
        #   -> (B, h, T_enc, d_k)
        V = self.W_v(encoder_output).reshape(B, T_enc, h, d_k).transpose(1, 2)
        #   -> (B, h, T_enc, d_k)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
        #   -> (B, h, T_dec, T_enc)

        # Apply encoder padding mask (if any)
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, float('-inf'))

        attn_weights = softmax(scores, dim=-1)  # (B, h, T_dec, T_enc)
        # NOTE: no causal mask here — decoder can see ALL encoder positions

        attn_output = attn_weights @ V           # (B, h, T_dec, d_k)
        attn_output = attn_output.transpose(1, 2).reshape(B, T_dec, d_model)

        out = self.W_o(attn_output)
        out = residual + out  # residual connection
        return out


class TransformerDecoderLayer:
    """Full decoder layer showing where cross-attention fits."""

    def forward(self, decoder_hidden, encoder_output):
        # Step 1: Masked self-attention (causal — decoder attends to itself)
        x = self.self_attn(decoder_hidden, decoder_hidden, decoder_hidden,
                           causal_mask=True)

        # Step 2: Cross-attention (decoder queries encoder)
        x = self.cross_attn(
            decoder_hidden=x,
            encoder_output=encoder_output
        )

        # Step 3: Feed-forward network
        x = self.ffn(x)

        return x
```

## Why it matters

Cross-attention is the bridge that makes encoder-decoder architectures work. Without it, the decoder would have no way to condition its generation on the input — it would simply be an unconditional language model. Cross-attention solves the problem of how to selectively route information from an arbitrary-length input into each step of the output generation process.

This mechanism is essential whenever two different representations need to interact: source language to target language in translation, image features to text tokens in captioning, document images to OCR text in document understanding, audio spectrograms to text in speech recognition, and conditioning signals to generated content in diffusion models. The soft, learned, differentiable nature of attention weights means the model discovers on its own which parts of the input are relevant at each generation step — no hard alignment or manual feature engineering required.

## Used in

- [[OCR-free Document Understanding Transformer (Donut)]]
