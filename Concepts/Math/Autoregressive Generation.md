Token-by-token left-to-right sequence generation where each new token is conditioned on all previously generated tokens, forming the backbone of modern language models and seq2seq decoders.

## What is it?

Autoregressive generation is the process of producing a sequence one element at a time, where each new element is sampled from a probability distribution conditioned on all elements generated so far (and optionally on some input context). This left-to-right, token-by-token factorization is the standard way that language models (GPT, BART, T5) and seq2seq decoders generate text, code, or any discrete sequence.

The core idea is to decompose the joint probability of an entire sequence into a product of conditional probabilities using the chain rule. Rather than trying to predict the whole output at once, the model predicts one token, appends it to the context, then predicts the next token given the extended context, and so on until a special end-of-sequence token is produced or a maximum length is reached. This makes the generation problem tractable: at each step, the model only needs to produce a distribution over the vocabulary, not over all possible sequences.

During training, the model is typically trained with **teacher forcing**: instead of feeding its own predictions back as input, it receives the ground-truth previous tokens at every step. This allows fully parallel computation of all positions via causal masking. At inference time, however, generation is inherently sequential — each token must be produced before the next one can be predicted. Different decoding strategies (greedy, beam search, sampling with temperature, top-k, top-p/nucleus) control the trade-off between output quality, diversity, and computational cost.

## How it works

![[basics_autoregressive.png]]

[🔗 Open interactive Autoregressive Visualizer](../../interactive/autoregressive.html)

### The Autoregressive Factorization

Given an input $x$ (e.g., an encoded document image) and target sequence $y = (y_1, y_2, \ldots, y_T)$, the joint probability is factorized as:

$$P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)$$

Where $y_{<t} = (y_1, y_2, \ldots, y_{t-1})$ denotes all tokens generated before step $t$.

The training objective is to maximize the log-likelihood:

$$\mathcal{L} = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)$$

Which is equivalent to minimizing the cross-entropy loss at each position.

### Causal Masking

During training, all positions are computed in parallel using a causal (lower-triangular) attention mask that prevents position $t$ from attending to any position $t' > t$:

$$\text{CausalMask}_{t, t'} =
\begin{cases}
0 & \text{if } t' \leq t \\
-\infty & \text{if } t' > t
\end{cases}$$

This ensures that even though computation is parallel, each position only "sees" previous tokens — matching the autoregressive factorization.

### Teacher Forcing (Training)

During training, the decoder receives the ground-truth tokens as input at every position:

$$\hat{y}_t = \text{Decoder}(y_1, y_2, \ldots, y_{t-1}, x)$$

The model never sees its own predictions during training. This enables efficient parallel training (all time steps computed at once with causal masking) but can cause **exposure bias**: at inference time the model may encounter token sequences it never saw during training because it is consuming its own imperfect outputs.

### Decoding Strategies (Inference)

**Greedy decoding:** At each step, pick the highest-probability token:
$$y_t = \arg\max_{v \in \mathcal{V}} P(v \mid y_{<t}, x)$$

**Beam search:** Maintain $B$ candidate sequences (beams) and expand each by all vocabulary tokens, keeping the top-$B$ sequences by cumulative log-probability:
$$\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i \mid y_{<i}, x)$$

**Sampling with temperature:** Sample from a softened distribution:
$$P_\tau(y_t = v) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$
where $z_v$ are the logits and $\tau$ is the temperature. $\tau < 1$ sharpens (more greedy), $\tau > 1$ flattens (more random).

**Top-k sampling:** Restrict sampling to the $k$ highest-probability tokens.

**Top-p (nucleus) sampling:** Restrict to the smallest set of tokens whose cumulative probability exceeds $p$.

### Pseudocode: Greedy Autoregressive Decoding

```python
def greedy_decode(model, encoder_output, max_len, bos_token, eos_token):
    """
    Greedy autoregressive decoding for an encoder-decoder model.

    Args:
        model:          trained encoder-decoder (e.g., Donut, BART, T5)
        encoder_output: (1, T_enc, d_model) — encoded input (image, text, etc.)
        max_len:        maximum number of tokens to generate
        bos_token:      beginning-of-sequence token id
        eos_token:      end-of-sequence token id
    Returns:
        generated_ids: list of token ids
    """
    # Start with just the BOS token
    generated_ids = [bos_token]

    for step in range(max_len):
        # Prepare decoder input: all tokens generated so far
        decoder_input = tensor(generated_ids).unsqueeze(0)  # (1, t)

        # Forward pass through decoder
        # Cross-attention connects decoder to encoder_output
        logits = model.decoder(
            input_ids=decoder_input,
            encoder_output=encoder_output
        )  # (1, t, vocab_size)

        # Get logits for the LAST position only
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Greedy: pick the highest probability token
        next_token = argmax(next_token_logits)

        # Append to generated sequence
        generated_ids.append(next_token)

        # Stop if we generated EOS
        if next_token == eos_token:
            break

    return generated_ids
```

### Pseudocode: Beam Search Decoding

```python
def beam_search_decode(model, encoder_output, max_len, bos_token, eos_token,
                       beam_width=5):
    """
    Beam search: maintain top-B hypotheses at each step.

    Returns:
        best_sequence: list of token ids (highest-scoring complete hypothesis)
    """
    # Each beam: (log_prob, token_ids)
    beams = [(0.0, [bos_token])]
    completed = []

    for step in range(max_len):
        all_candidates = []

        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue

            decoder_input = tensor(seq).unsqueeze(0)
            logits = model.decoder(
                input_ids=decoder_input,
                encoder_output=encoder_output
            )
            log_probs = log_softmax(logits[0, -1, :])  # (vocab_size,)

            # Expand this beam by top-B tokens
            top_k_probs, top_k_ids = topk(log_probs, beam_width)

            for i in range(beam_width):
                new_score = score + top_k_probs[i].item()
                new_seq = seq + [top_k_ids[i].item()]
                all_candidates.append((new_score, new_seq))

        # Keep top-B candidates
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

        # Early stop: all beams have ended
        if all(seq[-1] == eos_token for _, seq in beams):
            completed.extend(beams)
            break

    # Return highest-scoring completed sequence
    completed.sort(key=lambda x: x[0], reverse=True)
    return completed[0][1] if completed else beams[0][1]
```

### Pseudocode: Teacher Forcing (Training)

```python
def train_step(model, encoder_output, target_ids):
    """
    One training step with teacher forcing.

    Args:
        encoder_output: (B, T_enc, d_model)
        target_ids:     (B, T) — ground truth token ids
    """
    # Shift target right: decoder input is [BOS, y1, y2, ..., y_{T-1}]
    decoder_input = target_ids[:, :-1]   # (B, T-1)
    # Labels are [y1, y2, ..., y_T] (the next token at each position)
    labels = target_ids[:, 1:]           # (B, T-1)

    # Forward with causal mask (applied internally)
    logits = model.decoder(
        input_ids=decoder_input,
        encoder_output=encoder_output
    )  # (B, T-1, vocab_size)

    # Cross-entropy loss at every position (parallelized, not sequential)
    loss = cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1)
    )

    loss.backward()
    optimizer.step()
```

## Why it matters

Autoregressive generation is the dominant paradigm for producing structured, variable-length outputs in deep learning. It provides a principled probabilistic framework (exact likelihood via the chain rule) and naturally handles sequences of any length. The factorization into per-step conditionals makes the problem tractable and allows the model to make fine-grained decisions at each position.

The approach was introduced in the context of neural sequence models (RNNs, then Transformers) to replace earlier approaches like CTC or non-autoregressive models that struggled with complex output dependencies. Autoregressive models excel at capturing long-range dependencies in the output — each token can depend on the entire history — which is critical for coherent text generation, structured output (JSON, XML, code), and tasks where output tokens are interdependent.

The main trade-off is inference speed: generation is inherently sequential (each token depends on the previous), so it scales linearly with output length. This has motivated research into speculative decoding, non-autoregressive models, and KV-cache optimization, but autoregressive generation remains the standard approach for its superior output quality and flexibility.

## Used in

- [[OCR-free Document Understanding Transformer (Donut)]]
