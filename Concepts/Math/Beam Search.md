# Beam Search

> **TL;DR:** A decoding strategy that keeps the top-$k$ most probable partial sequences at each step, balancing quality and computational cost between greedy decoding and exhaustive search.

---

## What is it?

When a language model generates text, it produces a probability distribution over the vocabulary at each step. The challenge: how do you pick the sequence of tokens that forms the best overall output? Greedy decoding takes the most probable token at each step, but this is locally optimal and can miss globally better sequences. Exhaustive search over all possible sequences is intractable (exponential in sequence length).

Beam search is the practical middle ground. It maintains $k$ candidates (called **beams**) at each step, expanding each beam with all possible next tokens, scoring the results, and keeping only the top $k$. This explores a broader search space than greedy decoding without the exponential cost of full search.

Beam search dominated machine translation and summarization for years. For creative or open-ended generation, sampling methods (top-k, top-p, temperature) are preferred because beam search tends to produce repetitive and "safe" text.

---

## How it works

![[basics_beam_search.png]]

[🔗 Open interactive Beam Search Explorer](../interactive/beam_search.html)

### Pseudocode: greedy decoding

```python
def greedy_decode(model, prompt, max_len):
    """Always pick the single most probable next token."""
    tokens = prompt
    for _ in range(max_len):
        logits = model(tokens)             # shape: (vocab_size,)
        next_token = argmax(logits)
        tokens = tokens + [next_token]
        if next_token == EOS:
            break
    return tokens
```

### Pseudocode: beam search

```python
def beam_search(model, prompt, max_len, k=5):
    """
    Keep top-k partial sequences at each step.
    k: beam width (number of candidates to maintain)
    """
    # Each beam: (token_sequence, cumulative_log_prob)
    beams = [(prompt, 0.0)]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for seq, score in beams:
            if seq[-1] == EOS:
                completed.append((seq, score))
                continue

            logits = model(seq)
            log_probs = log_softmax(logits)

            # Expand this beam with every possible next token
            for token_id in range(vocab_size):
                new_seq = seq + [token_id]
                new_score = score + log_probs[token_id]
                all_candidates.append((new_seq, new_score))

        # Keep only top-k candidates
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:k]

        if len(beams) == 0:
            break

    # Return best completed sequence (or best beam if none completed)
    all_results = completed + beams
    # Apply length normalization to avoid bias toward short sequences
    best = max(all_results, key=lambda x: x[1] / len(x[0]) ** alpha)
    return best[0]
```

### Length normalization

Beam search sums log-probabilities, so longer sequences accumulate more negative scores and are penalized. Length normalization fixes this:

$$\text{score}(y) = \frac{\log P(y|x)}{|y|^\alpha}$$

where $\alpha \in [0, 1]$ controls the strength ($\alpha = 0$: no normalization, $\alpha = 1$: full normalization by length). Typical value: $\alpha = 0.6$.

### Sampling methods (alternatives to beam search)

**Temperature scaling:**

$$P(x_i) = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

Low temperature ($\tau < 1$): sharper, more deterministic. High temperature ($\tau > 1$): flatter, more random.

**Top-k sampling:**

```python
def top_k_sample(logits, k=50, temperature=1.0):
    logits = logits / temperature
    top_k_logits, top_k_indices = topk(logits, k)
    probs = softmax(top_k_logits)
    chosen = sample(top_k_indices, probs)    # random sample from top-k
    return chosen
```

Keep only the $k$ most probable tokens, zero out the rest, renormalize, and sample.

**Top-p (nucleus) sampling:**

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_probs = sort(softmax(logits), descending=True)
    cumulative = cumsum(sorted_probs)
    # Find smallest set of tokens whose cumulative prob >= p
    cutoff_index = first_index_where(cumulative >= p)
    # Zero out everything below cutoff, renormalize, sample
    probs = softmax(logits, mask=top_tokens_only)
    return sample(probs)
```

Instead of a fixed $k$, keep the smallest set of tokens whose cumulative probability exceeds $p$. Adapts to the shape of the distribution -- narrow distributions use fewer tokens, broad distributions use more.

### Comparison

| Method | Quality | Speed | Diversity | Best for |
|---|---|---|---|---|
| **Greedy** | Suboptimal | Fast (1x) | None | Quick prototyping |
| **Beam search** ($k$) | High | $k$x slower | Low | Translation, summarization |
| **Top-k sampling** | Variable | ~1x | High | Creative generation |
| **Top-p sampling** | Variable | ~1x | High | Open-ended generation |
| **Temperature** | Depends on $\tau$ | ~1x | Controllable | Combined with sampling |

---

## Why it matters

- **Quality vs cost trade-off**: Beam search provides a principled way to explore the output space more thoroughly than greedy, at a predictable computational cost.
- **Dominant in structured tasks**: For tasks where there is a single "correct" output (translation, summarization), beam search consistently outperforms greedy decoding.
- **Sampling for creativity**: For chatbots and creative writing, sampling methods produce more diverse and natural-sounding text. Beam search tends to produce generic, repetitive outputs in these settings.
- **Understanding the trade-offs**: Choosing the right decoding strategy is as important as choosing the right model. A great model with bad decoding will underperform.

---

## Used in

- Machine translation (beam search is the standard)
- Text summarization (beam search with length penalty)
- Speech recognition (beam search over token lattices)
- ChatGPT / Claude / open-source LLMs (top-p sampling for conversational generation)

---

**See also:** [[Autoregressive Generation]], [[Softmax]]
