Byte Pair Encoding (BPE) turns raw text into a sequence of integer token IDs by iteratively merging the most frequent character pairs into subword units, giving models a fixed vocabulary that handles rare and unseen words gracefully.

## What is it?

Before a language model can process text, every string must be converted into a sequence of integers from a fixed vocabulary. **Tokenization** is this conversion step, and **Byte Pair Encoding** (Sennrich et al., 2016) is the dominant algorithm for building the vocabulary and performing the segmentation.

BPE starts from individual characters (or bytes) and greedily merges the most frequent adjacent pair into a new token, repeating until the vocabulary reaches a target size. The result is a **subword** vocabulary: common words like "the" become single tokens, while rare words are split into recognizable pieces. For example:

- "unhappiness" -> ["un", "happiness"] (meaningful morphemes)
- "transformers" -> ["transform", "ers"]

This is a sweet spot between word-level tokenization (huge vocabulary, cannot handle unseen words) and character-level tokenization (tiny vocabulary, very long sequences, harder to learn meaning):

| Approach | Vocabulary size | Sequence length | OOV handling |
|---|---|---|---|
| Word-level | 100K+ | Short | Cannot handle unseen words |
| Character-level | ~256 | Very long | Handles everything, but slow |
| **Subword (BPE)** | 30K-100K | Moderate | Splits rare words into known pieces |

## How it works

![[basics_tokenization.png]]

[🔗 Open interactive Tokenization Explorer](../../interactive/tokenization.html)

### BPE training: building the vocabulary

Starting from a corpus, the algorithm proceeds as follows:

1. **Initialize** the vocabulary with all individual characters (or bytes) found in the training data.
2. **Count** all adjacent token pairs across the corpus.
3. **Merge** the most frequent pair into a single new token and add it to the vocabulary.
4. **Repeat** steps 2-3 until the vocabulary reaches the desired size.

### Concrete example: training on "lowest"

Start with character-level tokenization of the word "lowest" (appearing many times in the corpus):

```
Step 0: ["l", "o", "w", "e", "s", "t"]
```

Suppose the most frequent pairs across the whole corpus are:

```
Step 1: merge ("l", "o") -> "lo"
         ["lo", "w", "e", "s", "t"]

Step 2: merge ("lo", "w") -> "low"
         ["low", "e", "s", "t"]

Step 3: merge ("e", "s") -> "es"
         ["low", "es", "t"]

Step 4: merge ("es", "t") -> "est"
         ["low", "est"]
```

The word "lowest" is now represented as two subword tokens: `["low", "est"]`. The suffix "est" will also tokenize "highest", "fastest", etc., so the model learns shared morphology.

### Tokenization at inference time

Given a trained vocabulary and merge rules, encoding new text applies the learned merges greedily in priority order (most frequent merges first). For example, a sentence like:

```
"Machine learning is amazing"
```

might be tokenized into token IDs (example from a GPT-style tokenizer):

```
[15496, 4673, 374, 8056]
```

Each ID maps to a subword entry in the vocabulary. Common words map to a single token; rare words are broken into multiple subword pieces.

### Pseudocode: BPE training algorithm

```python
def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    corpus: list of words (pre-tokenized by whitespace)
    vocab_size: target vocabulary size
    Returns: merge_rules (ordered list of pair merges)
    """
    # Step 1: Initialize -- split every word into characters
    # Each word is a tuple of symbols, with frequency count
    word_freqs = {}
    for word in corpus:
        symbols = tuple(word) + ("</w>",)  # end-of-word marker
        word_freqs[symbols] = word_freqs.get(symbols, 0) + 1

    # Initial vocabulary: all unique characters
    vocab = set()
    for symbols in word_freqs:
        vocab.update(symbols)

    merge_rules = []

    while len(vocab) < vocab_size:
        # Step 2: Count all adjacent pairs
        pair_counts = {}
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            break

        # Step 3: Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merge_rules.append(best_pair)

        # Merge that pair everywhere
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)

        new_word_freqs = {}
        for symbols, freq in word_freqs.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                    symbols[i] == best_pair[0] and
                    symbols[i + 1] == best_pair[1]):
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_word_freqs[tuple(new_symbols)] = freq
        word_freqs = new_word_freqs

    return merge_rules, vocab
```

### Variants: WordPiece, SentencePiece, and BPE

| Method | Merge criterion | Pre-tokenization | Used by |
|---|---|---|---|
| **BPE** | Most frequent pair | Whitespace + rules | GPT-2, GPT-3/4, LLaMA, OLMo |
| **WordPiece** | Pair that maximizes likelihood ($\frac{P(ab)}{P(a)P(b)}$) | Whitespace + rules | BERT, DistilBERT |
| **SentencePiece** | BPE or Unigram on raw bytes (no pre-tokenization) | None (language-agnostic) | T5, mT5, LLaMA (via SP-BPE) |

- **BPE** uses raw frequency counts to choose merges. Simple and fast.
- **WordPiece** uses a likelihood-based score, preferring merges where the combined token is much more common than its parts independently. This tends to produce slightly different vocabularies.
- **SentencePiece** (Kudo & Richardson, 2018) removes the dependency on language-specific pre-tokenization (whitespace splitting) by treating the input as a raw byte stream. It can apply either BPE or a Unigram language model internally.

### ModernBERT tokenizer

[[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)|ModernBERT]] uses the OLMo tokenizer, which is a BPE-based tokenizer with a vocabulary of **50,368 tokens**. This is notably larger than the original BERT WordPiece vocabulary (30,522 tokens), giving the model finer-grained subword representations and better coverage of code, multilingual text, and technical terminology.

## Why it matters

Tokenization is the first and last step of every NLP pipeline: the quality of the vocabulary directly affects model performance, sequence length efficiency, and the ability to handle multilingual or domain-specific text. BPE provides a principled, data-driven way to build a vocabulary that balances coverage with compactness. A poorly chosen tokenizer can fragment important words into too many pieces (wasting context length) or merge unrelated characters (obscuring meaning). Understanding tokenization is essential for diagnosing model behavior -- when a model fails on a rare word, the first question is always "how was it tokenized?"

## Used in

- [[A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT)|ModernBERT]] (OLMo tokenizer, 50,368 BPE vocab)
- [[Hypothetical Document Embeddings (HyDE)|HyDE]] (inherits the tokenizer of its backbone model)
- [[Embeddings]] (tokenization is the step before embedding lookup)
