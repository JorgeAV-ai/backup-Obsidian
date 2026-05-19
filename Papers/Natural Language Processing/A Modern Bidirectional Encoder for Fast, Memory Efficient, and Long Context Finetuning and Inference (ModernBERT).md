Modernized BERT encoder with decoder-only LLM efficiency techniques: alternating attention, Flash Attention, unpadding, and 8192 context length.

> [!quote] Information
> * @ Conference NONE
> * paper Paper [Link](https://arxiv.org/pdf/2412.13663)
> * git Github [Link](https://github.com/AnswerDotAI/ModernBERT)
> * hf Huggingface [Link](https://huggingface.co/papers/2412.13663)
> *  tag Tags
> 	[[Encoder-Only Transformers]]
> 	[[Towards Enhanced Efficiency]]
> 	[[Self-Attention mechanisms]]
> * calendar Date 19 December 2024
> * ? Motivation:

> 		Nowadays the so called GenAI models, such as LLama, ChatGPT, etc. Are models with an insane quantity of parameters. The purpose of the paper is improve the original BERT, adding latest optimizations, thus making it possible to obtain a better version, more scalable, long context and local-global attention.
> *  Dataset Datasets:
> 	[[GLUE]]
> 	[[BEIR]]
> 	[[MS-MARCO]]
> 	[[MLDR]]
> 	[[CodeSearchNet]]
> 	[[StackOverflow-QA]]

### 1. Introduction
#### 1.1 Background
We all remember the dominance  of BERT in the vast majority of NLP tasks before the advent of LLMs. During these years of LLMs (encoder-decoder) improvements, many new techniques dropped with the idea of improving efficiency and long context, but only some of them have been applied to only encoder. In particular, modern efficiency techniques from decoder-only LLMs such as Flash Attention, unpadding, and alternating attention have shown strong results but had not yet been adapted to encoder-only architectures. Additionally, including code data in the training mixture and extending context length post-training by adjusting RoPE theta parameters were hypothesized to benefit encoder models, following trends observed in the decoder-only setting.
#### 1.2 Objectives
This paper improves the basic BERT with some changes to improve its efficiency and longer context. The goal is to create the strongest encoder-only model at base and large sizes, incorporating modern training and architectural techniques that have only been applied to decoder-only models so far.
#### 1.3 What's New
* First encoder-only model to incorporate modern efficiency techniques from decoder-only LLMs (Flash Attention, unpadding, alternating attention)
* Context length extended to 8,192 tokens via RoPE theta adjustment post-training
* Trained on code data — only evaluated encoder with programming data in the mixture
* First MLM encoder to surpass DeBERTaV3 on GLUE

### 2. Methodology
#### 2.1 Data
Both ModernBERT models are trained on 2 trillion tokens of primarily English data from a variety of data sources, including web documents, code, and scientific literature, following common modern data mixtures. The final data mixture was chosen based on a series of ablations. The paper does not disclose the exact datasets or proportions used.

Training objective: Masked Language Modeling (MLM) with a 30% masking rate. No Next-Sentence Prediction (NSP) objective.

Tokenizer: modern BPE tokenizer, OLMo version, which provides better token efficiency and performance on code-related tasks. Vocabulary of 50368 (multiple of 64 for GPU optimization) and includes 83 unused tokens.

Sequence Packing: greedy algorithm, ensuring batch size uniformity packing sequences (over 99% efficiency).

#### 2.2 Model Architecture
The architecture keeps the standard transformer architecture with additional advances efficiency-oriented:

- **Bias Layers** disabled except final decoder linear layer, with the hypothesis that a decoder biased could mitigate weight tying's negative effects.
- **Rotatory Positional Embedding (RoPE)**, BERT base contains Positional Embeddings, empirically demonstrated an improvement with RoPE, improving long-context performance.
- **Normalization**, use of pre-normalization block (with standard layer norm) which is known to help stabilize training. They add a LayerNorm after the embedding large but it's removed in the first attention layer.
- **Activation**, variant version of GeLU, GeGLU.
- **Alternating Attention**, alternate between global attention --> every token within a sequence attends to every other token, and local attention --> tokens attend only to each other within small sliding window. In this case, ModernBert employs Global Attention with RoPE theta of 160,000 and the remaining layers use a 128 token, with a local sliding window attention with a RoPE of theta of 10,000.
- **Unpadding**, removes useless padding tokens, concatenate all sequence from minibatch into a single sequence, and process it as a batch of one. (Usage of Flash Attention's variable length attention and RoPE implementations, leading to a 10-20 percent performance improvement over other unpadding methods).
- **Flash Attention**, compute and memory efficient attention kernels.
- **Torch.compile**, 10% improvement in throughput with negligible compilation overhead.

> [!info]- Comments
> *[[RoPE]] (Rotary Positional Embedding)*: Instead of adding absolute position embeddings to token representations, RoPE encodes position by rotating the query and key vectors. For position $m$ and dimension $i$, it applies a rotation matrix with angle $m \cdot \theta_i$ where $\theta_i = 10000^{-2i/d}$. The key property is that the dot product between rotated queries and keys depends only on their relative position, enabling natural length generalization. Adjusting the $\theta$ base (e.g., from 10,000 to 160,000) extends context length.
>
> *[[GeGLU]] (Gated GELU Linear Unit)*: A variant of the GLU (Gated Linear Unit) family that uses GELU as the activation. Defined as $\text{GeGLU}(x) = \text{GELU}(xW_1) \odot (xW_2)$ where $\odot$ is element-wise multiplication. One half is gated by GELU, the other is a linear projection. Empirically outperforms standard GELU in transformers.
>
> *[[Flash Attention]]*: An IO-aware exact attention algorithm that avoids materializing the full $N \times N$ attention matrix in GPU HBM (high bandwidth memory). Instead, it tiles the computation and keeps intermediate results in fast SRAM. This reduces memory from $O(N^2)$ to $O(N)$ and improves wall-clock speed by 2-4x for long sequences. It computes the EXACT same result as standard attention — it's not an approximation.
>
> *[[Unpadding]]*: Standard batching pads all sequences to the longest length, wasting compute on padding tokens. Unpadding removes all padding, concatenates sequences into one long sequence, and processes it as a batch of one using Flash Attention's variable-length support. This gives 10-20% throughput improvement, especially with mixed-length batches.

#### 2.3 Implementation Details
- **Optimizer**: StableAdamW, improves AdamW by adding Adafactor-style update clipping as a per-parameter learning rate adjustment --> Outperforms on downstream tasks and led to more stable training.
- **LR Schedule**: Trapezoidal Learning Rate (Warmup-Stable-Decay). First warmup, then trapezoidal schedule holds the LR constant for the majority of training, followed by a short LR decay (1 - sqrt).
- **ModernBERT-base**: trained at a constant LR of 8e-4 for 1.7 trillion tokens, following a 3 billion token warmup.
- **ModernBERT-large**: After the 2B token warmup, trained at 5e-4 for 900 billion tokens. They rolled back and restarted training for the remaining 800 billion tokens after large's loss plateaued for a few hundred billion tokens at 5e-4.
- **Batch Size Schedule**: smaller gradient accumulated batches, increasing over time to full batch size. Warmed up from 768 to 4,608 over 50 billion tokens and from 448 to 4,928 over 10 billion tokens, for ModernBERT-base and large, respectively.
- **Weight Init and tiling**: Base version initialized with random weights following Megatron initialization, for ModernBERT-large, it starts with Base weights.
- **Context Length Extension**: After training on 1.7T at 1024 Seq Length and RoPE theta of 10,000 --> Extends the native context length to 8192 tokens by increasing global attention layer's RoPE theta to 160,000 and training for an additional 300 billion tokens. First trains at a constant lower learning rate of 3e-4 for 250 billion tokens on an 8192 token mixture, then upsamples higher-quality sources and conducts the decay phase with a 1−sqrt LR schedule over 50 billion tokens.

### 3. Results

ModernBERT is the strongest overall model at both the base and large model sizes. It represents a Pareto improvement on all tasks over the original BERT and RoBERTa models, with better performance on every evaluation category.

Evaluation was performed across four categories:
- **NLU**: [[GLUE]] benchmark (9 classification tasks, average reported).
- **Short-Context Retrieval**: [[BEIR]] evaluation suite in both single-vector Dense Passage Retrieval (DPR) and multi-vector ColBERT settings, using nDCG@10. Models trained on [[MS-MARCO]] with 1.25M samples and mined hard negatives.
- **Long-Context Retrieval**: [[MLDR]] (English subset, 200,000+ long documents), evaluated out-of-domain (OOD) and in-domain (ID) for DPR, and OOD for ColBERT.
- **Code Retrieval**: [[CodeSearchNet]] (CSN, code-to-text) and [[StackOverflow-QA]] (SQA, hybrid text-code), evaluated via the CoIR framework.

Key findings:
- ModernBERT-base is the first MLM-trained encoder to surpass DeBERTaV3-base on GLUE.
- On ColBERT long-context retrieval (MLDR OOD), ModernBERT outperforms other long-context models by at least 9 nDCG@10 points.
- On code tasks, ModernBERT outperforms all other models, being the only evaluated encoder trained on a data mixture including programming data.

| Model       | BEIR (DPR) | MLDR OOD (DPR) | MLDR ID (DPR) | BEIR (ColBERT) | MLDR OOD (ColBERT) | GLUE | CSN  | SQA  |
| ----------- | ---------- | --------------- | ------------- | -------------- | ------------------- | ---- | ---- | ---- |
| **Base**    |            |                 |               |                |                     |      |      |      |
| BERT        | 38.9       | 23.9            | 32.2          | 49.0           | 28.1                | 84.7 | 41.2 | 59.5 |
| RoBERTa     | 37.7       | 22.9            | 32.8          | 48.7           | 28.2                | 86.4 | 44.3 | 59.6 |
| DeBERTaV3   | 20.2       | 5.4             | 13.4          | 47.1           | 21.9                | 88.1 | 17.5 | 18.6 |
| NomicBERT   | 41.0       | 26.7            | 30.3          | 49.9           | 61.3                | 84.0 | 41.6 | 61.4 |
| GTE-en-MLM  | 41.4       | 34.3            | 44.4          | 48.2           | 69.3                | 85.6 | 44.9 | 71.4 |
| ModernBERT  | 41.6       | 27.4            | 44.0          | 51.3           | 80.2                | 88.4 | 56.4 | 73.6 |
| **Large**   |            |                 |               |                |                     |      |      |      |
| BERT        | 38.9       | 23.3            | 31.7          | 49.5           | 28.5                | 85.2 | 41.6 | 60.8 |
| RoBERTa     | 41.4       | 22.6            | 36.1          | 49.8           | 28.8                | 88.9 | 47.3 | 68.1 |
| DeBERTaV3   | 25.6       | 7.1             | 19.2          | 46.7           | 23.0                | 91.4 | 21.2 | 19.7 |
| GTE-en-MLM  | 42.5       | 36.4            | 48.9          | 50.7           | 71.3                | 87.6 | 40.5 | 66.9 |
| ModernBERT  | 44.0       | 34.3            | 48.6          | 52.4           | 80.4                | 90.4 | 59.5 | 83.9 |

#### 3.1 Limitations
* Encoder-only architecture — not suitable for generative tasks (summarization, translation, dialogue)
* Exact data mixture is not disclosed, making full reproducibility difficult
* DeBERTaV3-large still beats ModernBERT-large on GLUE (91.4 vs 90.4), though ModernBERT processes tokens in half the time
* Training requires massive compute (2T tokens) — not easy to replicate or ablate from scratch
* Short-context DPR retrieval (BEIR) shows modest gains over existing models — the big wins are specifically in long-context and code tasks

### 4. Appendix

### 5. Connections
- Modernizes the original [[BERT]] architecture with techniques from the LLM era
- Compared against [[RoBERTa]], [[DeBERTaV3]], NomicBERT, GTE-en-MLM
- Uses [[RoPE]] (from decoder-only models) instead of absolute positional embeddings
- Uses GeGLU activation (variant of [[GELU]]) and StableAdamW optimizer
- Tokenizer based on OLMo's BPE tokenizer
- Evaluated on [[GLUE]], [[BEIR]], [[MS-MARCO]], [[MLDR]], [[CodeSearchNet]], [[StackOverflow-QA]]
