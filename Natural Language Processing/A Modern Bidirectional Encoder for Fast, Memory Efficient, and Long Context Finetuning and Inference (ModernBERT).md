> [!quote] Information 
> * @ Conference NONE 
> * paper Paper [Link](https://arxiv.org/pdf/2412.13663)
> * git Github [Link](https://github.com/AnswerDotAI/ModernBERT)
> * hf Huggingface [Link](https://huggingface.co/papers/2412.13663)
> * calendar Date 19 December 2024
> * ? Motivation: 
> 		Nowadays the so called GenAI models, such as LLama, ChatGPT, etc. Are models with an insane quantity of parameters. The purpose of the paper is improve the original BERT, adding latest optimizations, thus making it possible to obtain a better version, more scalable, long context and local-global attention.    
> *  Dataset Datasets:
> 	[[ImageNet]]
> 	[[MS COCO]]
> 	[[ADE20K]]
> 	
> * Fields Related fields: 
> 	[[Vision Transformers]]
> 	[[Towards Enhanced Efficiency]]
> 	[[Self-Attention mechanisms]]

## 1. Introduction

We all remember the dominance  of BERT in the vast majority of NLP tasks before the advent of LLMs. During these years of LLMs (encoder-decoder) improvements, many new techniques dropped with the idea of improving efficiency and long context, but none of them have been applied to only encoder. This paper improves the basic BERT with some changes to improve its efficiency and longer context.
## 2. Architecture
The architecture keeps the standard transformer architecture with additional advances efficiency-oriented. 

- Bias Layers disabled except final decoder linear layer, with the hypothesis that a decoder biased  could mitigate weight tying's negative effects.
- Rotatory Positional Embedding (RoPE), BERT base contains Positional Embeddings, empirically demonstrated an improvement with RoPE, improving long-context performance.
- Normalization, use of pre-normalization block (with standard layer norm) which is known to help stabilize training. They add a LayerNorm after the embedding large but it's removed in the first attention layer
- Activation, variant version of GeLU, GeGLU.
- Alternating Attention, alternate between global attention --> every token within a sequence attends to every other token, and local attention --> tokens attend only to each other within small sliding window. In this case, ModernBert employs Global Attention with RoPE theta of 160,000 and the remaining layers use a 128 token,  with a local sliding window attention with a RoPE of theta of 10,000.
- Unpadding, removes useless padding tokens, concatenate all sequence from minibatch into a single sequence, and process itt as a bach of one. (Usage of Flash Attention's variable length attention and RoPE implementations, leading to a 10-20 percent performance improvement over other unpadding methods).
- Flash Attention, compute and memory efficient attention kernels.
- Torch.compile, 10 percent improvement in throughput with negligible compilation overhead

2T tokens of English data from a variety of data sources, including web documents, code, sientific literature, etc.

Tokenizer: modern BPE tokenizer, OLMo version, which provides better token efficiency and performance on code-related tasks. Vocabulary of 50368 ( multiple of 64) and includes 83  unused tokens.

Sequence Packing: greedy algorithm, ensuring batch size uniformity packing sequences (over 99%)

Optimizer: StableAdamW, improves adamW by adding Adafactor-style, update clipping as a per-parameter learning rate adjustment --> Outperforms on downstream tasks and led to more stable training.

LR Schedule: Trapezoidal Learning Rate, or Warmup-Stable-Decay. First warmup, then, trapezoidal schedule holds the LR constant for the majority of training, followed by a short LR decay, we use 1 - sqrt LR decay.

ModernBert-base trained at a constant LR of 8e-4 for 1.7 trillion tokens, following a 3 billion token warmup. After the 2 B token warmup, they trained ModernBERT-large at 5e-4 for 900 billion tokens.  They rolled back and restarted training for the remainin 800 bilion tokens after large's loss plateaud for a few hundred billion tokens at 5e-4

Batch Size Schedule: smaller gradiendt accumulated batches, incresing over time to full batch size.  Warmed up, from 768 to 4,608 over 50 billion tokens and from 448 to 4,928 over 10 billion tokens, for ModernBert-base and large, respectively

Weight Init and tiling, Base version is Initialized with random weights following Megatron initalization, for ModernBERT-large, it starts with Base weights.

Context Length Extension, After training on 1.7T at 1024 Seq Length and RoPE theta of 10,000. --> Extends the native context length of ModernBert to  8192 tokens by increasing global attention layer's RoPE theta to 160,000 and train for an additional 300 billion tokens. We first train at a constant lower learning rate6 of 3e-4 for 250 billion tokens on an 8192 token mixture of the original pretraining dataset sampled then upsample higher-quality sources and conduct the decay phase with a 1−sqrt LR schedule over 50 billion tokens



## 3. Data


## 4. Results



| Model      | BEIR |      |      |      |      |      |      |      |     |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- |
|            | 38.9 | 23.9 | 32.2 | 49.0 | 28.1 | 84.7 | 41.2 | 59.5 |     |
| RoBERTa    | 37.7 | 22.9 | 32.8 | 48.7 | 28.2 | 86.4 | 44.3 | 59.6 |     |
| DeBERTaV3  | 20.2 | 5.4  | 13.4 | 47.1 | 21.9 | 88.1 | 17.5 | 18.6 |     |
| NomicBERT  | 41.0 | 26.7 | 30.3 | 49.9 | 61.3 | 84.0 | 41.6 | 61.4 |     |
| GTE-en-MLM | 41.4 | 34.3 | 44.4 | 48.2 | 69.3 | 85.6 | 44.9 | 71.4 |     |
| ModernBERT | 41.6 | 27.4 | 44.0 | 51.3 | 80.2 | 88.4 | 56.4 | 73.6 |     |
| Large      | 38.9 | 23.3 | 31.7 | 49.5 | 28.5 | 85.2 | 41.6 | 60.8 |     |
| RoBERTa    | 41.4 | 22.6 | 36.1 | 49.8 | 28.8 | 88.9 | 47.3 | 68.1 |     |
| DeBERTaV3  | 25.6 | 7.1  | 19.2 | 46.7 | 23.0 | 91.4 | 21.2 | 19.7 |     |
| GTE-en-MLM | 42.5 | 36.4 | 48.9 | 50.7 | 71.3 | 87.6 | 40.5 | 66.9 |     |
| ModernBERT | 44.0 | 34.3 | 48.6 | 52.4 | 80.4 | 90.4 | 59.5 | 83.9 |     |
## 5. Appendix
