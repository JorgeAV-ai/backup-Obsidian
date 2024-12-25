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

-  Bias Layers disabled except final decoder linear layer, with the hypothesis that a decoder biased  could mitigate weight tying's negative effects.
- Rotatory Positional Embedding, BERT base contains Positional Embeddings, empirically demonstrated an improvement with RoPE, improving long-context performance.
- Normalization, use of pre-normalization block (with standard layer norm) which is knowkn to help stabilize training. They add a LayerNorm after the embedding large but it's removed in the first attentio layer
- Activation, 
- Unpadding, 


## 3. Data
## 4. Results

## 5. Appendix
