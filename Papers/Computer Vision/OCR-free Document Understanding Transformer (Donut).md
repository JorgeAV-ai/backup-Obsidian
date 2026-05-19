OCR-free document understanding via end-to-end encoder-decoder transformer with synthetic multilingual pretraining.

> [!quote] Information
> * @ Conference ECCV'22
> * paper Paper [Link](https://arxiv.org/pdf/2111.15664)
> * git Github [Link](https://github.com/clovaai/donut)
> * hf Huggingface [Link](https://huggingface.co/papers/2111.15664)
> *  tag Tags
> 	[[Visual Document Understanding (VDU)]]
> 	[[Document Information Extraction]]
> 	[[Optical Character Recognition]]
> 	[[End-to-End Transformer]]
> * calendar Date 6 October 2022
> * ? Motivation:
> 	Visual Document Understanding without reading text (OCR)
> *  Dataset Datasets:
> 	[[RVL-CDIP]]
> 	[[CORD]]
> 	[[DocVQA]]
> 	[[Ticket]]
> 	[[Business Card]]
> 	[[Receipt]]

### 1. Introduction
#### 1.1 Background
Most [[Visual Document Understanding (VDU)]] pipelines follow a two-stage approach: first extract text using an [[Optical Character Recognition]] engine, then feed the recognized text into a downstream model (e.g., [[LayoutLM]], [[BERT]]). While effective, this paradigm has three fundamental limitations:
1. **High computational cost** -- OCR itself is expensive, often dominating total inference time.
2. **Inflexibility** -- off-the-shelf OCR models are language- and domain-specific; switching to a new script or document type requires a different OCR engine or fine-tuning.
3. **Error propagation** -- OCR mistakes cascade into the downstream task with no way to recover.

These issues become especially painful in multilingual or low-resource scenarios where reliable OCR is hard to obtain.

Donut rests on several key assumptions that motivate the OCR-free design:
* A sufficiently large visual encoder can learn to "read" text implicitly from pixel-level features, eliminating the need for an explicit OCR step.
* Pre-training on a massive corpus of document images (real + synthetic) provides enough reading ability for the model to transfer to diverse downstream tasks.
* Structured output (JSON-like token sequences with special tokens) can faithfully represent document semantics for any VDU task.

#### 1.2 Objectives
The paper proposes **Donut** (Document understanding transformer), an [[End-to-End Transformer]] that maps a document image directly to structured output **without any OCR module**. The key goals are:
* Achieve competitive or superior accuracy compared to OCR-dependent methods across classification, information extraction, and VQA tasks.
* Drastically reduce inference latency by removing the OCR bottleneck.
* Support multiple languages (English, Chinese, Japanese, Korean) through a synthetic data generator called **SynthDoG**.

#### 1.3 What's New
* First end-to-end transformer for VDU that completely removes the OCR pipeline
* SynthDoG: synthetic document generator for multilingual pretraining (EN, CN, JP, KR)
* ~2x faster inference than OCR-dependent methods by eliminating the OCR bottleneck

### 2. Methodology
#### 2.1 Data
**Pre-training:**
* [[IIT-CDIP]] Test Collection 1.0 -- ~11 million scanned English documents (real).
* **SynthDoG** (Synthetic Document Generator) -- 0.5M synthetic document images **per language** for English, Chinese, Japanese, and Korean (2M total). SynthDoG composes background textures, document bodies (paragraphs rendered with diverse fonts), and optional layout elements. This makes pre-training flexible across languages without needing real labeled data in each language.

**Fine-tuning / Evaluation:**
* [[RVL-CDIP]] -- 400K document images across 16 categories (classification).
* [[CORD]] -- receipt parsing with hierarchical key-value structure.
* [[Ticket]] -- Korean train ticket parsing.
* [[Business Card]] -- Japanese business card parsing.
* [[Receipt]] -- Japanese receipt parsing.
* [[DocVQA]] -- document visual question answering (12K+ images, 50K+ questions).

#### 2.2 Model Architecture
Donut is an encoder-decoder [[Transformer]]:

**Encoder -- [[Swin Transformer]]-B**
* Layer configuration: {2, 2, 14, 2} with window size 10.
* Input resolution: **2560 x 1920** during pre-training (high resolution is critical for reading small text).
* Outputs a sequence of visual feature embeddings that serve as cross-attention context for the decoder.

> [!info]- Comments
> *[[Swin Transformer]]*: A hierarchical vision transformer that uses shifted window-based self-attention. Instead of computing global attention across all patches (quadratic cost), it partitions the image into non-overlapping local windows and computes attention within each window. The "shifted" part means that in alternating layers, the window partition is shifted by half the window size, allowing cross-window connections. This makes complexity linear with image size $O(n)$ instead of quadratic $O(n^2)$.
>
> *[[Cross-Attention]]*: In the decoder, each layer attends to the encoder's output features. This is how the decoder "looks at" the image while generating tokens — Query comes from the decoder, Key and Value come from the encoder: $\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc})$.

**Decoder -- mBART-based**
* Uses the first four layers of a multilingual BART decoder (Asian-BART-ECJK variant, covering English, Chinese, Japanese, Korean).
* Generates output tokens **autoregressively**, conditioned on the encoder features and a task-specific prompt token (e.g., `<s_cord-receipt>`, `<s_docvqa>`).
* Output is a JSON-like token sequence with special structural tokens (`<s_fieldname>`, `</s_fieldname>`) that can be deterministically parsed into key-value pairs, classes, or answers.
* Maximum decoder length: 1536 tokens.

> [!info]- Comments
> *[[Autoregressive Generation]]*: The decoder generates tokens one at a time, left-to-right. Each token is conditioned on all previous tokens AND the encoder features. This is the same generation strategy used by GPT models, but here it generates structured JSON tokens instead of natural language.
>
> *Task-specific prompt tokens*: The idea of using a special token (e.g., `<s_cord-receipt>`) to tell the model which task to perform is similar to how T5 uses task prefixes. It's elegant because the same architecture handles classification, extraction, and QA — only the prompt changes.

**Total parameters: ~143M** -- notably smaller than many OCR-dependent pipelines when you account for the OCR model parameters.

From what I see, the elegance here is that the entire "reading" and "understanding" happen in a single forward pass. The decoder prompt token is what tells the model which task to perform, making the architecture very flexible.

#### 2.3 Implementation Details
- **Pre-training:** 200K steps, batch size 196, on **64 NVIDIA A100 GPUs**.
- **Optimizer:** Adam with learning rate in the range 1e-5 to 1e-4.
- **Input resolution:** 2560 x 1920 for pre-training; reduced to 1280 x 960 for some downstream tasks to save compute.
- **Pre-training objective:** the model learns to read all text in the document image (pseudo-OCR task via cross-entropy loss over the token sequence).
- **Fine-tuning:** task-specific prompt tokens are prepended; the model is fine-tuned end-to-end on each downstream dataset.
- **SynthDoG generation pipeline:** background image + document body (rendered text with random fonts/sizes) + optional noise/degradation. 0.5M images per language.

### 3. Results

#### 3.1 Document Classification -- [[RVL-CDIP]]

| Model | OCR? | Accuracy (%) | Time (ms) |
| --- | --- | --- | --- |
| BERT | Yes | 89.81 | 1392 |
| RoBERTa | Yes | 90.06 | 1392 |
| LayoutLM | Yes | 91.78 | 1396 |
| LayoutLM (w/ image) | Yes | 94.42 | 1426 |
| LayoutLMv2 | Yes | 95.25 | 1489 |
| **Donut** | **No** | **95.30** | **752** |

Donut matches or slightly beats LayoutLMv2 while being **~2x faster** since it skips the OCR step entirely.

#### 3.2 Document Information Extraction

Metric: Field-level F1 and accuracy (Tree Edit Distance-based).

**[[CORD]] (Receipts):**

| Model | OCR? | F1 | Accuracy | Time (s) |
| --- | --- | --- | --- | --- |
| BERT | Yes | 73.0 | 65.5 | 1.6 |
| BROS | Yes | 74.7 | 70.0 | 1.7 |
| LayoutLM | Yes | 78.4 | 81.3 | 1.7 |
| LayoutLMv2 | Yes | 78.9 | 82.4 | 1.7 |
| **Donut** | **No** | **84.1** | **90.9** | **1.2** |

**[[Ticket]] (Korean train tickets):**

| Model | OCR? | F1 | Accuracy | Time (s) |
| --- | --- | --- | --- | --- |
| LayoutLMv2 | Yes | 87.2 | 90.1 | 1.8 |
| **Donut** | **No** | **94.1** | **98.7** | **0.6** |

**[[Business Card]] (Japanese):**

| Model | OCR? | F1 | Accuracy | Time (s) |
| --- | --- | --- | --- | --- |
| LayoutLMv2 | Yes | 52.2 | 83.0 | 1.6 |
| **Donut** | **No** | **57.8** | **84.4** | **1.4** |

**[[Receipt]] (Japanese):**

| Model | OCR? | F1 | Accuracy | Time (s) |
| --- | --- | --- | --- | --- |
| LayoutLMv2 | Yes | 72.9 | 78.0 | 2.6 |
| **Donut** | **No** | **78.6** | **88.6** | **1.9** |

From what I see, Donut dominates on every extraction benchmark, with especially large gains on the Korean Ticket dataset (+6.9 F1) where OCR quality is presumably lower.

#### 3.3 Document VQA -- [[DocVQA]]

| Model | OCR? | ANLS (test) | Time (ms) |
| --- | --- | --- | --- |
| BERT | Yes | 63.5 | 1517 |
| LayoutLM | Yes | 69.8 | 1519 |
| LayoutLMv2 | Yes | 78.1 | 1610 |
| **Donut** | **No** | **67.5** | **782** |
| LayoutLMv2-Large-QG | Yes | 86.7 | 1698 |

On DocVQA, Donut underperforms the strongest OCR-based models. This makes sense -- VQA requires fine-grained text comprehension where explicit OCR still has an edge. Still, Donut is **~2x faster** and competitive with mid-range OCR models.

#### 3.4 Speed Summary
Across all tasks, Donut achieves roughly **2-3x speedup** over OCR-dependent pipelines. The savings come entirely from removing the OCR stage, which typically accounts for 40-60% of total inference time in traditional pipelines.

#### 3.5 Limitations
* DocVQA results show that for tasks requiring precise text extraction, OCR-based methods still hold an advantage.
* The model requires very high input resolution (2560x1920) to read small text, which increases memory and compute.
* Pre-training is expensive (64 A100 GPUs), though fine-tuning is manageable.

### 4. Appendix

#### Ablation Highlights
* **Pre-training data scale matters:** Without pre-training, performance drops catastrophically. The jump from 0 to 11M real documents is the single biggest factor.
* **SynthDoG helps multilingual tasks:** Adding 0.5M synthetic documents per language yields clear gains for Japanese and Korean tasks, showing that the synthetic generator successfully bridges the language gap.
* **Input resolution is critical:** Moving from 1280x960 to 2560x1920 gives meaningful accuracy gains (small text becomes readable), but at the cost of higher compute. The 2560x1920 setting is used for pre-training; downstream tasks can often get away with lower resolution.
* **Model size:** Swin-B (143M total) hits a strong accuracy/efficiency sweet spot. Going to Swin-L gives diminishing returns.

### 5. Connections
- Encoder based on [[Swin Transformer]]-B, decoder based on [[mBART]]
- Compared against [[LayoutLM]], [[LayoutLMv2]], [[BERT]], [[RoBERTa]] -- wins on speed, competitive on accuracy
- Evaluated on [[RVL-CDIP]], [[CORD]], [[DocVQA]]
- Part of the [[Visual Document Understanding (VDU)]] field
- Follow-up: [[Pix2Struct]], [[Nougat]] take similar OCR-free approaches
