Replace redundant self-attention layers in ViTs with a lightweight parametric function for 19-25% speedup with equal or better accuracy.

> [!quote] Information
> * @ Conference ICLR 2024
> * paper Paper [Link](https://arxiv.org/pdf/2301.02240)
> * git Github -- (not publicly released)
> * hf Huggingface [Link](https://huggingface.co/papers/2301.02240)
> *  tag Tags
> 	[[Vision Transformers]]
> 	[[Towards Enhanced Efficiency]]
> 	[[Self-Attention mechanisms]]
> * calendar Date 17 January 2023
> * ? Motivation:
> 		Self-attention maps are highly correlated across consecutive transformer layers (cosine similarity up to 0.97), making many MSA computations redundant. Instead of making attention cheaper, we can skip computing it entirely in many layers.
> *  Dataset Datasets:
> 	[[ImageNet]]
> 	[[ADE20K]]

### 1. Introduction
#### 1.1 Background
[[Vision Transformers]] have achieved remarkable performance across a wide range of tasks, but they rely on computationally expensive [[Self-Attention mechanisms]] at every layer, with quadratic complexity $O(n^2d)$ with respect to the number of tokens $n$. This makes scaling ViTs to higher resolutions or deploying them on edge devices challenging.

The authors (from Qualcomm AI Research and QUVA Lab, University of Amsterdam) identify a key redundancy: **self-attention maps are highly correlated across consecutive layers**. Specifically, they measure the cosine similarity between CLS token attention maps $A_{l-1}^{[CLS]}$ and $A_l^{[CLS]}$ and find values as high as **0.97**. A CKA (Centered Kernel Alignment) analysis further shows that MSA output representations $Z^{MSA}$ are particularly correlated in layers 2-8 of a 12-layer ViT.

> This is a really elegant observation -- rather than trying to make attention itself cheaper (like [[Linear Attention]] or token pruning), they argue we can simply skip computing it in many layers.

The paper rests on three core assumptions: (1) self-attention operations are highly correlated across consecutive transformer layers, making many MSA computations redundant; (2) a lightweight parametric function can effectively approximate the transformation that MSA would perform, as long as it captures local spatial and channel interactions; and (3) the first and last few layers of the transformer are more unique and should retain full MSA computation.

#### 1.2 Objectives
The paper aims to:
* Exploit attention redundancy across layers to reduce computation
* Propose a lightweight parametric function that replaces full MSA in skipped layers
* Achieve equal or better accuracy than baseline ViTs with significantly improved throughput
* Demonstrate generalization across classification, segmentation, denoising, and self-supervised learning

#### 1.3 What's New
* Key insight: attention maps are 0.97 cosine-similar across consecutive layers — most MSA computations are redundant
* Parametric function Φ (FC→DwC→FC→ECA) replaces full MSA in skipped layers at O(nd²) vs O(n²d)
* No gating or blending — full MSA replacement in layers 3-8
* Generalizes across classification, segmentation, denoising, and self-supervised learning

### 2. Methodology
#### 2.1 Data
* [[ImageNet]]-1K for image classification (300 epochs, following DeiT training settings)
* [[ImageNet]]-1K for self-supervised learning with [[DINO]]
* [[ADE20K]] for semantic segmentation
* SIDD for image denoising
* DAVIS for video denoising

#### 2.2 Model Architecture
Standard attention computes:
$$A := \sigma\left(\frac{QK^T}{\sqrt{d}}\right), \quad Z^{MSA} := AV$$

In SkipAt, for layers where MSA is skipped, the attention output is completely replaced by a parametric function $\Phi$ applied to the previous layer's MSA output:
$$Z_l^{MSA} := \Phi(Z_{l-1}^{MSA})$$

The parametric function $\Phi$ is defined as:
$$\hat{Z}_l^{MSA} := \text{ECA}(\text{FC}_2(\text{DwC}(\text{FC}_1(Z_{l-1}^{MSA}))))$$

Where:
- **FC1**: Linear layer expanding channels $d \rightarrow 2d$
- **DwC**: Depthwise convolution with $5 \times 5$ kernel (captures local spatial information)
- **FC2**: Linear layer reducing channels $2d \rightarrow d$
- **ECA**: Efficient Channel Attention -- global average pooling followed by adaptive 1x1 convolution and sigmoid activation

> [!info]- Comments
> *[[Depthwise Convolution]] (DwC)*: A convolution where each input channel is convolved with its own separate filter, instead of mixing across channels. For a tensor with $d$ channels, standard conv uses $d \times d \times k^2$ parameters, but DwC uses only $d \times k^2$ — one filter per channel. It captures spatial patterns within each channel independently. Usually followed by a pointwise (1x1) conv to mix channels (together called Depthwise Separable Convolution, from MobileNet).
>
> *[[ECA|Efficient Channel Attention (ECA)]]*: A lightweight attention mechanism that models channel dependencies. It applies global average pooling to get a per-channel descriptor, then uses a 1D convolution (with adaptive kernel size) instead of two FC layers (as in SE-Net). The output is a sigmoid-activated channel weight vector. Much cheaper than Squeeze-and-Excitation but similarly effective.
>
> *[[CKA]] (Centered Kernel Alignment)*: A similarity metric used to compare neural network representations across layers. Unlike cosine similarity (which compares individual vectors), CKA compares the full representation matrices, accounting for invariances to orthogonal transformations and scaling. CKA ≈ 1 means the layers produce nearly equivalent representations.

The complexity of $\Phi$ is $O(nd^2)$ compared to $O(n^2d)$ for MSA, making it more efficient especially when the number of tokens $n$ is large.

**Skip Configuration**: Based on CKA analysis, MSA is skipped in layers 3-8 (out of 12 total), keeping full attention in the first 2 and last 4 layers. There is no learnable gating or alpha blending -- the MSA is fully replaced by $\Phi$.

> The design choice of no gating is interesting. They found that the parametric function alone (with the residual connection from the transformer block) is sufficient.

#### 2.3 Implementation Details
- ViT-T/16, ViT-S/16, ViT-B/16 as base architectures (trained following DeiT settings)
- 300 epochs from scratch on [[ImageNet]]-1K
- Batch size: 2048 (ViT-T), 1024 (ViT-S/B)
- Resolution: 224x224
- Also applied to Uformer (image denoising) and UniFormer (video denoising)

### 3. Results
#### 3.1 ImageNet-1K Classification

| Model | Top-1 Acc (%) | Throughput (img/s) | Speedup |
|-------|:---:|:---:|:---:|
| ViT-T/16 | 72.8 | 5,800 | -- |
| ViT-T/16 + SkipAt | **72.9** | **6,900** | +19% |
| ViT-S/16 | 79.8 | 3,200 | -- |
| ViT-S/16 + SkipAt | **80.2** | **3,800** | +21% |
| ViT-B/16 | 81.8 | 1,200 | -- |
| ViT-B/16 + SkipAt | **82.2** | **1,500** | +25% |

> Consistently better accuracy AND faster throughput -- not just a trade-off. The bigger the model, the larger the throughput improvement.

#### 3.2 Self-Supervised Learning (DINO)
SkipAt achieves **73.3%** k-NN accuracy in **96 GPU-hours** vs. baseline's **73.6%** in **131 GPU-hours** -- a 26% reduction in pretraining time with comparable accuracy.

#### 3.3 Semantic Segmentation on ADE20K

| Model | mIoU | Throughput (img/s) | Speedup |
|-------|:---:|:---:|:---:|
| ViT-S + Segmenter | 44.4 | 19,500 | -- |
| ViT-S + SkipAt + Segmenter | **45.3** | **27,200** | +40% |
| ViT-B + Segmenter | 45.6 | 11,100 | -- |
| ViT-B + SkipAt + Segmenter | **46.3** | **15,500** | +40% |

#### 3.4 Image Denoising on SIDD

| Model | PSNR (dB) | SSIM | Throughput (img/s) |
|-------|:---:|:---:|:---:|
| Uformer-S | 39.77 | 0.959 | 15,100 |
| Uformer-S + SkipAt | **39.84** | **0.960** | **18,900** |
| Uformer-B | 39.89 | 0.960 | 9,200 |
| Uformer-B + SkipAt | **39.94** | **0.960** | **10,900** |

#### 3.5 Video Denoising on DAVIS

| Model | PSNR (dB) | FLOPs Reduction |
|-------|:---:|:---:|
| UniFormer | 35.24 | -- |
| UniFormer + SkipAt | 35.16 | -17% |

#### 3.6 Ablation Studies (ViT-T/16)

**Parametric function variants:**

| Variant | Top-1 Acc (%) | Throughput (img/s) |
|---------|:---:|:---:|
| Identity (no transform) | 61.1 | 8,500 |
| Depthwise conv only | 65.6 | 7,800 |
| SkipAt (full, 5x5 kernel) | **67.7** | 6,900 |

**Kernel size for DwC:**

| Kernel Size | Top-1 Acc (%) |
|:-----------:|:---:|
| 3x3 | 67.1 |
| 5x5 | **67.7** |
| 7x7 | 67.4 |

**Channel expansion ratio:**

| Ratio | Top-1 Acc (%) |
|:-----:|:---:|
| 0.5x | 64.4 |
| 1.0x | 65.9 |
| 2.0x | **67.7** |

**Mobile latency (Samsung Galaxy S22):**

| Resolution | Baseline ViT-T | SkipAt | Improvement |
|:----------:|:---:|:---:|:---:|
| 224x224 | 5.65 ms | 4.76 ms | -19% |
| 384x384 | -- | -- | -34% |

> The 34% latency reduction at 384x384 is notable -- the gains increase with resolution since SkipAt's $O(nd^2)$ complexity scales better than $O(n^2d)$ when token count grows.

#### 3.7 Limitations
* Skip configuration (layers 3-8) is fixed and hand-designed based on CKA analysis — not adaptive per input or learned end-to-end
* Video denoising (DAVIS) shows a slight PSNR drop (-0.08 dB), suggesting the parametric function may lose fine-grained spatial details in some tasks
* Only evaluated on ViT-based architectures (ViT, Uformer, UniFormer) — generalization to non-ViT transformers (e.g., DETR, SegFormer) is untested
* The parametric function uses a fixed 5x5 depthwise convolution, which may not be optimal for all spatial scales or token configurations

### 4. Appendix
#### 4.1 Comparison with Other Efficient Methods
SkipAt outperforms token pruning methods such as SViTE, A-ViT, DynamicViT, and SPViT on the accuracy-efficiency trade-off, as the parametric function preserves all tokens rather than discarding them.

#### 4.2 Connection to FasterViT
The GitHub repository links to [[FasterViT]], which is a follow-up work from the same group that builds a full architecture using similar ideas of hierarchical attention and efficient computation. The Hugging Face link points to FasterViT (arxiv 2306.06189), suggesting SkipAt's ideas were incorporated into that architecture.

### 5. Connections
- From Qualcomm AI Research and QUVA Lab (University of Amsterdam)
- Ideas incorporated into [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]] — same research group's follow-up work
- Different approach from token pruning (SViTE, A-ViT, DynamicViT) — preserves all tokens instead of discarding
- Different from [[Linear Attention]] approaches — skips attention entirely rather than approximating it
- Applied to ViT (DeiT), Uformer (denoising), UniFormer (video) — architecture-agnostic
