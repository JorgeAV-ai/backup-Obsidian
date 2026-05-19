Hierarchical Vision Transformer that uses shifted window-based self-attention to achieve linear computational complexity, serving as a general-purpose backbone for dense vision tasks.

> [!quote] Information
> * @ Conference ICCV 2021
> * paper Paper [Link](https://arxiv.org/pdf/2103.14030)
> * git Github [Link](https://github.com/microsoft/Swin-Transformer)
> * hf Huggingface [Link](https://huggingface.co/papers/2103.14030)
> *  tag Tags
> 	[[Vision Transformers]]
> 	[[Hierarchical Representations]]
> 	[[Self-Attention mechanisms]]
> * calendar Date 25 March 2021
> * ? Motivation:
> 		Standard Vision Transformers compute global self-attention over all image patches with quadratic complexity, making them impractical for dense prediction tasks (detection, segmentation) that require high-resolution, multi-scale feature maps. Additionally, vision differs from language in that visual elements vary greatly in scale, and existing transformer architectures lack the hierarchical, multi-resolution structure that CNN backbones naturally provide.
> *  Dataset Datasets:
> 	[[ImageNet]]
> 	[[MS COCO]]
> 	[[ADE20K]]

### 1. Introduction
#### 1.1 Background
Vision Transformers (ViT) demonstrated that pure transformer architectures can match or surpass CNNs on image classification by treating images as sequences of patches. However, ViT processes all patches with global self-attention at a single resolution, resulting in quadratic complexity with respect to image size and lacking the multi-scale feature hierarchy that dense prediction frameworks (FPN, UPerNet, Mask R-CNN) require. Prior to Swin, transformers were largely confined to classification and struggled to serve as general-purpose vision backbones.

#### 1.2 Objectives
Design a hierarchical vision transformer that: (1) computes self-attention in local windows for linear complexity, (2) enables cross-window information flow via a shifted window mechanism, and (3) produces multi-scale feature maps compatible with existing dense prediction frameworks, making it a drop-in CNN backbone replacement.

#### 1.3 What's New
* Window-based self-attention (W-MSA) that restricts attention to non-overlapping local windows of fixed size $M \times M$, reducing complexity from $O(N^2)$ to $O(M^2 \cdot N)$ — linear in image size
* Shifted window multi-head self-attention (SW-MSA) that shifts the window partition by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ in alternating blocks, enabling cross-window connections without extra parameters
* Efficient cyclic shift + attention masking implementation that avoids the cost of padding or variable-sized windows
* Hierarchical architecture with patch merging layers producing feature maps at 4 resolutions (1/4, 1/8, 1/16, 1/32), directly compatible with FPN, UPerNet, and Mask R-CNN
* Relative position bias in attention instead of absolute positional embeddings, improving generalization across resolutions

### 2. Methodology
#### 2.1 Data
* [[ImageNet]]-1K: 1.28M training and 50K validation images, 1000 categories (classification).
* [[ImageNet]]-22K: 14.2M images with 21,841 classes (pre-training for larger models).
* [[MS COCO]]: 118K training images for object detection and instance segmentation with Cascade Mask R-CNN.
* [[ADE20K]]: 25K images (20K train, 2K val, 3K test) for semantic segmentation with UPerNet.

#### 2.2 Model Architecture

The architecture has four stages. Each stage consists of multiple Swin Transformer blocks that alternate between regular window attention (W-MSA) and shifted window attention (SW-MSA). Between stages, a patch merging layer downsamples spatial resolution by 2x while doubling channels.

![[basics_swin_transformer.png]]

[🔗 Open interactive Swin Window Visualizer](../interactive/swin_windows.html)

| Stage | Resolution          | Channels | Description |
|-------|---------------------|----------|-------------|
| 1     | $H/4 \times W/4$    | $C$      | Patch partition + linear embedding |
| 2     | $H/8 \times W/8$    | $2C$     | Patch merging + Swin blocks |
| 3     | $H/16 \times W/16$  | $4C$     | Patch merging + Swin blocks |
| 4     | $H/32 \times W/32$  | $8C$     | Patch merging + Swin blocks |

**Model Variants:**

| Model   | C   | Layers          | Params | FLOPs |
|---------|-----|-----------------|--------|-------|
| Swin-T  | 96  | {2, 2, 6, 2}   | 29M    | 4.5G  |
| Swin-S  | 96  | {2, 2, 18, 2}  | 50M    | 8.7G  |
| Swin-B  | 128 | {2, 2, 18, 2}  | 88M    | 15.4G |
| Swin-L  | 192 | {2, 2, 18, 2}  | 197M   | 34.5G |

**Patch Partitioning and Linear Embedding**

The input image of size $H \times W \times 3$ is split into non-overlapping patches of size $4 \times 4$, producing $\frac{H}{4} \times \frac{W}{4}$ patch tokens each with raw dimension $4 \times 4 \times 3 = 48$. A linear layer projects these to dimension $C$.

> [!info]- Comments
> Unlike ViT which uses 16x16 patches, Swin uses smaller 4x4 patches. This finer granularity is important because the hierarchical structure with patch merging will progressively reduce resolution — starting at 4x4 gives the model more spatial detail to work with in early stages.

**Window-Based Multi-Head Self-Attention (W-MSA)**

The feature map is partitioned into non-overlapping windows of size $M \times M$ (default $M = 7$). Self-attention is computed independently within each window.

Complexity comparison — Global MSA vs Window-based MSA:
$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$
$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

Where $hw$ is the total number of patches. The key difference: global attention scales as $(hw)^2$ while windowed attention scales as $M^2 \cdot hw$ — linear in image size since $M$ is fixed.

```python
def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C) input feature map
        window_size: M (int), size of each square window
    Returns:
        windows: (num_windows * B, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size,
                      W // window_size, window_size, C)
    # (B, num_h, M, num_w, M, C) -> (B, num_h, num_w, M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Merge windows back into a feature map.

    Args:
        windows: (num_windows * B, M, M, C)
        window_size: M (int)
        H, W: original feature map spatial dims
    Returns:
        x: (B, H, W, C)
    """
    num_h = H // window_size
    num_w = W // window_size
    B = windows.shape[0] // (num_h * num_w)
    x = windows.reshape(B, num_h, num_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, H, W, -1)
    return x
```

> [!info]- Comments
> The window partition is a pure reshape + permute operation with zero learnable parameters and zero FLOPs. The computational saving comes entirely from the fact that attention within an $M \times M$ window is $O(M^4)$ per window, and there are $\frac{hw}{M^2}$ windows, giving total cost $O(M^2 \cdot hw)$ instead of $O((hw)^2)$.

**Shifted Window Multi-Head Self-Attention (SW-MSA)**

To allow cross-window information flow, alternating blocks shift the window partition by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels. The shifted partition creates windows of varying sizes at the image borders. To maintain computational efficiency, Swin uses a cyclic shift followed by attention masking:

1. Cyclically shift the feature map by $(-\lfloor M/2 \rfloor, -\lfloor M/2 \rfloor)$
2. Compute windowed attention with a mask that prevents attention between patches from non-adjacent original regions
3. Reverse the cyclic shift

```python
class SwinTransformerBlock:
    """One block: LN -> (S)W-MSA -> residual -> LN -> MLP -> residual"""

    def forward(self, x, shift=False):
        """
        Args:
            x: (B, H, W, C)
            shift: if True, use shifted window attention (SW-MSA)
        """
        B, H, W, C = x.shape
        M = self.window_size  # default 7
        shortcut = x
        x = layer_norm(x)

        # --- Cyclic shift (only for SW-MSA) ---
        if shift:
            shift_size = M // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = self.compute_shifted_mask(H, W, M, shift_size)
        else:
            attn_mask = None

        # --- Partition into windows ---
        windows = window_partition(x, M)          # (num_win*B, M, M, C)
        windows = windows.reshape(-1, M * M, C)   # (num_win*B, M^2, C)

        # --- Windowed multi-head self-attention ---
        Q = windows @ W_q   # (num_win*B, M^2, d)
        K = windows @ W_k
        V = windows @ W_v
        attn = (Q @ K.transpose(-2, -1)) / sqrt(d)
        attn = attn + relative_position_bias      # learned per head
        if attn_mask is not None:
            attn = attn + attn_mask               # mask cross-region pairs
        attn = softmax(attn, dim=-1)
        out = attn @ V                            # (num_win*B, M^2, d)
        out = out @ W_o                           # project back to C

        # --- Merge windows back ---
        out = out.reshape(-1, M, M, C)
        x = window_reverse(out, M, H, W)          # (B, H, W, C)

        # --- Reverse cyclic shift ---
        if shift:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        # --- Residual + MLP ---
        x = shortcut + x
        x = x + mlp(layer_norm(x))
        return x
```

> [!info]- Comments
> *Cyclic shift trick*: The naive approach to shifted windows would produce up to 9 sub-windows of varying sizes at the borders, requiring padding or separate attention computations. The cyclic shift is elegant — by rolling the feature map and masking, all windows remain the same $M \times M$ size and can be batched efficiently. The attention mask sets logits to $-\infty$ for token pairs that should not attend to each other (i.e., tokens from non-adjacent original regions that ended up in the same window after the cyclic shift).
>
> *Relative position bias*: Instead of absolute positional embeddings, Swin adds a learnable bias $B \in \mathbb{R}^{M^2 \times M^2}$ to attention logits, parameterized by a smaller bias matrix $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$ indexed by relative positions. This generalizes better across different image sizes during transfer learning.

**Patch Merging**

Between stages, a patch merging layer concatenates features from $2 \times 2$ neighboring patches (reducing spatial resolution by 2x) and applies a linear layer to reduce the channel dimension from $4C$ to $2C$.

```
def PatchMerge(x):
    # x: (B, H, W, C)
    x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = concat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
    x = LayerNorm(x)
    x = Linear(4C, 2C)(x)                 # (B, H/2, W/2, 2C)
    return x
```

> [!info]- Comments
> This is analogous to strided convolution in CNNs for downsampling, but done via a deterministic gather + linear projection. It creates the hierarchical multi-scale structure that makes Swin compatible with FPN and similar dense prediction frameworks.

**Full Swin Transformer Forward Pass**

```python
def swin_transformer_forward(image):
    # Patch embedding: (B, H, W, 3) -> (B, H/4, W/4, C)
    x = patch_embed(image)  # 4x4 patches, linear projection to C dims

    features = []
    for stage_idx in range(4):
        for block_idx in range(num_blocks[stage_idx]):
            shift = (block_idx % 2 == 1)  # alternate W-MSA and SW-MSA
            x = swin_block[stage_idx][block_idx](x, shift=shift)

        features.append(x)  # save for FPN / dense prediction heads

        if stage_idx < 3:  # no merging after last stage
            x = patch_merge(x)  # 2x2 merge: halve resolution, double channels

    return features  # multi-scale: [H/4, H/8, H/16, H/32]
```

#### 2.3 Implementation Details
- **Optimizer**: AdamW with initial learning rate of 1e-3 and weight decay of 0.05.
- **Training schedule**: 300 epochs on ImageNet-1K with cosine decay LR scheduler including 20-epoch linear warm-up.
- **Batch size**: 1024 across 8 V100 GPUs.
- **Data augmentation**: RandAugment, Mixup (alpha=0.8), CutMix (alpha=1.0), random erasing (prob=0.25).
- **Regularization**: stochastic depth (drop path) with rates up to 0.5 for larger models, label smoothing of 0.1.
- **Window size**: $M = 7$ for all models.
- **ImageNet-22K pre-training**: 90 epochs with LR 1e-3, fine-tuned on 1K for 30 epochs with LR 1e-5.

### 3. Results

#### 3.1 Image Classification

Benchmark on [[ImageNet]]-1K at 224x224 resolution:

| Model         | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|---------------|:----------:|:---------:|:-------------:|
| DeiT-S        |    22      |    4.6    |     79.8      |
| DeiT-B        |    86      |   17.5    |     81.8      |
| Swin-T        |    29      |    4.5    |   **81.3**    |
| Swin-S        |    50      |    8.7    |   **83.0**    |
| Swin-B        |    88      |   15.4    |   **83.5**    |

ImageNet-22K pre-trained, fine-tuned at 384x384:

| Model         | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|---------------|:----------:|:---------:|:-------------:|
| ViT-B/16 (384)|   87      |   55.5    |     77.9      |
| ViT-L/16 (384)|  307      |  190.7    |     76.5      |
| Swin-B (384)  |    88      |   47.0    |   **84.5**    |
| Swin-L (384)  |   197      |  103.9    |   **87.3**    |

#### 3.2 Object Detection (COCO)

Cascade Mask R-CNN framework on [[MS COCO]]:

| Backbone | Params (M) | FLOPs (G) | AP box | AP mask |
|----------|:----------:|:---------:|:------:|:-------:|
| ResNet-50  |   82     |   739     |  46.3  |  40.1   |
| Swin-T     |   86     |   745     |**50.4**|**43.7** |
| Swin-S     |  107     |   838     |**51.9**|**45.0** |
| Swin-B     |  145     |   982     |**51.9**|**45.0** |
| Swin-L (HTC++)|  284  |  1470     |**58.7**|**51.1** |

#### 3.3 Semantic Segmentation (ADE20K)

UPerNet framework on [[ADE20K]]:

| Backbone   | Params (M) | FLOPs (G) | mIoU (ss) | mIoU (ms) |
|------------|:----------:|:---------:|:---------:|:---------:|
| ResNet-101 |    86      |   1029    |   44.9    |     -     |
| DeiT-S     |    52      |   1099    |   44.0    |     -     |
| Swin-T     |    60      |   945     | **44.5**  | **45.8**  |
| Swin-S     |    81      |   1038    | **47.6**  | **49.5**  |
| Swin-B     |   121      |   1188    | **48.1**  | **49.7**  |
| Swin-L     |   234      |   -       | **52.1**  | **53.5**  |

#### 3.4 Limitations
- Window-based attention is inherently local: information flow across distant windows requires multiple layers of stacking, limiting the effective receptive field in shallow layers compared to global attention models.
- Fixed window size ($M = 7$) may not be optimal across all resolutions and tasks; the architecture does not dynamically adapt window size to input characteristics.
- The cyclic shift mechanism, while efficient, adds implementation complexity and is specific to regular grid structures — it does not trivially extend to irregular or non-grid data.
- Patch merging is a rigid 2x downsampling scheme; unlike CNN pooling with learnable strides, it offers no flexibility in the downsampling ratio.
- Performance gains on ImageNet-1K without ImageNet-22K pre-training are moderate over DeiT (81.3 vs 79.8 for comparable size), with the major advantages appearing primarily on dense prediction tasks and with large-scale pre-training.

### 4. Appendix

**Authors**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo (Microsoft Research Asia).

**Key ablation results**:
- Shifted window vs. sliding window: shifted windows are 2x faster while achieving similar accuracy, because sliding windows create more windows due to overlap.
- Relative position bias vs. absolute position embedding: relative bias yields +1.2% on ImageNet, +1.3 AP on COCO, and +0.5 mIoU on ADE20K.
- Removing the shifted window mechanism (using only W-MSA, no SW-MSA) drops ImageNet top-1 by 1.1%, COCO AP by 2.8, and ADE20K mIoU by 2.5 — demonstrating the importance of cross-window connections.

### 5. Connections
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]: extends Swin's windowed attention with carrier tokens for more efficient cross-window interaction in a hybrid CNN-Transformer architecture
- [[OCR-free Document Understanding Transformer (Donut)]]: uses Swin Transformer as the visual encoder backbone for document understanding without OCR
- [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]]: proposes skipping redundant attention computations in vision transformers, applicable to Swin-style architectures for further efficiency gains
