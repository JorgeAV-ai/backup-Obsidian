Hybrid CNN-Transformer with Hierarchical Attention (carrier tokens) for fast high-resolution image throughput.

> [!quote] Information
> * @ Conference ICLR 2024
> * paper Paper [Link](https://arxiv.org/pdf/2306.06189)
> * git Github [Link](https://github.com/NVlabs/FasterViT)
> * hf Huggingface [Link](https://huggingface.co/papers/2306.06189)
> *  tag Tags
> 	[[Vision Transformers]]
> 	[[Towards Enhanced Efficiency]]
> 	[[Self-Attention mechanisms]]
> * calendar Date 1 April 2024
> * ? Motivation:
> 		Vision transformers models still face high computational cost due to quadratic complexity in self-attentions. In addition, it is well known that CNNs provides great local features representations, feature that Vision Transformers lack of,  since they provide global feature learning.
> *  Dataset Datasets:
> 	[[ImageNet]]
> 	[[MS COCO]]
> 	[[ADE20K]]

### 1. Introduction
#### 1.1 Background
Novel hybrid architecture, tailored for high-resolution input images, while maintaining a fast image throughput. CNNs excel at capturing local features while Vision Transformers excel at global feature learning — a hybrid architecture can combine both strengths. Carrier tokens can effectively summarize local window information, enabling efficient cross-window interaction without full global attention. Hierarchical attention complexity can grow almost linearly with input resolution by making local windowed attention the compute bottleneck.

#### 1.2 Objectives
For each transformer block, we use an interleaved pattern of local and, newly proposed, Hierarchical Attention blocks to capture both short and long-range spatial dependencies and efficiently model the cross-window interactions (HAT). It learns carrier tokens as a summary of each local window and efficiently models the cross-interaction between these regions.

![[Pasted image 20241124161913.png]]
Complexity of the Hierarchical Attention grows almost linearly with input image resolution, as the number of regions increases, due to the local windowed attention being the compute bottleneck.

#### 1.3 What's New
* Hierarchical Attention (HAT) mechanism using carrier tokens that summarize local windows and enable cross-window interaction at reduced cost
* Hybrid CNN-Transformer architecture with interleaved local and hierarchical attention blocks
* Near-linear complexity scaling with input resolution (vs. quadratic for standard ViTs)
* State-of-the-art throughput-accuracy trade-off across classification, detection, and segmentation

### 2. Methodology
#### 2.2 Data
* [[ImageNet]]-1K: 1.2M training and 50K validation images, 1000 categories (classification).
* [[ImageNet]]-21K: 14M images with 21,841 classes (pre-training).
* [[MS COCO]]: Detection and segmentation with Cascade Mask-RCNN.
* [[ADE20K]]: Semantic segmentation with UperNet.

#### 2.3 Model Architecture

![[Pasted image 20241124162245.png]]

**Patch Embeddings (Stem)**

```
kernel_size = 3
stride = 2
padding = 1

def FasterViTStem():
	x = Conv2d(num_channels, input_dim, kernel_size, stride, padding, bias=False)
	x = BatchNorm(input_dim)
	x = ReLU()
	x = Conv2d(input_dim, embed_dim, kernel_size, stride, padding, bias=False)
	x = BatchNorm(input_dim)
	x = ReLU()

```

> [!info]- Comments
> *[[BatchNorm]]:* goal of a Batchnorm is to reduce the covariate shift and makes the distribution more stable, accelerating the training and achieving higher accuracy.
>
> *[[ReLU]]*: stands as Rectified linear Unit --> $ReLU(x) = max(0,x)$
>
> Convolution formula   $\dfrac{Input - kernel + 2*Padding}{Stride}$

**Residual Conv Blocks**

$$
\hat{x} = GELU(BN(Conv_{3\times3}(x))), \\
$$
$$
x = BN(Conv_{3\times3}(\hat{x}) + x)
$$
```
def ResidualConvBlock(input):

	x = Conv2D(dim, dim, kernel_size, stride, padding)
	x = Batchnorm2d(dim, epsilon)
	x = GELU()
	x = Conv2D(dim, dim, kernel_size, stride, padding)
	x = Batchnorm2d(dim, epsilon)
	x = input + x

```

> [!info]- Comments
> *[[GELU]]*: stands as Gaussian Error Linear Unit, formula is a bit more complicated than ReLU, is derived from an approximation of the cumulative distribution (CDF) of the standard normal distribution
> $$GELU(x) = x \times CDF(x) = x \frac{1}{2}(1  + erf(\frac{x}{\sqrt{2}}))$$ being *erf* the Gaussian error function. Due to the complex function, it is used commonly an approximation of CDF, $1 + erf (\frac{x}{\sqrt{2}}) = 1 + tanh(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3))$.  Leaving the formula as follows:
> $$GELU_{tanh}(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}(x + 0.044715x^3)}))$$
>
> There is also a simpler sigmoid-based approximation:
> $$GELU_{sigmoid}(x) = x \times sigmoid(1.702 \times x)$$


**Downsample**

```
def DownSample(dim):
	LayerNorm2d(dim)
	Conv2d(dim, dim*2, kernel_size, stride, padding, bias=False)

```


**Hierarchical Attention Block (HAT)**

![[Pasted image 20241124164924.png]]
*Carrier Token*: Summarize role of the entire local window.

*First Attention block* is applied on CTs to summarize and pass global information.
Then  Window tokens + CTs are concatenated --> every local window has access only to its own set of CTs.
Performing self attention on concatenated tokens we facilitate local and global information exchange at reduced cost.
Feature map is divided into $n \times n$ local windows, being $n$  $\dfrac{H \times W}{k^2}$ , where $k$ is the window size.
We Initialize CTs by pooling them to $L = 2^c$  tokens per window, being $c$ something called *control latency*, fixed to 1.
$$\hat{x_c} = Conv_{3\times3}(x)$$
$$\hat{x}_{ct} = AvgPool_{H \times W \rightarrow n^2 L}(\hat{x_c})$$

```
window_size = 7
ct_size = 1
output_size_ct = int(ct_size * input_res/window_size)
stride_size_ct = int(input_resolution/output_size)
kernel_size_ct = input_res - (output_size - 1) * stride_size
kernel_size_pos_embed = 3
padding = 1

def InitializeCarrierToken():
	conv2D(dim, dim, kernel_size_pos_embed, padding, groups=dim)
	AvgPool2D(kernel_size_ct, stride_size_ct)
	ConvertTo(B,C,H // window_size, window_size, W // window_size, window_size)
	ReshapeTo(-1,H*W,C)

```

**HAT block**:
$$ \hat{x_{ct}} = \hat{x_{ct}} + \gamma_{1} \cdot MHSA(LN(\hat{x_{ct}}))$$
$$ \hat{x_{ct}} = \hat{x_{ct}} + \gamma_{2} \cdot MLP_{d \rightarrow 4d \rightarrow d}(LN(\hat{x_{ct}}))$$
where MHSA represents Multi head self attention and MLP is a 2-layer MLP with GeLU act. function

#### 2.4 Implementation Details
- Trained FasterViT models using **LAMB optimizer** for 300 epochs with a learning rate of 5e-3 and a total batch size of 4096 using **32 A100 GPUs**.
- **Data Augmentation**: same strategies as in previous efforts.
- **Exponential Moving Average (EMA)** used during training.
- For **pre-training on ImageNet-21K**: 90 epochs with LR 4e-3. Fine-tuned for 60 epochs with LR 7e-5.

### 3. Results

#### 3.1 Image Classification

Benchmark on [[ImageNet]]-1K (BS 128 with A100).

**Conv-Based**

| Model            | Image Size (Px) | Param (M) | Flops (G) | Throughput (Img/Sec) | Top-1 |
| ---------------- | --------------- | --------- | --------- | -------------------- | ----- |
| ConvNeXt-T       | 224             | 28.6      | 4.5       | 3196                 | 82.0  |
| ConvNeXt-S       | 224             | 50.2      | 8.7       | 2008                 | 83.1  |
| ConvNeXt-B       | 224             | 88.6      | 15.4      | 1485                 | 83.8  |
| RegNetY-040      | 288             | 20.6      | 6.6       | 3227                 | 83.0  |
| ResNetV2-101     | 224             | 44.5      | 7.8       | 4019                 | 82.0  |
| EfficientNetV2-S | 384             | 21.5      | 8.0       | 1735                 | 83.9  |

**Transformer-Based**

| Model          | Image Size (Px) | Param (M) | Flops (G) | Throughput (Img/Sec) | Top-1 |
| -------------- | :-------------: | :-------: | :-------: | :------------------: | :---: |
| Swin-T         |       224       |   28.3    |    4.4    |         2758         | 81.3  |
| Swin-S         |       224       |   49.6    |    8.5    |         1720         | 83.2  |
| SwinV2-T       |       256       |   28.3    |    4.4    |         1674         | 81.8  |
| SwinV2-S       |       256       |   49.7    |    8.5    |         1043         | 83.8  |
| SwinV2-B       |       256       |   87.9    |   15.1    |         535          | 84.6  |
| Twins-B        |       224       |   56.1    |    8.3    |         1926         | 83.1  |
| DeiT3-L        |       224       |   304.4   |   59.7    |         535          | 84.8  |
| PoolFormer-M58 |       224       |   73.5    |   11.6    |         884          | 82.4  |

**FasterViT (Hybrid Transformer)**

| Model       | Image Size (Px) | Param (M) | Flops (G) | Throughput (Img/Sec) |  Top-1   |
| ----------- | :-------------: | :-------: | :-------: | :------------------: | :------: |
| FasterViT-0 |       224       |   31.4    |    3.3    |       **5802**       | **82.1** |
| FasterViT-1 |       224       |   53.4    |    5.3    |       **4188**       | **83.2** |
| FasterViT-2 |       224       |   75.9    |    8.7    |       **3161**       | **84.2** |
| FasterViT-3 |       224       |   159.5   |   18.2    |       **1780**       | **84.9** |
| FasterViT-4 |       224       |   424.6   |   36.6    |       **849**        | **85.4** |
| FasterViT-5 |       224       |   957.5   |   113.0   |       **449**        | **85.6** |
| FasterViT-6 |       224       |  1360.0   |   142.0   |       **352**        | **85.8** |



![[Pasted image 20241125233536.png| 400]]


#### 3.2 Detection and Segmentation

We used the [[MS COCO]] dataset to finetune a Cascade Mask-RCNN network. Trained all models with AdamW optimizer with an initial learning rate of 1e-4, a 3x schedule, weight decay of 5e-2 and a total batch size of 16 on 8 A100 GPUs.

| Backbone      | throu. (im/sec) | AP mask            | AP box                 |
| ------------- | --------------- | ------------------ | ---------------------- |
|               |                 | Box   50   75      | Mask  50      75       |
| Swin-T        | 161             | 50.4  69  54.7     | 43.7   66.6   47.3     |
| ConvNeXt-T    | 166             | 50.4  69  54.8     | 43.7   66.5   47.3     |
| DeiT-Small/16 | 269             | 48.0  67  51.7     | 41.4   64.2   44.3     |
| FasterViT-2   | 287             | **52.1  71  56.6** | **45.4   68.4   49.0** |
|               |                 |                    |                        |
| Swin-S        | 119             | 51.9  70.7  56.3   | 45.0  68.2  48.8       |
| X101-32       | 124             | 48.1  66.5  52.4   | 41.6  63.9  45.2       |
| ConvNeXt-S    | 128             | 51.9  70.8  56.5   | 45.0  68.4  49.1       |
| FasterViT-3   | 159             | 52.4  71.1  56.7   | 45.4  68.7  49.3       |
|               |                 |                    |                        |
| X101-64       | 86              | 48.3  66.4  52.3   | 41.7  64.0  45.1       |
| Swin-B        | 90              | 51.9  70.5  56.4   | 45.0  68.1  48.9       |
| ConvNeXt-B    | 101             | 52.7  71.3  57.2   | 45.6  68.9  49.5       |
| FasterViT-4   | 117             | 52.9  71.6  57.7   | 45.8  69.1  49.8       |


#### 3.3 Semantic Segmentation

We employed [[ADE20K]] dataset to finetune an UperNet network with pre-trained FasterViT backbones. Trained all models with AdamW optimizer, LR 6e-5, weight decay 1e-2 and total batch size of 16 on 8 A100 GPUs.

| Model       | Throughput | FLOPs | IoU(ss/ms) |
| ----------- | ---------- | ----- | ---------- |
| Swin-T      | 350        | 945   | 44.5/45    |
| ConvNeXt-T  | 363        | 939   | -/46.7     |
| FasterViT-2 | 377        | 974   | 47.2/48.4  |
|             |            |       |            |
| Twins-SVT-B | 204        | -     | 47.7/48.9  |
| Swin-S      | 219        | 1038  | 47.6/49.5  |
| ConvNeXt-S  | 234        | 1027  | -/49.6     |
| FasterViT-3 | 254        | 1076  | 48.7/49.7  |
|             |            |       |            |
| Twins-SVT-L | 164        | -     | 48.8/50.2  |
| Swin-B      | 172        | 1188  | 48.1/49.7  |
| ConvNeXt-B  | 189        | 1170  | -/49.9     |
| FasterViT-4 | 202        | 1290  | 49.1/50.3  |

#### 3.4 Limitations
- Largest variants (FasterViT-5/6) have massive parameter counts (957M–1.3B), making them impractical for edge deployment despite the architecture's efficiency focus.
- Throughput advantage narrows at larger model scales: FasterViT-6 achieves only 352 img/s compared to DeiT3-L at 535 img/s, meaning the larger FasterViT variants trade throughput for accuracy rather than maintaining both.
- The carrier token mechanism adds a fixed overhead (pooling, global attention on CTs, concatenation) that may not pay off for very low-resolution inputs where standard windowed attention is already cheap.
- The HAT block pseudocode in this note leaves the positional embedding function (`PosEmbeds`) empty, suggesting the positional encoding details are non-trivial and not straightforward to summarize.

### 4. Appendix

### 5. Connections
- Uses [[Swin Transformer]] windowed attention as baseline, extends it with carrier tokens for cross-window interaction
- Incorporates ideas from [[Skip-Attention, Improving Vision Transformers by Paying Less Attention]] — same research group, SkipAt ideas were incorporated into FasterViT's design
- Compared against [[ConvNeXt]], [[DeiT]], [[Swin Transformer]] — consistently better throughput-accuracy trade-off
- Hybrid CNN-Transformer approach similar to CoAtNet but with the novel HAT mechanism
