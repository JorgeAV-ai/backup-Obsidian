> [!quote] Information 
> * paper Paper [Link](https://arxiv.org/pdf/2306.06189)
> * calendar Date 1 April 2024
> * ? Motivation: 
> 		Vision transformers models still face high computational cost due to quadratic complexity in self-attentions. In addition, it is well known that CNNs provides great local features representations, feature that Vision Transformers lack of,  since they provide global feature learning.
> *  Dataset Datasets:
> 	[[ImageNet]]
> 	[[MS COCO]]
> 	[[ADE20K]]
> 	
> * Fields Related fields: 
> 	[[Vision Transformers]]
> 	[[Towards Enhanced Efficiency]]
> 	[[Self-Attention mechanisms]]
> 	


## 1. Introduction

Novel hybrid architecture, tailored for high-resolution input images, while maintaining a fast image throughput.

For each transformer block, we use an interleaved pattern of local and, newly proposed, Hierarchical Attention blocks to capture both short and long-range spatial dependencies and efficiently model the cross-window interactions (HAT). It learns carrier tokens as a summary of each local window and efficiently models the cross-interaction between these regions.

![[Pasted image 20241124161913.png]]
Complexity of the Hierarchical Attention grows almost linearly with input image resolution, as the number of regions increases, due to the local windowed attention being the compute bottleneck.


## 2 Architecture

![[Pasted image 20241124162245.png]]

##### 2.1 **Patch Embeddings (Stem)**

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

*Small reminder, the goal of a Batchnorm is to reduce the covariate shift and makes the distribution more stable, accelerating the training and achieving higher accuracy.* 

*Additional reminder: ReLU states as Rectified linear Unit --> $ReLU(x) = max(0,x)$*

*third reminder: formula Convolution --> * $\dfrac{Input - kernel + 2*Padding}{Stride}$



##### 2.2 **Residual Conv Block**s: 

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

##### 2.3 **Downsample**: 
reduced by 2 between stages 2D Layer norm + conv Layer 3x3, Stride=2

```
def DownSample():
	

```


##### 2.4 **Hierarchical attention Block** :

![[Pasted image 20241124164924.png]]
*Carrier Token*: Summarize role of the entire local window. 
*First Attention block* is applied on CTs to summarize and pass global information. 
Then  Window tokens + CTs are concatenated --> every local window has access only to its own set of CTs. 
Performing self attention on concatenated tokens we facilitate local and global information exchange at reduced cost.
Feature map is divided into $n \times n$ local windows, being $n$  $\dfrac{H \times W}{k^2}$ , where $k$ is the window size.
We Initialize CTs by pooling them to $L = 2^c$  tokens per window, being $c$ something called *control latency*, fixed to 1.
$$ \hat{x_c} = Conv_{3\times3}(x)$$
$$\hat{x}_{ct} = AvgPool_{H \times W \rightarrow n^2 L}(\hat{x_c})
$$


**HAT block**:
$$ \hat{x_{ct}} = \hat{x_{ct}} + \gamma_{1} \cdot MHSA(LN(\hat{x_{ct}}))$$
$$ \hat{x_{ct}} = \hat{x_{ct}} + \gamma_{2} \cdot MLP_{d \rightarrow 4d \rightarrow d}(LN(\hat{x_{ct}}))$$
where MHSA represents Multi head self attention and MLP is a 2-layer MLP with GeLU act. function

## 3 Results
(measured with bs of 128 and A100, Imagenet-1K)
![[Pasted image 20241124173435.png | center| 400]]

![[Pasted image 20241125233536.png| 400]]


**Image Classification**: We employ the ImageNet-1K dataset (Deng et al., 2009) for classification that includes 1.2M and 50K training and validation images. The dataset has 1000 categories and we report the performance in terms of top-1 accuracy. In addition, we use ImageNet-21K dataset which has 14M images with 21841 classes for pretraining.

Trained FasterViT models using LAMB optimizer (You et al., 2019) optimizer for 300 epochs with a learning rate of 5e-3 and a total batch size of 4096 using 32 A100 GPUs.

For Data Augmentation, they used the same strategies as in previous efforts (?)
They also used Exponential Moving Average (EMA).

For pre-training on ImageNet-21K, we train the models for 90 epochs with a learning rate of 4e-3. In addition, we fine-tune the models for 60 epochs with a learning rate of 7e-5.


**Detection and Segmentation** We used the MS COCO dataset to finetune a Cascade Mask-RCNN network. For this purpose, we trained all models with AdamW optimizer with an initial learning rate of 1e-4, a 3 x schedule, weight decay of 5e-2 and a total batch size of 16 on 8 A100 GPUs

**Semantic Segmentation**: we employed ADE20K dataset  to finetune an UperNet network with pre-trained FasterViT backbones. Specifically, we trained all models with Adam-W optimizer and by using a learning rate of 6e-5, weight decay of 1e-2 and total batch size of 16 on 8 A100 GPUs