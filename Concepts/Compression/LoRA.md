Low-Rank Adaptation (LoRA) fine-tunes large pretrained models by freezing the original weights and injecting small, trainable low-rank matrices, reducing the number of trainable parameters by orders of magnitude while matching full fine-tuning performance on many tasks.

## What is it?

Fine-tuning a large language model means updating all of its weight matrices, which for a model like GPT-3 (175B parameters) requires storing a full copy of gradients and optimizer states — hundreds of gigabytes of GPU memory. LoRA (Hu et al., 2021) observes that the weight updates during fine-tuning tend to have **low intrinsic rank**: you do not need to update all $d \times d$ entries of a weight matrix to adapt it to a new task.

Instead of modifying the pretrained weight matrix $W \in \mathbb{R}^{d \times d}$ directly, LoRA adds a parallel low-rank path:

$$h = (W + BA)x = Wx + BAx$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$, with rank $r \ll d$. The original $W$ is **frozen** (no gradients), and only $B$ and $A$ are trained. This reduces trainable parameters from $d^2$ to $2 \cdot r \cdot d$ — a massive reduction.

**Example:** For a weight matrix of size $4096 \times 4096$ (16.7M parameters), LoRA with $r = 8$ adds only $2 \times 8 \times 4096 = 65{,}536$ trainable parameters — a **256x reduction** for that single matrix. Across the full model, LoRA typically trains only **0.01%-0.1%** of total parameters.

## How it works

![[basics_lora.png]]

[🔗 Open interactive LoRA Explorer](../../interactive/lora.html)

### The LoRA layer

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # Freeze original weights
        self.original = original_layer
        self.original.weight.requires_grad = False

        # Low-rank trainable matrices
        self.A = nn.Linear(d_in, r, bias=False)    # down-projection
        self.B = nn.Linear(r, d_out, bias=False)   # up-projection

        # Initialization: A ~ Normal, B = 0 (so BA = 0 at start)
        nn.init.kaiming_normal_(self.A.weight)
        nn.init.zeros_(self.B.weight)

        # Scaling factor
        self.scaling = alpha / r

    def forward(self, x):
        base_out = self.original(x)           # Wx
        lora_out = self.B(self.A(x))          # BAx
        return base_out + lora_out * self.scaling
```

Key details:
- $B$ is initialized to **zero**, so at the start of training the LoRA path contributes nothing and the model behaves exactly as the pretrained one.
- The scaling factor $\alpha / r$ controls the magnitude of the LoRA update. $\alpha$ is a hyperparameter (commonly set to $2r$).
- LoRA is typically applied to the **attention projection matrices** ($W_Q$, $W_V$, and sometimes $W_K$, $W_O$), though it can be applied to any linear layer.

### Merging after training

Once fine-tuning is complete, the LoRA weights can be **merged** back into the original weights, incurring zero additional inference cost:

```python
def merge_lora(original_layer, lora_A, lora_B, scaling):
    # W_merged = W + scaling * B @ A
    delta = scaling * (lora_B.weight @ lora_A.weight)
    original_layer.weight.data += delta
    return original_layer   # now a standard layer, no extra computation
```

After merging, the model is a normal model with no architectural changes. This is a major practical advantage: you can ship a single merged checkpoint with no LoRA-specific inference code.

### Choosing the rank $r$

| Rank $r$ | Trainable params (per $4096 \times 4096$ layer) | Expressiveness |
|---|---|---|
| 4 | 32,768 | Low — works for simple domain adaptation |
| 8 | 65,536 | Good default for most tasks |
| 16 | 131,072 | Higher capacity, good for complex tasks |
| 32 | 262,144 | Near full fine-tuning quality on hard tasks |
| 64+ | 524,288+ | Diminishing returns; consider full fine-tuning |

In practice, $r = 8$ or $r = 16$ is the most common choice. The original paper shows that even $r = 4$ performs surprisingly well, supporting the low-rank hypothesis.

### QLoRA

QLoRA (Dettmers et al., 2023) combines LoRA with aggressive quantization: the base model is loaded in **4-bit** precision (using NF4 quantization), while the LoRA adapters remain in FP16/BF16. This makes it possible to fine-tune a 65B parameter model on a single 48GB GPU. The 4-bit weights are dequantized to FP16 on the fly during the forward pass, so the LoRA gradient computation happens in full precision. See [[Quantization]] for details on the quantization side.

## Why it matters

LoRA democratized fine-tuning of large models. Before LoRA, adapting GPT-3 or LLaMA to a specific task required multiple high-end GPUs just for optimizer states. With LoRA, a 7B model can be fine-tuned on a single consumer GPU, and a 70B model becomes feasible on a single A100. Beyond memory savings, LoRA adapters are tiny files (often 10-50 MB), making it easy to store and swap between hundreds of task-specific adapters on top of a single base model. This is the foundation of the open-source fine-tuning ecosystem (HuggingFace PEFT, etc.).

## Used in

- [[Transformer]]
- [[Quantization]]
