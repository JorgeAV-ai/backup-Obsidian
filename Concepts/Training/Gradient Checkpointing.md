# Gradient Checkpointing

> **TL;DR:** Trade compute for memory: instead of storing all intermediate activations for the backward pass, store only a few checkpoints and recompute the rest. Uses ~33% more compute but ~60--70% less activation memory.

---

## What is it?

During standard backpropagation, a neural network must store the activations (intermediate outputs) of every layer during the forward pass, because these are needed to compute gradients during the backward pass. For a network with $N$ layers, this means activation memory scales as $O(N)$. For large models (billions of parameters, long sequences), this activation memory often exceeds GPU capacity long before the parameter memory does.

Gradient checkpointing (also called **activation checkpointing** or **rematerialization**) is a memory optimization technique introduced by Chen et al. (2016). The idea: instead of storing every activation, store only a subset of them (the "checkpoints") and **recompute** the missing activations from the nearest checkpoint during the backward pass. You pay with extra compute (rerunning parts of the forward pass) but save a large fraction of memory.

This is the single most important technique for fitting large models into limited GPU memory, and it is used in virtually all large-scale training setups.

---

## How it works

![[basics_gradient_checkpointing.png]]

[🔗 Open interactive Gradient Checkpointing Demo](../../interactive/gradient_checkpointing.html)

### Standard forward/backward (no checkpointing)

In a network with layers $f_1, f_2, \dots, f_N$:

**Forward pass:** compute and store all activations.

$$a_0 \xrightarrow{f_1} a_1 \xrightarrow{f_2} a_2 \xrightarrow{f_3} \cdots \xrightarrow{f_N} a_N$$

All $a_0, a_1, \dots, a_N$ are kept in memory.

**Backward pass:** use stored activations to compute gradients layer by layer (from $N$ back to 1).

**Memory:** $O(N)$ activations stored.
**Compute:** $1 \times$ forward + $1 \times$ backward.

### Checkpointed forward/backward

**Strategy:** divide $N$ layers into $\sqrt{N}$ segments of $\sqrt{N}$ layers each. Store only the activation at the boundary of each segment.

**Forward pass:** compute all activations, but only keep the ones at segment boundaries.

$$a_0 \xrightarrow{f_1 \dots f_{\sqrt{N}}} \underset{\text{save}}{a_{\sqrt{N}}} \xrightarrow{f_{\sqrt{N}+1} \dots f_{2\sqrt{N}}} \underset{\text{save}}{a_{2\sqrt{N}}} \xrightarrow{\cdots} a_N$$

Intermediate activations within each segment are discarded.

**Backward pass:** for each segment, recompute the intermediate activations from the saved checkpoint, then compute gradients normally for that segment.

**Memory:** $O(\sqrt{N})$ stored activations (just the checkpoints, plus one segment's worth of recomputed activations at a time).
**Compute:** $1 \times$ forward + $1 \times$ backward + $1 \times$ partial recomputation $\approx 1.33 \times$ total.

### Pseudocode: standard vs checkpointed

```python
# ====== Standard training (no checkpointing) ======
def standard_forward_backward(layers, x, target):
    # Forward: store all activations
    activations = [x]
    for layer in layers:
        x = layer(x)
        activations.append(x)         # store every activation

    loss = loss_fn(x, target)

    # Backward: use stored activations
    grad = loss.backward_initial()
    for i in reversed(range(len(layers))):
        grad = layers[i].backward(grad, activations[i])

    # Memory: O(N) activations
    return loss


# ====== Checkpointed training ======
def checkpointed_forward_backward(layers, x, target, segment_size):
    """
    segment_size: typically sqrt(N), where N = len(layers)
    """
    N = len(layers)
    segments = split_into_chunks(layers, segment_size)

    # Forward: store only segment boundary activations
    checkpoints = [x]
    for segment in segments:
        for layer in segment:
            x = layer(x)              # compute but DON'T store intermediates
        checkpoints.append(x)         # store only segment output

    loss = loss_fn(x, target)

    # Backward: recompute within each segment
    grad = loss.backward_initial()
    for seg_idx in reversed(range(len(segments))):
        segment = segments[seg_idx]
        seg_input = checkpoints[seg_idx]

        # Recompute activations for this segment
        seg_activations = [seg_input]
        a = seg_input
        for layer in segment:
            a = layer(a)
            seg_activations.append(a)

        # Now compute gradients normally within this segment
        for i in reversed(range(len(segment))):
            grad = segment[i].backward(grad, seg_activations[i])

    # Memory: O(sqrt(N)) checkpoints + O(sqrt(N)) recomputed at a time
    return loss
```

### Memory comparison

| Approach | Activations stored | Compute overhead |
|---|---|---|
| **Standard** | $O(N)$ | 1x |
| **Checkpoint every $\sqrt{N}$** | $O(\sqrt{N})$ | ~1.33x |
| **Checkpoint every layer** (extreme) | $O(1)$ | ~2x |

### Practical numbers

For a Transformer with 96 layers (like GPT-3 scale):
- **Standard:** store 96 layers of activations
- **Checkpointing ($\sqrt{96} \approx 10$ segments):** store ~10 checkpoints + recompute up to 10 layers at a time
- **Memory savings:** ~60--70% reduction in activation memory
- **Compute cost:** ~33% more total FLOPs

---

## Why it matters

- **Enables training large models**: Without gradient checkpointing, many large models simply do not fit in GPU memory. It is the difference between "possible" and "impossible."
- **Scales sub-linearly**: The $O(\sqrt{N})$ memory scaling means doubling model depth only increases activation memory by $\sim 1.4 \times$, not $2 \times$.
- **Complements other memory techniques**: Works alongside [[Mixed Precision (FP16 BF16)]] (which halves memory per activation), optimizer state sharding (ZeRO), and model parallelism.
- **Easy to use**: In PyTorch, wrapping a module with `torch.utils.checkpoint.checkpoint()` is often a one-line change.
- **Adjustable trade-off**: You can choose how many checkpoints to use, trading off anywhere between full storage (fast, memory-heavy) and full recomputation (slow, memory-light).

---

## Used in

- GPT-3, GPT-4, and most large-scale LLM training
- All major training frameworks: Megatron-LM, DeepSpeed, FSDP
- PyTorch: `torch.utils.checkpoint`
- JAX: `jax.checkpoint` (called `remat`)
- Any training setup where activation memory is the bottleneck

---

**See also:** [[Backpropagation]], [[Mixed Precision (FP16 BF16)]]
