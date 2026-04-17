Quantization makes large language models practical to run by compressing their numerical representation, but every bit removed creates trade-offs in accuracy, robustness, and behavior.

## 1. Introduction: Why Quantization Exists

### 1.1 The memory wall

Large language models do not only scale into a compute problem; they scale into a memory problem. As parameter counts grow, the cost of storing and moving weights becomes one of the main constraints on inference.

### 1.2 Why scaling LLMs makes this unavoidable

A large model in FP16 or BF16 can require far more memory than consumer hardware can provide comfortably. Quantization exists because, without compression, many useful models remain expensive or impractical to run.

### 1.3 What this article will explain

This article explains how quantization works, why naive quantization fails in large models, how modern methods respond to that failure, and what costs appear when precision is reduced.

![[quantization_memory_wall.png]]

## 2. The Basic Intuition Behind Quantization

### 2.1 From high precision to low-bit representation

Quantization maps values from a higher-precision space into a smaller set of discrete levels. The gain is smaller memory and faster movement of data; the cost is reduced numerical resolution.

![[quantization_intuition.png]]

### 2.2 The quantization formula, reconstruction, and error

The quantization pipeline can be understood as a sequence of simple operations: scale a value, shift it into a discrete grid, round it, clip it to the allowed range, and then reconstruct it approximately during inference.

![[quantization_formula.png]]

### 2.3 What is actually lost when precision goes down

Lower precision does not randomly damage the model. It reduces the number of distinct values the model can represent, which increases approximation error and makes nearby values harder to distinguish.

## 3. Why Naive Quantization Fails in LLMs

### 3.1 Why simple rounding breaks large models

If model weights are just numbers, it is tempting to think that quantization is only a matter of rounding them more aggressively. In practice, large transformers are much more fragile than that intuition suggests.

### 3.2 Outliers, step-size inflation, and massive activations

The main failure mode is not rounding by itself but what rounding does when a tensor contains extreme values. Outliers can stretch the scale so much that the useful resolution of ordinary values collapses.

### 3.3 Why this problem gets worse at scale

As models grow, these pathologies become more structural. Large transformers develop outlier channels, token-level activation spikes, and behaviors that make naive quantization much less reliable.

![[quantization_outlier_distortion.png]]

## 4. How Quantization Is Actually Applied Today

### 4.1 Post-Training Quantization vs Quantization-Aware Training

Modern quantization usually begins with either post-training quantization, which adapts an already trained model, or quantization-aware training, which exposes the model to quantization during training or fine-tuning.

### 4.2 What modern methods try to preserve

Different methods try to preserve different things: salient weights, local scaling behavior, throughput, bitrate flexibility, or quality under very low precision.

### 4.3 Main methods and formats in practice

Today’s ecosystem is not a random list of acronyms. Formats and methods such as GGUF, AWQ, GPTQ, EXL2, and QAT exist because different deployment settings and error profiles demand different compromises.

### 4.4 When GGUF, AWQ, GPTQ, EXL2, or QAT make sense

Once the trade-offs are clear, the method landscape becomes easier to reason about. The right choice depends on whether the priority is local portability, GPU throughput, accuracy retention, or robustness at very low bit-widths.

## 5. The Hidden Costs of Quantization

### 5.1 Capability degradation: reasoning, context, and multimodal reliability

Reducing precision can degrade reasoning quality, long-context behavior, and multimodal reliability. These degradations are often uneven: some tasks remain stable while others deteriorate sharply.

### 5.2 Silent failures in agents and real-world behavior

Quantized models can also fail in quieter ways. Tool use, multi-step planning, and agentic workflows may become more brittle even when the model still appears fluent at the surface level.

### 5.3 Security, bias, and alignment distortions

Compression can also alter safety and behavioral properties. Quantization does not automatically make a model unsafe or biased, but it can distort the mechanisms that support alignment, guardrails, and stable behavior.

## 6. Conclusion: Quantization as a Core Design Layer

### 6.1 What quantization really gives us

Quantization makes large models cheaper to store, easier to move, and more practical to deploy. It is one of the main reasons advanced models can run outside specialized infrastructure.

### 6.2 What it can quietly take away

The same compression that makes deployment practical can also remove useful resolution from the model’s internal computation. That loss may show up as degraded reasoning, silent instability, or shifted behavior.

### 6.3 Where research is going next

Current research is pushing toward smarter low-bit methods, better protection of sensitive activations, hardware-aware formats, and finer evaluation of what compression changes beyond benchmark accuracy.
