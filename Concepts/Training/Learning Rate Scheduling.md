---
tags:
  - basics
  - optimization
---

**TL;DR:** The learning rate changes over training following a schedule. Warmup prevents early instability, then cosine/step/linear decay reduces the rate toward zero. Trapezoidal schedules (warmup-stable-decay) are gaining traction.

## What is it?

A **learning rate schedule** defines how the learning rate $\alpha$ varies across training steps. Rather than using a fixed value, nearly all modern training pipelines modulate $\alpha$ to balance fast early progress with fine-grained late convergence.

## How it works

![[basics_lr_schedule.png]]

[🔗 Open interactive LR Schedule Explorer](../interactive/lr_schedule.html)

### 1. Linear Warmup

Ramp $\alpha$ linearly from 0 (or a small value) to the target learning rate over $T_w$ steps.

$$\alpha_t = \alpha_{\max} \cdot \frac{t}{T_w}, \quad t \leq T_w$$

```
def warmup(step, max_lr, warmup_steps):
    return max_lr * (step / warmup_steps)
```

**Why warmup?** At initialization, Adam's moment estimates $m$ and $v$ are zero. Early gradients can be large and noisy. A large learning rate at this point destabilizes training. Warmup lets the moment estimates "warm up" before applying the full learning rate.

### 2. Cosine Decay

After warmup, decay $\alpha$ following a cosine curve from $\alpha_{\max}$ to $\alpha_{\min}$:

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w} \cdot \pi\right)\right)$$

```
def cosine_decay(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return warmup(step, max_lr, warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
```

Cosine decay is gentle: most of training happens near $\alpha_{\max}$, with a smooth landing.

### 3. Step Decay

Multiply $\alpha$ by a factor $\gamma$ every $S$ steps:

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / S \rfloor}$$

```
def step_decay(step, initial_lr, gamma, step_size):
    return initial_lr * (gamma ** (step // step_size))
```

Common in older CNN training (e.g., divide by 10 at epochs 30, 60, 90). Less common in transformer training.

### 4. Trapezoidal (Warmup-Stable-Decay)

Three phases: linear warmup, constant rate, then decay. Used in ModernBERT.

```
def trapezoidal(step, max_lr, min_lr, warmup_steps, stable_steps, total_steps):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)          # warmup
    elif step < warmup_steps + stable_steps:
        return max_lr                                    # stable
    else:
        decay_steps = total_steps - warmup_steps - stable_steps
        progress = (step - warmup_steps - stable_steps) / decay_steps
        return max_lr - (max_lr - min_lr) * sqrt(progress)  # 1-sqrt decay
```

### Visual summary

```
Cosine:       /‾‾‾‾\___          (warmup + smooth decay)
Step:         /‾‾‾‾|__|__        (warmup + discrete drops)
Trapezoidal:  /‾‾‾‾‾‾‾‾\_       (warmup + stable + decay)
```

## Concrete examples

| Paper | Schedule | Details |
|---|---|---|
| **ModernBERT** | Trapezoidal | 3B token warmup, stable at $8 \times 10^{-4}$, 1-sqrt decay |
| **FasterViT** | Cosine | Cosine schedule with LAMB optimizer |
| **Most LLMs** | Cosine | Warmup 1-2k steps, cosine to ~10% of peak |

## Why it matters

- Without warmup, Adam's biased moment estimates cause erratic early updates
- Without decay, the model oscillates around the minimum instead of converging tightly
- The choice of schedule affects final performance: cosine tends to outperform linear decay
- Trapezoidal schedules are useful for long training runs where you want a long stable phase

## Used in

- All modern training pipelines: pretraining, fine-tuning, instruction tuning
- Every paper in this vault uses some form of LR scheduling

---

See also: [[Adam]], [[Backpropagation]]
