# RLHF

> **TL;DR:** Reinforcement Learning from Human Feedback aligns LLMs with human preferences through a three-stage pipeline: supervised fine-tuning, reward model training, and RL optimization. DPO simplifies this by skipping the reward model entirely.

---

## What is it?

RLHF (Reinforcement Learning from Human Feedback) is the technique that turns a capable-but-unaligned language model into one that is helpful, harmless, and honest. A base LLM trained on internet text can write fluently, but it will also confidently produce toxic, incorrect, or dangerous content. SFT (supervised fine-tuning) on curated examples helps, but it cannot cover every situation and the model may still produce bad outputs with high confidence.

RLHF solves this by training the model to optimize for what humans actually prefer. Instead of hand-writing rules for every possible situation, you collect human judgments on model outputs and train the model to produce outputs that humans would rate highly.

The technique was popularized by InstructGPT (Ouyang et al., 2022) and is the key ingredient that made ChatGPT feel qualitatively different from GPT-3.

---

## How it works

![[basics_rlhf.png]]

[🔗 Open interactive RLHF Demo](../../interactive/rlhf.html)

### The three-stage RLHF pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**

Start with a pretrained LLM. Fine-tune it on high-quality (prompt, response) pairs written by humans. This gives the model the right "format" -- it learns to follow instructions, answer questions, and refuse harmful requests.

**Stage 2: Reward Model Training**

Collect human preference data: show human annotators two (or more) model responses to the same prompt and ask which is better. Train a reward model $R(x, y)$ that takes a prompt $x$ and response $y$ and outputs a scalar score predicting human preference.

The reward model is trained with a pairwise ranking loss:

$$\mathcal{L}_{\text{RM}} = -\log\sigma(R(x, y_w) - R(x, y_l))$$

where $y_w$ is the preferred (winning) response and $y_l$ is the rejected (losing) response.

**Stage 3: RL Optimization (PPO)**

Use the reward model as a reward signal to fine-tune the SFT model with Proximal Policy Optimization (PPO). The objective:

$$\max_\pi \, \mathbb{E}_{x \sim D, \, y \sim \pi(\cdot|x)} \left[ R(x, y) - \beta \, \text{KL}(\pi(\cdot|x) | \pi_{\text{ref}}(\cdot|x)) \right]$$

The KL penalty prevents the model from drifting too far from the SFT model (reference policy $\pi_{\text{ref}}$), which would lead to reward hacking -- producing gibberish that exploits the reward model.

### DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) is an elegant alternative that skips the reward model entirely. The key insight: you can reparameterize the RLHF objective so that the optimal policy has a closed-form relationship with the reward function. This lets you optimize human preferences directly on the language model.

**DPO loss:**

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)$$

where:
- $\pi_\theta$ is the model being trained
- $\pi_{\text{ref}}$ is the frozen reference model (the SFT model)
- $y_w, y_l$ are the preferred and rejected responses
- $\beta$ controls how far the model can deviate from the reference

Intuitively, DPO increases the probability of preferred responses relative to the reference model and decreases the probability of rejected responses.

### Pseudocode: DPO training step

```python
def dpo_training_step(model, ref_model, batch, beta=0.1):
    """
    batch contains: prompts, chosen_responses, rejected_responses
    model: the policy being trained (trainable)
    ref_model: frozen copy of the SFT model
    """
    prompts, y_win, y_lose = batch

    # Compute log-probabilities under both models
    log_pi_win  = model.log_prob(y_win, given=prompts)
    log_pi_lose = model.log_prob(y_lose, given=prompts)
    log_ref_win  = ref_model.log_prob(y_win, given=prompts)    # no gradient
    log_ref_lose = ref_model.log_prob(y_lose, given=prompts)   # no gradient

    # Compute log-ratios
    log_ratio_win  = log_pi_win - log_ref_win
    log_ratio_lose = log_pi_lose - log_ref_lose

    # DPO loss
    logits = beta * (log_ratio_win - log_ratio_lose)
    loss = -log_sigmoid(logits).mean()

    loss.backward()
    optimizer.step()
    return loss.item()
```

### RLHF vs DPO comparison

| | RLHF (PPO) | DPO |
|---|---|---|
| **Components** | SFT + Reward Model + PPO | SFT + Direct Optimization |
| **Complexity** | High (4 models in memory during PPO) | Low (2 models: policy + reference) |
| **Stability** | Tricky to tune (PPO hyperparameters) | More stable, standard supervised loss |
| **Data** | Can generate on-policy data | Offline only (fixed preference dataset) |
| **Performance** | Strong with good tuning | Competitive, sometimes better |

---

## Why it matters

- **Alignment is the bottleneck**: Raw capability (pretraining) is necessary but not sufficient. Without RLHF/DPO, models are unreliable -- they hallucinate confidently, follow harmful instructions, and produce outputs misaligned with user intent.
- **SFT is not enough**: Supervised fine-tuning can only teach the model behaviors that are explicitly demonstrated. RLHF generalizes preferences across novel situations.
- **Human preferences are nuanced**: What makes a response "good" is complex and context-dependent. Learning from comparative judgments captures subtleties that rules cannot.
- **DPO democratized alignment**: By eliminating the reward model and PPO, DPO made alignment accessible to smaller teams without massive RL infrastructure.

---

## Used in

- InstructGPT / ChatGPT (OpenAI) -- the original RLHF application to LLMs
- Claude (Anthropic) -- RLHF with Constitutional AI (RLAIF)
- Llama 2 Chat (Meta) -- RLHF for open-source LLMs
- Zephyr, OpenChat, and many open-source models -- DPO alignment

---

**See also:** [[Softmax]], [[Autoregressive Generation]]
