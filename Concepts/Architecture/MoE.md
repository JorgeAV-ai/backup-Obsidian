---
tags:
  - basics
  - architecture
---

**TL;DR:** Mixture of Experts activates only a subset of parameters per input token. A router selects top-k experts out of N, scaling model capacity without proportionally scaling compute. Mixtral 8x7B: 47B params, ~13B active.

## What is it?

**Mixture of Experts (MoE)** is an architecture pattern where each input is processed by only a few "expert" subnetworks selected by a learned routing function. This decouples total model capacity (all parameters) from per-token compute cost (only active parameters).

In a standard transformer, every token passes through the same FFN. In an MoE transformer, the FFN is replaced by N parallel expert FFNs, and a **router** (gating network) decides which top-k experts process each token.

## How it works

![[basics_moe.png]]

[🔗 Open interactive MoE Visualizer](../../interactive/moe.html)

### Formula

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

Where:
- $E_i(x)$ is the output of expert $i$ applied to input $x$
- $G(x)_i$ is the gate value for expert $i$ — only the **top-k** gate values are non-zero
- In practice, $G(x) = \text{TopK}(\text{Softmax}(W_g \cdot x))$, setting all but the top-k values to zero

### Router / Gating

```
# Token-level routing
router_logits = x @ W_gate           # [batch*seq, num_experts]
router_probs = softmax(router_logits)
top_k_values, top_k_indices = topk(router_probs, k)
top_k_values = top_k_values / top_k_values.sum()  # renormalize
```

### Pseudocode: MoE layer with top-k routing

```
class MoELayer:
    def __init__(self, num_experts, top_k, hidden_dim):
        self.experts = [FFN(hidden_dim) for _ in range(num_experts)]
        self.gate = Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x: [batch * seq_len, hidden_dim]

        # 1. Route: compute gating scores
        logits = self.gate(x)                          # [tokens, num_experts]
        probs = softmax(logits, dim=-1)
        top_k_probs, top_k_idx = topk(probs, k=self.top_k)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 2. Dispatch: send each token to its selected experts
        output = zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_idx[:, i]               # which expert for each token
            weight = top_k_probs[:, i]                  # gating weight

            for e in range(num_experts):
                mask = (expert_idx == e)                # tokens assigned to expert e
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    output[mask] += weight[mask].unsqueeze(-1) * expert_out

        return output
```

### Load balancing loss

Without regularization, the router often collapses — sending most tokens to one or two experts while the rest are unused. The **load balancing loss** encourages uniform expert utilization:

$$\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $p_i$ = mean router probability for expert $i$
- This loss is minimized when both $f_i$ and $p_i$ are uniform ($1/N$)

This auxiliary loss is added to the main training loss with a small coefficient (e.g., 0.01).

### Concrete example: Mixtral 8x7B

| Property | Value |
|---|---|
| Total experts | 8 |
| Active experts per token | 2 (top-2 routing) |
| Expert size | ~7B params each |
| Total parameters | ~47B |
| Active parameters per token | ~13B |
| Effective compute | Similar to a dense 13B model |
| Capacity | Much larger than a dense 13B model |

Each transformer layer has its own set of 8 experts and its own router. The attention layers are shared (not expert-ified).

## Why it matters

- **Scale capacity without scaling compute**: a 47B MoE model runs at roughly the cost of a 13B dense model
- **Conditional computation**: different experts can specialize in different types of tokens or domains
- **Training efficiency**: more parameters means more capacity to absorb data, but compute stays manageable
- **Trade-off**: MoE models require more memory (all experts must be loaded) and more communication bandwidth in distributed settings

## Used in

- **Mixtral 8x7B** (Mistral AI): top-2 routing, 8 experts
- **Switch Transformer** (Google): top-1 routing for simplicity
- **GShard** (Google): early large-scale MoE for machine translation
- **GPT-4** (rumored to use MoE architecture)

---

See also: [[Transformer]], [[Softmax]]
