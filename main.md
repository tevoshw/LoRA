# LoRA (Low-Rank Adaptation)

## Core Idea

Fine-tuning produces new weights:

$$W_{final} = W_{original} + \Delta W$$

$\Delta W$ is the **change** training made to the weights. It has the same shape as $W$: $(d, k)$.

Key observation: $\Delta W$ has **low rank**. It is redundant (some columns are combinations of others), so it spends memory and compute on directions that barely matter.

**LoRA's fix:** assume low rank from the start and train two small matrices instead of the full $\Delta W$:

$$\Delta W \approx B \cdot A$$

$$W_{final} = W_{original} + \frac{\alpha}{r} \, B A$$

## Understanding B·A

| Matrix | Shape | Role |
|--------|-------|------|
| $W$ | $(d, k)$ | Original weights, **frozen** |
| $B$ | $(d, r)$ | Expands back to output: "what to write" |
| $A$ | $(r, k)$ | Compresses the input: "what to look at" |
| $BA$ | $(d, k)$ | Same shape as $\Delta W$, but rank $\le r$ |

- $r$ is the **rank**: how many independent directions the update can have. It is the width of the bottleneck, and input and output stay at $d$ and $k$ dimensions.
- $BA$ is a sum of $r$ rank-1 terms, so it **cannot** hold more than $r$ directions. Noise and weak directions have no room. The limit comes from the shape, not from learning.
- LoRA does not remove redundancy after the fact. It never builds the full $\Delta W$.

**Forward pass:**

$$h = Wx + \frac{\alpha}{r} \, B(Ax)$$

**Parameters** ($d = k = 4096$, $r = 16$): $\Delta W$ has 16.7M, LoRA has about 131K.

**Initialization:** $A$ is random, $B$ is zero, so $BA = 0$ and training starts from the original model.

## Scaling: α/r

- Dividing by $r$ **normalizes**: more terms in the sum would otherwise inflate the update when $r$ grows.
- $\alpha$ sets the **intensity** of the adapter.
- Rule of thumb: $\alpha = r$ or $2r$.

## QLoRA

Same mechanics as LoRA, but the frozen $W$ is **quantized to 4 bits (NF4)**. Only $A$ and $B$ stay in high precision (bf16) and receive gradients. The goal is lower memory, not a different method.

> LoRA/QLoRA is not a universal solution. It fits **behavior changes** (tone, format, narrow tasks), not large-scale knowledge injection.

## Trade-off: Full FT vs LoRA

| | Full FT | LoRA / QLoRA |
|---|---|---|
| Capacity | Higher (any rank) | Limited by $r$ |
| Forgetting | **More** (updates everything) | Less ($W$ intact, low-rank update) |
| Memory and compute | Very high | Low |
| Storage | One full model per task | Small adapters (MBs) |

**Forgetting** means losing PT capabilities because weights were overwritten for the new task.

LoRA **learns less and forgets less**.

## When to Use What

| Scenario | Choice |
|----------|--------|
| Tone, style, format, narrow task | LoRA / QLoRA |
| Limited GPU | QLoRA |
| One assistant per client | LoRA (one adapter each) |
| Knowledge from documents | RAG |
| New language or huge domain corpus | Full FT (or high-rank LoRA) |

**Practical order:** start with LoRA, then raise $r$ and target all linear layers, and move to full FT only if evaluation shows a real gap.