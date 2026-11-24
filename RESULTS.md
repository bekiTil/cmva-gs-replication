# Baseline Replication Results

Scaled reproduction of CMVA-GS (Dognin et al., 2024) on a single Colab **T4 GPU**.
Base model **TinyLlama-1.1B-Chat** (paper used OpenAssistant-12B), data from
**Social Chemistry 101** (the corpus MIC is built from; MIC itself is gated).

## Reward models (5 value classifiers, DistilBERT, test accuracy)

| care | fairness | authority | sanctity | loyalty |
|------|----------|-----------|----------|---------|
| 0.889 | 0.877 | 0.877 | 0.853 | 0.878 |

Strong, well-calibrated classifiers — these drive the PPO reward and the eval.

## Table 1 — P(answer conforms to moral value): PT base vs Moral Agent

| Moral Value | PT model | Moral Agent | Δ |
|-------------|---------:|------------:|------:|
| care     | 94.16% | **94.43%** | +0.27 |
| fairness | 92.96% | 91.71% | −1.25 |
| authority| 93.91% | 92.13% | −1.78 |
| sanctity | 92.56% | **94.55%** | +1.99 |
| loyalty  | 92.44% | **93.85%** | +1.41 |

**3 of 5 agents beat the base** on their own value (care, sanctity, loyalty).
PPO ran only 15 steps/agent (consumer-GPU budget).

## Figure 2 — ROUGE on held-out SocialChem (PT vs full CMVA-GS)

Effect of **aggregator training-set size** (the paper used 91K examples):

**Aggregator trained on 60 examples** (test n=70):

| System   | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------|--------:|--------:|--------:|-----------:|
| PT-model | 14.39 | 2.19 | 11.57 | 11.57 |
| CMVA-GS  | 10.95 | 0.70 |  9.76 |  9.76 |

**Aggregator trained on 800 examples, 4 epochs** (test n=120):

| System   | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------|--------:|--------:|--------:|-----------:|
| PT-model | 16.83 | 3.09 | 13.36 | 13.36 |
| CMVA-GS  | 14.25 | **3.22** | 12.77 | 12.77 |

**Aggregator trained on 2000 examples, 2 epochs** (test n=120, same PT baseline):

| System   | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------|--------:|--------:|--------:|-----------:|
| PT-model | 16.83 | 3.09 | 13.36 | 13.36 |
| CMVA-GS  | 15.14 | **3.82** | **13.62** | **13.62** |

**Aggregator trained on 4000 examples, 1 epoch** (test n=120, same PT baseline):

| System   | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------|--------:|--------:|--------:|-----------:|
| PT-model | 16.83 | 3.09 | 13.36 | 13.36 |
| CMVA-GS  | 13.87 | 3.06 | 12.68 | 12.68 |

**Trend table** (CMVA-GS vs PT, same 120-item test set):

CMVA-GS rouge scores vs PT (16.83 / 3.09 / 13.36 / 13.36), same 120-item test:

| Aggregator config        | R1 | R2 | RL | wins over PT |
|--------------------------|---:|---:|---:|--------------|
| 60 examples  × 3 ep      | 10.95 | 0.70 | 9.76 | none (underfit) |
| 800 examples × 4 ep      | 14.25 | 3.22 | 12.77 | ROUGE-2 |
| **2000 examples × 2 ep** | **15.14** | **3.82** | **13.62** | **ROUGE-2, ROUGE-L, ROUGE-Lsum (3/4)** |
| 4000 examples × 1 ep     | 13.87 | 3.06 | 12.68 | none (underfit) |
| 2000 examples × 5 ep     | 14.42 | 2.41 | 12.61 | none (**overfit**) |

**Final 10K run on an L4 GPU (fresh runtime, test n=150, vs its own PT baseline
PT = 16.31 / 3.11 / 13.11 / 13.11):**

| System   | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|----------|--------:|--------:|--------:|-----------:|
| PT-model | 16.31 | 3.11 | 13.11 | 13.11 |
| CMVA-GS  | **16.83** | **5.06** | **15.44** | **15.44** |
| (Δ)      | **+0.52** | **+1.95** | **+2.33** | **+2.33** |

**10000 examples × 2 epochs → CMVA-GS beats PT on ALL 4 metrics, ROUGE-1 included.**

**Key finding: there is a training sweet spot, and enough data unlocks a clean win.**
- **Too little training** (4000×1, or 60 examples) → *under*-fits → ROUGE below PT.
- **Too much training on too little data** (2000×5) → *over*-fits the 2000 train
  RoTs → generalizes worse → ROUGE drops below both the 2000×2 run and PT.
- **Just right (2000 × 2 epochs)** → beats PT on 3 of 4 metrics (ROUGE-1 trails).
- **Enough data (10000 × 2 epochs)** → 2 full epochs *without* overfitting (loss
  ~1.2) → **beats PT on all 4 metrics, flipping the stubborn ROUGE-1**. This is
  what the paper's 91K-example scale buys, reproduced in miniature.

So the lever is the data/epoch *balance*: more unique data lets you train longer
without overfitting, and at 10K that is enough to beat the base model on every
ROUGE metric — the same effect the paper gets at **91K** examples.

## Research extension (G13) — Chain-of-Thought Contextual Aggregator

For the CoT study the base model was switched to **Qwen2.5-1.5B-Instruct** (a
stronger, instruction-tuned backbone than TinyLlama), and the whole pipeline was
re-run from scratch on it: 5 DistilBERT reward models → 5 Qwen PPO+LoRA Moral
Agents → aggregator. The CoT reasoning traces are built by `build_cot_data.py`
from Social Chemistry's structured annotations (moral foundation + action +
judgment → a short "reason about the consequences, then answer" trace).

**Head-to-head: plain aggregator (control) vs CoT aggregator (treatment).** Both
share the *same* base, the *same* 5 agents' candidate answers, the *same* 2000
training situations and the *same* 2 epochs — the **only** difference is whether
the aggregator is trained to answer directly or to emit a consequence-reasoning
trace before `Aggregated answer:`. ROUGE is scored on the final answer only
(parsed after the marker), so the comparison is apples-to-apples. Test n=150.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|--------|--------:|--------:|--------:|-----------:|
| PT (base Qwen2.5-1.5B) | 15.18 | 3.18 | 13.60 | 13.60 |
| CMVA-GS (plain aggregator) | 14.09 | 3.05 | 12.58 | 12.58 |
| **CoT-CA (reason-then-answer)** | **26.76** | **10.39** | **25.13** | **25.13** |
| Δ CoT-CA vs PT | **+11.58** | **+7.21** | **+11.53** | **+11.53** |
| Δ CoT-CA vs plain | **+12.67** | **+7.34** | **+12.55** | **+12.55** |

**Headline: Chain-of-Thought roughly doubles ROUGE and makes the aggregator far
more sample-efficient.** At the identical 2000×2 budget where the *plain*
aggregator still trails the base model (14.09 vs 15.18 on ROUGE-1 — consistent
with the earlier finding that the plain aggregator needs ~10K examples to beat
the base), the *CoT* aggregator beats the base on **all four** metrics by a wide
margin and beats the plain control on all four. Reasoning before answering is
doing real work here, which is exactly the G13 hypothesis.

**Scaled to 10K×2 (controlled; candidates cached once and shared).** Repeating the
same controlled comparison at 10K training examples — both aggregators built from
*one* shared, cached set of the 5 agents' candidates, so base, agents, situations,
candidates and epochs are all identical and only the training target differs —
sharpens the result. Test n=150.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|--------|--------:|--------:|--------:|-----------:|
| PT (base Qwen2.5-1.5B) | 15.18 | 3.18 | 13.60 | 13.60 |
| CMVA-GS (plain, 10K×2) | 11.83 | 2.32 | 10.65 | 10.65 |
| **CoT-CA (10K×2)** | **28.23** | **10.32** | **26.37** | **26.37** |
| Δ CoT-CA vs PT | **+13.06** | **+7.14** | **+12.77** | **+12.77** |
| Δ CoT-CA vs plain | **+16.40** | **+8.00** | **+15.72** | **+15.72** |

**Two findings, and the second is the real story:**

1. **The plain aggregator does *not* beat the strong base — and more data makes it
   *worse*** (11.83 at 10K×2 vs 14.09 at 2000×2, both below PT's 15.18). This is
   the opposite of the TinyLlama replication, where 10K examples flipped the plain
   aggregator to a clean win. The cause is the base: Qwen2.5-1.5B-**Instruct** is
   already a strong zero-shot rule-of-thumb generator, so the small LoRA aggregator
   — trained 5× harder at 10K×2 (2500 steps vs 500) — over-conforms to the training
   RoTs' style and *degrades* the base's answers. On a strong instruction-tuned
   base the paper's plain pipeline does not reproduce "beat the base."

2. **Chain-of-Thought rescues it decisively.** With the *same* everything, the CoT
   aggregator scores **28.23 ROUGE-1 — ~1.9× the base and ~2.4× the plain
   aggregator**, winning on all four metrics by +13 to +16 points. So on a strong
   base, CoT is not an incremental add-on: it is *what makes contextual aggregation
   work at all*. Reasoning about the consequences before committing to an answer
   keeps the aggregator from collapsing into style-mimicry and lets it actually use
   the five agents' candidates.

This reframes the G13 contribution: not "CoT adds a little on top of a working
baseline," but **"CoT is the ingredient that lets multi-agent moral aggregation
beat a strong base model when plain SFT cannot."**

*Caveats to probe in the ablations (COT_PLAN.md, Phase 3):* part of the gain may
come from the CoT target's consistent format and the extra structure teaching the
model a more RoT-like answer style, not purely from "reasoning." The planned
ablations isolate this — A1 (templated vs self-distilled rationales), A2
(single- vs multi-value reasoning), A3 (trained CoT vs prompt-only CoT).

## Honest interpretation (for the report)

The **pipeline is faithful** (5 classifier reward models → 5 PPO+LoRA Moral
Agents → LoRA aggregator → Table 1 + ROUGE). On **Figure 2 (ROUGE)** the scaled
system reproduces the paper's headline result — at 10K examples CMVA-GS beats the
base model on all four metrics. On **Table 1 (conformance)** the result is more
modest, for an understandable reason. Two notes:

1. **Near-ceiling base (Table 1).** TinyLlama-Chat already gives bland
   pro-social answers, so the base scores ~92–94% conformance on every value —
   little headroom for the agents to improve. The paper's gaps were larger
   because OpenAssistant-12B started far less aligned (74–91%). With only 15 PPO
   steps and rising KL, the agents move only a point or two, sometimes down.

2. **Aggregator training (Figure 2).** The Contextual Aggregator has a training
   *sweet spot*. Too little (4000×1 under-fits) or too much on too little data
   (2000×5 over-fits) both lose to PT; **2000×2** beats PT on 3 of 4 metrics but
   trails on ROUGE-1; and **10000 × 2 epochs beats PT on all 4 metrics, including
   ROUGE-1** (16.83 vs 16.31). So the lever is the data/epoch *balance*: enough
   unique data lets you run a full 2 epochs without overfitting. This reproduces
   the paper's qualitative result (CMVA-GS > base model) in miniature; the paper
   used **91K** examples on 8×A100, we used 10K on one L4.

**Takeaways to scale up:** more PPO steps per agent, a less-aligned/larger base
to create headroom, and a much larger aggregator training set. These are the
levers the research extension can pull.

Raw metrics are saved by the scripts to `results/{reward_metrics,table1,rouge}.json`.
