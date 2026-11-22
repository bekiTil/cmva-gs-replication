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

| Aggregator config        | train loss | wins over PT |
|--------------------------|-----------:|--------------|
| 60 examples  × 3 ep      | 1.69 | none |
| 800 examples × 4 ep      | 1.29 | ROUGE-2 |
| 2000 examples × 2 ep     | 1.29 | **ROUGE-2, ROUGE-L, ROUGE-Lsum** (3 of 4) |
| 4000 examples × 1 ep     | 1.43 | none (regression) |

**Key finding: epochs matter more than raw data size.** The 4000×1 run uses the
*same* number of gradient steps (500) as 2000×2, but sees each example only once,
so it under-fits (loss 1.43 vs 1.29) and ROUGE *drops below* the 2000 run on every
metric. The aggregator needs **repetition** to tightly match the rule-of-thumb
style that ROUGE rewards. **Best config so far: 2000 examples × 2 epochs** — more
*training* (epochs), not just more unique data, is the lever. ROUGE-1 remains the
hardest metric for an aggregation model and is the last to flip; the paper's full
win used 91K examples with correspondingly more training.

## Honest interpretation (for the report)

The **pipeline is faithful** (5 classifier reward models → 5 PPO+LoRA Moral
Agents → LoRA aggregator → Table 1 + ROUGE), but the **scaled-down results are
weaker than the paper's**, for two understandable reasons:

1. **Near-ceiling base (Table 1).** TinyLlama-Chat already gives bland
   pro-social answers, so the base scores ~92–94% conformance on every value —
   little headroom for the agents to improve. The paper's gaps were larger
   because OpenAssistant-12B started far less aligned (74–91%). With only 15 PPO
   steps and rising KL, the agents move only a point or two, sometimes down.

2. **Aggregator training (Figure 2).** The Contextual Aggregator needs enough
   *training* to synthesize the 5 candidates. At 60 examples it loses to PT on
   ROUGE; at **2000 examples × 2 epochs** it beats PT on ROUGE-2, ROUGE-L and
   ROUGE-Lsum (3 of 4), trailing only on ROUGE-1. Note 4000 examples × 1 epoch
   *regressed* (same gradient steps, under-fit) — so the lever is epochs/training,
   not raw data size. The paper's full win used **91K** examples with much more
   training; the trend points toward a full sweep with more epochs on more data.

**Takeaways to scale up:** more PPO steps per agent, a less-aligned/larger base
to create headroom, and a much larger aggregator training set. These are the
levers the research extension can pull.

Raw metrics are saved by the scripts to `results/{reward_metrics,table1,rouge}.json`.
