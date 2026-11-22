# CMVA-GS Baseline Replication

Reproduction of the baseline from **"Contextual Moral Value Alignment Through
Context-Based Aggregation"** (Dognin et al., IBM Research, arXiv:2403.12805),
scaled to run on a single consumer GPU / Apple Silicon.

## What is replicated

| Paper component | This repo | Paper spec | Our scaled version |
|---|---|---|---|
| Base LLM | `TinyLlama-1.1B-Chat` | OpenAssistant 12B | 1.1B, LoRA |
| Data | Social Chemistry 101 | MIC (built from SocialChem) | same 5 foundations |
| Reward models | 5 value classifiers | classifier per value | DistilBERT, binary |
| Moral Agents | 5 PPO agents | PPO (TRL) per value | PPO + LoRA (TRL) |
| Aggregator (CA) | LoRA SFT | LoRA SFT | LoRA SFT |
| Eval | Table 1 + ROUGE | expected reward + ROUGE | same metrics |

The 5 moral values are **care, fairness, authority, sanctity, loyalty**
(Moral Foundations), exactly as in the paper.

> Note on data: the paper uses MIC, which is gated on HuggingFace. MIC is built
> from **Social Chemistry 101**, which is openly downloadable and carries the
> identical 5 moral-foundation labels and binarized moral judgments the paper's
> reward models rely on. We use SocialChem so the pipeline is fully reproducible
> without credentials. Swap in MIC by replacing `data/prepared/` if you have access.

## Pipeline

```
prepare_data.py        -> per-value reward datasets + prompt/pairs pools
train_reward_models.py -> 5 classifiers (reward models)
eval_conformance.py    -> P(conform) per value, for a given policy
train_ppo_agent.py     -> one Moral Agent (PPO+LoRA) per value
build_table.py         -> assembles Table 1 (PT vs Moral Agents)
train_aggregator.py    -> Contextual Aggregator (LoRA SFT)
eval_rouge.py          -> Figure 2 (ROUGE: PT-model vs CMVA-GS)
```

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # local (Apple MPS / CPU)
```

GPU-heavy steps (the 5 PPO agents, aggregator) are much faster on CUDA — see
`notebooks/colab_cmva_gs.ipynb` to run them on a free Colab T4.

## Hardware used for development
Apple Silicon (MPS), 17 GB unified memory, no CUDA. Reward models + data prep
run locally in minutes; PPO agents are the slow step (use Colab GPU).
