#!/usr/bin/env bash
# One-shot 10K pipeline for Colab: rebuilds everything a fresh VM is missing,
# caches the 10K agent candidates once, then trains + evals BOTH aggregators.
#   1. baseline: plain CMVA-GS aggregator (10K x2)  -> must beat PT (paper's claim)
#   2. treatment: CoT aggregator (10K x2) on the SAME cached candidates
# Run from Colab with a single cell:
#   !wget -q -O /content/run_10k.sh https://raw.githubusercontent.com/bekiTil/cmva-gs-replication/main/run_10k.sh && bash /content/run_10k.sh
set -e

cd /content
pip -q install transformers==5.0.0 trl==0.11.4 peft==0.22.0 datasets accelerate evaluate rouge-score scikit-learn beautifulsoup4 sentencepiece

if [ ! -d cmva-gs-replication ]; then
  git clone -q https://github.com/bekiTil/cmva-gs-replication.git
fi
cd cmva-gs-replication
git pull -q || true

echo "===== [1/8] data ====="
if [ ! -f data/raw/social-chem-101/social-chem-101.v1.0.tsv ]; then
  mkdir -p data/raw
  wget -q -O data/raw/sc.zip https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip
  (cd data/raw && unzip -o -q sc.zip)
fi
[ -d data/prepared/reward ] || python3 src/prepare_data.py --raw data/raw/social-chem-101/social-chem-101.v1.0.tsv
[ -f data/prepared/cot/train.jsonl ] || python3 src/build_cot_data.py

echo "===== [2/8] reward models ====="
[ -f models/reward/loyalty/model.safetensors ] || python3 src/train_reward_models.py --max_train 12000 --epochs 2

echo "===== [3/8] PPO moral agents ====="
if [ ! -d models/agents/loyalty ]; then
  for v in care fairness authority sanctity loyalty; do
    python3 src/train_ppo_agent.py --value "$v" --steps 20 --n_prompts 1000
  done
fi

echo "===== [4/8] cache 10K candidates (once) ====="
[ -f data/prepared/candidates/train10k.jsonl ] || \
  python3 src/gen_candidates.py --n_train 10000 --out data/prepared/candidates/train10k.jsonl

echo "===== [5/8] train PLAIN aggregator (baseline, 10K x2) ====="
python3 src/train_aggregator.py --candidates data/prepared/candidates/train10k.jsonl --n_train 10000 --epochs 2

echo "===== [6/8] eval PLAIN vs PT ====="
python3 src/eval_rouge.py --n 150 --agg_dir models/aggregator --label CMVA-GS --out results/rouge_plain10k.json

echo "===== [7/8] train CoT aggregator (10K x2, same candidates) ====="
python3 src/train_aggregator_cot.py --candidates data/prepared/candidates/train10k.jsonl --n_train 10000 --epochs 2

echo "===== [8/8] eval CoT vs PT ====="
python3 src/eval_rouge.py --n 150 --cot --agg_dir models/aggregator_cot --label CoT-CA --out results/rouge_cot10k.json

echo "ALL DONE"
echo "=== PLAIN (baseline) rouge_plain10k.json ==="; cat results/rouge_plain10k.json
echo "=== COT rouge_cot10k.json ==="; cat results/rouge_cot10k.json
