#!/usr/bin/env bash
# Resumable 10K pipeline for Colab. Persists every expensive artifact to Google
# Drive as it is produced and restores it on a fresh VM, so you can close the PC
# and resume by re-running the same cell (which mounts Drive, then runs this).
#
#   Drive layout:  MyDrive/cmva_gs/{reward, agents, candidates, aggregator,
#                                   aggregator_cot, rouge_plain10k.json, rouge_cot10k.json}
#
#   Steps: data -> reward -> agents -> 10K candidate cache (incremental)
#          -> plain aggregator (baseline) -> eval vs PT
#          -> CoT aggregator -> eval
set -e

DRIVE="${DRIVE_DIR:-/content/drive/MyDrive/cmva_gs}"
mkdir -p "$DRIVE" "$DRIVE/candidates"

cd /content
pip -q install transformers==5.0.0 trl==0.11.4 peft==0.22.0 datasets accelerate evaluate rouge-score scikit-learn beautifulsoup4 sentencepiece
[ -d cmva-gs-replication ] || git clone -q https://github.com/bekiTil/cmva-gs-replication.git
cd cmva-gs-replication
git pull -q || true

echo "===== [0/8] restore persisted artifacts from Drive ====="
mkdir -p models data/prepared/candidates results
[ -d "$DRIVE/reward" ]         && cp -ru "$DRIVE/reward"         models/reward         || true
[ -d "$DRIVE/agents" ]         && cp -ru "$DRIVE/agents"         models/agents         || true
[ -d "$DRIVE/aggregator" ]     && cp -ru "$DRIVE/aggregator"     models/aggregator     || true
[ -d "$DRIVE/aggregator_cot" ] && cp -ru "$DRIVE/aggregator_cot" models/aggregator_cot || true

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
cp -ru models/reward "$DRIVE/reward"

echo "===== [3/8] PPO moral agents ====="
if [ ! -d models/agents/loyalty ]; then
  for v in care fairness authority sanctity loyalty; do
    python3 src/train_ppo_agent.py --value "$v" --steps 20 --n_prompts 1000
  done
fi
cp -ru models/agents "$DRIVE/agents"

echo "===== [4/8] cache 10K candidates (incremental, straight onto Drive) ====="
# Per-agent files land in $DRIVE/candidates, so partial progress survives a VM reset.
# A complete train10k.jsonl (e.g. snapshotted from an earlier run) short-circuits it.
CAND="$DRIVE/candidates/train10k.jsonl"
[ -f "$CAND" ] || python3 src/gen_candidates.py --n_train 10000 --out "$CAND"

echo "===== [5/8] train PLAIN aggregator (baseline, 10K x2) ====="
[ -f models/aggregator/adapter_model.safetensors ] || python3 src/train_aggregator.py --candidates "$CAND" --n_train 10000 --epochs 2
cp -ru models/aggregator "$DRIVE/aggregator"

echo "===== [6/8] eval PLAIN vs PT ====="
python3 src/eval_rouge.py --n 150 --agg_dir models/aggregator --label CMVA-GS --out results/rouge_plain10k.json
cp -f results/rouge_plain10k.json "$DRIVE/"

echo "===== [7/8] train CoT aggregator (10K x2, same candidates) ====="
[ -f models/aggregator_cot/adapter_model.safetensors ] || python3 src/train_aggregator_cot.py --candidates "$CAND" --n_train 10000 --epochs 2
cp -ru models/aggregator_cot "$DRIVE/aggregator_cot"

echo "===== [8/8] eval CoT vs PT ====="
python3 src/eval_rouge.py --n 150 --cot --agg_dir models/aggregator_cot --label CoT-CA --out results/rouge_cot10k.json
cp -f results/rouge_cot10k.json "$DRIVE/"

echo "ALL DONE"
echo "=== PLAIN (baseline) rouge_plain10k.json ==="; cat results/rouge_plain10k.json
echo "=== COT rouge_cot10k.json ==="; cat results/rouge_cot10k.json
