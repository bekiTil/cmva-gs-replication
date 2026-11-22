#!/bin/bash
# End-to-end CMVA-GS baseline replication.
set -e
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
PY="python3 -I"
cd src 2>/dev/null && cd .. # noop guard

RAW=data/raw/social-chem-101/social-chem-101.v1.0.tsv
BASE=${BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
VALUES=(care fairness authority sanctity loyalty)

echo "== 1. Preprocess =="
[ -d data/prepared/reward ] || python3 -I src/prepare_data.py --raw "$RAW"

echo "== 2. Reward models =="
python3 -I src/train_reward_models.py --max_train 12000 --epochs 2

echo "== 3. Base (PT) conformance =="
python3 -I src/eval_conformance.py --base_model "$BASE" --label PT --n_prompts 200

echo "== 4. Train 5 Moral Agents (PPO) =="
for v in "${VALUES[@]}"; do
  python3 -I src/train_ppo_agent.py --value "$v" --base_model "$BASE" --steps 50
done

echo "== 5. Agent conformance =="
for v in "${VALUES[@]}"; do
  python3 -I src/eval_conformance.py --base_model "$BASE" --adapter "models/agents/$v" \
    --label "agent_$v" --n_prompts 200
done

echo "== 6. Table 1 =="
python3 -I src/build_table.py

echo "== 7. Aggregator (LoRA SFT) =="
python3 -I src/train_aggregator.py --base_model "$BASE" --n_train 400

echo "== 8. ROUGE (Figure 2) =="
python3 -I src/eval_rouge.py --base_model "$BASE" --n 150

echo "DONE. See results/"
