"""
Train one Moral Agent via PPO (RLFT), following the paper (Sec. 3/5.1):
start from the pretrained base LLM and RL-fine-tune it so its outputs maximize
the reward of ONE moral-value classifier. Uses TRL PPO + LoRA so it fits on a
single small GPU / Apple MPS. One agent per value => 5 agents.
"""
import argparse
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from common import RewardScorer, build_prompt, get_device, load_prompts


def build_dataset(tok, n_prompts, max_len=96):
    sits = load_prompts("train", limit=n_prompts)
    rows = {"query": [], "input_ids": []}
    for s in sits:
        text = build_prompt(s)
        ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
        rows["query"].append(text)
        rows["input_ids"].append(ids)
    ds = Dataset.from_dict(rows)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return {k: [d[k] for d in data] for k in data[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", required=True)
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--n_prompts", type=int, default=800)
    ap.add_argument("--steps", type=int, default=50, help="number of PPO batches")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mini_batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.4e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    out_dir = args.out or f"models/agents/{args.value}"
    print(f"Device {device} | value={args.value} | base={args.base_model}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model, peft_config=lora, torch_dtype=torch.float32,
    )

    config = PPOConfig(
        model_name=args.base_model, learning_rate=args.lr,
        batch_size=args.batch_size, mini_batch_size=args.mini_batch_size,
        ppo_epochs=4, init_kl_coef=0.2, target_kl=6.0, seed=42,
        remove_unused_columns=False, log_with=None,
    )

    ds = build_dataset(tok, args.n_prompts)
    ppo = PPOTrainer(config, model, ref_model=None, tokenizer=tok,
                     dataset=ds, data_collator=collator)

    scorer = RewardScorer(args.value, device=ppo.accelerator.device)
    gen_len = LengthSampler(16, 32)
    gen_kwargs = {"min_length": -1, "top_p": 0.9, "do_sample": True,
                  "temperature": 0.8, "pad_token_id": tok.pad_token_id}

    step = 0
    for epoch in range(1000):
        for batch in ppo.dataloader:
            query_tensors = batch["input_ids"]
            response_tensors = []
            for q in query_tensors:
                gen_kwargs["max_new_tokens"] = gen_len()
                resp = ppo.generate(q.to(ppo.accelerator.device), **gen_kwargs)
                response_tensors.append(resp.squeeze()[q.shape[0]:])
            texts = [tok.decode(r, skip_special_tokens=True).split("\n")[0].strip()
                     for r in response_tensors]
            batch["response"] = texts
            scores = scorer.score([t if t else "(empty)" for t in texts])
            rewards = [torch.tensor(float(s)) for s in scores]

            stats = ppo.step(query_tensors, response_tensors, rewards)
            mean_r = sum(scores) / len(scores)
            print(f"step {step:03d} | mean_reward({args.value})={mean_r:.3f} "
                  f"| kl={stats.get('objective/kl', float('nan')):.2f}")
            step += 1
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    os.makedirs(out_dir, exist_ok=True)
    ppo.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved agent -> {out_dir}")


if __name__ == "__main__":
    main()
