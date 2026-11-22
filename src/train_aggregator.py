"""
Train the Contextual Aggregator (CA) via supervised fine-tuning with LoRA,
following the paper (Sec. 4/5.1): the aggregator takes the question + the moral
profile + the context of the 5 agents' answers and is trained with cross-entropy
to produce the ground-truth answer.

Input  = situation + moral-profile weights + the 5 Moral Agents' candidate answers
Target = ground-truth rule-of-thumb (SocialChem RoT)
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from common import VALUES, build_prompt, get_device


def load_agent(base_model, value, tok, device):
    from peft import PeftModel
    adir = f"models/agents/{value}"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32).to(device)
    if os.path.isdir(adir):
        model = PeftModel.from_pretrained(model, adir).to(device)
    model.eval()
    return model


def gen(model, tok, situation, device, max_new_tokens=28):
    enc = tok(build_prompt(situation), return_tensors="pt", truncation=True,
              max_length=96).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             top_p=0.9, temperature=0.8, pad_token_id=tok.pad_token_id)
    g = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return g.split("\n")[0].strip()


def build_agg_input(situation, profile, candidates):
    prof = ", ".join(f"{v}={w:.1f}" for v, w in zip(VALUES, profile))
    cand = "\n".join(f"- ({v}) {c}" for v, c in zip(VALUES, candidates))
    return (f"Situation: {situation}\nMoral profile: {prof}\n"
            f"Candidate answers:\n{cand}\nAggregated answer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--out", default="models/aggregator")
    args = ap.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from common import PROFILES
    profile = PROFILES[args.profile]

    # --- Build aggregator dataset by running the 5 agents ---
    pairs = [json.loads(l) for l in open("data/prepared/pairs/train.jsonl")][: args.n_train]
    print(f"Generating candidate answers from 5 agents for {len(pairs)} situations ...")
    # One agent in GPU memory at a time (5 full 1.1B models do not fit on a T4).
    cand_by_value = {}
    for v in VALUES:
        agent = load_agent(args.base_model, v, tok, device)
        cand_by_value[v] = [gen(agent, tok, p["situation"], device) for p in pairs]
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  candidates done: {v}")
    examples = []
    for i, p in enumerate(pairs):
        cands = [cand_by_value[v][i] for v in VALUES]
        inp = build_agg_input(p["situation"], profile, cands)
        examples.append({"text": inp + " " + p["target"] + tok.eos_token,
                         "prompt_len_text": inp})
    os.makedirs("results", exist_ok=True)
    json.dump(examples[:3], open("results/agg_samples.json", "w"), indent=2)

    # --- SFT with LoRA, masking the prompt tokens ---
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(ex):
        full = tok(ex["text"], truncation=True, max_length=256)
        plen = len(tok(ex["prompt_len_text"], truncation=True, max_length=256)["input_ids"])
        labels = list(full["input_ids"])
        for j in range(min(plen, len(labels))):
            labels[j] = -100
        full["labels"] = labels
        return full

    ds = Dataset.from_list(examples).map(tokenize, remove_columns=["text", "prompt_len_text"])

    targs = TrainingArguments(
        output_dir=f"{args.out}/_ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=20, save_strategy="no", report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved aggregator -> {args.out}")


if __name__ == "__main__":
    main()
