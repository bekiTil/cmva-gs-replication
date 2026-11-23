"""
CoT Contextual Aggregator (G13): the aggregator is trained to emit an explicit
consequence-reasoning trace BEFORE the final answer, instead of answering directly.

Target = "Reasoning: <consequences> ... Aggregated answer: <rot>"
(reasoning comes from data/prepared/cot/, built by build_cot_data.py).

This is the TREATMENT; train_aggregator.py is the no-CoT CONTROL. Same base,
same candidates, same data scale -> the only difference is CoT, isolating its effect.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from common import VALUES, PROFILES, build_prompt, generate_batch, get_device
from train_aggregator import load_agent


def build_agg_input_cot(situation, profile, candidates):
    prof = ", ".join(f"{v}={w:.1f}" for v, w in zip(VALUES, profile))
    cand = "\n".join(f"- ({v}) {c}" for v, c in zip(VALUES, candidates))
    return (f"Situation: {situation}\nMoral profile: {prof}\n"
            f"Candidate answers:\n{cand}\n"
            f"Reason about the consequences, then give the final answer.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--out", default="models/aggregator_cot")
    args = ap.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    profile = PROFILES[args.profile]

    rows = [json.loads(l) for l in open("data/prepared/cot/train.jsonl")][: args.n_train]
    print(f"Generating candidates from 5 agents for {len(rows)} situations ...")
    agent_prompts = [build_prompt(r["situation"]) for r in rows]
    cand_by_value = {}
    for v in VALUES:
        agent = load_agent(args.base_model, v, tok, device)
        cand_by_value[v] = generate_batch(agent, tok, agent_prompts, device,
                                           max_new_tokens=28, bs=32)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  candidates done:", v)

    examples = []
    for i, r in enumerate(rows):
        cands = [cand_by_value[v][i] for v in VALUES]
        inp = build_agg_input_cot(r["situation"], profile, cands)
        tgt = r["reasoning"] + " Aggregated answer: " + r["target"]
        examples.append({"text": inp + " " + tgt + tok.eos_token,
                         "prompt_len_text": inp})
    os.makedirs("results", exist_ok=True)
    json.dump(examples[:3], open("results/cot_samples.json", "w"), indent=2)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def tokenize(ex):
        full = tok(ex["text"], truncation=True, max_length=320)
        plen = len(tok(ex["prompt_len_text"], truncation=True, max_length=320)["input_ids"])
        labels = list(full["input_ids"])
        for j in range(min(plen, len(labels))):
            labels[j] = -100
        full["labels"] = labels
        return full

    ds = Dataset.from_list(examples).map(tokenize, remove_columns=["text", "prompt_len_text"])
    targs = TrainingArguments(
        output_dir=f"{args.out}/_ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=20, save_strategy="no", report_to=[])
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, padding=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saved CoT aggregator ->", args.out)


if __name__ == "__main__":
    main()
