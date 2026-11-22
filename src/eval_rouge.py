"""
Reproduce the paper's Figure 2: ROUGE (1/2/L/Lsum) of each system's output vs
the ground-truth answer on held-out data. Compares the base model (PT-model)
against the full CMVA-GS system (5 Moral Agents + Contextual Aggregator).
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import VALUES, PROFILES, build_prompt, generate_batch, get_device
from train_aggregator import build_agg_input, load_agent


def rouge_eval(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"],
                                      use_stemmer=True)
    agg = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in agg:
            agg[k] += s[k].fmeasure
    return {k: v / len(preds) for k, v in agg.items()}


def gen_text(model, tok, prompt, device, max_new_tokens=32):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=160).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--profile", default="balanced")
    args = ap.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = [json.loads(l) for l in open("data/prepared/pairs/test.jsonl")][: args.n]
    refs = [p["target"] for p in pairs]
    results = {}

    # --- PT-model baseline ---
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device)
    base.eval()
    pt_preds = generate_batch(base, tok, [build_prompt(p["situation"]) for p in pairs],
                              device, max_new_tokens=32, bs=32, do_sample=False)
    results["PT-model"] = rouge_eval(pt_preds, refs)
    del base

    # --- CMVA-GS: 5 agents -> aggregator ---
    agg_dir = "models/aggregator"
    if os.path.isdir(agg_dir):
        profile = PROFILES[args.profile]
        # candidates one agent at a time (5 full 1.1B models do not fit on a T4)
        agent_prompts = [build_prompt(p["situation"]) for p in pairs]
        cand_by_value = {}
        for v in VALUES:
            agent = load_agent(args.base_model, v, tok, device)
            cand_by_value[v] = generate_batch(agent, tok, agent_prompts, device,
                                              max_new_tokens=24, bs=32, do_sample=False)
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        from peft import PeftModel
        agg_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32).to(device)
        agg = PeftModel.from_pretrained(agg_base, agg_dir).to(device)
        agg.eval()
        agg_inputs = [build_agg_input(p["situation"], profile,
                                      [cand_by_value[v][i] for v in VALUES])
                      for i, p in enumerate(pairs)]
        cmva_preds = generate_batch(agg, tok, agg_inputs, device,
                                    max_new_tokens=32, bs=32, do_sample=False)
        results["CMVA-GS"] = rouge_eval(cmva_preds, refs)
    else:
        print("aggregator not found; skipping CMVA-GS row")

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/rouge.json", "w"), indent=2)
    print("\n=== Figure 2: ROUGE on held-out SocialChem ===")
    print(f"{'System':10s} " + " ".join(f"{m:>10s}" for m in ['rouge1','rouge2','rougeL','rougeLsum']))
    for sys_name, r in results.items():
        print(f"{sys_name:10s} " + " ".join(f"{r[m]*100:10.2f}" for m in ['rouge1','rouge2','rougeL','rougeLsum']))
    print("\nSaved results/rouge.json")


if __name__ == "__main__":
    main()
