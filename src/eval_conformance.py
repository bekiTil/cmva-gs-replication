"""
Evaluate how well a policy's responses conform to each moral value.

Reproduces the quantity in the paper's Table 1: the probability (expected
reward) that a model's answers follow each moral value, measured by the 5
reward-model classifiers. Run once for the base model (PT row) and once per
Moral Agent; build_table.py assembles the final table.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import VALUES, RewardScorer, build_prompt, get_device, load_prompts


def generate_responses(model, tok, prompts, device, max_new_tokens=32, bs=16):
    responses = []
    model.eval()
    for i in range(0, len(prompts), bs):
        batch = [build_prompt(p) for p in prompts[i : i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=0.9, temperature=0.8, pad_token_id=tok.pad_token_id,
            )
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            text = text.split("\n")[0].strip()  # first line only
            responses.append(text if text else "(empty)")
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (a Moral Agent)")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--label", required=True, help="row name, e.g. 'PT' or 'agent_care'")
    ap.add_argument("--out", default="results/conformance")
    args = ap.parse_args()

    device = get_device()
    print("Device:", device, "| model:", args.base_model, "| adapter:", args.adapter)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    ).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(device)

    prompts = load_prompts("test", limit=args.n_prompts)
    print(f"Generating for {len(prompts)} prompts ...")
    responses = generate_responses(model, tok, prompts, device)

    print("Scoring with 5 reward models ...")
    means = {}
    scorers = {v: RewardScorer(v, device=device) for v in VALUES}
    for v in VALUES:
        probs = scorers[v].score(responses)
        means[v] = sum(probs) / len(probs)

    os.makedirs(args.out, exist_ok=True)
    rec = {"label": args.label, "n": len(responses), "means": means,
           "samples": list(zip(prompts[:5], responses[:5]))}
    with open(os.path.join(args.out, f"{args.label}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[{args.label}] " + "  ".join(f"{v}={means[v]*100:.1f}%" for v in VALUES))


if __name__ == "__main__":
    main()
