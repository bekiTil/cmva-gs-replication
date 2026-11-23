"""
Generate the 5 Moral Agents' candidate answers ONCE for a set of situations and
cache them to disk, so the plain and CoT aggregators can be trained on the
*identical* candidates without regenerating them (the slow step at 10K scale).

Reads data/prepared/cot/train.jsonl (which carries situation + reasoning + the
ground-truth RoT target), so the cache has everything both aggregators need.
Output line (VALUES order for cands):
  {"situation": ..., "target": <rot>, "reasoning": <trace>, "cands": [c_care, ...]}

Using one shared candidate set also makes the plain-vs-CoT comparison maximally
controlled: same base, same agents, same situations, same candidates -> the only
difference is the aggregator's training target (direct answer vs reason-then-answer).
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from common import VALUES, build_prompt, generate_batch, get_device
from train_aggregator import load_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--src", default="data/prepared/cot/train.jsonl")
    ap.add_argument("--out", default="data/prepared/candidates/train.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=28)
    args = ap.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.src)][: args.n_train]
    print(f"Generating candidates for {len(rows)} situations from 5 agents ...")
    prompts = [build_prompt(r["situation"]) for r in rows]
    cand_by_value = {}
    for v in VALUES:
        agent = load_agent(args.base_model, v, tok, device)
        cand_by_value[v] = generate_batch(agent, tok, prompts, device,
                                           max_new_tokens=args.max_new_tokens, bs=32)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  candidates done: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "situation": r["situation"],
                "target": r.get("target", ""),
                "reasoning": r.get("reasoning", ""),
                "cands": [cand_by_value[v][i] for v in VALUES],
            }) + "\n")
    print(f"Saved {len(rows)} cached candidate rows -> {args.out}")


if __name__ == "__main__":
    main()
