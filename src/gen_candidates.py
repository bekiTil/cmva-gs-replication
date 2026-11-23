"""
Generate the 5 Moral Agents' candidate answers ONCE and cache them, so the plain
and CoT aggregators train on the *identical* candidates without regenerating them
(the slow step at 10K scale).

Incremental + resumable: each agent's candidates are written to
<outdir>/_cand_<value>.jsonl as soon as that agent finishes, and an agent whose
file already exists (with the right row count) is skipped. The final --out file
is assembled from the per-agent files. Point --out at a Google Drive path and the
per-agent files land on Drive too, so a VM restart resumes instead of restarting.

Reads data/prepared/cot/train.jsonl (situation + reasoning + RoT target), so the
cache carries everything both aggregators need. Final line (VALUES order):
  {"situation": ..., "target": <rot>, "reasoning": <trace>, "cands": [c_care, ...]}
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
    ap.add_argument("--out", default="data/prepared/candidates/train10k.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.src)][: args.n_train]
    n = len(rows)
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    prompts = [build_prompt(r["situation"]) for r in rows]

    cand_by_value = {}
    for v in VALUES:
        pf = os.path.join(outdir, f"_cand_{v}.jsonl")
        if os.path.isfile(pf):
            cached = [json.loads(l) for l in open(pf)]
            if len(cached) == n:
                cand_by_value[v] = cached
                print(f"  [resume] loaded cached candidates: {v}", flush=True)
                continue
        agent = load_agent(args.base_model, v, tok, device)
        cv = generate_batch(agent, tok, prompts, device,
                            max_new_tokens=args.max_new_tokens, bs=32)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tmp = pf + ".tmp"
        with open(tmp, "w") as f:
            for c in cv:
                f.write(json.dumps(c) + "\n")
        os.replace(tmp, pf)  # atomic: a complete per-agent file or none
        cand_by_value[v] = cv
        print(f"  candidates done: {v}", flush=True)

    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "situation": r["situation"],
                "target": r.get("target", ""),
                "reasoning": r.get("reasoning", ""),
                "cands": [cand_by_value[v][i] for v in VALUES],
            }) + "\n")
    print(f"Saved {n} cached candidate rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
