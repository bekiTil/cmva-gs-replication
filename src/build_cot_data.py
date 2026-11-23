"""
Build Chain-of-Thought reasoning targets for the CoT aggregator (G13).

Templated track: turn Social Chemistry's structured annotations
(moral foundation + action + judgment + rule-of-thumb) into a short
"reason about the consequences, then answer" trace:

  Reasoning: The main moral concern is <value>. The action "<action>" <consequence>,
  so it is <judgment>. Weighing the consequences, the aligned guidance is:
  Aggregated answer: <rot>

Output: data/prepared/cot/<split>.jsonl  with {situation, reasoning, target}.
(The self-distilled track is produced separately by the Qwen base model.)
"""
import argparse
import json
import os

import pandas as pd

# foundation -> (readable name, good-consequence phrase, bad-consequence phrase)
FOUNDATION = {
    "care-harm": ("care and avoiding harm",
                  "helps or protects others from harm",
                  "could cause harm or suffering to others"),
    "fairness-cheating": ("fairness",
                          "treats people fairly and honestly",
                          "is unfair or dishonest to others"),
    "authority-subversion": ("respect for authority",
                             "respects legitimate authority and social order",
                             "undermines authority or social order"),
    "sanctity-degradation": ("sanctity and dignity",
                             "upholds purity and dignity",
                             "degrades purity or dignity"),
    "loyalty-betrayal": ("loyalty",
                         "shows loyalty to one's group or relationships",
                         "betrays one's group or relationships"),
}
RAW = "data/raw/social-chem-101/social-chem-101.v1.0.tsv"
OUT = "data/prepared/cot"


def pick_foundation(mf):
    for key in FOUNDATION:
        if key in str(mf):
            return key
    return "care-harm"  # default


def build_reasoning(row):
    key = pick_foundation(row.get("rot-moral-foundations", ""))
    name, good, bad = FOUNDATION[key]
    judg = row.get("action-moral-judgment", 0)
    try:
        judg = float(judg)
    except (TypeError, ValueError):
        judg = 0.0
    good_side = judg >= 0
    consequence = good if good_side else bad
    jword = "acceptable" if good_side else "not okay"
    action = str(row.get("action", "")).strip().rstrip(".")
    return (f"Reasoning: The main moral concern is {name}. The action "
            f'"{action}" {consequence}, so it is {jword}. Weighing the '
            f"consequences, the aligned guidance is:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()
    print("Loading", args.raw)
    df = pd.read_csv(args.raw, sep="\t", low_memory=False)
    df = df.dropna(subset=["situation", "rot", "action"])
    df["_split"] = df["split"].map(
        lambda s: "val" if s == "dev" else (s if s in ("train", "test") else None))
    # one row per situation (first), keeping the fields we need
    df = df.dropna(subset=["_split"]).drop_duplicates("situation")
    os.makedirs(OUT, exist_ok=True)
    for split in ("train", "val", "test"):
        sub = df[df["_split"] == split]
        out = os.path.join(OUT, f"{split}.jsonl")
        with open(out, "w") as f:
            for _, r in sub.iterrows():
                f.write(json.dumps({
                    "situation": str(r["situation"]).strip(),
                    "reasoning": build_reasoning(r),
                    "target": str(r["rot"]).strip(),
                }) + "\n")
        print(f"[cot] {split}={len(sub)} -> {out}")
    # show a sample
    s = json.loads(open(os.path.join(OUT, "train.jsonl")).readline())
    print("\nSAMPLE reasoning:\n", s["reasoning"], "\n-> target:", s["target"])


if __name__ == "__main__":
    main()
