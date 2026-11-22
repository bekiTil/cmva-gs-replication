"""
Prepare data for the CMVA-GS baseline replication.

Source: Social Chemistry 101 (Forbes et al., 2020) — the dataset the paper's
Moral Integrity Corpus (MIC) is built from. It carries the same 5 Moral
Foundations the paper uses, with a moral judgment we binarize exactly as the
paper describes ("0 for negative, 1 for neutral/positive").

Outputs (written to data/prepared/):
  reward/<value>.{train,val,test}.jsonl   -> text + binary label, per value
  prompts/{train,val,test}.jsonl          -> situations used as prompts for
                                             agent (PPO) training + evaluation
"""
import argparse
import json
import os
import random
from collections import Counter

import pandas as pd

# The 5 moral foundations the paper uses (names as they appear in SocialChem,
# mapped to the short names used in the paper's Table 1).
FOUNDATIONS = {
    "care-harm": "care",
    "fairness-cheating": "fairness",
    "authority-subversion": "authority",
    "sanctity-degradation": "sanctity",
    "loyalty-betrayal": "loyalty",
}

RAW_DEFAULT = "data/raw/social-chem-101.v1.0.tsv"
OUT_DIR = "data/prepared"


def load_raw(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    return df


def binarize_judgment(series):
    """Map SocialChem judgment to {0,1}: 0 = negative, 1 = neutral/positive."""
    # action-moral-judgment is a float in roughly [-2, 2]; >=0 => acceptable.
    return (series.fillna(0).astype(float) >= 0).astype(int)


def build(df, seed=42):
    os.makedirs(os.path.join(OUT_DIR, "reward"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "prompts"), exist_ok=True)

    # Pick the text + judgment + foundations columns robustly.
    mf_col = "rot-moral-foundations"
    judg_col = "action-moral-judgment" if "action-moral-judgment" in df.columns else "rot-judgment"
    text_col = "action" if "action" in df.columns else "rot"
    sit_col = "situation"
    split_col = "split" if "split" in df.columns else None

    print(f"Using: foundations='{mf_col}', judgment='{judg_col}', text='{text_col}'")

    df = df.dropna(subset=[mf_col, text_col]).copy()
    df["label"] = binarize_judgment(df[judg_col])

    def split_of(row):
        if split_col and isinstance(row[split_col], str):
            s = row[split_col].strip().lower()
            if s in ("train", "dev", "validation", "val", "test"):
                return "val" if s in ("dev", "validation") else s
        return None

    df["_split"] = df.apply(split_of, axis=1)
    # Fallback random split for rows without one.
    rng = random.Random(seed)
    missing = df["_split"].isna()
    df.loc[missing, "_split"] = [
        rng.choices(["train", "val", "test"], weights=[0.8, 0.1, 0.1])[0]
        for _ in range(missing.sum())
    ]

    # ---- Per-value reward-model datasets ----
    summary = {}
    for sc_name, short in FOUNDATIONS.items():
        mask = df[mf_col].str.contains(sc_name, na=False, regex=False)
        sub = df[mask]
        counts = {}
        for split in ("train", "val", "test"):
            rows = sub[sub["_split"] == split]
            out = os.path.join(OUT_DIR, "reward", f"{short}.{split}.jsonl")
            with open(out, "w") as f:
                for _, r in rows.iterrows():
                    f.write(json.dumps({
                        "text": str(r[text_col]).strip(),
                        "label": int(r["label"]),
                    }) + "\n")
            counts[split] = (len(rows), dict(Counter(rows["label"].tolist())))
        summary[short] = counts
        print(f"[{short}] " + " ".join(f"{s}={counts[s][0]}" for s in counts))

    # ---- Prompt pool (unique situations) for agent training + eval ----
    sits = df.dropna(subset=[sit_col])[[sit_col, "_split"]].drop_duplicates(sit_col)
    for split in ("train", "val", "test"):
        rows = sits[sits["_split"] == split]
        out = os.path.join(OUT_DIR, "prompts", f"{split}.jsonl")
        with open(out, "w") as f:
            for _, r in rows.iterrows():
                f.write(json.dumps({"situation": str(r[sit_col]).strip()}) + "\n")
        print(f"[prompts] {split}={len(rows)}")

    # ---- Situation -> guidance pairs (aggregator targets + ROUGE refs) ----
    rot_col = "rot" if "rot" in df.columns else text_col
    os.makedirs(os.path.join(OUT_DIR, "pairs"), exist_ok=True)
    pairs = df.dropna(subset=[sit_col, rot_col]).drop_duplicates(sit_col)
    for split in ("train", "val", "test"):
        rows = pairs[pairs["_split"] == split]
        out = os.path.join(OUT_DIR, "pairs", f"{split}.jsonl")
        with open(out, "w") as f:
            for _, r in rows.iterrows():
                f.write(json.dumps({"situation": str(r[sit_col]).strip(),
                                    "target": str(r[rot_col]).strip()}) + "\n")
        print(f"[pairs] {split}={len(rows)}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. Wrote", OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(load_raw(args.raw), seed=args.seed)
