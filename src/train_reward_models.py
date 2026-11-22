"""
Train one reward model (binary moral-value classifier) per moral foundation.

Each classifier estimates P(response follows value) in [0,1], exactly the role
of the reward models in the paper (Sec. 4/5): "a reward is the probability of a
LLM simulated response being in the good class of a moral value classifier."

Backbone: distilbert-base-uncased (small, fast; swappable). Runs on Apple MPS.
Outputs: models/reward/<value>/ and results/reward_metrics.json
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

VALUES = ["care", "fairness", "authority", "sanctity", "loyalty"]


def train_one(value, args, device):
    data_files = {
        "train": f"data/prepared/reward/{value}.train.jsonl",
        "val": f"data/prepared/reward/{value}.val.jsonl",
        "test": f"data/prepared/reward/{value}.test.jsonl",
    }
    ds = load_dataset("json", data_files=data_files)
    if args.max_train and len(ds["train"]) > args.max_train:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(args.max_train))
    if args.max_eval and len(ds["val"]) > args.max_eval:
        ds["val"] = ds["val"].shuffle(seed=42).select(range(args.max_eval))

    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(b):
        return tok(b["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(enc, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    out_dir = f"models/reward/{value}"

    targs = TrainingArguments(
        output_dir=f"{out_dir}/_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to=[],
        fp16=False,
        dataloader_num_workers=0,
    )

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "acc": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    m = trainer.evaluate(ds["test"])
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[{value}] test acc={m['eval_acc']:.4f} f1={m['eval_f1']:.4f}")
    return {"acc": m["eval_acc"], "f1": m["eval_f1"],
            "n_train": len(ds["train"]), "n_test": len(ds["test"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--values", nargs="+", default=VALUES)
    ap.add_argument("--max_train", type=int, default=20000)
    ap.add_argument("--max_eval", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    results = {}
    for v in args.values:
        print(f"\n===== Training reward model: {v} =====")
        results[v] = train_one(v, args, device)
    os.makedirs("results", exist_ok=True)
    with open("results/reward_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/reward_metrics.json")
    for v, r in results.items():
        print(f"  {v:10s} acc={r['acc']:.3f} f1={r['f1']:.3f}")


if __name__ == "__main__":
    main()
