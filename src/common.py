"""Shared helpers: device, reward scoring, prompt formatting, moral profiles."""
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

VALUES = ["care", "fairness", "authority", "sanctity", "loyalty"]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_prompt(situation):
    """Instruction prompt given to a Moral Agent / base model for a situation."""
    return (
        "Below is a social situation. Respond with one sentence describing the "
        "right thing to do.\n"
        f"Situation: {situation}\nResponse:"
    )


class RewardScorer:
    """Loads one value's classifier and returns P(text follows the value) in [0,1]."""

    def __init__(self, value, device=None, model_dir=None):
        self.value = value
        self.device = device or get_device()
        path = model_dir or f"models/reward/{value}"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, texts, batch_size=64, max_len=64):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, truncation=True, max_length=max_len,
                           padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, 1]  # P(class 1 = follows value)
            out.extend(probs.detach().cpu().tolist())
        return out


def load_prompts(split="test", limit=None):
    path = f"data/prepared/prompts/{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line)["situation"])
            if limit and len(rows) >= limit:
                break
    return rows


# Moral profile vectors (c = [c_care, c_fairness, c_authority, c_sanctity, c_loyalty]),
# used by the aggregator. A few illustrative profiles.
PROFILES = {
    "care_focused": [1.0, 0.2, 0.0, 0.0, 0.2],
    "fairness_focused": [0.2, 1.0, 0.2, 0.0, 0.0],
    "authority_focused": [0.0, 0.2, 1.0, 0.2, 0.2],
    "balanced": [0.5, 0.5, 0.5, 0.5, 0.5],
}
