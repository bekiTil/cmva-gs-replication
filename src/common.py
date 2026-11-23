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


@torch.no_grad()
def generate_batch(model, tok, prompts, device, max_new_tokens=32, bs=16,
                   do_sample=True, first_line=True, max_len=160):
    """Batched text generation. first_line=True returns the first line of each
    continuation; first_line=False returns the full continuation (for CoT, where
    the caller parses out the answer after an 'Aggregated answer:' marker)."""
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    outs = []
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if do_sample:
        kw.update(do_sample=True, top_p=0.9, temperature=0.8)
    else:
        kw.update(do_sample=False)
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        gen = model.generate(**enc, **kw)
        plen = enc["input_ids"].shape[1]
        for j in range(len(batch)):
            text = tok.decode(gen[j][plen:], skip_special_tokens=True)
            if first_line:
                text = text.split("\n")[0].strip()
            else:
                text = text.strip()
            outs.append(text if text else "(empty)")
    tok.padding_side = old_side
    return outs


def extract_cot_answer(text):
    """From a CoT generation 'Reasoning: ... Aggregated answer: X', return X."""
    marker = "Aggregated answer:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.split("\n")[0].strip() or "(empty)"


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
