# Research Plan — G13: Chain-of-Thought for Contextual Moral Value Alignment

Builds directly on the reproduced CMVA-GS baseline (see `RESULTS.md`).

## 1. Research question & hypothesis

**Slide G13:** *"Force the model to reason about the consequences before labeling
the value."*

**Question:** Does making the system produce an explicit **consequence-reasoning
trace before assigning the moral value / final answer** improve contextual
alignment and interpretability over the baseline CMVA-GS?

**Hypotheses**
- **H1 (performance):** CoT ≥ baseline on ROUGE + value-conformance, with the
  **largest gains on multi-value / conflicting situations**, where reasoning about
  trade-offs matters most.
- **H2 (interpretability):** the reasoning trace exposes *why* a value was chosen,
  enabling error analysis the baseline cannot support, and improves
  **answer↔value consistency**.

## 2. Where CoT goes (primary + alternatives)

Baseline Contextual Aggregator (CA):
`situation + moral profile + 5 candidate answers → aligned answer (RoT)`

**Primary — CoT-Augmented Aggregator (CoT-CA):** train the aggregator to emit a
reasoning trace, *then* the answer:

```
Situation: <s>
Moral profile: care=.., fairness=.., authority=.., sanctity=.., loyalty=..
Candidate answers:
 - (care) ...
 - (fairness) ...
 ...
Reasoning: The key moral tension is <value>. If <action>, the consequence is
<harm/benefit> to <party>. Because the profile weights <value> highly, the
aligned choice is <...>.
Aggregated answer: <RoT>
```

**Alternatives / ablations (secondary):**
- **CoT Moral Agents** — each PPO agent reasons-then-acts during generation.
- **CoT value classifier** — the reward model reasons about context before scoring.
- **Self-consistency** — sample k reasoning paths, majority-vote the answer.

## 3. Where the reasoning traces come from (no external API needed)

Social Chemistry 101 already carries the structured fields to **templatize
rationales**, so CoT targets are fully reproducible from the dataset itself:
- `rot-moral-foundations` → which value(s) are at stake
- `action` + `action-moral-judgment` → the action and whether it is good/bad
- `rot` → the ground-truth rule of thumb (the final answer)
- extra attributes (`rot-char-targeting`, `action-agency`, …) → consequence/party

Build `Reasoning: <templated consequence sentence> → Aggregated answer: <rot>`.
**Optional stronger variant:** distill free-form rationales from a larger LLM
(GPT-4 / a bigger open model) and SFT on those — list as an extension if API is
available.

## 4. Method / training

- `build_cot_data.py`: SocialChem fields → `{input, reasoning, target}` examples.
- `train_aggregator_cot.py`: fork of `train_aggregator.py`; target becomes
  `"Reasoning: … Aggregated answer: <rot>"`, same LoRA SFT + prompt masking.
- Reuse the trained 5 Moral Agents + reward models from the baseline unchanged.

## 5. Evaluation (head-to-head, matched data scale)

Compare **CoT-CA vs baseline CA** at the same aggregator data size (start 2000,
then the 10K setting that already wins all ROUGE metrics):
1. **ROUGE-1/2/L** of the *final answer* (parse text after `Aggregated answer:`)
   vs RoT — identical protocol to the baseline.
2. **Value-conformance** (the 5 reward-model scores) — Table-1-style.
3. **Subset split:** single-value vs multi-value situations (count of
   `rot-moral-foundations`); test whether CoT helps more on multi-value.
4. **Interpretability metrics:** (a) *value-identification accuracy* — does the
   reasoning name the ground-truth foundation? (b) *consistency* — does the final
   answer follow the stated reasoning?
5. **Error analysis (30% of the grade):** curated cases where CoT helps, hurts,
   or reasons plausibly-but-wrongly (hallucinated consequences).
6. **Ablations:** prompt-only CoT (no training) vs trained CoT; reasoning length;
   CoT in aggregator vs agents vs classifier; self-consistency k.

## 6. Timeline (course weeks)

- Wk 1–3 — baseline replication ✅ (done; CMVA-GS beats base on all 4 ROUGE).
- Wk 3–5 — implement CoT-CA: data builder, CoT SFT, eval parsing.
- Wk 6 — experiments, subset analysis, ablations, hyperparameters.
- Wk 7 — 4-page ACL report + 15-min presentation.

## 7. Risks & mitigations

- **TinyLlama is a weak reasoner** → trained/distilled CoT ≫ zero-shot CoT; try a
  slightly larger base (e.g. Qwen2.5-1.5B) if the L4 allows.
- **CoT tokens can dilute ROUGE precision** → always score the *final answer only*,
  and report reasoning-quality metrics separately.
- **Templated rationales may be formulaic** → optionally mix in LLM-distilled
  rationales for variety.
- **No gain at small scale** → a *negative but analyzed* result is acceptable per
  the rubric; the interpretability contribution (H2) stands regardless.

## 8. Deliverables

GitHub repo (this one), 4-page ACL-style report, 15-min presentation. New code:
`build_cot_data.py`, `train_aggregator_cot.py`, CoT-aware eval + analysis scripts.
