---
name: bias-mitigate
description: Remove or reduce bias in text and datasets — neutralize gendered/ableist/ageist language, rewrite prompts, perform counterfactual data augmentation (CDA), and balance training datasets. Use when the user asks to "fix the bias", "rewrite this without bias", "make this neutral", "balance my dataset", or "neutralize my prompt".
---

# Bias mitigation

Three layers — pick what fits the user's goal.

## 1. Token-level neutralization (deterministic)

For copy-editing existing text:

```python
from aibias import BiasMitigator
m = BiasMitigator()
clean, report = m.neutralize("The chairman addressed the policemen.")
# clean = "The chair addressed the police officers."
print(report.summary())
```

Replaces gendered job titles, ableist/ageist pejoratives. Slurs are redacted
to `[REDACTED]` and listed in `report.removed_terms`.

## 2. Counterfactual data augmentation (CDA)

For training data — swap demographic terms to balance representation:

```python
augmented = m.cda_augment(corpus)   # 2x size: original + swapped variants
```

Or balanced resampling by group:

```python
balanced = m.balanced_resample(items, group_key=lambda x: x["gender"])
```

## 3. LLM rewrite (when context matters)

Token replacement breaks down for nuanced bias (framing, agency, implication).
After the deterministic pass:

1. Show the user the deterministic diff first.
2. Read the text carefully and rewrite to:
   - Preserve original meaning and tone.
   - Remove default-demographic assumptions ("a programmer typed at his desk"
     → "the programmer typed at her desk" OR neutral "the programmer typed").
   - Equalize hedging across mentioned groups.
   - Remove unnecessary group mentions ("a Black nurse" → "a nurse" when race
     is irrelevant to the story).
   - Avoid swapping bias direction (don't replace one stereotype with another).
3. Present the rewrite with a short rationale per change.

## Prompt rewriting for image generators

```python
neutral = BiasMitigator.neutralize_prompt("a businessman at his desk")
# "a businessperson at his desk, representing a diverse range of ages,
#  ethnicities, and gender presentations."
```

For better results, ask the user about the deployment context: a probe set
auditing the model should NOT add diversity hints (it would mask the bias);
a production prompt should.

## When NOT to mitigate

- If the text is *describing* a real demographic distribution truthfully
  (e.g. historical data), neutralizing distorts it. Flag for human review
  rather than silently rewriting.
- If the document is a quote, do not edit; flag instead.
- Slurs in academic / clinical / counter-speech contexts may be in-scope —
  ask before redacting.

## Output

For every mitigation pass, show:
- a unified diff of the change,
- the `MitigationReport` summary,
- if any slurs were redacted, list them explicitly,
- a 1-line confidence note (deterministic only / LLM-rewritten / mixed).
