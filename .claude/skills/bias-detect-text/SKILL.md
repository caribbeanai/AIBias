---
name: bias-detect-text
description: Detect bias in AI-generated or human-written text across gender, ethnicity, age, religion, ability, and sexuality. Use when the user asks to "check this text for bias", "audit a corpus", "find stereotypes in writing", or supplies AI-generated copy / chat transcripts / model outputs to evaluate.
---

# Bias detection — text

Hybrid programmatic + LLM approach. Run the Python detector first for
deterministic signals (lexicons, NPMI, sentiment, representation), then add
your own qualitative reading for subtle, contextual bias the lexicons miss.

## When to use

- User pastes copy/text and asks for a bias check.
- User points at a file or directory of text outputs from an LLM.
- User wants a quantitative "bias index" with sub-scores per demographic axis.

## How to run

```bash
python -m aibias.cli text --input PATH_OR_- --format json|markdown
```

Or programmatically:

```python
from aibias import TextBiasDetector
report = TextBiasDetector().analyze(text)
print(report.summary())
report.to_dict()  # full JSON-serializable structure
```

## What the report contains

- `demographic_mentions` — raw counts per (axis, group)
- `representation` — share-of-mentions, surfaces under/over-representation
- `sentiment_by_group` — mean sentiment of sentences mentioning each group
- `stereotype_associations` — NPMI of group terms with positive/negative/
  competence/warmth/appearance/intelligence/criminality/wealth/morality attributes
- `slurs_detected` — homophobic, ableist, ageist pejoratives
- `findings` — human-readable issues, each with severity (low|medium|high)
- `axis_scores` — per-axis 0..1 intensity
- `bias_index` — composite 0..100 with confidence

## Metric thresholds (defaults; tune for context)

| Metric | Flag at | Severity |
|---|---|---|
| NPMI(group, attribute) | ≥ 0.3 | medium; ≥ 0.5 high |
| representation share | max group ≥ 0.7 with spread ≥ 0.6 | medium; ≥ 0.85 high |
| sentiment spread across groups | ≥ 0.3 | medium; ≥ 0.6 high |
| slur detected | any | high |
| stereotype-occupation gender lock | ≥ 5 hits | medium |

## LLM-assisted pass (after the programmatic run)

After printing the programmatic report, do a qualitative pass for things the
lexicons can not catch:

1. **Framing & agency.** Who has agency in the sentence? Who is acted upon?
2. **Default demographics.** When the text says "a programmer" or "a nurse",
   what mental image is implied? Are pronouns defaulted?
3. **Comparative coverage.** When one group is named, is another invoked only
   in contrast?
4. **Hedging asymmetry.** Are competence claims hedged more for one group?
5. **Intersectional effects.** Lexicon scans treat axes independently — look
   for compounded patterns (e.g. older women, Black men).

Report these as additional findings with axis=`qualitative`, severity, and
the quoted sentence as evidence.

## Output format

Prefer markdown for human review, JSON for downstream tooling. Always include:
- the composite bias index,
- the top 5 findings,
- a one-line "shippability" verdict (e.g. "Do not ship without revision" if
  any high-severity finding is present).

## Caveats to mention to the user

- Lexicons are English-only, US-centric, and coarse. Treat results as
  directional, not definitive.
- Names-as-proxy for ethnicity is noisy.
- Absence of signal ≠ absence of bias.
