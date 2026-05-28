# AIBias

A comprehensive toolkit and set of Skills to **detect, quantify, and mitigate
bias in AI-generated content** — text, images, and classifier outputs — across
gender, ethnicity, age, religion, ability, sexuality, and other axes.

Ships as:
- a Python package (`aibias`) usable from code or CLI, and
- five Claude Code Skills (in `.claude/skills/`) that orchestrate the package
  together with LLM reasoning for nuanced, context-aware bias review.

---

## Skills

| Skill | Purpose |
|---|---|
| `bias-detect-text` | Find demographic mentions, slurs, stereotype associations, sentiment skew, and representation imbalances in written content. |
| `bias-detect-image` | Audit generated images and image sets for skin-tone distribution (Monk Skin Tone), perceived demographic representation, and context stereotypes. |
| `bias-quantify` | Compute fairness metrics: SPD, disparate impact, equal opportunity, equalized odds, WEAT, NPMI, Theil index, composite Bias Index. |
| `bias-mitigate` | Neutralize gendered / ableist / ageist language, redact slurs, run counterfactual data augmentation, balance datasets, rewrite generation prompts. |
| `bias-audit` | End-to-end audit combining all of the above into a compliance-grade JSON + Markdown report (EU AI Act / NIST AI RMF / ISO 42001 aligned). |

Each Skill is a markdown file in `.claude/skills/<skill>/SKILL.md` with
trigger phrases, workflow, and metric thresholds.

---

## Quickstart

```bash
pip install -e ".[all]"
```

```python
from aibias import TextBiasDetector, ImageBiasDetector, BiasMitigator, BiasAuditor

# 1. Detect bias in text
report = TextBiasDetector().analyze("The chairman dismissed the female engineer.")
print(report.summary())

# 2. Detect bias in an image set
img_report = ImageBiasDetector().analyze(["gen1.png", "gen2.png", "gen3.png"])

# 3. Mitigate
clean, mrep = BiasMitigator().neutralize("The policemen and stewardesses arrived.")

# 4. End-to-end audit
auditor = BiasAuditor()
report = auditor.audit(
    texts=[...],
    image_paths=[...],
    classifier_outputs={
        "positive_rates": {"male": 0.62, "female": 0.41},
        "tpr_by_group":   {"male": 0.85, "female": 0.70},
        "fpr_by_group":   {"male": 0.10, "female": 0.15},
        "privileged_group": "male",
    },
)
auditor.save(report, "audit_out/")
```

## CLI

```bash
python -m aibias.cli text     --input "She was emotional, he was rational." --format json
python -m aibias.cli image    --inputs ./generations --format text
python -m aibias.cli mitigate --input ./article.txt
python -m aibias.cli audit    --text outputs/*.txt --images gen/ --out audit_out/
```

---

## Bias metrics included

### Classification fairness
- **Statistical Parity Difference** — `P(Y=1|g) − P(Y=1|priv)`
- **Disparate Impact** (EEOC 80% rule)
- **Equal Opportunity Difference** (TPR gap)
- **Equalized Odds Difference** (TPR + FPR gap)
- **Theil Index** (generalized entropy / inequality)

### Representation & content
- **Representation Ratio** (observed / baseline share per group)
- **Sentiment Skew** across groups
- **Normalized Pointwise Mutual Information** for group × attribute associations
- **WEAT effect size** (Word Embedding Association Test, Caliskan 2017)
- **Stereotype Association Score** across competence / warmth / valence axes

### LM-specific
- **Log-Probability Bias Score** (Webster 2020) for paired prompt completions

### Composite
- **Bias Index** (0–100) — L2-aggregated, with per-axis sub-scores and a
  sample-size-based confidence value.

---

## Demographic axes covered

`gender`, `ethnicity`, `age`, `religion`, `ability`, `sexuality` — plus
sentiment skew as a cross-cutting axis. Lexicons are extensible; see
`aibias/lexicons/`.

---

## Mitigation toolkit

- Token-level neutralization (gendered job titles, ableist/ageist pejoratives, slur redaction).
- Counterfactual data augmentation (CDA, Lu et al. 2018).
- Group-balanced resampling for training data.
- Prompt rewriting for image generators.
- LLM-assisted contextual rewrites (driven by the `bias-mitigate` SKILL.md).

---

## Compliance mapping

The `bias-audit` Skill produces output aligned with:
- **EU AI Act** Art. 10(2)(f) (training-data bias) and Art. 15 (accuracy/robustness)
- **NIST AI RMF** GOVERN-1.1 and MEASURE-2.11
- **ISO/IEC 42001** AI management system fairness controls

---

## Project layout

```
aibias/
  __init__.py        # public API
  text.py            # TextBiasDetector
  image.py           # ImageBiasDetector
  mitigate.py        # BiasMitigator
  audit.py           # BiasAuditor (orchestrator)
  metrics.py         # all bias / fairness metrics
  cli.py             # `python -m aibias.cli ...`
  lexicons/          # per-axis term lists (gender, ethnicity, age, ...)
.claude/skills/
  bias-detect-text/SKILL.md
  bias-detect-image/SKILL.md
  bias-quantify/SKILL.md
  bias-mitigate/SKILL.md
  bias-audit/SKILL.md
tests/               # pytest suite
examples/            # quickstart and end-to-end audit
```

---

## Caveats

- Lexicons are English-only and US-centric. Extend for production use.
- Names-as-proxy for ethnicity is noisy; results are directional.
- Perceived attributes (gender, age, ethnicity) inferred from images are not
  identity claims and should not be applied to identifiable real individuals.
- Fairness definitions can mathematically conflict (Kleinberg–Chouldechova);
  pick and document the one that fits your context.

---

## License

MIT — see `LICENSE`.
