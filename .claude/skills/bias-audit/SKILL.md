---
name: bias-audit
description: End-to-end AI bias audit combining text, image, and classifier-output analysis into a single compliance-grade report (NIST AI RMF / EU AI Act / ISO 42001 aligned). Use when the user asks for a full audit, a fairness review of a model or product, a compliance report, or "audit this AI system".
---

# Bias audit — end-to-end

The orchestrator. Combines the text, image, and quantitative skills into a
single report appropriate for compliance documentation.

## Inputs you can mix and match

- `texts`: list of strings (LLM outputs, generated copy, chat logs).
- `image_paths`: list of paths (generated images).
- `classifier_outputs`: dict with `positive_rates`, `tpr_by_group`,
  `fpr_by_group`, `privileged_group`.

## How to run

```python
from aibias import BiasAuditor
auditor = BiasAuditor()
report = auditor.audit(
    texts=[...],
    image_paths=[...],
    classifier_outputs={
        "positive_rates": {"male": 0.6, "female": 0.4},
        "tpr_by_group":   {"male": 0.82, "female": 0.68},
        "fpr_by_group":   {"male": 0.10, "female": 0.18},
        "privileged_group": "male",
    },
)
auditor.save(report, "audit_out/")
print(report.to_markdown())
```

## Deliverables produced

- `audit_out/bias_audit.json` — machine-readable.
- `audit_out/bias_audit.md`   — for stakeholders / compliance evidence.

The markdown report includes:
- Overall bias index (0..100) and per-axis sub-scores.
- All findings ranked by severity.
- Classifier fairness metrics: SPD, DI, equalized-odds gaps.
- Concrete recommendations (auto-generated; you should add 1-3 more
  tailored to the user's context).
- Timestamp and inputs summary.

## Workflow

1. **Clarify scope**. Ask: which axes matter? what's the deployment context?
   what's the baseline distribution (real-world, target, or uniform)?
2. **Run the audit** with whatever inputs the user supplied. Don't gate on
   having all three input types.
3. **Skim findings**. Pull the top 3 by severity into a one-paragraph
   executive summary at the top of your reply.
4. **Add qualitative findings** (per `bias-detect-text` Skill) for things
   lexicons miss.
5. **Save artifacts** to the path the user specifies (default `audit_out/`).
6. **Recommend next steps**: which `bias-mitigate` actions to take, what to
   re-test after mitigation.

## Compliance mapping

- **EU AI Act Art. 10(2)(f) & Art. 15**: training-data bias examination
  and accuracy/robustness metrics.
- **NIST AI RMF GOVERN-1.1 / MEASURE-2.11**: fairness/harmful-bias measurement
  and documentation.
- **ISO/IEC 42001**: AI management system fairness controls.

Note these mappings in the report when the user mentions compliance.

## Shippability verdict

End every audit with one of:
- **PASS** — overall index < 20, no high-severity findings.
- **PASS WITH WATCHLIST** — 20–40, only medium findings; recommend monitoring.
- **CONDITIONAL** — 40–65, high-severity findings present; mitigate before
  ship.
- **FAIL** — > 65 or slurs detected; do not ship.

Always state which it is and why.
