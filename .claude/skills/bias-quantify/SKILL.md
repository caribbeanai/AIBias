---
name: bias-quantify
description: Compute quantitative fairness metrics — statistical parity, disparate impact, equal opportunity, equalized odds, WEAT, NPMI, Theil index, composite bias index — for classifier outputs, content corpora, or embedding spaces. Use when the user has predictions/labels grouped by demographic attribute or wants to put a number on bias.
---

# Bias quantification

A focused metric calculator. Use this when the question is "how biased,
quantitatively?" rather than "where is the bias?".

## Metric chooser

| You have... | Use |
|---|---|
| Predictions {group: [0/1, ...]} | `demographic_parity`, then `disparate_impact` |
| TPR / FPR per group | `equal_opportunity_difference`, `equalized_odds_difference` |
| Per-individual outcome scores | `theil_index` |
| Word embeddings + target/attribute sets | `weat_effect_size` |
| Co-occurrence counts | `normalized_pointwise_mutual_information` |
| Counts per group | `representation_ratio` (with baseline) |
| LM token probabilities for paired prompts | `log_probability_bias_score` |
| All of the above | `bias_index` to combine |

## Interpretation guide

- **Disparate impact (DI)**: EEOC "80% rule" — flag if DI < 0.8 or > 1.25.
- **Statistical parity diff (SPD)**: |SPD| > 0.1 commonly considered material.
- **Equal opportunity diff (EOD)**: |EOD| > 0.1 = unequal TPR.
- **WEAT effect size d**: |d| > 0.5 moderate, > 0.8 large association.
- **NPMI ∈ [-1, 1]**: > 0.3 = strong positive association; < -0.3 = avoidance.
- **Theil index**: 0 = perfect equality, grows unbounded.
- **Composite bias index**: 0..100 (this repo's L2 aggregate).

## Example

```python
from aibias import (
    disparate_impact, statistical_parity_difference,
    equal_opportunity_difference, bias_index,
)

pos_rates = {"male": 0.62, "female": 0.41, "nonbinary": 0.38}
di  = disparate_impact(pos_rates, privileged_group="male")
spd = statistical_parity_difference(pos_rates, "male")
# di = {"female": 0.66, "nonbinary": 0.61}  -> both violate 80% rule
```

## When metrics conflict (and they will)

DI and EOD often disagree (Kleinberg–Chouldechova impossibility result). Be
explicit with the user about which fairness definition they're optimizing
for — they cannot all be satisfied simultaneously unless base rates are equal.

State the chosen definition in the report header.

## Reporting checklist

1. Sample size per group (small N → low confidence; flag if any group < 30).
2. Baseline used for representation (real-world, target, or uniform).
3. Confidence interval or bootstrap range, not just the point estimate.
4. Which fairness definition is being optimized.
5. Whether metrics were computed pre- or post-mitigation.
