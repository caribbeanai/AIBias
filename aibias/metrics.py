"""Quantitative bias metrics.

References:
- Dwork et al. 2012 ("Fairness through awareness") — statistical parity.
- Feldman et al. 2015 — disparate impact (80% rule).
- Hardt et al. 2016 — equal opportunity / equalized odds.
- Caliskan et al. 2017 — WEAT (Word Embedding Association Test).
- Bolukbasi et al. 2016 — gender direction.
- Bordia & Bowman 2019 — NPMI-based corpus bias.
- Webster et al. 2020 — log-probability bias score.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Classification fairness metrics (for downstream models / decisions)
# ---------------------------------------------------------------------------

def statistical_parity_difference(
    positive_rates: Mapping[str, float],
    privileged_group: str,
) -> dict[str, float]:
    """SPD = P(Y=1|A=g) - P(Y=1|A=priv) for every unprivileged group g.

    A perfectly fair model has SPD = 0. |SPD| > 0.1 is commonly flagged.
    """
    priv = positive_rates[privileged_group]
    return {g: r - priv for g, r in positive_rates.items() if g != privileged_group}


def demographic_parity(
    predictions_by_group: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Per-group positive prediction rate P(Y_hat = 1 | A = g)."""
    out = {}
    for g, preds in predictions_by_group.items():
        out[g] = sum(preds) / len(preds) if preds else 0.0
    return out


def disparate_impact(
    positive_rates: Mapping[str, float],
    privileged_group: str,
) -> dict[str, float]:
    """DI = P(Y=1|unpriv) / P(Y=1|priv).

    The "80% rule" (EEOC) flags DI < 0.8 or > 1.25 as adverse impact.
    """
    priv = positive_rates[privileged_group]
    if priv == 0:
        return {g: float("inf") for g in positive_rates if g != privileged_group}
    return {g: r / priv for g, r in positive_rates.items() if g != privileged_group}


def equal_opportunity_difference(
    tpr_by_group: Mapping[str, float],
    privileged_group: str,
) -> dict[str, float]:
    """EOD = TPR_unpriv - TPR_priv. Zero = equal true positive rates."""
    priv = tpr_by_group[privileged_group]
    return {g: t - priv for g, t in tpr_by_group.items() if g != privileged_group}


def equalized_odds_difference(
    tpr_by_group: Mapping[str, float],
    fpr_by_group: Mapping[str, float],
    privileged_group: str,
) -> dict[str, dict[str, float]]:
    """Returns TPR and FPR gaps per group vs privileged."""
    return {
        "tpr_gap": equal_opportunity_difference(tpr_by_group, privileged_group),
        "fpr_gap": equal_opportunity_difference(fpr_by_group, privileged_group),
    }


def theil_index(values: Sequence[float]) -> float:
    """Generalized entropy / Theil index — inequality across individuals."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    total = 0.0
    for v in values:
        if v <= 0:
            continue
        total += (v / mean) * math.log(v / mean)
    return total / n


# ---------------------------------------------------------------------------
# Representation metrics (for generated content corpora)
# ---------------------------------------------------------------------------

def representation_ratio(
    counts: Mapping[str, int],
    baseline: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Share-of-mentions per group, optionally normalized by baseline share.

    baseline maps group -> expected share in [0,1]. When provided returns the
    ratio observed/expected; ratio of 1.0 means perfectly representative.
    """
    total = sum(counts.values()) or 1
    observed = {g: c / total for g, c in counts.items()}
    if baseline is None:
        return observed
    return {g: observed.get(g, 0.0) / baseline[g] if baseline[g] else float("inf")
            for g in baseline}


def sentiment_skew(
    sentiments_by_group: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Mean sentiment per group. Diff across groups indicates valence bias."""
    return {
        g: (sum(s) / len(s)) if s else 0.0
        for g, s in sentiments_by_group.items()
    }


# ---------------------------------------------------------------------------
# Association / embedding-based bias metrics
# ---------------------------------------------------------------------------

def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    num = sum(a * b for a, b in zip(u, v))
    du = math.sqrt(sum(a * a for a in u))
    dv = math.sqrt(sum(b * b for b in v))
    return num / (du * dv) if du and dv else 0.0


def weat_effect_size(
    X: Sequence[Sequence[float]],
    Y: Sequence[Sequence[float]],
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
) -> float:
    """WEAT effect size d (Caliskan et al. 2017).

    X, Y: target word embeddings (e.g. male names vs female names).
    A, B: attribute word embeddings (e.g. career vs family).
    Returns Cohen's d. |d| > 0.5 = moderate, > 0.8 = large association.
    """
    def s(w):
        return (sum(_cosine(w, a) for a in A) / len(A)) - (
            sum(_cosine(w, b) for b in B) / len(B)
        )
    sx = [s(x) for x in X]
    sy = [s(y) for y in Y]
    mean_diff = (sum(sx) / len(sx)) - (sum(sy) / len(sy))
    pooled = sx + sy
    pooled_mean = sum(pooled) / len(pooled)
    sd = math.sqrt(sum((p - pooled_mean) ** 2 for p in pooled) / (len(pooled) - 1))
    return mean_diff / sd if sd else 0.0


def stereotype_association(
    cooccurrence: Mapping[tuple[str, str], int],
    group_terms: Mapping[str, set[str]],
    attribute_terms: Mapping[str, set[str]],
) -> dict[str, dict[str, float]]:
    """Per-(group, attribute_set) association score = mean co-occurrence count."""
    out: dict[str, dict[str, float]] = {}
    for g, gset in group_terms.items():
        out[g] = {}
        for attr, aset in attribute_terms.items():
            pairs = [cooccurrence.get((t, a), 0) for t in gset for a in aset]
            out[g][attr] = sum(pairs) / len(pairs) if pairs else 0.0
    return out


def normalized_pointwise_mutual_information(
    joint: int, marg_x: int, marg_y: int, total: int,
) -> float:
    """NPMI in [-1, 1]. +1 = perfect association, 0 = independence, -1 = never co-occur."""
    if joint == 0 or marg_x == 0 or marg_y == 0 or total == 0:
        return -1.0
    p_xy = joint / total
    p_x = marg_x / total
    p_y = marg_y / total
    pmi = math.log(p_xy / (p_x * p_y))
    h = -math.log(p_xy)
    if h == 0:
        return 1.0  # perfect co-occurrence
    return pmi / h


def log_probability_bias_score(
    p_target_given_priv: float,
    p_target_given_unpriv: float,
) -> float:
    """LPBS = log P(target|priv) - log P(target|unpriv) (Webster et al. 2020).

    Used to compare LM probabilities of completions across template variants.
    Positive => model favors privileged-group completion.
    """
    if p_target_given_priv <= 0 or p_target_given_unpriv <= 0:
        raise ValueError("Probabilities must be > 0")
    return math.log(p_target_given_priv) - math.log(p_target_given_unpriv)


# ---------------------------------------------------------------------------
# Composite indices
# ---------------------------------------------------------------------------

@dataclass
class BiasIndex:
    """Composite 0-100 bias score with sub-scores.

    0 = no detected bias; 100 = severe bias across all axes.
    """
    overall: float
    by_axis: dict[str, float]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 2),
            "by_axis": {k: round(v, 2) for k, v in self.by_axis.items()},
            "confidence": round(self.confidence, 2),
        }


def bias_index(
    axis_scores: Mapping[str, float],
    sample_size: int = 0,
) -> BiasIndex:
    """Aggregate per-axis bias scores (each in [0,1]) into a 0-100 index.

    Uses L2 norm so a single severe axis still drives the score up.
    Confidence rises with sample_size (asymptotes at 1.0 around N=500).
    """
    if not axis_scores:
        return BiasIndex(0.0, {}, 0.0)
    vals = list(axis_scores.values())
    l2 = math.sqrt(sum(v * v for v in vals) / len(vals))
    overall = min(100.0, l2 * 100.0)
    confidence = 1 - math.exp(-sample_size / 150) if sample_size else 0.3
    return BiasIndex(
        overall=overall,
        by_axis={k: round(v * 100, 2) for k, v in axis_scores.items()},
        confidence=confidence,
    )


def cooccurrence(tokens: Iterable[str], window: int = 5) -> Counter:
    """Symmetric co-occurrence counts within a sliding window."""
    toks = [t.lower() for t in tokens]
    counts: Counter = Counter()
    for i, t in enumerate(toks):
        for j in range(max(0, i - window), min(len(toks), i + window + 1)):
            if i == j:
                continue
            counts[(t, toks[j])] += 1
    return counts
