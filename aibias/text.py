"""Text bias detection.

Hybrid approach:
1. Lexicon-based scans for demographic mentions, slurs, ageist/ableist terms,
   stereotype attribute co-occurrences (no external models required).
2. Optional sentiment via VADER or transformers if installed.
3. LLM-assisted analysis (Claude) for subtle/contextual bias — invoked by the
   Skill instructions, not by this Python module directly.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

from .lexicons import (
    ALL_DEMOGRAPHIC_AXES,
    GENDER_TERMS,
    GENDER_OCCUPATIONS,
    POSITIVE_ATTRIBUTES,
    NEGATIVE_ATTRIBUTES,
    COMPETENCE_TERMS,
    WARMTH_TERMS,
    STEREOTYPE_SEEDS,
)
from .lexicons.age import AGEIST_NEGATIVE_TERMS
from .lexicons.ability import ABLEIST_TERMS
from .lexicons.sexuality import HOMOPHOBIC_TERMS
from .metrics import (
    normalized_pointwise_mutual_information,
    representation_ratio,
    cooccurrence,
    bias_index,
)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class Finding:
    axis: str               # e.g. "gender", "ethnicity"
    kind: str               # e.g. "slur", "stereotype_cooccurrence", "underrepresentation"
    severity: str           # "low" | "medium" | "high"
    message: str
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0      # 0..1, axis-local intensity


@dataclass
class TextBiasReport:
    text_preview: str
    token_count: int
    demographic_mentions: dict[str, dict[str, int]]
    representation: dict[str, dict[str, float]]
    sentiment_by_group: dict[str, float]
    stereotype_associations: dict[str, dict[str, float]]
    slurs_detected: list[str]
    findings: list[Finding]
    axis_scores: dict[str, float]
    bias_index: dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d

    def summary(self) -> str:
        bi = self.bias_index
        lines = [
            f"Bias Index: {bi['overall']}/100 (confidence {bi['confidence']})",
            "Axis scores: " + ", ".join(
                f"{k}={v}" for k, v in bi["by_axis"].items()
            ),
            f"Findings: {len(self.findings)}",
        ]
        for f in self.findings[:10]:
            lines.append(f"  [{f.severity}] {f.axis}/{f.kind}: {f.message}")
        return "\n".join(lines)


class TextBiasDetector:
    """Programmatic text-bias detector.

    Designed to run with zero external dependencies. If `vaderSentiment` or
    `transformers` is installed, sentiment scoring will use them automatically.
    """

    def __init__(self, window: int = 8):
        self.window = window
        self._sentiment_fn = self._load_sentiment()

    # ------------------------------------------------------------------ public
    def analyze(self, text: str) -> TextBiasReport:
        tokens = tokenize(text)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        demographic_mentions = self._count_demographics(tokens)
        slurs = self._find_slurs(tokens)
        stereotype = self._stereotype_associations(tokens)
        sentiment_by_group = self._sentiment_by_group(sentences)
        representation = self._representation(demographic_mentions)

        findings: list[Finding] = []
        axis_scores: dict[str, float] = defaultdict(float)

        # Slurs / ableist / ageist terms -> immediate high severity
        if slurs:
            for slur in slurs:
                axis = self._axis_of_slur(slur)
                findings.append(Finding(
                    axis=axis, kind="slur_or_pejorative", severity="high",
                    message=f"Pejorative/slur term detected: '{slur}'",
                    evidence=[slur], score=1.0,
                ))
                axis_scores[axis] = max(axis_scores[axis], 0.9)

        # Representation imbalances
        for axis, shares in representation.items():
            if not shares:
                continue
            mx = max(shares.values())
            mn = min(shares.values())
            if mx - mn >= 0.6 and mx >= 0.7:
                dominant = max(shares, key=shares.get)
                findings.append(Finding(
                    axis=axis, kind="representation_imbalance",
                    severity="medium" if mx < 0.85 else "high",
                    message=(
                        f"Group '{dominant}' dominates mentions on axis '{axis}' "
                        f"({mx*100:.0f}% share)."
                    ),
                    evidence=[f"{g}={v:.2f}" for g, v in shares.items()],
                    score=mx - mn,
                ))
                axis_scores[axis] = max(axis_scores[axis], mx - mn)

        # Stereotype associations
        for axis, by_group in stereotype.items():
            for group, attrs in by_group.items():
                for attr, score in attrs.items():
                    if score >= 0.3:  # NPMI threshold
                        findings.append(Finding(
                            axis=axis, kind="stereotype_association",
                            severity="high" if score > 0.5 else "medium",
                            message=(
                                f"'{group}' strongly co-occurs with "
                                f"'{attr}' descriptors (NPMI={score:.2f})."
                            ),
                            evidence=[f"{group}~{attr}={score:.2f}"],
                            score=score,
                        ))
                        axis_scores[axis] = max(axis_scores[axis], score)

        # Sentiment skew
        if len(sentiment_by_group) >= 2:
            vals = list(sentiment_by_group.values())
            spread = max(vals) - min(vals)
            if spread >= 0.3:
                low = min(sentiment_by_group, key=sentiment_by_group.get)
                findings.append(Finding(
                    axis="sentiment", kind="valence_skew",
                    severity="high" if spread > 0.6 else "medium",
                    message=(
                        f"Sentiment differs by {spread:.2f} across groups; "
                        f"'{low}' carries the most negative valence."
                    ),
                    evidence=[f"{g}={v:.2f}" for g, v in sentiment_by_group.items()],
                    score=min(1.0, spread),
                ))
                axis_scores["sentiment"] = spread

        # Gendered-occupation patterns
        for group, occs in self._gendered_occupations(tokens).items():
            if occs:
                findings.append(Finding(
                    axis="gender", kind="occupational_stereotype",
                    severity="medium",
                    message=(
                        f"Stereotypically-{group} occupations referenced: "
                        f"{', '.join(sorted(occs)[:5])}"
                    ),
                    evidence=sorted(occs),
                    score=min(1.0, len(occs) / 5),
                ))
                axis_scores["gender"] = max(axis_scores["gender"], min(1.0, len(occs) / 5))

        bi = bias_index(dict(axis_scores), sample_size=len(tokens))

        return TextBiasReport(
            text_preview=text[:240] + ("..." if len(text) > 240 else ""),
            token_count=len(tokens),
            demographic_mentions=demographic_mentions,
            representation=representation,
            sentiment_by_group=sentiment_by_group,
            stereotype_associations=stereotype,
            slurs_detected=slurs,
            findings=findings,
            axis_scores=dict(axis_scores),
            bias_index=bi.to_dict(),
        )

    # ----------------------------------------------------------------- helpers
    def _count_demographics(self, tokens: list[str]) -> dict[str, dict[str, int]]:
        token_set = Counter(tokens)
        out: dict[str, dict[str, int]] = {}
        for axis, groups in ALL_DEMOGRAPHIC_AXES.items():
            out[axis] = {}
            for group, terms in groups.items():
                out[axis][group] = sum(token_set.get(t, 0) for t in terms)
        return out

    def _representation(
        self, demographic_mentions: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, float]]:
        return {
            axis: representation_ratio(counts)
            for axis, counts in demographic_mentions.items()
        }

    def _find_slurs(self, tokens: list[str]) -> list[str]:
        token_set = set(tokens)
        hits = []
        for vocab in (HOMOPHOBIC_TERMS, ABLEIST_TERMS, AGEIST_NEGATIVE_TERMS):
            hits.extend(sorted(token_set & vocab))
        return hits

    def _axis_of_slur(self, slur: str) -> str:
        if slur in HOMOPHOBIC_TERMS:
            return "sexuality"
        if slur in ABLEIST_TERMS:
            return "ability"
        if slur in AGEIST_NEGATIVE_TERMS:
            return "age"
        return "other"

    def _stereotype_associations(
        self, tokens: list[str]
    ) -> dict[str, dict[str, dict[str, float]]]:
        cooc = cooccurrence(tokens, window=self.window)
        total_pairs = sum(cooc.values()) or 1
        marg = Counter()
        for (a, b), c in cooc.items():
            marg[a] += c

        attribute_sets = {
            "positive": POSITIVE_ATTRIBUTES,
            "negative": NEGATIVE_ATTRIBUTES,
            "competence": COMPETENCE_TERMS,
            "warmth": WARMTH_TERMS,
            **STEREOTYPE_SEEDS,
        }

        out: dict[str, dict[str, dict[str, float]]] = {}
        for axis, groups in ALL_DEMOGRAPHIC_AXES.items():
            out[axis] = {}
            for group, terms in groups.items():
                out[axis][group] = {}
                for attr_name, attr_terms in attribute_sets.items():
                    npmis = []
                    for t in terms:
                        for a in attr_terms:
                            j = cooc.get((t, a), 0)
                            if j == 0:
                                continue
                            npmis.append(
                                normalized_pointwise_mutual_information(
                                    j, marg[t], marg[a], total_pairs
                                )
                            )
                    out[axis][group][attr_name] = (
                        sum(npmis) / len(npmis) if npmis else 0.0
                    )
        return out

    def _sentiment_by_group(self, sentences: list[str]) -> dict[str, float]:
        if not self._sentiment_fn:
            return {}
        by_group: dict[str, list[float]] = defaultdict(list)
        for sent in sentences:
            toks = set(tokenize(sent))
            score = self._sentiment_fn(sent)
            for axis, groups in ALL_DEMOGRAPHIC_AXES.items():
                for group, terms in groups.items():
                    if toks & terms:
                        by_group[f"{axis}:{group}"].append(score)
        return {g: sum(v) / len(v) for g, v in by_group.items() if v}

    def _gendered_occupations(self, tokens: list[str]) -> dict[str, set[str]]:
        text_lower = " ".join(tokens)
        return {
            group: {occ for occ in occs if occ in text_lower}
            for group, occs in GENDER_OCCUPATIONS.items()
        }

    def _load_sentiment(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            return lambda s: sia.polarity_scores(s)["compound"]
        except Exception:
            pass
        # Lightweight built-in fallback: positive-minus-negative word ratio.
        pos = POSITIVE_ATTRIBUTES
        neg = NEGATIVE_ATTRIBUTES

        def fallback(s: str) -> float:
            toks = tokenize(s)
            if not toks:
                return 0.0
            p = sum(1 for t in toks if t in pos)
            n = sum(1 for t in toks if t in neg)
            return (p - n) / max(1, len(toks))

        return fallback
