"""Bias mitigation utilities.

Programmatic transforms:
- Pronoun / gendered-term neutralization
- Slur scrubbing
- Counterfactual data augmentation (swap demographic terms to balance corpora)
- Group-balanced resampling for training data
- Prompt-rewriting helpers for generation pipelines

For nuanced rewriting (preserving meaning while removing bias) the SKILL.md
delegates to the LLM (Claude). This module supplies the deterministic backbone.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .lexicons import GENDER_TERMS, GENDERED_PRONOUNS
from .lexicons.ability import ABLEIST_TERMS
from .lexicons.age import AGEIST_NEGATIVE_TERMS
from .lexicons.sexuality import HOMOPHOBIC_TERMS


# Word-level neutralization map. Keys are case-insensitive.
NEUTRAL_REPLACEMENTS = {
    "chairman": "chair",
    "chairwoman": "chair",
    "policeman": "police officer",
    "policewoman": "police officer",
    "policemen": "police officers",
    "policewomen": "police officers",
    "fireman": "firefighter",
    "firewoman": "firefighter",
    "firemen": "firefighters",
    "mailman": "mail carrier",
    "mailwoman": "mail carrier",
    "mailmen": "mail carriers",
    "businessman": "businessperson",
    "businesswoman": "businessperson",
    "manpower": "workforce",
    "mankind": "humankind",
    "man-made": "human-made",
    "mothering": "parenting",
    "fathering": "parenting",
    "stewardess": "flight attendant",
    "actress": "actor",
    "waitress": "server",
    "salesman": "salesperson",
    "saleswoman": "salesperson",
    "freshman": "first-year student",
    "manhole": "utility hole",
    # Ableist
    "crazy": "unreasonable",
    "insane": "extreme",
    "lame": "unimpressive",
    "dumb": "uninformed",
    "retarded": "delayed",
    "psycho": "unstable",
}

# Counterfactual swap pairs (used in CDA / Counterfactual Data Augmentation).
CDA_PAIRS: list[tuple[str, str]] = [
    ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
    ("husband", "wife"), ("king", "queen"), ("uncle", "aunt"),
    ("nephew", "niece"), ("mr", "ms"), ("sir", "madam"),
]


@dataclass
class MitigationReport:
    original_length: int
    final_length: int
    replacements: dict[str, int] = field(default_factory=dict)
    removed_terms: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Replacements: {sum(self.replacements.values())}"]
        for k, v in sorted(self.replacements.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {k} -> ({v}x)")
        if self.removed_terms:
            lines.append(f"Removed/flagged terms: {', '.join(self.removed_terms)}")
        return "\n".join(lines)


class BiasMitigator:
    def __init__(self, extra_map: Mapping[str, str] | None = None):
        self._map = dict(NEUTRAL_REPLACEMENTS)
        if extra_map:
            self._map.update({k.lower(): v for k, v in extra_map.items()})

    # ------------------------------------------------------- token-level fixes
    def neutralize(self, text: str) -> tuple[str, MitigationReport]:
        """Replace gendered / ableist / ageist terms with neutral equivalents."""
        counts: Counter = Counter()
        removed: list[str] = []
        out = text

        # 1. Slur/pejorative scrubbing FIRST (otherwise softening rules could
        #    silently sanitize a slur instead of flagging it).
        for vocab, label in (
            (HOMOPHOBIC_TERMS, "homophobic"),
            (ABLEIST_TERMS, "ableist"),
            (AGEIST_NEGATIVE_TERMS, "ageist"),
        ):
            for term in vocab:
                pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                new_out, n = pattern.subn("[REDACTED]", out)
                if n:
                    counts[f"{term} -> [REDACTED] ({label})"] = n
                    removed.append(term)
                    out = new_out

        # 2. Neutral replacements (longest source first to handle multi-word).
        for src, tgt in sorted(self._map.items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
            new_out, n = pattern.subn(
                lambda m, tgt=tgt: _match_case(m.group(0), tgt),
                out,
            )
            if n:
                counts[f"{src} -> {tgt}"] = n
                out = new_out

        return out, MitigationReport(len(text), len(out), dict(counts), removed)

    # --------------------------------------------- counterfactual augmentation
    def counterfactual_swap(self, text: str) -> str:
        """Swap demographic terms with their counterfactual counterpart.

        Useful for CDA training-data balancing (Lu et al. 2018).
        """
        pairs = CDA_PAIRS + [(b, a) for a, b in CDA_PAIRS]
        # Use placeholders to avoid double-swap collisions.
        placeholder = "\x00{}\x00"
        out = text
        for i, (src, _tgt) in enumerate(pairs):
            out = re.sub(
                rf"\b{re.escape(src)}\b",
                lambda m, i=i: placeholder.format(i),
                out, flags=re.IGNORECASE,
            )
        for i, (_src, tgt) in enumerate(pairs):
            out = out.replace(placeholder.format(i), tgt)
        return out

    def cda_augment(self, texts: Sequence[str]) -> list[str]:
        """Return original + counterfactual versions, deduplicated."""
        seen = set()
        out = []
        for t in texts:
            for variant in (t, self.counterfactual_swap(t)):
                if variant not in seen:
                    seen.add(variant)
                    out.append(variant)
        return out

    # --------------------------------------------------- dataset rebalancing
    @staticmethod
    def balanced_resample(
        items: Sequence[tuple],
        group_key,
        seed: int = 0,
        target_per_group: int | None = None,
    ) -> list[tuple]:
        """Down/up-sample to equalize group sizes.

        items: arbitrary tuples; group_key extracts the demographic label.
        """
        rng = random.Random(seed)
        buckets: dict = defaultdict(list)
        for it in items:
            buckets[group_key(it)].append(it)
        if not buckets:
            return []
        if target_per_group is None:
            target_per_group = min(len(v) for v in buckets.values())
        out = []
        for v in buckets.values():
            if len(v) >= target_per_group:
                out.extend(rng.sample(v, target_per_group))
            else:
                out.extend(v + rng.choices(v, k=target_per_group - len(v)))
        rng.shuffle(out)
        return out

    # ----------------------------------------------------- prompt rewriting
    @staticmethod
    def neutralize_prompt(prompt: str) -> str:
        """Heuristic prompt-rewrite for image generation: strip default-demographic
        assumptions ("a businessman" -> "a businessperson") and add a diversity hint.
        """
        rewritten = prompt
        for src, tgt in NEUTRAL_REPLACEMENTS.items():
            rewritten = re.sub(rf"\b{re.escape(src)}\b", tgt, rewritten, flags=re.IGNORECASE)
        if "diverse" not in rewritten.lower():
            rewritten = rewritten.rstrip(". ") + ", representing a diverse range of ages, ethnicities, and gender presentations."
        return rewritten


def _match_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt
