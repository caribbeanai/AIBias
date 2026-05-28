"""Lexicons of demographic terms and stereotypical attribute words.

Sources are research lexicons commonly used in NLP fairness studies
(WEAT, SEAT, StereoSet, CrowS-Pairs, Winogender) — adapted and consolidated.
These are intentionally compact and English-focused; extend for production use.
"""
from .gender import GENDER_TERMS, GENDER_OCCUPATIONS, GENDERED_PRONOUNS
from .ethnicity import ETHNICITY_TERMS, RACIAL_GROUP_NAMES
from .age import AGE_TERMS, AGE_GROUPS
from .religion import RELIGION_TERMS
from .ability import ABILITY_TERMS
from .sexuality import SEXUALITY_TERMS
from .attributes import (
    POSITIVE_ATTRIBUTES,
    NEGATIVE_ATTRIBUTES,
    COMPETENCE_TERMS,
    WARMTH_TERMS,
    STEREOTYPE_SEEDS,
)

ALL_DEMOGRAPHIC_AXES = {
    "gender": GENDER_TERMS,
    "ethnicity": ETHNICITY_TERMS,
    "age": AGE_TERMS,
    "religion": RELIGION_TERMS,
    "ability": ABILITY_TERMS,
    "sexuality": SEXUALITY_TERMS,
}
