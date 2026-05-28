"""Age lexicon."""

AGE_GROUPS = ("child", "young_adult", "middle_aged", "older_adult")

AGE_TERMS = {
    "child": {
        "child", "kid", "children", "kids", "youngster", "toddler", "baby",
        "infant", "youth", "teen", "teenager", "adolescent",
    },
    "young_adult": {
        "young", "youthful", "twentysomething", "millennial", "gen z",
        "young adult", "youngster",
    },
    "middle_aged": {
        "middle-aged", "middle aged", "midlife", "forties", "fifties",
        "gen x", "adult",
    },
    "older_adult": {
        "old", "elderly", "senior", "seniors", "geriatric", "aged",
        "retired", "retiree", "boomer", "old-timer", "septuagenarian",
        "octogenarian", "pensioner", "grandparent", "grandma", "grandpa",
    },
}

AGEIST_NEGATIVE_TERMS = {
    "senile", "decrepit", "feeble", "useless", "slow", "outdated",
    "out-of-touch", "doddering", "frail", "incompetent",
}
AGEIST_POSITIVE_STEREOTYPES = {
    "wise", "experienced", "grandfatherly", "grandmotherly",
}
