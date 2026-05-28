"""Attribute words used as targets in association tests (WEAT-style).

POSITIVE / NEGATIVE: pleasantness valence (Greenwald 1998 IAT seed words).
COMPETENCE / WARMTH: stereotype content model (Fiske 2002).
STEREOTYPE_SEEDS: common stereotypical descriptors for probe construction.
"""

POSITIVE_ATTRIBUTES = {
    "love", "joy", "peace", "wonderful", "pleasure", "friend", "laughter",
    "happy", "lucky", "diamond", "loyal", "honest", "rainbow", "freedom",
    "honor", "miracle", "sunrise", "family", "paradise", "vacation",
    "gentle", "warm", "talented", "brilliant", "kind", "caring", "smart",
}

NEGATIVE_ATTRIBUTES = {
    "abuse", "crash", "filth", "murder", "sickness", "accident", "death",
    "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute",
    "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill",
    "rotten", "vomit", "agony", "prison", "criminal", "violent", "lazy",
    "stupid", "evil",
}

COMPETENCE_TERMS = {
    "competent", "skilled", "intelligent", "capable", "efficient",
    "effective", "professional", "qualified", "expert", "accomplished",
    "ambitious", "assertive", "decisive", "rational", "analytical",
}

WARMTH_TERMS = {
    "warm", "friendly", "kind", "caring", "compassionate", "nurturing",
    "gentle", "supportive", "empathetic", "understanding", "sociable",
    "emotional", "sensitive", "modest", "agreeable",
}

# Common stereotype dimensions used in StereoSet (Nadeem 2020) and CrowS-Pairs (Nangia 2020).
STEREOTYPE_SEEDS = {
    "appearance": {"attractive", "beautiful", "ugly", "fat", "thin", "tall", "short"},
    "intelligence": {"smart", "intelligent", "stupid", "dumb", "bright", "slow"},
    "criminality": {"criminal", "thug", "thief", "violent", "dangerous", "law-abiding"},
    "wealth": {"rich", "wealthy", "poor", "impoverished", "destitute", "affluent"},
    "morality": {"honest", "dishonest", "moral", "immoral", "trustworthy", "shady"},
}
