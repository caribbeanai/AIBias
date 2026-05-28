"""Gender lexicon. Includes binary terms used historically in WEAT/SEAT studies;
non-binary terms included for representation analysis."""

GENDERED_PRONOUNS = {
    "masculine": {"he", "him", "his", "himself"},
    "feminine": {"she", "her", "hers", "herself"},
    "neutral": {"they", "them", "their", "theirs", "themself", "themselves"},
}

GENDER_TERMS = {
    "masculine": {
        "he", "him", "his", "man", "men", "male", "boy", "boys", "father",
        "dad", "son", "brother", "uncle", "nephew", "husband", "gentleman",
        "sir", "mister", "mr", "king", "prince", "lord", "guy", "gentlemen",
    },
    "feminine": {
        "she", "her", "hers", "woman", "women", "female", "girl", "girls",
        "mother", "mom", "daughter", "sister", "aunt", "niece", "wife",
        "lady", "madam", "ms", "mrs", "queen", "princess", "gal",
    },
    "non_binary": {
        "they", "them", "their", "nonbinary", "non-binary", "enby",
        "genderqueer", "agender", "genderfluid",
    },
}

# Historically gender-skewed occupations used in fairness probes (Bolukbasi 2016, Caliskan 2017).
GENDER_OCCUPATIONS = {
    "stereotypically_masculine": {
        "engineer", "programmer", "developer", "scientist", "ceo", "executive",
        "doctor", "surgeon", "pilot", "soldier", "lawyer", "architect",
        "mechanic", "carpenter", "electrician", "plumber", "firefighter",
        "construction worker", "truck driver", "police officer",
    },
    "stereotypically_feminine": {
        "nurse", "teacher", "secretary", "receptionist", "assistant",
        "homemaker", "babysitter", "housekeeper", "librarian", "cashier",
        "hairdresser", "social worker", "flight attendant", "stylist",
        "kindergarten teacher", "dental hygienist",
    },
}
