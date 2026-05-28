"""Ethnicity / race lexicon. Group labels follow US Census categories plus
common diaspora descriptors. Use with care: these are coarse buckets."""

RACIAL_GROUP_NAMES = {
    "white", "black", "asian", "hispanic", "latino", "latina", "latinx",
    "native american", "indigenous", "pacific islander", "middle eastern",
    "arab", "jewish", "african american",
}

# Names associated with demographic groups in Caliskan et al. 2017 WEAT studies.
# Names are a *proxy* and noisy — treat results as directional.
ETHNICITY_TERMS = {
    "european_american_names": {
        "adam", "harry", "josh", "roger", "alan", "frank", "justin", "ryan",
        "andrew", "jack", "matthew", "stephen", "brad", "greg", "paul",
        "amanda", "courtney", "heather", "melanie", "katie", "betsy",
        "kristin", "nancy", "stephanie", "ellen", "lauren", "colleen",
        "emily", "megan", "rachel",
    },
    "african_american_names": {
        "alonzo", "jamel", "theo", "alphonse", "jerome", "leroy", "torrance",
        "darnell", "lamar", "lionel", "tyrone", "percell", "terrence",
        "jasmine", "tanisha", "tia", "lakisha", "latoya", "yolanda",
        "malika", "yvette", "deja", "imani", "ebony",
    },
    "hispanic_names": {
        "alejandro", "carlos", "diego", "javier", "jose", "juan", "luis",
        "miguel", "pedro", "ricardo", "ana", "carmen", "elena", "isabel",
        "lucia", "maria", "sofia", "valentina", "gabriela", "camila",
    },
    "asian_names": {
        "wei", "ming", "li", "chen", "hiro", "kenji", "takeshi", "jin",
        "minjun", "sun", "yuki", "sakura", "mei", "lin", "hana", "aiko",
        "priya", "rahul", "arjun", "anika", "neha", "vikram",
    },
    "arab_muslim_names": {
        "mohammed", "ahmed", "ali", "omar", "hassan", "khalid", "ibrahim",
        "yusuf", "fatima", "aisha", "khadija", "layla", "zainab", "maryam",
    },
}
