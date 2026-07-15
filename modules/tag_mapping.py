from typing import Optional

PENN_TO_MGNN: dict[str, Optional[str]] = {
    # Nouns
    "NN": "NNC",
    "NNS": "NNC",
    "NNP": "NNP",
    "NNPS": "NNP",

    # Verbs
    "VB": "VBW",
    "VBD": "VBTS",
    "VBG": "VBTR",
    "VBN": "VBTS",
    "VBP": "VBW",
    "VBZ": "VBW",
    "MD": "VBW",
    "EX": "VBH",

    # Adjectives
    "JJ": "JJD",
    "JJR": "JJD",
    "JJS": "JJD",

    # Adverbs
    "RB": "RBD",
    "RBR": "RBD",
    "RBS": "RBD",

    # Numbers
    "CD": "CDB",

    # Prepositions
    "IN": "CCU",

    # to
    "TO": "CCT",

    # Possessives
    "POS": "PRSP",
    "PRP$": "PRSP",
}

SPECIAL_WORD_MAPPING: dict[str, Optional[str]] = {
    # Determiners
    "a": "DTC",
    "an": "DTC",
    "the":  "DTP",
    # Conjunctions
    # CCA = coordinating conjunctions; CCB = case/connective particle.
    "and": "CCA",
    "nor": "CCA",
    "but":  "CCT",
    "or":  "CCT" ,
    "yet": "CCT" ,
    "for": "CCT" ,
    "so":  "CCR" ,
    # Pronouns
    "i": "PRS" ,
    "me": "PRS" ,
    "you": "PRS" ,
    "he": "PRS" ,
    "him":  "PRS" ,
    "she":  "PRS" ,
    "her":  "PRS" ,
    "it":   "PRS" ,
    "we":   "PRP" ,
    "us":   "PRP" ,
    "they": "PRP" ,
}


def english_to_mgnn(word: str, penn: str) -> str:
    """
    Converts an English word and its Penn Treebank tag
    into the equivalent MGNN tag.
    """

    word = word.lower()

    if word in SPECIAL_WORD_MAPPING:
        return SPECIAL_WORD_MAPPING[word]

    return PENN_TO_MGNN.get(
        penn,
        "FW"
    )