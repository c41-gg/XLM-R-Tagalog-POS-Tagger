"""
MGNN tagset (Nocon & Borra, 2016) decomposed into five orthogonal heads
for multi-head XLM-R fine-tuning:

    category  -- coarse POS (NN, PR, DT, CC, LM, VB, JJ, RB, CD, TS, FW, PM)
    subtype   -- fine-grained *type* tag within category (verb tense/mood,
                 noun type, adjective type, adverb type...)
    focus     -- verb-only trigger/focus marker (actor/object/etc.), or NONE
    degree    -- adjective-only comparison marker (comparative/superlative),
                 or NONE
    extra     -- attached tags that does not align to the two previous 
                 category like ligature/linker (CCP) suffix and 
                 contractions such as Lexical Markers (LM) for "ay",
                 conjunction (CCA) for "at", and (PRSP) for "'s"  for
                 english possesive form

Source: https://www.sketchengine.eu/mgnn-tagalog-part-of-speech-tagset/
        (Nocon, N. and Borra, A., SMTPOST, 2016)

IMPORTANT: RBI ("Enclitics") is NOT a cross-cutting suffix -- it is a normal
adverb *subtype*, used when the enclitic particle (na, pa, rin, daw...) is
its own token. The actual cross-cutting attachable morpheme is CCP (the
ligature -ng/na/-g), which is why it gets its own head here instead of RBI.

Adjectives factor the same way verbs do: JJD/JJN are the base *type*
(mutually exclusive, like NNC/NNP), while JJC/JJCC/JJCS/JJCN are the
*degree* family (comparative/superlative markers) that stack onto a type,
e.g. napakalaki = JJCS (superlative) + JJD (descriptive) -> "JJCS_JJD".
This mirrors verbs stacking a focus tag onto a tense tag (VBTR_VBAF).
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------
# Atomic tag -> (category, subtype, focus, degree, extras)
# Exactly one of (subtype, focus, degree) is non-None per atomic tag.
# ---------------------------------------------------------------
# Every atomic (non-compound) tag from the official 69-tag base set,
# plus the two pipeline-internal tags your aligner emits (EMOJI/IGNORED)
# which are NOT part of MGNN proper.

EXTRA_TAGS = {
    "CCP",     # ligature
    "CCA",     # attached "at"
    "LM",      # attached "ay"
    "PRSP",    # English possessive 's
}


ATOMIC_TAGS = {
    # Nouns
    "NNC":  ("NN", "NNC", None, None),
    "NNP":  ("NN", "NNP", None, None),
    "NNPA": ("NN", "NNPA", None, None),
    "NNCA": ("NN", "NNCA", None, None),

    # Pronouns
    "PRS":  ("PR", "PRS", None, None),
    "PRP":  ("PR", "PRP", None, None),
    "PRSP": ("PR", "PRSP", None, None),
    "PRO":  ("PR", "PRO", None, None),
    "PRQ":  ("PR", "PRQ", None, None),
    "PRQP": ("PR", "PRQP", None, None),
    "PRL":  ("PR", "PRL", None, None),
    "PRC":  ("PR", "PRC", None, None),
    "PRF":  ("PR", "PRF", None, None),
    "PRI":  ("PR", "PRI", None, None),

    # Determiners
    "DTC":  ("DT", "DTC", None, None),
    "DTCP": ("DT", "DTCP", None, None),
    "DTP":  ("DT", "DTP", None, None),
    "DTPP": ("DT", "DTPP", None, None),

    # Conjunctions (CCP is the ligature -- handled specially below)
    "CCT": ("CC", "CCT", None, None),
    "CCR": ("CC", "CCR", None, None),
    "CCB": ("CC", "CCB", None, None),
    "CCA": ("CC", "CCA", None, None),
    "CCU": ("CC", "CCU", None, None),
    "CCP": ("CC", "CCP", None, None ), 

    # Lexical marker
    "LM": ("LM", None, None, None),

    # Verbs -- mood/class + tense/aspect share the "subtype" slot
    "VBW":  ("VB", "VBW", None, None),   # neutral/infinitive
    "VBS":  ("VB", "VBS", None, None),   # auxiliary/modal
    "VBH":  ("VB", "VBH", None, None),   # existential
    "VBN":  ("VB", "VBN", None, None),   # non-existential
    "VBTS": ("VB", "VBTS", None, None),  # perfective
    "VBTR": ("VB", "VBTR", None, None),  # imperfective
    "VBTF": ("VB", "VBTF", None, None),  # contemplative
    "VBTP": ("VB", "VBTP", None, None),  # recent past

    # Verb focus (orthogonal axis -- own head)
    "VBAF": ("VB", None, "AF", None),  # actor focus
    "VBOF": ("VB", None, "OF", None),  # object/goal focus
    "VBOB": ("VB", None, "OB", None),  # benefactive focus
    "VBOL": ("VB", None, "OL", None),  # locative focus
    "VBOI": ("VB", None, "OI", None),  # instrumental focus
    "VBRF": ("VB", None, "RF", None),  # referential/measurement focus

    # Adjectives -- JJD/JJN are base type. JJC/JJCC/JJCS/JJCN are the
    # degree (comparison) family and go in their own head, not subtype.
    "JJD": ("JJ", "JJD", None, None),   # descriptive (base type)
    "JJN": ("JJ", "JJN", None, None),   # number-adjective (base type)
    "JJCC":  ("JJ", None, None, "COMP"),   # comparative
    "JJC": ("JJ", None, None, "COMPEQ"), # comparative (equality)
    "JJCS": ("JJ", None, None, "SUP"),    # superlative
    "JJCN": ("JJ", None, None, "COMPN"),  # comparative (numeral)

    # Adverbs (RBI = enclitic-as-its-own-token, a normal subtype here)
    "RBD": ("RB", "RBD", None, None),
    "RBN": ("RB", "RBN", None, None),
    "RBK": ("RB", "RBK", None, None),
    "RBP": ("RB", "RBP", None, None),
    "RBB": ("RB", "RBB", None, None),
    "RBR": ("RB", "RBR", None, None),
    "RBQ": ("RB", "RBQ", None, None),
    "RBT": ("RB", "RBT", None, None),
    "RBF": ("RB", "RBF", None, None),
    "RBW": ("RB", "RBW", None, None),
    "RBM": ("RB", "RBM", None, None),
    "RBL": ("RB", "RBL", None, None),
    "RBI": ("RB", "RBI", None, None),
    "RBJ": ("RB", "RBJ", None, None),
    "RBS": ("RB", "RBS", None, None),

    # Cardinal numbers
    "CDB": ("CD", "CDB", None, None),

    # Standalone categories, no subtype
    "TS": ("TS", None, None, None),
    "FW": ("FW", None, None, None),

    # Punctuation
    "PMP":  ("PM", "PMP", None, None),
    "PME":  ("PM", "PME", None, None),
    "PMQ":  ("PM", "PMQ", None, None),
    "PMC":  ("PM", "PMC", None, None),
    "PMSC": ("PM", "PMSC", None, None),
    "PMS":  ("PM", "PMS", None, None),

    # Pipeline-internal (NOT MGNN -- emitted by alignment.py, not the tagger)
    "EMOJI":   ("PIPELINE", "EMOJI", None, None),
    "IGNORED": ("PIPELINE", "IGNORED", None, None),
}


@dataclass
class DecomposedTag:
    category: Optional[str] = None
    subtype: Optional[str] = None
    focus: Optional[str] = None
    degree: Optional[str] = None
    extras: list[str] = field(default_factory=list)
    extra_subtypes: list = field(default_factory=list)  # same-axis stacking
    unknown_parts: list = field(default_factory=list)    # tags not in table


def decompose(tag: str) -> DecomposedTag:
    """Split a (possibly compound) MGNN tag into its five heads."""

    result = DecomposedTag()

    for part in tag.split("_"):
        if part not in ATOMIC_TAGS:
            result.unknown_parts.append(part)
            continue

        category, subtype, focus, degree = ATOMIC_TAGS[part]

        if part in EXTRA_TAGS and result.category is not None:
            result.extras.append(part)
            continue

        if focus:
            result.focus = focus
            if result.category is None:
                result.category = category

            continue

        if degree:
            result.degree = degree
            if result.category is None:
                result.category = category

            continue

        # normal category+subtype (type) tag

        if result.category is None:
            result.category = category
            result.subtype = subtype

        elif result.category == category and result.subtype is not None:
            # genuine same-axis stacking -- two *type* values at once
            result.extra_subtypes.append(subtype)

        elif result.subtype is None:
            result.subtype = subtype

        else:
            result.extra_subtypes.append(f"{category}:{subtype}")



    return result





def recompose(d: DecomposedTag) -> str:
    """Best-effort rejoin, for round-trip validation against source data."""

    parts = []

    if d.subtype:
        parts.append(d.subtype)
    elif d.category and not d.focus and not d.degree:
        parts.append(d.category)
    parts.extend(d.extra_subtypes)

    if d.focus:
        for tag, (_, _, foc, _) in ATOMIC_TAGS.items():
            if foc == d.focus:
                parts.append(tag)
                break

    if d.degree:

        for tag, (_, _, _, deg) in ATOMIC_TAGS.items():
            if deg == d.degree:
                parts.append(tag)
                break

    if d.extras:

        parts.extend(d.extras)

    return "_".join(parts)


if __name__ == "__main__":

    # quick smoke test

    for tag in ["NNC", "JJD_CCP", "VBTR_VBAF", "VBTS_VBOF", "JJCS_JJD",
                "PRI_CCP", "RBI_CCP", "VBTR_VBRF", "JJN_CCP", "JJCC"]:

        d = decompose(tag)

        print(f"{tag:<15} -> category={d.category} subtype={d.subtype} "
              f"focus={d.focus} degree={d.degree} extras={d.extras} "
              f"extra={d.extra_subtypes} unknown={d.unknown_parts}")
