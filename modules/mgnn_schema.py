"""
MGNN tagset (Nocon & Borra, 2016) decomposed into five orthogonal heads
for multi-head XLM-R fine-tuning:

    category  -- coarse POS (NN, PR, DT, CC, LM, VB, JJ, RB, CD, TS, FW, PM)
    subtype   -- fine-grained *type* tag within category (verb tense/mood,
                 noun type, adjective type, adverb type...)
    focus     -- verb-only trigger/focus marker (actor/object/etc.), or NONE
    degree    -- adjective-only comparison marker (comparative/superlative),
                 or NONE
    ligature  -- whether the ligature/linker (CCP) is attached, True/False

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
# Atomic tag -> (category, subtype, focus, degree, is_ligature)
# Exactly one of (subtype, focus, degree) is non-None per atomic tag.
# ---------------------------------------------------------------
# Every atomic (non-compound) tag from the official 69-tag base set,
# plus the two pipeline-internal tags your aligner emits (EMOJI/IGNORED)
# which are NOT part of MGNN proper.

ATOMIC_TAGS = {
    # Nouns
    "NNC":  ("NN", "NNC", None, None, False),
    "NNP":  ("NN", "NNP", None, None, False),
    "NNPA": ("NN", "NNPA", None, None, False),
    "NNCA": ("NN", "NNCA", None, None, False),

    # Pronouns
    "PRS":  ("PR", "PRS", None, None, False),
    "PRP":  ("PR", "PRP", None, None, False),
    "PRSP": ("PR", "PRSP", None, None, False),
    "PRO":  ("PR", "PRO", None, None, False),
    "PRQ":  ("PR", "PRQ", None, None, False),
    "PRQP": ("PR", "PRQP", None, None, False),
    "PRL":  ("PR", "PRL", None, None, False),
    "PRC":  ("PR", "PRC", None, None, False),
    "PRF":  ("PR", "PRF", None, None, False),
    "PRI":  ("PR", "PRI", None, None, False),

    # Determiners
    "DTC":  ("DT", "DTC", None, None, False),
    "DTCP": ("DT", "DTCP", None, None, False),
    "DTP":  ("DT", "DTP", None, None, False),
    "DTPP": ("DT", "DTPP", None, None, False),

    # Conjunctions (CCP is the ligature -- handled specially below)
    "CCT": ("CC", "CCT", None, None, False),
    "CCR": ("CC", "CCR", None, None, False),
    "CCB": ("CC", "CCB", None, None, False),
    "CCA": ("CC", "CCA", None, None, False),
    "CCU": ("CC", "CCU", None, None, False),
    "CCP": ("CC", "CCP", None, None, True),   # ligature -- sets ligature=True

    # Lexical marker
    "LM": ("LM", None, None, None, False),

    # Verbs -- mood/class + tense/aspect share the "subtype" slot
    "VBW":  ("VB", "VBW", None, None, False),   # neutral/infinitive
    "VBS":  ("VB", "VBS", None, None, False),   # auxiliary/modal
    "VBH":  ("VB", "VBH", None, None, False),   # existential
    "VBN":  ("VB", "VBN", None, None, False),   # non-existential
    "VBTS": ("VB", "VBTS", None, None, False),  # perfective
    "VBTR": ("VB", "VBTR", None, None, False),  # imperfective
    "VBTF": ("VB", "VBTF", None, None, False),  # contemplative
    "VBTP": ("VB", "VBTP", None, None, False),  # recent past

    # Verb focus (orthogonal axis -- own head)
    "VBAF": ("VB", None, "AF", None, False),  # actor focus
    "VBOF": ("VB", None, "OF", None, False),  # object/goal focus
    "VBOB": ("VB", None, "OB", None, False),  # benefactive focus
    "VBOL": ("VB", None, "OL", None, False),  # locative focus
    "VBOI": ("VB", None, "OI", None, False),  # instrumental focus
    "VBRF": ("VB", None, "RF", None, False),  # referential/measurement focus

    # Adjectives -- JJD/JJN are base type. JJC/JJCC/JJCS/JJCN are the
    # degree (comparison) family and go in their own head, not subtype.
    "JJD": ("JJ", "JJD", None, None, False),   # descriptive (base type)
    "JJN": ("JJ", "JJN", None, None, False),   # number-adjective (base type)
    "JJCC":  ("JJ", None, None, "COMP", False),   # comparative
    "JJC": ("JJ", None, None, "COMPEQ", False), # comparative (equality)
    "JJCS": ("JJ", None, None, "SUP", False),    # superlative
    "JJCN": ("JJ", None, None, "COMPN", False),  # comparative (numeral)

    # Adverbs (RBI = enclitic-as-its-own-token, a normal subtype here)
    "RBD": ("RB", "RBD", None, None, False),
    "RBN": ("RB", "RBN", None, None, False),
    "RBK": ("RB", "RBK", None, None, False),
    "RBP": ("RB", "RBP", None, None, False),
    "RBB": ("RB", "RBB", None, None, False),
    "RBR": ("RB", "RBR", None, None, False),
    "RBQ": ("RB", "RBQ", None, None, False),
    "RBT": ("RB", "RBT", None, None, False),
    "RBF": ("RB", "RBF", None, None, False),
    "RBW": ("RB", "RBW", None, None, False),
    "RBM": ("RB", "RBM", None, None, False),
    "RBL": ("RB", "RBL", None, None, False),
    "RBI": ("RB", "RBI", None, None, False),
    "RBJ": ("RB", "RBJ", None, None, False),
    "RBS": ("RB", "RBS", None, None, False),

    # Cardinal numbers
    "CDB": ("CD", "CDB", None, None, False),

    # Standalone categories, no subtype
    "TS": ("TS", None, None, None, False),
    "FW": ("FW", None, None, None, False),

    # Punctuation
    "PMP":  ("PM", "PMP", None, None, False),
    "PME":  ("PM", "PME", None, None, False),
    "PMQ":  ("PM", "PMQ", None, None, False),
    "PMC":  ("PM", "PMC", None, None, False),
    "PMSC": ("PM", "PMSC", None, None, False),
    "PMS":  ("PM", "PMS", None, None, False),

    # Pipeline-internal (NOT MGNN -- emitted by alignment.py, not the tagger)
    "EMOJI":   ("PIPELINE", "EMOJI", None, None, False),
    "IGNORED": ("PIPELINE", "IGNORED", None, None, False),
}


@dataclass
class DecomposedTag:
    category: Optional[str] = None
    subtype: Optional[str] = None
    focus: Optional[str] = None
    degree: Optional[str] = None
    ligature: bool = False
    extra_subtypes: list = field(default_factory=list)  # same-axis stacking
    unknown_parts: list = field(default_factory=list)    # tags not in table


def decompose(tag: str) -> DecomposedTag:
    """Split a (possibly compound) MGNN tag into its five heads."""
    result = DecomposedTag()

    for part in tag.split("_"):
        if part not in ATOMIC_TAGS:
            result.unknown_parts.append(part)
            continue

        category, subtype, focus, degree, is_ligature = ATOMIC_TAGS[part]

        if is_ligature:
            result.ligature = True
            if result.category is None:
                result.category = category
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
        for tag, (_, _, foc, _, _) in ATOMIC_TAGS.items():
            if foc == d.focus:
                parts.append(tag)
                break
    if d.degree:
        for tag, (_, _, _, deg, _) in ATOMIC_TAGS.items():
            if deg == d.degree:
                parts.append(tag)
                break
    if d.ligature:
        parts.append("CCP")
    return "_".join(parts)


if __name__ == "__main__":
    # quick smoke test
    for tag in ["NNC", "JJD_CCP", "VBTR_VBAF", "VBTS_VBOF", "JJCS_JJD",
                "PRI_CCP", "RBI_CCP", "VBTR_VBRF", "JJN_CCP", "JJCC"]:
        d = decompose(tag)
        print(f"{tag:<15} -> category={d.category} subtype={d.subtype} "
              f"focus={d.focus} degree={d.degree} ligature={d.ligature} "
              f"extra={d.extra_subtypes} unknown={d.unknown_parts}")