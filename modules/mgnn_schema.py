"""
MGNN tagset (Nocon & Borra, 2016) decomposed into four orthogonal heads
for multi-head XLM-R fine-tuning:

    category  -- coarse POS (NN, PR, DT, CC, LM, VB, JJ, RB, CD, TS, FW, PM)
    subtype   -- fine-grained tag within category (includes verb tense/mood)
    focus     -- verb-only trigger/focus marker, or NONE
    ligature  -- whether the ligature/linker (CCP) is attached, True/False

Source: https://www.sketchengine.eu/mgnn-tagalog-part-of-speech-tagset/
        (Nocon, N. and Borra, A., SMTPOST, 2016)

IMPORTANT: RBI ("Enclitics") is NOT a cross-cutting suffix -- it is a normal
adverb *subtype*, used when the enclitic particle (na, pa, rin, daw...) is
its own token. The actual cross-cutting attachable morpheme is CCP (the
ligature -ng/na/-g), which is why it gets its own head here instead of RBI.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------
# Atomic tag -> (category, subtype-or-None, is_focus, is_ligature)
# ---------------------------------------------------------------
# Every atomic (non-compound) tag from the official 69-tag base set,
# plus the two pipeline-internal tags your aligner emits (EMOJI/IGNORED)
# which are NOT part of MGNN proper.

ATOMIC_TAGS = {
    # Nouns
    "NNC":  ("NN", "NNC", False, False),
    "NNP":  ("NN", "NNP", False, False),
    "NNPA": ("NN", "NNPA", False, False),
    "NNCA": ("NN", "NNCA", False, False),

    # Pronouns
    "PRS":  ("PR", "PRS", False, False),
    "PRP":  ("PR", "PRP", False, False),
    "PRSP": ("PR", "PRSP", False, False),
    "PRO":  ("PR", "PRO", False, False),
    "PRQ":  ("PR", "PRQ", False, False),
    "PRQP": ("PR", "PRQP", False, False),
    "PRL":  ("PR", "PRL", False, False),
    "PRC":  ("PR", "PRC", False, False),
    "PRF":  ("PR", "PRF", False, False),
    "PRI":  ("PR", "PRI", False, False),

    # Determiners
    "DTC":  ("DT", "DTC", False, False),
    "DTCP": ("DT", "DTCP", False, False),
    "DTP":  ("DT", "DTP", False, False),
    "DTPP": ("DT", "DTPP", False, False),

    # Conjunctions (CCP is the ligature -- handled specially below)
    "CCT": ("CC", "CCT", False, False),
    "CCR": ("CC", "CCR", False, False),
    "CCB": ("CC", "CCB", False, False),
    "CCA": ("CC", "CCA", False, False),
    "CCU": ("CC", "CCU", False, False),
    "CCP": ("CC", "CCP", False, True),   # ligature -- also sets ligature=True

    # Lexical marker
    "LM": ("LM", None, False, False),

    # Verbs -- mood/class + tense/aspect share the "subtype" slot
    "VBW":  ("VB", "VBW", False, False),   # neutral/infinitive
    "VBS":  ("VB", "VBS", False, False),   # auxiliary/modal
    "VBH":  ("VB", "VBH", False, False),   # existential
    "VBN":  ("VB", "VBN", False, False),   # non-existential
    "VBTS": ("VB", "VBTS", False, False),  # perfective
    "VBTR": ("VB", "VBTR", False, False),  # imperfective
    "VBTF": ("VB", "VBTF", False, False),  # contemplative
    "VBTP": ("VB", "VBTP", False, False),  # recent past

    # Verb focus (orthogonal axis -- own head)
    "VBAF": ("VB", None, "AF", False),  # actor focus
    "VBOF": ("VB", None, "OF", False),  # object/goal focus
    "VBOB": ("VB", None, "OB", False),  # benefactive focus
    "VBOL": ("VB", None, "OL", False),  # locative focus
    "VBOI": ("VB", None, "OI", False),  # instrumental focus
    "VBRF": ("VB", None, "RF", False),  # referential/measurement focus

    # Adjectives
    "JJD":  ("JJ", "JJD", False, False),
    "JJC":  ("JJ", "JJC", False, False),
    "JJCC": ("JJ", "JJCC", False, False),
    "JJCS": ("JJ", "JJCS", False, False),
    "JJCN": ("JJ", "JJCN", False, False),
    "JJN":  ("JJ", "JJN", False, False),

    # Adverbs (RBI = enclitic-as-its-own-token, a normal subtype here)
    "RBD": ("RB", "RBD", False, False),
    "RBN": ("RB", "RBN", False, False),
    "RBK": ("RB", "RBK", False, False),
    "RBP": ("RB", "RBP", False, False),
    "RBB": ("RB", "RBB", False, False),
    "RBR": ("RB", "RBR", False, False),
    "RBQ": ("RB", "RBQ", False, False),
    "RBT": ("RB", "RBT", False, False),
    "RBF": ("RB", "RBF", False, False),
    "RBW": ("RB", "RBW", False, False),
    "RBM": ("RB", "RBM", False, False),
    "RBL": ("RB", "RBL", False, False),
    "RBI": ("RB", "RBI", False, False),
    "RBJ": ("RB", "RBJ", False, False),
    "RBS": ("RB", "RBS", False, False),

    # Cardinal numbers
    "CDB": ("CD", "CDB", False, False),

    # Standalone categories, no subtype
    "TS": ("TS", None, False, False),
    "FW": ("FW", None, False, False),

    # Punctuation
    "PMP":  ("PM", "PMP", False, False),
    "PME":  ("PM", "PME", False, False),
    "PMQ":  ("PM", "PMQ", False, False),
    "PMC":  ("PM", "PMC", False, False),
    "PMSC": ("PM", "PMSC", False, False),
    "PMS":  ("PM", "PMS", False, False),

    # Pipeline-internal (NOT MGNN -- emitted by alignment.py, not the tagger)
    "EMOJI":   ("PIPELINE", "EMOJI", False, False),
    "IGNORED": ("PIPELINE", "IGNORED", False, False),
}

FOCUS_TAGS = {t for t, v in ATOMIC_TAGS.items() if v[2] is not False}


@dataclass
class DecomposedTag:
    category: Optional[str] = None
    subtype: Optional[str] = None
    focus: Optional[str] = None
    ligature: bool = False
    extra_subtypes: list = field(default_factory=list)  # same-axis stacking
    unknown_parts: list = field(default_factory=list)    # tags not in table


def decompose(tag: str) -> DecomposedTag:
    """Split a (possibly compound) MGNN tag into its four heads."""
    result = DecomposedTag()

    for part in tag.split("_"):
        if part not in ATOMIC_TAGS:
            result.unknown_parts.append(part)
            continue

        category, subtype, focus, is_ligature = ATOMIC_TAGS[part]

        if is_ligature:
            result.ligature = True
            # CCP also has category=CC/subtype=CCP but we don't let it
            # overwrite a real category -- it's cross-cutting by design.
            if result.category is None:
                result.category = category
            continue

        if focus:
            result.focus = focus
            if result.category is None:
                result.category = category
            continue

        # normal category+subtype tag
        if result.category is None:
            result.category = category
            result.subtype = subtype
        elif result.category == category and result.subtype is not None:
            # same-axis stacking, e.g. JJCS_JJD (two JJ subtypes at once)
            result.extra_subtypes.append(subtype)
        elif result.subtype is None:
            result.subtype = subtype
        else:
            # different category showing up mid-tag -- shouldn't happen
            # in well-formed MGNN tags, but record it rather than silently
            # dropping it
            result.extra_subtypes.append(f"{category}:{subtype}")

    return result


def recompose(d: DecomposedTag) -> str:
    """Best-effort rejoin, for round-trip validation against source data."""
    parts = []
    if d.subtype:
        parts.append(d.subtype)
    elif d.category and not d.focus:
        parts.append(d.category)
    parts.extend(d.extra_subtypes)
    if d.focus:
        # find the atomic tag name for this focus value
        for tag, (_, _, foc, _) in ATOMIC_TAGS.items():
            if foc == d.focus:
                parts.append(tag)
                break
    if d.ligature:
        parts.append("CCP")
    return "_".join(parts)


if __name__ == "__main__":
    # quick smoke test
    for tag in ["NNC", "JJD_CCP", "VBTR_VBAF", "VBTS_VBOF", "JJCS_JJD",
                "PRI_CCP", "RBI_CCP", "VBTR_VBRF"]:
        d = decompose(tag)
        print(f"{tag:<15} -> category={d.category} subtype={d.subtype} "
              f"focus={d.focus} ligature={d.ligature} "
              f"extra={d.extra_subtypes} unknown={d.unknown_parts}")