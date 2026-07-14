from modules.token_types import TaggedToken, JavaTaggedToken
import unicodedata
# ------------------------------------------------------------
# Java punctuation tags
# ------------------------------------------------------------

PUNCT_TAGS = {
    "PMS",
    "PMC",
    "PMP",
    "PMQ",
    "PME",
    "PMSC",
}

# ------------------------------------------------------------
# Secondary tags for contractions
# ------------------------------------------------------------

SECONDARY_TAGS = {
    "t": "CCA",   # at
    "y": "LM",    # ay
    "s": "PRSP",  # 's
}

# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

NORMALIZATION_MAP = {
     "-lrb-": "(",
    "-rrb-": ")",
    "-lsb-": "[",
    "-rsb-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    "’": "'",
    "‘": "'",
    "…": "...",
    "“": '"',
    "”": '"',
    "»": '"',
    "«": '"',
    "€": "$",
    "£": "#"}

def normalize(text: str) -> str:
    text = text.lower()
    
    for old, new in NORMALIZATION_MAP.items():
        text = text.replace(old, new)
    
    if len(text) == 1:
        if unicodedata.category(text) == "No" and not unicodedata.numeric(text.encode("utf-8")).is_integer():
            text = unicodedata.normalize("NFKC", text).replace("\u2044", "/")

        if unicodedata.category(text) == "Nl" and unicodedata.name(text.encode("utf-8"), "").startswith("ROMAN NUMERAL"):
            text = unicodedata.normalize('NFKC', text)

    return text
# ------------------------------------------------------------
# Rule 1
# WORD + ' + t
# WORD + ' + y
# ------------------------------------------------------------

def contraction_rule(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int
):

    if j + 2 >= len(java_tokens):
        return None

    first = java_tokens[j]
    second = java_tokens[j + 1]
    third = java_tokens[j + 2]
    print(first.token, first.tag)
    print(second.token, second.tag)
    print(third.token, third.tag)

    if normalize(second.token) != "'":
        return None

    suffix = normalize(third.token)

    if suffix not in SECONDARY_TAGS:
        return None

    candidate = (
        normalize(first.token)
        + "'"
        + suffix
    )

    if candidate != normalize(original.token):
        return None

    suffix_tag = SECONDARY_TAGS[suffix]

    tag = first.tag
    tag = f"{tag}_{suffix_tag}"

    return tag, j + 3

# ------------------------------------------------------------
# Rule 2
# n + ' + ya
# s + ' + ya
# t + ' + yak
# etc.
# ------------------------------------------------------------
def dropped_vowel_rule(original: TaggedToken, java_tokens: list[JavaTaggedToken], j: int):

    if j + 1 >= len(java_tokens):
        return None

    first = java_tokens[j]
    second = java_tokens[j + 1]
    second_norm = normalize(second.token)

    # 3-way: word, standalone apostrophe, separate suffix token
    if second_norm == "'" and j + 2 < len(java_tokens):

        third = java_tokens[j + 2]        
        candidate = ( normalize(first.token)+ "'" + normalize(third.token))

        if candidate == normalize(original.token):
            return first.tag, j + 3

    # 2-way: word, then apostrophe+suffix fused into one token
    if second_norm.startswith("'"):

        candidate = normalize(first.token) + second_norm

        if candidate == normalize(original.token):
            return first.tag, j + 2

    return None

# ------------------------------------------------------------
# Rule 3
# Generic merge fallback
# ------------------------------------------------------------

def leading_apostrophe_rule(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int,
):

    if j + 1 >= len(java_tokens):
        return None

    if normalize(java_tokens[j].token) != "'":
        return None

    candidate = "'" + normalize(java_tokens[j + 1].token)

    if candidate != normalize(original.token):
        return None

    return java_tokens[j + 1].tag, j + 2

def generic_merge(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int
):
    target = normalize(original.token).replace(" ", "")

    merged = ""
    primary_tag = None

    start = j
    MAX_MERGE = 5

    while j < len(java_tokens) and j - start < MAX_MERGE:

        jt = java_tokens[j]

        piece = normalize(jt.token)
        merged += piece.replace(" ", "")

        if primary_tag is None and jt.tag not in PUNCT_TAGS:
            primary_tag = jt.tag

        if merged == target:
            return primary_tag, j + 1

        if len(merged) > len(target):
            break

        j += 1

    return None

# ------------------------------------------------------------
# Ingnorables
# ------------------------------------------------------------

def is_emoji(token: str) -> bool:
    return bool(token) and all(ord(ch) > 0xFFFF for ch in token)

def ignorable(token: str):
    if token in {
        "\u200b",
        "\u200c",
        "\u200d",
        "\u200e",
        "\u200f",
        "\ufeff",
        "\ufffd"
    }:
        return True
    
    if len(token) == 1:

        cp = ord(token)

        if 0x80 <= cp <= 0x9F:
            return True
        
        if 0x2160 <= cp <= 0x2188:
            return True
        
        if 0xFE00 <= cp <= 0xFE0F:
            return True
        
        return False

    return False

# ------------------------------------------------------------
# Main aligner
# ------------------------------------------------------------

def apply_match(original: TaggedToken, result):
    tag, next_j = result
    original.mgnn_tag = tag
    return next_j

def alignment_error(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int,
):
    print("=" * 60)
    print("ALIGNMENT ERROR")
    print("=" * 60)

    print("Original token:")
    print(original.token)
    print()

    print("Remaining Java tokens:")

    for token in java_tokens[j:j + 10]:
        print(token)

def align(
    original_tokens: list[TaggedToken],
    java_tokens: list[JavaTaggedToken]
) -> bool:

    i = 0
    j = 0

    while i < len(original_tokens):
        

        if j >= len(java_tokens):
            return False

        original = original_tokens[i]

        # ----------------------------------------------------
        # Ignored
        # ----------------------------------------------------

        if is_emoji(original.token):
            original.mgnn_tag = "EMOJI"
            i += 1
            continue  

        if ignorable(original.token):
            original.mgnn_tag = "IGNORED"
            i += 1
            continue     

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        if normalize(original.token) == normalize(java_tokens[j].token):
            original.mgnn_tag = java_tokens[j].tag
            i += 1
            j += 1
            continue

        # ----------------------------------------------------
        # Rule 1
        # WORD + ' + t / y
        # ----------------------------------------------------

        result = contraction_rule(original,java_tokens,j)

        if result:
            j = apply_match(original, result)
            i += 1
            continue

        # ----------------------------------------------------
        # Rule 2
        # n'ya
        # s'ya
        # t'yak
        # ----------------------------------------------------

        result = dropped_vowel_rule(original,java_tokens,j)

        if result:
            j = apply_match(original, result)
            i += 1
            continue

        # ----------------------------------------------------
        # Rule 3
        # Generic merge
        # ----------------------------------------------------

        result = leading_apostrophe_rule(original,java_tokens,j)

        if result:
            j = apply_match(original, result)
            i += 1
            continue

        result = generic_merge(original, java_tokens,j)

        if result:
            j = apply_match(original, result)
            i += 1
            continue

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        alignment_error(original, java_tokens, j)

        return False

    return True