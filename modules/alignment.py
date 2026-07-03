import regex as re
from modules.token_types import TaggedToken, JavaTaggedToken

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

def normalize(text: str) -> str:

    return (
        text.lower()
            .replace("-lrb-", "(")
            .replace("-rrb-", ")")
            .replace("-lsb-", "[")
            .replace("-rsb-", "]")
            .replace("``", "\"")
            .replace("''", "\"")
            .replace("`", "'")
            .replace("’", "'")
            .replace("‘", "'")
            .replace("…", "...")
            .replace("“", '"')
            .replace("”", '"')
            
            
    )



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

    if suffix_tag and "_" not in tag:
        tag = f"{tag}_{suffix_tag}"

    return tag, j + 3


# ------------------------------------------------------------
# Rule 2
# n + ' + ya
# s + ' + ya
# t + ' + yak
# etc.
# ------------------------------------------------------------

def dropped_vowel_rule(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int
):

    if j + 2 >= len(java_tokens):
        return None

    first = java_tokens[j]
    second = java_tokens[j + 1]
    third = java_tokens[j + 2]

    if normalize(second.token) != "'":
        return None

    candidate = (
        normalize(first.token)
        + "'"
        + normalize(third.token)
    )

    if candidate != normalize(original.token):
        return None

    return first.tag, j + 3


# ------------------------------------------------------------
# Rule 3
# Generic merge fallback
# ------------------------------------------------------------

def leading_apostrophe_rule(original, java_tokens, j):

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

        if piece in {"'", '"'}:
            merged += piece
        elif jt.tag in PUNCT_TAGS:
            merged += piece
        else:
            merged += piece

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
    return bool(re.fullmatch(r"\p{Extended_Pictographic}+", token))

def ignorable(token: str):
    if token in {
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    }:
        return True

    return False

# ------------------------------------------------------------
# Main aligner
# ------------------------------------------------------------

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

        result = contraction_rule(
            original,
            java_tokens,
            j
        )

        if result:

            tag, j = result

            original.mgnn_tag = tag

            i += 1
            continue

        # ----------------------------------------------------
        # Rule 2
        # n'ya
        # s'ya
        # t'yak
        # ----------------------------------------------------

        result = dropped_vowel_rule(
            original,
            java_tokens,
            j
        )

        if result:

            tag, j = result

            original.mgnn_tag = tag

            i += 1
            continue

        # ----------------------------------------------------
        # Rule 3
        # Generic merge
        # ----------------------------------------------------

        result = leading_apostrophe_rule(
            original,
            java_tokens,
            j
        )

        if result:
            tag, j = result
            original.mgnn_tag = tag
            i += 1
            continue

        result = generic_merge(
            original,
            java_tokens,
            j
        )

        if result:

            tag, j = result

            original.mgnn_tag = tag

            i += 1
            continue


        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        print("=" * 60)
        print("ALIGNMENT ERROR")
        print("=" * 60)

        print("Original token:")
        print(original.token)

        print()

        print("Remaining Java tokens:")

        for token in java_tokens[j:j + 10]:
            print(token)

        return False

    return True