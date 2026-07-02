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
}


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

def normalize(text: str) -> str:

    return (
        text.lower()
            .replace("-lrb-", "(")
            .replace("-rrb-", ")")
            .replace("``", "\"")
            .replace("''", "\"")
            .replace("`", "'")
            .replace("’", "'")
            .replace("‘", "'")
            .replace("…", "...")
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

    tag = first.tag

    if "_" not in tag:
        tag = f"{tag}_{SECONDARY_TAGS[suffix]}"

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

def generic_merge(
    original: TaggedToken,
    java_tokens: list[JavaTaggedToken],
    j: int
):

    target = normalize(original.token)

    merged = ""
    primary_tag = None

    start = j

    while j < len(java_tokens):

        jt = java_tokens[j]

        merged += normalize(jt.token)

        if primary_tag is None and jt.tag not in PUNCT_TAGS:
            primary_tag = jt.tag

        if merged == target:
            return primary_tag, j + 1

        if len(merged) > len(target):
            break

        j += 1

    return None


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