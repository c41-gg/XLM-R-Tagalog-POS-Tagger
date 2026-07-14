"""
Interactive test script for HybridPOSTagger.

Usage:
    python test_hybrid_pos.py `
        --java-jar Library/FSPOST/stanford-postagger.jar `
        --tagalog-model Library/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger

Then type sentences at the prompt. Type 'quit' or 'exit' (or Ctrl+D) to stop.

Logs each stage of HybridPOSTagger.tag() as it happens (tokenize -> FSPOST
tag -> FW detection -> English fallback -> merge), instead of only showing
the final result. This calls the same underlying pieces HybridPOSTagger
uses (tokenize, tl.tag, en.tag, english_to_mgnn) directly rather than
tagger.tag(sentence) as one opaque call, and rather than editing
hybrid_pos.py itself -- that module is shared with the real pipeline
(run_phase1.py), and it shouldn't get step-by-step console spam added to
it just to support this debug script.
"""

import argparse

from modules.hybrid_pos import HybridPOSTagger
from modules.tokenizer import tokenize
from modules.tag_mapping import english_to_mgnn
from modules.mgnn_schema import decompose


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--java-jar", required=True)
    parser.add_argument("--tagalog-model", required=True)

    return parser.parse_args()


def print_tokens(tokens, tag_attr="mgnn_tag"):

    if not tokens:
        print("  (no tokens)")
        return

    for t in tokens:
        tag = getattr(t, tag_attr, None) or "None"
        print(f"  [{t.index}] {t.token!r:<20} -> {tag}")


def print_result(tokens):

    if not tokens:
        print("(no tokens)")
        return

    col_token = max(len("TOKEN"), max(len(t.token) for t in tokens))
    col_tag = max(len("TAG"), max(len(t.mgnn_tag or "None") for t in tokens))

    header = (
        f"{'#':<3} {'TOKEN':<{col_token}} {'TAG':<{col_tag}} "
        f"{'CATEGORY':<10}{'SUBTYPE':<10}{'FOCUS':<7}{'DEGREE':<8}{'EXTRAS'}"
    )
    print(header)
    print("-" * len(header))

    for t in tokens:

        tag = t.mgnn_tag or "None"

        if t.mgnn_tag:
            d = decompose(t.mgnn_tag)
            category = d.category or ""
            subtype = d.subtype or ""
            focus = d.focus or ""
            degree = d.degree or ""
            extras = ",".join(d.extras) if d.extras else ""
            if d.unknown_parts:
                extras += f" [UNKNOWN:{d.unknown_parts}]"
            if d.extra_subtypes:
                extras += f" [SAME-AXIS-STACK:{d.extra_subtypes}]"
        else:
            category = subtype = focus = degree = extras = ""

        print(
            f"{t.index:<3} {t.token:<{col_token}} {tag:<{col_tag}} "
            f"{category:<10}{subtype:<10}{focus:<7}{degree:<8}{extras}"
        )


def tag_with_logging(tagger, sentence):
    """
    Mirrors HybridPOSTagger.tag() step by step, printing the state after
    each stage instead of only returning the final result.
    """

    print("[STEP 1] Tokenizing sentence...")
    tokens = tokenize(sentence)
    print_tokens(tokens, tag_attr="token")  # nothing tagged yet, just tokens
    print(f"  -> {len(tokens)} token(s)\n")

    print("[STEP 2] Running FSPOST (Tagalog tagger)...")
    tagger.tl.tag(tokens)  # mutates tokens in place, sets mgnn_tag
    print_tokens(tokens)
    print()

    print("[STEP 3] Scanning for FW (foreign word) tokens needing English fallback...")
    fw_indices = [
        i for i, t in enumerate(tokens)
        if t.mgnn_tag == "FW"
    ]

    if not fw_indices:
        print("  -> No FW tokens found. Skipping English fallback.\n")
        print("[STEP 4] Final result (unchanged from FSPOST output):")
        return tokens

    print(f"  -> Found {len(fw_indices)} FW token(s) at index(es): {fw_indices}\n")

    print("[STEP 4] Running English POS tagger on FW tokens...")
    fw_words = [tokens[i] for i in fw_indices]
    english_tags = tagger.en.tag(fw_words)
    print_tokens(english_tags, tag_attr="penn_tag")
    print()

    print("[STEP 5] Remapping English Penn tags -> MGNN tags...")
    for idx, en_token in zip(fw_indices, english_tags):
        old_tag = tokens[idx].mgnn_tag
        new_tag = english_to_mgnn(en_token.token, en_token.penn_tag)
        tokens[idx].mgnn_tag = new_tag
        print(f"  [{idx}] {en_token.token!r:<20} {old_tag} (penn={en_token.penn_tag}) -> {new_tag}")
    print()

    print("[STEP 6] Final merged result:")
    return tokens


def main():

    args = parse_args()

    print("Loading HybridPOSTagger (this may take a moment)...")

    tagger = HybridPOSTagger(args.java_jar, args.tagalog_model)

    print("Ready. Type a sentence to tag it, or 'quit'/'exit' to stop.\n")

    while True:

        try:
            sentence = input(">> ").strip()
        except EOFError:
            print()
            break

        if not sentence:
            continue

        if sentence.lower() in {"quit", "exit"}:
            break

        print()

        try:
            tokens = tag_with_logging(tagger, sentence)
            print_result(tokens)
            print()
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()