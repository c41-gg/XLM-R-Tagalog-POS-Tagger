"""
Interactive test script for FSPOSTTagger (the Tagalog-only FSPOST wrapper,
without the English fallback HybridPOSTagger adds on top).

Usage:
    python test_tagalog_pos.py `
        --java-jar Library/FSPOST/stanford-postagger.jar `
        --tagalog-model Library/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger

Then type sentences at the prompt. Type 'quit' or 'exit' (or Ctrl+D) to stop.

Logs each internal stage of FSPOSTTagger.tag() as it happens (tokenize ->
validate -> build sentence string -> run Java subprocess -> parse Java
output -> align), instead of only showing the final tagged result or a
single alignment-failure exception. Calls the tagger's own pieces
(validate_tokens, parse) plus the align() function directly rather than
tagger.tag(tokens) as one call, and rather than adding step-by-step
console spam into tagalog_pos.py itself -- that module is shared with the
real pipeline (run_phase1.py / hybrid_pos.py).
"""

import argparse
import subprocess
import tempfile
import os

from modules.tokenizer import tokenize
from modules.tagalog_pos import FSPOSTTagger
from modules.alignment import align
from modules.json_writer import build_entry



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--java-jar", required=True)
    parser.add_argument("--tagalog-model", required=True)

    return parser.parse_args()


def print_tokens(tokens, tag_attr=None):

    if not tokens:
        print("  (no tokens)")
        return

    for t in tokens:
        if tag_attr is None:
            print(f"  [{t.index}] {t.token!r}")
        else:
            tag = getattr(t, tag_attr, None) or "None"
            print(f"  [{t.index}] {t.token!r:<20} -> {tag}")


def print_java_parsed(parsed):

    if not parsed:
        print("  (no parsed tokens)")
        return

    for i, jt in enumerate(parsed):
        print(f"  [{i}] {jt.token!r:<20} -> {jt.tag}")

def print_result(tokens):

    if not tokens:
        print("(no tokens)")
        return

    entry = build_entry(tokens)

    col_token = max(len("TOKEN"), max(len(t) for t in entry["tokens"]))
    col_tag = max(len("TAG"), max(len(tag or "None") for tag in entry["labels"]))

    header = (
        f"{'#':<3} "
        f"{'TOKEN':<{col_token}} "
        f"{'TAG':<{col_tag}} "
        f"{'CATEGORY':<10}"
        f"{'SUBTYPE':<10}"
        f"{'FOCUS':<10}"
        f"{'DEGREE':<10}"
        f"{'EXTRAS'}"
    )

    print(header)
    print("-" * len(header))

    unresolved_lookup = {}

    for item in entry.get("unresolved", []):
        unresolved_lookup[item["index"]] = item

    for i, token in enumerate(entry["tokens"]):

        extras = ", ".join(entry["extras"][i])

        if i in unresolved_lookup:
            u = unresolved_lookup[i]

            if "unknown_parts" in u:
                extras += f" [UNKNOWN: {u['unknown_parts']}]"

            if "extra_subtypes" in u:
                extras += f" [SAME-AXIS-STACK: {u['extra_subtypes']}]"

        print(
            f"{i:<3} "
            f"{token:<{col_token}} "
            f"{(entry['labels'][i] or 'None'):<{col_tag}} "
            f"{str(entry['categories'][i] or ''):<10}"
            f"{str(entry['subtypes'][i] or ''):<10}"
            f"{str(entry['focuses'][i] or ''):<10}"
            f"{str(entry['degrees'][i] or ''):<10}"
            f"{extras}"
        )

def tag_with_logging(tagger, sentence):
    """
    Mirrors FSPOSTTagger.tag() step by step, printing the state after
    each stage instead of only returning the final result (or raising
    on alignment failure with no visibility into what led up to it).
    """

    print("[STEP 1] Tokenizing sentence...")
    tokens = tokenize(sentence)
    print_tokens(tokens)
    print(f"  -> {len(tokens)} token(s)\n")

    if not tokens:
        print("[STEP 2] No tokens to tag.")
        return tokens

    print("[STEP 2] Validating tokens (checking for empty/whitespace-only)...")
    tagger.validate_tokens(tokens)  # prints its own warnings, if any
    print("  -> done\n")

    joined_sentence = " ".join(t.token for t in tokens)
    print("[STEP 3] Sentence string sent to Java (space-joined tokens):")
    print(f"  {joined_sentence!r}\n")

    print("[STEP 4] Running Java FSPOST tagger subprocess...")

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".txt", encoding="utf-8"
    ) as temp_file:
        temp_file.write(joined_sentence)
        temp_file_path = temp_file.name

    try:
        command = [
            "java", "-mx3g", "-cp", tagger.jar_path,
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model", tagger.model_path,
            "-textFile", temp_file_path,
        ]

        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")

        if process.returncode != 0:
            print(f"  -> Java process failed (exit code {process.returncode})")
            print(f"  stderr: {process.stderr}")
            raise RuntimeError(process.stderr)

        print("  -> Raw Java stdout:")
        print(f"  {process.stdout!r}\n")

        print("[STEP 5] Parsing Java output into (token, tag) pairs...")
        parsed = tagger.parse(process.stdout)
        print_java_parsed(parsed)
        print(f"  -> {len(parsed)} parsed token(s)\n")

    finally:
        os.unlink(temp_file_path)

    print("[STEP 6] Aligning original tokens against Java's parsed tokens...")
    success = align(tokens, parsed)

    if not success:
        failure_index = next(
            (t.index for t in tokens if t.mgnn_tag is None), None
        )
        failure_token = tokens[failure_index].token if failure_index is not None else "?"
        print(f"  -> ALIGNMENT FAILED at original index {failure_index} (token={failure_token!r})")
        print(f"     {len(tokens)} input tokens vs {len(parsed)} parsed tokens\n")
        raise RuntimeError(
            f"Alignment failed at original index {failure_index} "
            f"(token={failure_token!r}): {len(tokens)} input tokens vs "
            f"{len(parsed)} parsed tokens."
        )

    print("  -> Alignment succeeded\n")

    print("[STEP 7] Final tagged tokens:")
    return tokens


def main():

    args = parse_args()

    print("Loading FSPOSTTagger (this may take a moment)...")

    tagger = FSPOSTTagger(args.java_jar, args.tagalog_model)

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
            print_tokens(tokens, tag_attr="mgnn_tag")
            print_result(tokens)
            print()
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()