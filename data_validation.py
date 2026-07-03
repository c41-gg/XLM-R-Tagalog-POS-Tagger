"""
Validation script for Phase 1 silver POS data.

Checks:
  1. MGNN tag distribution / class imbalance
  2. FW-leakage: how many tokens ended up tagged FW that are
     actually recognizable English words (i.e. the hybrid
     English fallback should have caught them but didn't)

Usage:
    python validate_phase1.py --input data/processed/phase1.jsonl

Optional:
    --whitelist path/to/tags.txt   # one MGNN tag per line, flags
                                    # any tag in the data NOT in this list
    --top-n 30                     # how many example tokens to print per bucket
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import enchant
    _EN_DICT = enchant.Dict("en_US")
    _HAS_ENCHANT = True
except Exception:
    _EN_DICT = None
    _HAS_ENCHANT = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--whitelist", default=None)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--rare-threshold", type=float, default=0.1,
                   help="Flag tags below this %% of total tokens as rare classes")
    return p.parse_args()


def load_jsonl(path):
    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Handles legacy runs that wrote Python repr instead of JSON
                try:
                    records.append(eval(line, {"__builtins__": {}}, {}))
                    skipped += 0
                except Exception:
                    print(f"  [!] Could not parse line {line_no}, skipping")
                    skipped += 1
    if skipped:
        print(f"  [!] Skipped {skipped} unparsable lines")
    return records


def load_whitelist(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# ---------------------------------------------------------------
# 1. Tag distribution
# ---------------------------------------------------------------

def tag_distribution(records, whitelist, rare_threshold):
    counter = Counter()
    total_sentences = len(records)
    total_tokens = 0

    for r in records:
        for label in r.get("labels", []):
            counter[label] += 1
            total_tokens += 1

    print("\n" + "=" * 60)
    print("TAG DISTRIBUTION")
    print("=" * 60)
    print(f"Sentences: {total_sentences}   Tokens: {total_tokens}")
    print(f"Unique tags: {len(counter)}\n")

    print(f"{'TAG':<12}{'COUNT':>10}{'%':>8}")
    print("-" * 32)

    rare = []
    unknown = []

    for tag, count in counter.most_common():
        pct = 100 * count / total_tokens if total_tokens else 0
        flag = ""
        if pct < rare_threshold:
            flag = "  <- rare"
            rare.append((tag, count, pct))
        if whitelist is not None and tag not in whitelist:
            flag += "  <- NOT IN WHITELIST"
            unknown.append((tag, count, pct))
        print(f"{tag:<12}{count:>10}{pct:>7.2f}%{flag}")

    if whitelist is not None:
        missing = whitelist - set(counter.keys())
        if missing:
            print(f"\n[!] {len(missing)} whitelisted tags never appeared in the data:")
            print("    " + ", ".join(sorted(missing)))

    if rare:
        print(f"\n[!] {len(rare)} tag(s) below {rare_threshold}% of all tokens "
              f"-- consider whether you have enough examples to train on these:")
        for tag, count, pct in rare:
            print(f"    {tag:<10} count={count:<6} ({pct:.3f}%)")

    return counter


# ---------------------------------------------------------------
# 2. FW-leakage
# ---------------------------------------------------------------

def fw_leakage(records, top_n):
    print("\n" + "=" * 60)
    print("FW-LEAKAGE ANALYSIS")
    print("=" * 60)

    if not _HAS_ENCHANT:
        print("[!] pyenchant not installed / en_US dict unavailable.")
        print("    Install with: pip install pyenchant  (needs libenchant on the system)")
        print("    Skipping FW-leakage check.\n")
        return

    fw_tokens = Counter()
    total_tokens = 0
    total_fw = 0

    for r in records:
        tokens = r.get("tokens", [])
        labels = r.get("labels", [])
        for tok, lab in zip(tokens, labels):
            total_tokens += 1
            if lab == "FW":
                total_fw += 1
                fw_tokens[tok.lower()] += 1

    if total_fw == 0:
        print("No tokens tagged FW in this dataset. Nothing to check.")
        return

    print(f"Total tokens: {total_tokens}")
    print(f"Tokens tagged FW: {total_fw} ({100*total_fw/total_tokens:.2f}%)")
    print(f"Unique FW surface forms: {len(fw_tokens)}\n")

    recognized = []   # english words that leaked through as FW
    unrecognized = []  # genuinely foreign / garbage / misspelled

    for word, count in fw_tokens.items():
        clean = word.strip(".,!?\"'()[]{}:;")
        if not clean or not clean.isalpha():
            unrecognized.append((word, count))
            continue
        try:
            is_en = _EN_DICT.check(clean) or _EN_DICT.check(clean.capitalize())
        except Exception:
            is_en = False
        if is_en:
            recognized.append((word, count))
        else:
            unrecognized.append((word, count))

    leaked_token_count = sum(c for _, c in recognized)
    print(f"FW tokens recognized as valid English words (LEAKAGE): "
          f"{leaked_token_count} tokens / {len(recognized)} unique "
          f"({100*leaked_token_count/total_fw:.2f}% of all FW-tagged tokens)")
    print(f"FW tokens NOT recognized as English (likely genuine foreign/garbage): "
          f"{total_fw - leaked_token_count} tokens / {len(unrecognized)} unique\n")

    recognized.sort(key=lambda x: -x[1])
    unrecognized.sort(key=lambda x: -x[1])

    print(f"Top {top_n} LEAKED tokens (English words still tagged FW -- "
          f"these are candidates for fixing the fallback / PENN_TO_MGNN gaps):")
    for word, count in recognized[:top_n]:
        print(f"    {word:<20} x{count}")

    print(f"\nTop {top_n} genuinely-foreign / unrecognized FW tokens "
          f"(expected -- non-English loanwords, misspellings, CC-100 noise):")
    for word, count in unrecognized[:top_n]:
        print(f"    {word:<20} x{count}")

    if leaked_token_count / total_fw > 0.15:
        print(f"\n[!] Over 15% of FW-tagged tokens are recognizable English words.")
        print("    Likely cause: PENN_TO_MGNN in tag_mapping.py doesn't cover every")
        print("    Penn tag NLTK's PerceptronTagger can emit (e.g. UH, WP, WRB, RP,")
        print("    DT, CC, SYM fall through to the 'FW' default). Check which Penn")
        print("    tags the leaked words are getting -- add explicit mappings for them.")


def main():
    args = parse_args()
    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    whitelist = load_whitelist(args.whitelist)

    tag_distribution(records, whitelist, args.rare_threshold)
    fw_leakage(records, args.top_n)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()