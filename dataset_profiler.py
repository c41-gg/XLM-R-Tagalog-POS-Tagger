import json
import argparse
from collections import Counter
from statistics import mean, median
from pathlib import Path
import csv
import ast

from modules.mgnn_schema import decompose


class DatasetProfiler:

    def __init__(self, jsonl_path):

        self.path = Path(jsonl_path)

        self.sentences = 0
        self.tokens = 0

        self.label_counter = Counter()
        self.token_counter = Counter()

        self.lengths = []

        self.category_counter = Counter()
        self.subtype_counter = Counter()
        self.focus_counter = Counter()
        self.ligature_count = 0

        self.unresolved_counter = Counter()  # tag -> count of tokens hitting it
        self.unresolved_reasons = Counter()  # "unknown_part" / "same_axis_stack"

        self.decomposition_mismatches = 0  # stored fields vs. freshly computed

        self.malformed = Counter()

    # -----------------------------------------------------------
    # Per-token decomposition: use stored fields if present,
    # otherwise fall back to computing them on the fly so this
    # still works on older phase1.jsonl files written before the
    # json_writer update.
    # -----------------------------------------------------------

    def _decomposed_fields(self, item, i, label):

        has_stored = all(
            key in item for key in
            ("categories", "subtypes", "focuses", "ligatures")
        )

        computed = decompose(label)
        computed_category = computed.category
        computed_subtype = computed.subtype
        computed_focus = computed.focus
        computed_ligature = computed.ligature

        if computed.unknown_parts:
            self.unresolved_reasons["unknown_part"] += 1
            self.unresolved_counter[label] += 1
        elif computed.extra_subtypes:
            self.unresolved_reasons["same_axis_stack"] += 1
            self.unresolved_counter[label] += 1

        if has_stored:
            category = item["categories"][i]
            subtype = item["subtypes"][i]
            focus = item["focuses"][i]
            ligature = item["ligatures"][i]

            if (category, subtype, focus, ligature) != (
                computed_category, computed_subtype,
                computed_focus, computed_ligature
            ):
                self.decomposition_mismatches += 1

            return category, subtype, focus, ligature

        return computed_category, computed_subtype, computed_focus, computed_ligature

    def profile(self):

        with open(self.path, encoding="utf-8") as f:

            for line in f:

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = ast.literal_eval(line)

                tokens = item["tokens"]
                labels = item["labels"]

                self.sentences += 1
                self.tokens += len(tokens)

                self.lengths.append(len(tokens))

                for i, (token, label) in enumerate(zip(tokens, labels)):

                    if label is None:
                        print(f"Found None label on token: {token}")
                        continue

                    self.token_counter[token] += 1
                    self.label_counter[label] += 1

                    if " " in label:
                        self.malformed[label] += 1
                        continue

                    category, subtype, focus, ligature = self._decomposed_fields(
                        item, i, label
                    )

                    if category:
                        self.category_counter[category] += 1
                    if subtype:
                        self.subtype_counter[subtype] += 1
                    if focus:
                        self.focus_counter[focus] += 1
                    if ligature:
                        self.ligature_count += 1

    def print_report(self):

        print("=" * 60)
        print("DATASET PROFILE")
        print("=" * 60)

        print(f"Sentences              : {self.sentences:,}")
        print(f"Tokens                 : {self.tokens:,}")
        print(f"Vocabulary             : {len(self.token_counter):,}")

        print()

        print(f"Average Length         : {mean(self.lengths):.2f}")
        print(f"Median Length          : {median(self.lengths):.0f}")
        print(f"Shortest Sentence      : {min(self.lengths)}")
        print(f"Longest Sentence       : {max(self.lengths)}")

        print()

        print(f"Unique Labels          : {len(self.label_counter)}")
        print(f"Unique Categories      : {len(self.category_counter)}")
        print(f"Unique Subtypes        : {len(self.subtype_counter)}")
        print(f"Unique Focus Values    : {len(self.focus_counter)}")

        print()

        print("Category Distribution")
        print("-------------------")
        for cat, count in self.category_counter.most_common():
            pct = 100 * count / self.tokens if self.tokens else 0
            print(f"{cat:12} {count:8,} ({pct:5.2f}%)")

        print()

        print("Focus Distribution (verb-only)")
        print("-------------------")
        if self.focus_counter:
            focus_total = sum(self.focus_counter.values())
            verb_total = self.category_counter.get("VB", 0)
            for foc, count in self.focus_counter.most_common():
                pct = 100 * count / focus_total
                print(f"{foc:12} {count:8,} ({pct:5.2f}% of focus-marked tokens)")
            if verb_total:
                coverage = 100 * focus_total / verb_total
                print(f"\nFocus coverage: {focus_total:,}/{verb_total:,} "
                      f"VB tokens carry an explicit focus tag ({coverage:.2f}%)")
        else:
            print("(none found)")

        print()

        ligature_pct = 100 * self.ligature_count / self.tokens if self.tokens else 0
        print(f"Ligature-attached Tokens : {self.ligature_count:,} ({ligature_pct:.2f}%)")

        print()

        if self.unresolved_counter:
            total_unresolved = sum(self.unresolved_counter.values())
            print(f"Unresolved Decompositions : {total_unresolved:,} tokens "
                  f"across {len(self.unresolved_counter)} distinct tag(s)")
            for reason, count in self.unresolved_reasons.most_common():
                print(f"  {reason:<18}: {count:,}")
            print("  Top offending tags:")
            for tag, count in self.unresolved_counter.most_common(10):
                print(f"    {tag:<15} x{count}")
        else:
            print("Unresolved Decompositions : None -- every tag decomposed cleanly")

        if self.decomposition_mismatches:
            print(f"\n[!] {self.decomposition_mismatches} token(s) where stored "
                  f"category/subtype/focus/ligature fields disagree with a fresh "
                  f"decompose() call -- check for schema drift between "
                  f"json_writer.py and mgnn_schema.py")

        print()

        print("Top 30 Labels")

        print("-------------------")

        for label, count in self.label_counter.most_common(30):

            pct = count / self.tokens * 100

            print(f"{label:20} {count:8} ({pct:5.2f}%)")

        print()

        rare = [
            (l, c)
            for l, c in self.label_counter.items()
            if c < 10
        ]

        print(f"Rare Labels (<10)      : {len(rare)}")

        if self.malformed:
            print(f"Malformed Labels       : {len(self.malformed)}")
        else:
            print("Malformed Labels       : None")

    def export_csv(self):

        with open("label_distribution.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Label", "Count"])
            for row in self.label_counter.most_common():
                writer.writerow(row)

        with open("token_distribution.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Token", "Count"])
            for row in self.token_counter.most_common():
                writer.writerow(row)

        with open("sentence_lengths.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["SentenceLength"])
            for length in self.lengths:
                writer.writerow([length])

        with open("category_distribution.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Category", "Count"])
            for row in self.category_counter.most_common():
                writer.writerow(row)

        with open("subtype_distribution.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Subtype", "Count"])
            for row in self.subtype_counter.most_common():
                writer.writerow(row)

        with open("focus_distribution.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Focus", "Count"])
            for row in self.focus_counter.most_common():
                writer.writerow(row)

        with open("unresolved_tags.csv", "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)
            writer.writerow(["Tag", "Count"])
            for row in self.unresolved_counter.most_common():
                writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="JSONL dataset"
    )

    args = parser.parse_args()

    profiler = DatasetProfiler(args.input)

    profiler.profile()

    profiler.print_report()

    profiler.export_csv()