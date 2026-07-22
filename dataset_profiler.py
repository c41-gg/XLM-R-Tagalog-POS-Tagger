"""
python dataset_profiler.py `
    --input data/processed/train/phase3_clean.jsonl `
    --output phase3-profile.json `
    --text-output phase3-profile.txt `
"""
import json
import argparse
from collections import Counter
from statistics import mean, median
from pathlib import Path
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
        self.degree_counter = Counter()
        self.extra_counter = Counter()

        self.unresolved_counter = Counter()  # tag -> count of tokens hitting it
        self.unresolved_reasons = Counter()  # "unknown_part" / "same_axis_stack"

        self.decomposition_mismatches = 0  # stored fields vs. freshly computed
        self.none_labels = 0

        self.malformed = Counter()

        self._report_path = None

    # -----------------------------------------------------------
    # Per-token decomposition: use stored fields if present,
    # otherwise fall back to computing them on the fly so this
    # still works on older phase1.jsonl files written before the
    # json_writer update.
    # -----------------------------------------------------------

    def _decomposed_fields(self, item, i, label):

        # NOTE: build_entry() in json_writer.py emits "degrees" and
        # "extras" (plural) -- these must match exactly or has_stored
        # is always False and the decomposition_mismatches check never
        # fires.
        has_stored = all(
            key in item for key in
            ("categories", "subtypes", "focuses", "degrees", "extras")
        )

        computed = decompose(label)
        computed_category = computed.category
        computed_subtype = computed.subtype
        computed_focus = computed.focus
        computed_degree = computed.degree
        computed_extra = computed.extras  # list[str], e.g. [] or ["CCP"]

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
            degree = item["degrees"][i]
            extra = item["extras"][i]

            if (category, subtype, focus, degree, extra) != (
                computed_category, computed_subtype,
                computed_focus, computed_degree, computed_extra
            ):
                self.decomposition_mismatches += 1

        # Distributions are always built from the fresh decompose() call,
        # never the stored fields. Stored data can go stale relative to
        # the current mgnn_schema.py (e.g. an older json_writer.py wrote
        # wrong values for a field that's since been fixed) -- trusting
        # it silently would mean a schema fix never actually shows up in
        # reporting until the whole dataset is regenerated. The stored
        # vs. computed comparison above still runs and still counts
        # mismatches, so drift is visible; it just no longer decides
        # which values get counted.
        return computed_category, computed_subtype, computed_focus, computed_degree, computed_extra

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
                        self.none_labels += 1
                        continue

                    self.token_counter[token] += 1
                    self.label_counter[label] += 1

                    if " " in label:
                        self.malformed[label] += 1
                        continue

                    category, subtype, focus, degree, extra = self._decomposed_fields(
                        item, i, label
                    )

                    if category:
                        self.category_counter[category] += 1
                    if subtype:
                        self.subtype_counter[subtype] += 1
                    if focus:
                        self.focus_counter[focus] += 1
                    if degree:
                        self.degree_counter[degree] += 1

                    # `extra` is a list (a token can carry 0, 1, or more
                    # attached tags -- CCP/CCA/LM/PRSP), so count each
                    # entry individually rather than using the list
                    # itself as a Counter key (unhashable -> crashes).
                    for e in extra:
                        self.extra_counter[e] += 1

    def write_text_report(self, output_path="dataset_profile.txt"):

        lines = []

        lines.append("=" * 60)
        lines.append("DATASET PROFILE")
        lines.append("=" * 60)

        lines.append(f"Sentences              : {self.sentences:,}")
        lines.append(f"Tokens                 : {self.tokens:,}")
        lines.append(f"Vocabulary             : {len(self.token_counter):,}")

        if self.none_labels:
            lines.append(f"None-labeled Tokens    : {self.none_labels:,} "
                          f"(upstream tagging/alignment failure -- excluded above)")

        lines.append("")

        lines.append(f"Average Length         : {mean(self.lengths):.2f}")
        lines.append(f"Median Length          : {median(self.lengths):.0f}")
        lines.append(f"Shortest Sentence      : {min(self.lengths)}")
        lines.append(f"Longest Sentence       : {max(self.lengths)}")

        lines.append("")

        lines.append(f"Unique Labels          : {len(self.label_counter)}")
        lines.append(f"Unique Categories      : {len(self.category_counter)}")
        lines.append(f"Unique Subtypes        : {len(self.subtype_counter)}")
        lines.append(f"Unique Focus Values    : {len(self.focus_counter)}")
        lines.append(f"Unique Degrees         : {len(self.degree_counter)}")
        lines.append(f"Unique Extra Tags      : {len(self.extra_counter)}")

        lines.append("")

        lines.append("Category Distribution")
        lines.append("-------------------")
        for cat, count in self.category_counter.most_common():
            pct = 100 * count / self.tokens if self.tokens else 0
            lines.append(f"{cat:12} {count:8,} ({pct:5.2f}%)")

        lines.append("")

        lines.append("Focus Distribution (verb-only)")
        lines.append("-------------------")
        if self.focus_counter:
            focus_total = sum(self.focus_counter.values())
            verb_total = self.category_counter.get("VB", 0)
            for foc, count in self.focus_counter.most_common():
                pct = 100 * count / focus_total
                lines.append(f"{foc:12} {count:8,} ({pct:5.2f}% of focus-marked tokens)")
            if verb_total:
                coverage = 100 * focus_total / verb_total
                lines.append(f"")
                lines.append(f"Focus coverage: {focus_total:,}/{verb_total:,} "
                              f"VB tokens carry an explicit focus tag ({coverage:.2f}%)")
        else:
            lines.append("(none found)")

        lines.append("")

        lines.append("Extra Tag Distribution")
        lines.append("----------------------")

        extra_total = sum(self.extra_counter.values())

        if extra_total:
            for extra, count in self.extra_counter.most_common():
                pct = 100 * count / extra_total
                lines.append(f"{extra:8} {count:8,} ({pct:5.2f}%)")
        else:
            lines.append("(none)")

        lines.append("")

        if self.unresolved_counter:
            total_unresolved = sum(self.unresolved_counter.values())
            lines.append(f"Unresolved Decompositions : {total_unresolved:,} tokens "
                          f"across {len(self.unresolved_counter)} distinct tag(s)")
            for reason, count in self.unresolved_reasons.most_common():
                lines.append(f"  {reason:<18}: {count:,}")
            lines.append("  Top offending tags:")
            for tag, count in self.unresolved_counter.most_common(10):
                lines.append(f"    {tag:<15} x{count}")
        else:
            lines.append("Unresolved Decompositions : None -- every tag decomposed cleanly")

        if self.decomposition_mismatches:
            lines.append(f"")
            lines.append(f"[!] {self.decomposition_mismatches} token(s) where stored "
                          f"category/subtype/focus/degree/extra fields disagree with a fresh "
                          f"decompose() call -- all distributions above are built from the "
                          f"fresh call, not the stored fields, so this count is diagnostic "
                          f"only. It usually means the JSONL was written by an older "
                          f"json_writer.py/mgnn_schema.py than the one now in modules/; "
                          f"consider regenerating the dataset once the pipeline is stable.")

        lines.append("")

        lines.append("Top 30 Labels")
        lines.append("-------------------")

        for label, count in self.label_counter.most_common(30):
            pct = count / self.tokens * 100
            lines.append(f"{label:20} {count:8} ({pct:5.2f}%)")

        lines.append("")

        rare = [
            (l, c)
            for l, c in self.label_counter.items()
            if c < 10
        ]

        lines.append(f"Rare Labels (<10)      : {len(rare)}")

        if self.malformed:
            lines.append(f"Malformed Labels       : {len(self.malformed)}")
        else:
            lines.append("Malformed Labels       : None")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        self._report_path = output_path

    def export_report(self, output_path="dataset_profile.json"):
        """
        Everything -- summary stats plus every distribution in full (not
        just the console's top-30) -- combined into one JSON file. One
        file to upload/keep per run instead of juggling label/token/
        category/subtype/focus/degree/extra/unresolved CSVs separately.
        """

        self._report_path = output_path

        report = {
            "summary": {
                "sentences": self.sentences,
                "tokens": self.tokens,
                "vocabulary": len(self.token_counter),
                "none_labeled_tokens": self.none_labels,
                "average_length": round(mean(self.lengths), 2) if self.lengths else 0,
                "median_length": median(self.lengths) if self.lengths else 0,
                "shortest_sentence": min(self.lengths) if self.lengths else 0,
                "longest_sentence": max(self.lengths) if self.lengths else 0,
                "unique_labels": len(self.label_counter),
                "unique_categories": len(self.category_counter),
                "unique_subtypes": len(self.subtype_counter),
                "unique_focus_values": len(self.focus_counter),
                "unique_degrees": len(self.degree_counter),
                "unique_extra_tags": len(self.extra_counter),
                "rare_labels_under_10": sum(
                    1 for c in self.label_counter.values() if c < 10
                ),
                "malformed_label_count": len(self.malformed),
                "decomposition_mismatches": self.decomposition_mismatches,
            },
            "focus_coverage": {
                "focus_marked_tokens": sum(self.focus_counter.values()),
                "verb_tokens": self.category_counter.get("VB", 0),
                "coverage_pct": round(
                    100 * sum(self.focus_counter.values())
                    / self.category_counter["VB"], 2
                ) if self.category_counter.get("VB") else 0,
            },
            "distributions": {
                "labels": dict(self.label_counter.most_common()),
                "tokens": dict(self.token_counter.most_common()),
                "categories": dict(self.category_counter.most_common()),
                "subtypes": dict(self.subtype_counter.most_common()),
                "focuses": dict(self.focus_counter.most_common()),
                "degrees": dict(self.degree_counter.most_common()),
                "extras": dict(self.extra_counter.most_common()),
            },
            "sentence_lengths": self.lengths,
            "unresolved": {
                "reasons": dict(self.unresolved_reasons.most_common()),
                "tags": dict(self.unresolved_counter.most_common()),
            },
            "malformed_labels": dict(self.malformed.most_common()),
        }

        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="JSONL dataset"
    )
    parser.add_argument(
        "--output",
        default="dataset_profile.json",
        help="Combined JSON report output path (default: dataset_profile.json)"
    )
    parser.add_argument(
        "--text-output",
        default="dataset_profile.txt",
        help="Human-readable text report output path (default: dataset_profile.txt)"
    )

    args = parser.parse_args()

    profiler = DatasetProfiler(args.input)

    profiler.profile()

    profiler.export_report(args.output)

    profiler.write_text_report(args.text_output)