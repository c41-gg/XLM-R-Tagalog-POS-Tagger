import json
import argparse
from collections import Counter
from statistics import mean, median
from pathlib import Path
import csv
import ast



class DatasetProfiler:

    def __init__(self, jsonl_path):

        self.path = Path(jsonl_path)

        self.sentences = 0
        self.tokens = 0

        self.label_counter = Counter()
        self.token_counter = Counter()

        self.lengths = []

        self.composite_counter = Counter()

        self.malformed = Counter()

    def profile(self):

        with open(self.path, encoding="utf-8") as f:

            for line in f:

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = ast.literal_eval(line)

                tokens = item["tokens"]
                labels = item["labels"]

                self.sentences += 1
                self.tokens += len(tokens)

                self.lengths.append(len(tokens))

                for token, label in zip(tokens, labels):
                    
                    if label is None:
                        print(f"Found None label on token: {token}")
                        continue


                    self.token_counter[token] += 1
                    self.label_counter[label] += 1

                    if "_" in label:

                        self.composite_counter[label] += 1

                        second = label.split("_", 1)[1]

                        if second == "CCP":
                            self.composite_counter["_CCP"] += 1

                        elif second.startswith("VB"):
                            self.composite_counter["_VB*F"] += 1

                        elif second.startswith("JJ"):
                            self.composite_counter["_JJ*"] += 1

                    if " " in label:
                        self.malformed[label] += 1

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

        single = sum(
            c for l, c in self.label_counter.items()
            if "_" not in l
        )

        composite = self.tokens - single

        print(f"Single-tag Tokens      : {single:,}")
        print(f"Composite-tag Tokens   : {composite:,}")

        if self.tokens:
            print(f"Composite Percentage   : {100*composite/self.tokens:.2f}%")

        print()

        print("Composite Breakdown")

        print("-------------------")

        print(f"_CCP                  : {self.composite_counter['_CCP']:,}")
        print(f"_VB*F                 : {self.composite_counter['_VB*F']:,}")
        print(f"_JJ*                  : {self.composite_counter['_JJ*']:,}")

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