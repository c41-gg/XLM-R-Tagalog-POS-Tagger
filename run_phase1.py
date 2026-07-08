"""
Use case:

python run_phase1.py `
  --input data/processed/corpus_clean2.txt `
  --output data/processed/phase2_2.jsonl `
  --log logs/phase2_2_errors.log `
  --java-jar Library/FSPOST/stanford-postagger.jar `
  --tagalog-model Library/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger `
  --batch-size 50 `
  --max-sentence-len 60 `
  --num-workers 4 `
  --target-sentences 10000 `
  --starting-sentence-index 0
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from modules.hybrid_pos import HybridPOSTagger
from modules.json_writer import build_entry


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/phase2_1.jsonl")
    parser.add_argument("--log", default="logs/phase2_2_errors.log")

    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-sentence-len", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--java-jar", required=True)
    parser.add_argument("--tagalog-model", required=True)

    parser.add_argument(
        "--target-sentences",
        type=int,
        default=-1,
        help="Number of valid sentences to process (-1 = process all)"
    )

    parser.add_argument(
        "--starting-sentence-index",
        type=int,
        default=0,
        help="Start processing from this VALID sentence index"
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Load corpus
# ------------------------------------------------------------

def load_sentences(path):

    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# ------------------------------------------------------------
# Sentence filter
# ------------------------------------------------------------

def is_valid(sentence, max_len):

    return len(sentence.split()) <= max_len


# ------------------------------------------------------------
# Worker initialization
# ------------------------------------------------------------

tagger = None

def init_worker(java_jar, model_path):

    global tagger

    tagger = HybridPOSTagger(java_jar, model_path)


# ------------------------------------------------------------
# Process sentence
# ------------------------------------------------------------

def process_sentence(sentence):

    global tagger

    try:

        tokens = tagger.tag(sentence)

        return build_entry(tokens)

    except Exception as e:

        return {
            "error": str(e),
            "sentence": sentence
        }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = parse_args()
    processed_count = 0

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    sentences = load_sentences(args.input)

    sentences = [
        s
        for s in sentences
        if is_valid(s, args.max_sentence_len)
    ]

    total_valid = len(sentences)

    if args.starting_sentence_index < 0:
        raise ValueError(
            "--starting-sentence-index must be >= 0"
        )

    if args.starting_sentence_index >= total_valid:
        raise ValueError(
            f"Starting index ({args.starting_sentence_index}) "
            f"is greater than the number of valid sentences "
            f"({total_valid})."
        )

    print(f"Total valid sentences : {total_valid:,}")
    print(f"Starting index        : {args.starting_sentence_index:,}")

    sentences = sentences[args.starting_sentence_index:]

    print(f"Remaining sentences   : {len(sentences):,}")
    print()

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(args.log).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    current_index = args.starting_sentence_index

    with open(args.output, "w", encoding="utf-8") as out_file, \
         open(args.log, "w", encoding="utf-8") as log_file:

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(
                args.java_jar,
                args.tagalog_model
            )
        ) as executor:

            for result in executor.map(
                process_sentence,
                sentences
            ):

                if (
                    args.target_sentences > 0
                    and processed_count >= args.target_sentences
                ):
                    print(
                        f"Reached target: {args.target_sentences}"
                    )
                    break

                if "error" in result:

                    log_file.write(
                        f"[Sentence {current_index}] "
                        f"{result['sentence']}\n"
                    )
                    log_file.write(
                        f"Error: {result['error']}\n\n"
                    )

                    current_index += 1
                    continue

                out_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    ) + "\n"
                )

                processed_count += 1
                current_index += 1

                if processed_count % args.batch_size == 0:
                    print(
                        f"Processed: {processed_count}"
                    )

    print("\nDONE")
    print(
        f"Total processed sentences: "
        f"{processed_count}"
    )


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    main()