"""
Use case:
python run_phase1.py `
  --input data/processed/corpus_clean2.txt `
  --output data/processed/phase1_1.jsonl `
  --java-jar Library/FSPOST/stanford-postagger.jar `
  --tagalog-model Library/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger `
  --batch-size 50 `
  --max-sentence-len 60 `
  --num-workers 4 `
  --target-sentences 1000

"""


import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from modules.hybrid_pos import HybridPOSTagger
from modules.json_writer import build_entry


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/phase1_8.jsonl")
    parser.add_argument("--log", default="logs/phase1_1_errors.log")

    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-sentence-len", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--java-jar", required=True)
    parser.add_argument("--tagalog-model", required=True)
    parser.add_argument("--target-sentences", type=int, default=-1)

    return parser.parse_args()


# -----------------------------
# LOAD CORPUS
# -----------------------------

def load_sentences(path):

    with open(path, "r", encoding="utf-8") as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]


# -----------------------------
# FILTER
# -----------------------------

def is_valid(sentence, max_len):

    return len(sentence.split()) <= max_len


# -----------------------------
# WORKER INITIALIZATION
# -----------------------------

tagger = None

def init_worker(java_jar, model_path):

    global tagger

    tagger = HybridPOSTagger(java_jar, model_path)


# -----------------------------
# PROCESS SINGLE SENTENCE
# -----------------------------

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


# -----------------------------
# MAIN
# -----------------------------

def main():

    args = parse_args()


    processed_count = 0

    sentences = load_sentences(args.input)

    sentences = [
        s for s in sentences
        if is_valid(s, args.max_sentence_len)
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_file, \
         open(args.log, "w", encoding="utf-8") as log_file:

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(args.java_jar, args.tagalog_model)
        ) as executor:

            batch = []

            for result in executor.map(process_sentence, sentences):

                if args.target_sentences > 0 and processed_count >= args.target_sentences:
                    print(f"Reached target: {args.target_sentences}")
                    break

                if "error" in result:

                    log_file.write(
                        f"[ERROR] {result['sentence']} | {result['error']}\n"
                    )
                    continue

                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                processed_count += 1

                if processed_count % args.batch_size == 0:
                    print(f"Processed: {processed_count}")
    
    print(f"\nDONE")
    print(f"Total processed sentences: {processed_count}")


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    main()