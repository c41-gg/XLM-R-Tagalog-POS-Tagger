import json
from pathlib import Path


FIELDS = [
    "tokens",
    "labels",
    "categories",
    "subtypes",
    "focuses",
    "degrees",
    "extras",
]


def clean_entry(entry: dict):
    labels = entry.get("labels", [])

    keep_indices = [
        i for i, label in enumerate(labels)
        if label != "IGNORED"
    ]

    if not keep_indices:
        return None

    cleaned = {}

    for key, value in entry.items():

        if key in FIELDS:
            cleaned[key] = [value[i] for i in keep_indices]

        else:
            cleaned[key] = value

    # ---------------------------------------
    # Update unresolved indices if present
    # ---------------------------------------

    if "unresolved" in cleaned:

        index_map = {
            old: new
            for new, old in enumerate(keep_indices)
        }

        new_unresolved = []

        for item in cleaned["unresolved"]:

            old_index = item["index"]

            if old_index in index_map:

                item = dict(item)
                item["index"] = index_map[old_index]
                new_unresolved.append(item)

        cleaned["unresolved"] = new_unresolved

    return cleaned


def clean_dataset(input_file, output_file):

    total_sentences = 0
    written_sentences = 0
    removed_tokens = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:

            if not line.strip():
                continue

            total_sentences += 1

            entry = json.loads(line)

            before = len(entry["tokens"])

            cleaned = clean_entry(entry)

            if cleaned is None:
                continue

            after = len(cleaned["tokens"])

            removed_tokens += before - after

            fout.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False
                ) + "\n"
            )

            written_sentences += 1

    print("Finished.")
    print(f"Sentences read      : {total_sentences:,}")
    print(f"Sentences written   : {written_sentences:,}")
    print(f"Tokens removed      : {removed_tokens:,}")


if __name__ == "__main__":

    INPUT = "data/processed/train/phase3.jsonl"
    OUTPUT = "data/processed/train/phase3_clean.jsonl"

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    clean_dataset(INPUT, OUTPUT)