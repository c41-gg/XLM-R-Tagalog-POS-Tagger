import argparse
import random
from pathlib import Path


def split_jsonl_lines(
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    validation_ratio: float = 0.1,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        raise ValueError(f"No non-empty lines found in {input_path}")

    if abs((train_ratio + test_ratio + validation_ratio) - 1.0) > 1e-9:
        raise ValueError("Train/test/validation ratios must sum to 1.0")

    rng = random.Random(seed)
    rng.shuffle(lines)

    n = len(lines)
    train_n = int(n * train_ratio)
    test_n = int(n * test_ratio)
    validation_n = n - train_n - test_n

    splits = {
        "train.txt": lines[:train_n],
        "test.txt": lines[train_n : train_n + test_n],
        "validation.txt": lines[train_n + test_n :],
    }

    for filename, subset in splits.items():
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as f:
            if subset:
                f.write("\n".join(subset) + "\n")
        print(f"Wrote {output_path} ({len(subset)} lines)")

    return {
        "total": n,
        "train": train_n,
        "test": test_n,
        "validation": validation_n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create shuffled train/test/validation splits from a JSONL file")
    parser.add_argument("--input", default="data/processed/train/phase3_clean.jsonl", help="Path to the input JSONL file")
    parser.add_argument("--output-dir", default="data/processed", help="Directory where train.txt, test.txt, and validation.txt will be written")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    result = split_jsonl_lines(args.input, args.output_dir, seed=args.seed)
    print(result)
