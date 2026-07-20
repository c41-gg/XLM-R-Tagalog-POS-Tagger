"""
Remove duplicate entries from JSONL files.
Duplicates are identified by matching token sequences.

Usage:
    python scripts/deduplicate_jsonl.py <input_file> [--output <output_file>]

Example:
    python scripts/deduplicate_jsonl.py data/processed/phase2_6.jsonl
    python scripts/deduplicate_jsonl.py data/processed/phase2_6.jsonl --output data/processed/phase2_6_dedup.jsonl
"""

import json
import argparse
from pathlib import Path


def deduplicate_jsonl(input_file: str, output_file: str = None) -> tuple[int, int]:
    """
    Remove duplicate entries from JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output file (defaults to overwriting input)
    
    Returns:
        Tuple of (total_lines, unique_lines)
    """
    
    if output_file is None:
        output_file = input_file
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    seen = set()
    unique_entries = []
    duplicates_removed = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                tokens = entry.get('tokens', [])
                
                # Create a hashable key from tokens (the unique identifier)
                token_key = tuple(tokens)
                
                if token_key not in seen:
                    seen.add(token_key)
                    unique_entries.append(line)
                else:
                    duplicates_removed += 1
            
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    # Write deduplicated entries
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in unique_entries:
            f.write(entry + '\n')
    
    total_lines = line_num
    unique_lines = len(unique_entries)
    
    return total_lines, unique_lines, duplicates_removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate JSONL entries based on token sequences"
    )
    parser.add_argument(
        "input_file",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: overwrites input file)"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Processing: {args.input_file}")
        total, unique, removed = deduplicate_jsonl(
            args.input_file,
            args.output
        )
        
        output_path = args.output or args.input_file
        print(f"\nResults:")
        print(f"  Total entries processed: {total}")
        print(f"  Unique entries: {unique}")
        print(f"  Duplicates removed: {removed}")
        print(f"  Written to: {output_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
