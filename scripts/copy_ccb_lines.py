"""
Copy lines with _CCB tags from source JSONL files and append to phase2_5.jsonl.

Usage:
    python scripts/copy_CCB_lines.py <source_file> <num_lines> [--output <output_file>]

Example:
    python scripts/copy_CCB_lines.py data/processed/phase2_5.jsonl 300 --output data/processed/phase2_6.jsonl
"""

import json
import argparse
from pathlib import Path


def has_CCB_tag(labels: list) -> bool:
    """Check if any label contains '_CCB' substring."""
    return any("_CCB" in label for label in labels if isinstance(label, str))


def copy_CCB_lines(source_file: str, num_lines: int, output_file: str = None) -> int:
    """
    Copy lines with _CCB tags from source to output file.
    
    Args:
        source_file: Path to source JSONL file
        num_lines: Number of lines with _CCB tags to copy
        output_file: Path to output file (defaults to data/processed/phase2_5.jsonl)
    
    Returns:
        Number of lines actually copied
    """
    
    if output_file is None:
        output_file = "data/processed/phase2_5.jsonl"
    
    source_path = Path(source_file)
    output_path = Path(output_file)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Read source file and collect lines with _CCB tags
    ccp_lines = []
    
    with open(source_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                labels = entry.get("labels", [])
                
                if has_CCB_tag(labels):
                    ccp_lines.append(line)
                    
                    if len(ccp_lines) >= num_lines:
                        break
            
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    # Append to output file
    if ccp_lines:
        with open(output_path, "a", encoding="utf-8") as f:
            for line in ccp_lines:
                f.write(line + "\n")
    
    copied = len(ccp_lines)
    print(f"Copied {copied} line(s) with _CCB tags from {source_file}")
    print(f"Appended to {output_file}")
    
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Copy JSONL lines with _CCB tags to phase2_5.jsonl"
    )
    parser.add_argument(
        "source_file",
        help="Source JSONL file path"
    )
    parser.add_argument(
        "num_lines",
        type=int,
        help="Number of lines with _CCB tags to copy"
    )
    parser.add_argument(
        "--output",
        default="data/processed/phase2_5.jsonl",
        help="Output file path (default: data/processed/phase2_5.jsonl)"
    )
    
    args = parser.parse_args()
    
    try:
        copied = copy_CCB_lines(
            args.source_file,
            args.num_lines,
            args.output
        )
        
        if copied < args.num_lines:
            print(
                f"Warning: Requested {args.num_lines} lines but only found "
                f"and copied {copied} lines with _CCB tags."
            )
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
