"""
Match sentences from a concordance text file to tagged sentences in a JSONL file.
Outputs matching sentences to a new text file.

Usage:
    python scripts/match_concordance_to_jsonl.py <text_file> <jsonl_file> <output_file>

Example:
    python scripts/match_concordance_to_jsonl.py `
        data/raw/tlTenTen19/concordance_preloaded_tltenten19_3_20260716180122.txt `
        data/processed/phase2_6.jsonl `
        data/processed/matched_sentences.txt
"""

import json
import re
import argparse
from pathlib import Path


def normalize_sentence(text: str) -> str:
    """Normalize text for matching by removing extra whitespace and converting to lowercase."""
    # Remove <coll> tags and extra whitespace
    text = re.sub(r'</?coll>', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def extract_sentences_from_concordance(text_file: str) -> list[dict]:
    """
    Extract sentences from concordance text file.
    
    Format expected:
    Reference,Sentence
    source | <s> sentence text </s>
    """
    sentences = []
    
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        # Skip header and empty lines
        if not line or 'Reference,Sentence' in line:
            continue
        
        # Extract source and sentence
        match = re.match(r'^([^|]+)\|\s*<s>\s*(.*?)\s*</s>\s*$', line)
        if match:
            source = match.group(1).strip()
            sentence_text = match.group(2).strip()
            
            sentences.append({
                'source': source,
                'text': sentence_text,
                'normalized': normalize_sentence(sentence_text)
            })
    
    return sentences


def extract_sentences_from_jsonl(jsonl_file: str) -> list[dict]:
    """Extract sentences from JSONL file by joining tokens."""
    sentences = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                tokens = entry.get('tokens', [])
                
                if tokens:
                    sentence_text = ' '.join(tokens)
                    sentences.append({
                        'line_num': line_num,
                        'text': sentence_text,
                        'normalized': normalize_sentence(sentence_text),
                        'labels': entry.get('labels', []),
                        'tokens': tokens
                    })
            
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    return sentences


def match_sentences(text_sentences: list[dict], jsonl_sentences: list[dict]) -> list[dict]:
    """Match sentences from text file to JSONL sentences."""
    matches = []
    
    # Create a lookup dict for faster matching
    jsonl_normalized = {s['normalized']: s for s in jsonl_sentences}
    
    for text_sent in text_sentences:
        normalized = text_sent['normalized']
        
        if normalized in jsonl_normalized:
            jsonl_sent = jsonl_normalized[normalized]
            matches.append({
                'source': text_sent['source'],
                'text': text_sent['text'],
                'jsonl_line': jsonl_sent['line_num'],
                'jsonl_tokens': jsonl_sent['tokens'],
                'jsonl_labels': jsonl_sent['labels']
            })
    
    return matches


def write_matches(matches: list[dict], output_file: str):
    """Write matched sentences to output file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("SOURCE | SENTENCE | JSONL_LINE | TAGS\n")
        f.write("=" * 120 + "\n\n")
        
        for match in matches:
            f.write(f"Source: {match['source']}\n")
            f.write(f"Sentence: {match['text']}\n")
            f.write(f"JSONL Line: {match['jsonl_line']}\n")
            f.write(f"Tags: {', '.join(match['jsonl_labels'])}\n")
            f.write("-" * 120 + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Match concordance sentences to tagged JSONL sentences"
    )
    parser.add_argument(
        "text_file",
        help="Concordance text file path"
    )
    parser.add_argument(
        "jsonl_file",
        help="JSONL file path with tagged sentences"
    )
    parser.add_argument(
        "output_file",
        help="Output text file for matched sentences"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Reading concordance file: {args.text_file}")
        text_sentences = extract_sentences_from_concordance(args.text_file)
        print(f"  Found {len(text_sentences)} sentence(s)")
        
        print(f"\nReading JSONL file: {args.jsonl_file}")
        jsonl_sentences = extract_sentences_from_jsonl(args.jsonl_file)
        print(f"  Found {len(jsonl_sentences)} sentence(s)")
        
        print(f"\nMatching sentences...")
        matches = match_sentences(text_sentences, jsonl_sentences)
        print(f"  Found {len(matches)} match(es)")
        
        if matches:
            print(f"\nWriting matches to: {args.output_file}")
            write_matches(matches, args.output_file)
            print(f"  Successfully wrote {len(matches)} matched sentence(s)")
        else:
            print("  No matches found.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
