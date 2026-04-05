#!/usr/bin/env python3
"""Clean thinking tokens and markdown from completions.

Fixes Qwen3.5 output that includes </think> tags and ```python blocks.
Run this on any results JSONL file before evaluation.

Usage: python clean_completions.py [file.jsonl ...]
       If no files given, cleans all JSONL files in data/results/
"""
import sys
import json
from pathlib import Path

def clean_completion(text: str) -> str:
    """Strip thinking tokens and markdown fences."""
    if "</think>" in text:
        text = text[:text.index("</think>")]
    if "```python" in text:
        text = text[:text.index("```python")]
    if "```" in text:
        text = text[:text.index("```")]
    return text


def clean_file(path: Path) -> int:
    """Clean a single JSONL file in place. Returns number of records cleaned."""
    records = []
    cleaned = 0
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            original = record["completion"]
            record["completion"] = clean_completion(original)
            if record["completion"] != original:
                cleaned += 1
            records.append(record)

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return cleaned


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        results_dir = Path(__file__).parent.parent / "data" / "results"
        files = sorted(results_dir.glob("*.jsonl"))

    for path in files:
        if path.name.endswith("_results.jsonl"):
            continue  # skip evaluation result files
        cleaned = clean_file(path)
        print(f"  {path.name}: {cleaned} completions cleaned")
