#!/usr/bin/env python3
"""Run HumanEval with individual emotion vectors at various multipliers."""

import argparse
from human_eval.data import read_problems

from llm_pharma.config import EMOTION_NAMES, DEFAULT_MULTIPLIERS, RESULTS_DIR
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.vectors.registry import load_vector, list_vectors
from llm_pharma.benchmark.humaneval_runner import generate_completions
from llm_pharma.benchmark.evaluator import evaluate_results

parser = argparse.ArgumentParser()
parser.add_argument("--emotions", nargs="*", default=None, help="Specific emotions to test (default: all)")
parser.add_argument("--multipliers", nargs="*", type=float, default=None, help="Specific multipliers (default: config)")
parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (just generate)")
args = parser.parse_args()

emotions = args.emotions or list_vectors()
multipliers = args.multipliers or DEFAULT_MULTIPLIERS

print("Loading model...")
model, tokenizer = load_model_and_tokenizer()

print("Loading HumanEval problems...")
problems = read_problems()

total_runs = len(emotions) * len(multipliers)
run_num = 0

for emotion_name in emotions:
    sv = load_vector(emotion_name)

    for mult in multipliers:
        run_num += 1
        run_name = f"single_{emotion_name}_{mult:+.1f}"
        output_path = RESULTS_DIR / f"{run_name}.jsonl"

        # Check if already completed
        if output_path.exists():
            existing_lines = sum(1 for _ in open(output_path))
            if existing_lines >= len(problems):
                print(f"[{run_num}/{total_runs}] {run_name}: already complete, skipping")
                continue

        print(f"\n[{run_num}/{total_runs}] {run_name} (emotion={emotion_name}, mult={mult})")
        generate_completions(
            model, tokenizer, problems,
            steering_vector=sv,
            multiplier=mult,
            output_path=output_path,
        )

        if not args.skip_eval:
            metrics = evaluate_results(output_path)
            print(f"  Result: {metrics}")

        clear_memory()

print(f"\nAll {total_runs} runs complete.")
