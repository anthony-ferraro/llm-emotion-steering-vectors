#!/usr/bin/env python3
"""Run HumanEval with emotion cocktails."""

import argparse
from human_eval.data import read_problems

from llm_pharma.config import COCKTAILS, COCKTAIL_MULTIPLIERS, RESULTS_DIR
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.benchmark.cocktails import build_cocktail
from llm_pharma.benchmark.humaneval_runner import generate_completions
from llm_pharma.benchmark.evaluator import evaluate_results

parser = argparse.ArgumentParser()
parser.add_argument("--cocktails", nargs="*", default=None, help="Specific cocktails to test")
parser.add_argument("--multipliers", nargs="*", type=float, default=None)
parser.add_argument("--skip-eval", action="store_true")
args = parser.parse_args()

target_cocktails = COCKTAILS
if args.cocktails:
    target_cocktails = [c for c in COCKTAILS if c.name in args.cocktails]

multipliers = args.multipliers or COCKTAIL_MULTIPLIERS

print("Loading model...")
model, tokenizer = load_model_and_tokenizer()

print("Loading HumanEval problems...")
problems = read_problems()

total_runs = len(target_cocktails) * len(multipliers)
run_num = 0

for cocktail in target_cocktails:
    print(f"\nBuilding cocktail: {cocktail.name} = {cocktail.recipe}")
    sv = build_cocktail(cocktail)

    for mult in multipliers:
        run_num += 1
        run_name = f"cocktail_{cocktail.name}_{mult:.1f}"
        output_path = RESULTS_DIR / f"{run_name}.jsonl"

        if output_path.exists():
            existing_lines = sum(1 for _ in open(output_path))
            if existing_lines >= len(problems):
                print(f"[{run_num}/{total_runs}] {run_name}: already complete, skipping")
                continue

        print(f"[{run_num}/{total_runs}] {run_name} (mult={mult})")
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

print(f"\nAll {total_runs} cocktail runs complete.")
