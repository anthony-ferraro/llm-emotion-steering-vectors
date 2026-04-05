#!/usr/bin/env python3
"""Combined pipeline: loads model once, runs baseline + steered benchmarks.

Avoids the 40-min model reload per script on network volume storage.
Uses unbuffered print for real-time log visibility.
"""
import sys
import os
import json
import argparse

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

from llm_pharma.config import (
    CORE_EMOTIONS, COCKTAILS, COCKTAIL_MULTIPLIERS,
    DEFAULT_MULTIPLIERS, RESULTS_DIR, STEERING_LAYERS,
    get_emotion,
)
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.vectors.registry import load_vector, list_vectors
from llm_pharma.vectors.caa import combine_vectors
from llm_pharma.benchmark.humaneval_runner import generate_completions
from llm_pharma.benchmark.cocktails import build_cocktail

# Import human_eval
from human_eval.data import read_problems

parser = argparse.ArgumentParser()
parser.add_argument("--skip-baseline", action="store_true")
parser.add_argument("--skip-singles", action="store_true")
parser.add_argument("--skip-cocktails", action="store_true")
parser.add_argument("--emotions", nargs="*", default=None, help="Subset of emotions")
parser.add_argument("--multipliers", nargs="*", type=float, default=None)
args = parser.parse_args()

emotions = args.emotions or list_vectors()
multipliers = args.multipliers or DEFAULT_MULTIPLIERS

print("=" * 60, flush=True)
print("LLM-PHARMA PIPELINE", flush=True)
print("=" * 60, flush=True)

# Load model ONCE
print("\n[1/4] Loading model...", flush=True)
model, tokenizer = load_model_and_tokenizer()

# Load problems ONCE
print("\n[2/4] Loading HumanEval problems...", flush=True)
problems = read_problems()
print(f"  {len(problems)} problems loaded", flush=True)

# --- BASELINE ---
if not args.skip_baseline:
    print("\n[3/4] Running baseline (no steering)...", flush=True)
    output_path = RESULTS_DIR / "baseline.jsonl"
    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if existing >= len(problems):
            print(f"  Baseline already complete ({existing} results), skipping", flush=True)
        else:
            generate_completions(model, tokenizer, problems, output_path=output_path)
    else:
        generate_completions(model, tokenizer, problems, output_path=output_path)
    print(f"  Baseline saved to {output_path}", flush=True)
else:
    print("\n[3/4] Skipping baseline", flush=True)

# --- SINGLE EMOTIONS ---
if not args.skip_singles:
    total_runs = len(emotions) * len(multipliers)
    run_num = 0
    print(f"\n[4a] Running single emotion steering ({total_runs} runs)...", flush=True)

    for emotion_name in emotions:
        sv = load_vector(emotion_name)

        for mult in multipliers:
            run_num += 1
            run_name = f"single_{emotion_name}_{mult:+.1f}"
            output_path = RESULTS_DIR / f"{run_name}.jsonl"

            if output_path.exists():
                existing = sum(1 for _ in open(output_path))
                if existing >= len(problems):
                    print(f"  [{run_num}/{total_runs}] {run_name}: complete, skipping", flush=True)
                    continue

            print(f"  [{run_num}/{total_runs}] {run_name}...", flush=True)
            generate_completions(
                model, tokenizer, problems,
                steering_vector=sv,
                multiplier=mult,
                output_path=output_path,
            )
            clear_memory()
else:
    print("\n[4a] Skipping single emotions", flush=True)

# --- COCKTAILS ---
if not args.skip_cocktails:
    total_runs = len(COCKTAILS) * len(COCKTAIL_MULTIPLIERS)
    run_num = 0
    print(f"\n[4b] Running cocktail steering ({total_runs} runs)...", flush=True)

    for cocktail in COCKTAILS:
        print(f"  Building {cocktail.name}: {cocktail.recipe}", flush=True)
        sv = build_cocktail(cocktail)

        for mult in COCKTAIL_MULTIPLIERS:
            run_num += 1
            run_name = f"cocktail_{cocktail.name}_{mult:.1f}"
            output_path = RESULTS_DIR / f"{run_name}.jsonl"

            if output_path.exists():
                existing = sum(1 for _ in open(output_path))
                if existing >= len(problems):
                    print(f"  [{run_num}/{total_runs}] {run_name}: complete, skipping", flush=True)
                    continue

            print(f"  [{run_num}/{total_runs}] {run_name}...", flush=True)
            generate_completions(
                model, tokenizer, problems,
                steering_vector=sv,
                multiplier=mult,
                output_path=output_path,
            )
            clear_memory()
else:
    print("\n[4b] Skipping cocktails", flush=True)

print("\n" + "=" * 60, flush=True)
print("PIPELINE COMPLETE", flush=True)
print(f"Results in {RESULTS_DIR}", flush=True)
print("=" * 60, flush=True)
