#!/usr/bin/env python3
"""Run HumanEval baseline (no steering)."""

from human_eval.data import read_problems
from llm_pharma.config import RESULTS_DIR
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.benchmark.humaneval_runner import generate_completions
from llm_pharma.benchmark.evaluator import evaluate_results

print("Loading model...")
model, tokenizer = load_model_and_tokenizer()

print("Loading HumanEval problems...")
problems = read_problems()
print(f"  {len(problems)} problems loaded")

output_path = RESULTS_DIR / "baseline.jsonl"
print(f"\nGenerating baseline completions -> {output_path}")
results = generate_completions(model, tokenizer, problems, output_path=output_path)

print(f"\nGenerated {len(results)} completions. Evaluating...")
clear_memory()
del model

metrics = evaluate_results(output_path)
print(f"\nBaseline results: {metrics}")
