#!/usr/bin/env python3
"""Analyze and visualize all benchmark results."""

from llm_pharma.config import RESULTS_DIR, FIGURES_DIR, EMOTION_NAMES, DEFAULT_MULTIPLIERS
from llm_pharma.analysis.results import compute_deltas, load_evaluation, per_problem_matrix
from llm_pharma.analysis.visualization import plot_delta_bar_chart, plot_dose_response
from llm_pharma.vectors.registry import load_vector, list_vectors
from llm_pharma.analysis.visualization import plot_emotion_geometry

print("=== LLM-Pharma Results Analysis ===\n")

# --- Baseline ---
baseline = load_evaluation("baseline")
if "error" in baseline:
    print(f"No baseline found: {baseline['error']}")
    print("Run scripts/05_run_baseline.py first.")
    exit(1)

print(f"Baseline pass@1: {baseline['pass_at_1']:.4f} ({baseline['passed']}/{baseline['total']})")

# --- Deltas ---
print("\n--- Pass@1 Deltas (sorted by effect) ---")
deltas = compute_deltas()
for d in deltas:
    sign = "+" if d["delta"] >= 0 else ""
    print(f"  {d['name']:>40s}: {d['pass_at_1']:.4f} ({sign}{d['delta_pct']:.1f}%)")

# --- Bar chart ---
print("\nGenerating delta bar chart...")
plot_delta_bar_chart(deltas)

# --- Dose-response curves ---
print("\nGenerating dose-response curves...")
available_emotions = list_vectors()
for emotion in available_emotions:
    multipliers = []
    pass_rates = []
    for mult in DEFAULT_MULTIPLIERS:
        run_name = f"single_{emotion}_{mult:+.1f}"
        eval_data = load_evaluation(run_name)
        if "error" not in eval_data:
            multipliers.append(mult)
            pass_rates.append(eval_data["pass_at_1"])

    if len(multipliers) >= 2:
        plot_dose_response(emotion, multipliers, pass_rates, baseline["pass_at_1"])

# --- Emotion geometry ---
print("\nGenerating emotion geometry plots...")
vectors = {name: load_vector(name) for name in available_emotions}
if vectors:
    plot_emotion_geometry(vectors)

# --- Summary ---
print("\n=== Top 5 Best Performing Conditions ===")
for d in deltas[:5]:
    print(f"  {d['name']}: pass@1={d['pass_at_1']:.4f} (delta={d['delta']:+.4f})")

print("\n=== Top 5 Worst Performing Conditions ===")
for d in deltas[-5:]:
    print(f"  {d['name']}: pass@1={d['pass_at_1']:.4f} (delta={d['delta']:+.4f})")

print(f"\nFigures saved to {FIGURES_DIR}")
print("Analysis complete.")
