#!/usr/bin/env python3
"""Compute residual stream norms and steering vector norms to calibrate multipliers."""
import os
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from llm_pharma.model_utils import load_model_and_tokenizer, compute_residual_stream_norm
from llm_pharma.vectors.registry import load_vector, list_vectors

print("Loading model...", flush=True)
model, tokenizer = load_model_and_tokenizer()

print("\nComputing residual stream norm...", flush=True)
texts = [
    "def fibonacci(n):",
    "Write a function that checks if a number is prime.",
    "The developer was debugging a failing test suite.",
    "Implement a binary search algorithm.",
    "Parse a CSV file and return the rows as dictionaries.",
]
norm = compute_residual_stream_norm(model, tokenizer, texts)
print(f"Mean residual stream norm: {norm:.1f}", flush=True)

print("\nSteering vector norms:", flush=True)
for name in list_vectors():
    sv = load_vector(name)
    vec_norms = [vec.float().norm().item() for vec in sv.layer_activations.values()]
    mean_vec_norm = sum(vec_norms) / len(vec_norms)
    ratio = mean_vec_norm / norm
    rec_mult = 0.5 * norm / mean_vec_norm  # to achieve 0.5x residual norm effect
    print(f"  {name:>14s}: norm={mean_vec_norm:.1f}  ratio={ratio:.4f}  rec_mult={rec_mult:.3f}", flush=True)

print(f"\nTo match Anthropic's 0.5x residual norm steering strength,", flush=True)
print(f"use multiplier ≈ {0.5 * norm / mean_vec_norm:.3f}", flush=True)
