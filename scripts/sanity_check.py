#!/usr/bin/env python3
"""Quick sanity check: compare steered vs unsteered output at various multipliers."""
import os
os.environ["PYTHONUNBUFFERED"] = "1"

from llm_pharma.model_utils import load_model_and_tokenizer
from llm_pharma.vectors.registry import load_vector

print("Loading model...", flush=True)
model, tokenizer = load_model_and_tokenizer()

prompt = "def fibonacci(n):\n    "
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[1]

# Baseline
print("=== NO STEERING ===", flush=True)
out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)

# Test calm at various multipliers
for mult in [0.05, 0.1, 0.2, 0.5]:
    sv = load_vector("calm")
    print(f"\n=== CALM x{mult} ===", flush=True)
    with sv.apply(model, multiplier=mult, min_token_index=None, token_indices=slice(0, input_len)):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)

# Test desperate at 0.1
sv = load_vector("desperate")
print("\n=== DESPERATE x0.1 ===", flush=True)
with sv.apply(model, multiplier=0.1, min_token_index=None, token_indices=slice(0, input_len)):
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)

print("\nDone.", flush=True)
