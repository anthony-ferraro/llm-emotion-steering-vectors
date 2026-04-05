#!/usr/bin/env python3
"""Validate trained emotion vectors."""

from llm_pharma.config import EMOTION_NAMES
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.emotions import load_pairs
from llm_pharma.vectors.registry import load_vector, list_vectors
from llm_pharma.validation.activation_check import validate_all_vectors
from llm_pharma.validation.logit_lens import logit_lens_report
from llm_pharma.validation.cosine_geometry import cosine_similarity_matrix
from llm_pharma.analysis.visualization import plot_emotion_geometry

# Load vectors
available = list_vectors()
print(f"Found {len(available)} vectors: {available}")

vectors = {name: load_vector(name) for name in available}

# Load model for activation checks
print("\nLoading model...")
model, tokenizer = load_model_and_tokenizer()

# --- 1. Activation check ---
print("\n=== Activation Validation ===")
val_pairs = {name: load_pairs(name, split="val") for name in available}
results = validate_all_vectors(model, tokenizer, vectors, val_pairs)

passed = 0
for name, r in results.items():
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  {name:>12s}: score={r['discrimination_score']:.3f}  [{status}]")
    if r["pass"]:
        passed += 1

print(f"\n{passed}/{len(results)} vectors passed activation validation")

# --- 2. Logit lens ---
print("\n=== Logit Lens ===")
logit_results = logit_lens_report(model, tokenizer, vectors)

# --- 3. Geometry ---
print("\n=== Geometry ===")
sim_matrix, names = cosine_similarity_matrix(vectors)
print(f"Cosine similarity matrix shape: {sim_matrix.shape}")
plot_emotion_geometry(vectors)

clear_memory()
del model
print("\nValidation complete. Check data/figures/ for plots.")
