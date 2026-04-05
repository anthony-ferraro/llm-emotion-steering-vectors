#!/usr/bin/env python3
"""Generate contrastive pairs for all target emotions."""

from llm_pharma.emotions import generate_all_pairs

print("Generating contrastive pairs...")
generate_all_pairs()
print("Done.")
