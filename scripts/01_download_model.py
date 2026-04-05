#!/usr/bin/env python3
"""Download and verify model."""

from llm_pharma.model_utils import load_model_and_tokenizer, get_device, clear_memory
from llm_pharma.config import MODEL_ID

print(f"Device: {get_device()}")
print(f"Downloading/loading {MODEL_ID}...")

model, tokenizer = load_model_and_tokenizer()

# Quick sanity check — generate a few tokens
inputs = tokenizer("def hello():\n    ", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(f"Sanity check: {tokenizer.decode(output[0], skip_special_tokens=True)}")

clear_memory()
del model
print("Model download verified.")
