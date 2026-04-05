#!/usr/bin/env python3
"""Train steering vectors for all emotions via CAA."""

from llm_pharma.config import CORE_EMOTIONS, STEERING_LAYERS, get_emotion
from llm_pharma.model_utils import load_model_and_tokenizer, clear_memory
from llm_pharma.emotions import load_pairs
from llm_pharma.vectors.caa import train_emotion_vector
from llm_pharma.vectors.registry import save_vector

print("Loading model...")
model, tokenizer = load_model_and_tokenizer()

print(f"\nTraining vectors for {len(CORE_EMOTIONS)} core emotions (layers {STEERING_LAYERS[0]}-{STEERING_LAYERS[-1]})...")

for name in CORE_EMOTIONS:
    emotion = get_emotion(name)
    print(f"\n--- {emotion.name} ---")
    pairs = load_pairs(emotion.name, split="train")
    print(f"  Loaded {len(pairs)} training pairs")

    sv = train_emotion_vector(model, tokenizer, pairs)

    path = save_vector(
        emotion.name,
        sv,
        metadata={
            "emotion": emotion.name,
            "num_pairs": len(pairs),
            "layers": STEERING_LAYERS,
        },
    )
    print(f"  Saved: {path}")
    clear_memory()

print("\nAll vectors trained and saved.")
