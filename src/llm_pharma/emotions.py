"""Emotion definitions and contrastive pair generation."""

import json
import random
from pathlib import Path
from dataclasses import dataclass

from llm_pharma.config import (
    EMOTIONS,
    Emotion,
    NEUTRAL_SEED,
    PAIRS_DIR,
    PAIRS_PER_EMOTION,
    TRAIN_PAIRS,
    VAL_PAIRS,
)

# Coding-adjacent scenario templates.
# Each has a {positive} and {negative} slot filled by the emotion seeds.
CODING_SCENARIOS = [
    # Debugging
    "A developer is debugging a failing test suite. {emotion_text} They open the test output and begin reading the stack trace.",
    "A developer encounters an unexpected null pointer exception in production. {emotion_text} They pull up the logs and start investigating.",
    "A developer's CI pipeline has been red for three commits. {emotion_text} They review the diff between the last passing and first failing commit.",
    "A developer finds that a function returns the wrong result for edge cases. {emotion_text} They set up a minimal reproduction.",
    "A developer notices the application is 10x slower after their latest change. {emotion_text} They start profiling the hot path.",
    # Writing new code
    "A developer starts implementing a new feature from a spec. {emotion_text} They plan out the data structures they'll need.",
    "A developer is writing a recursive algorithm for a tree traversal. {emotion_text} They think through the base cases.",
    "A developer needs to implement input validation for a web form. {emotion_text} They consider the edge cases.",
    "A developer is writing a function to parse a custom file format. {emotion_text} They sketch out the state machine.",
    "A developer begins implementing a sorting algorithm from scratch. {emotion_text} They reason about time complexity.",
    # Code review
    "A developer is reviewing a colleague's pull request. {emotion_text} They read through the changes file by file.",
    "A developer receives feedback on their own code review. {emotion_text} They consider the suggested changes.",
    # Design / architecture
    "A developer is designing the API for a new service. {emotion_text} They think about the interface consumers will use.",
    "A developer needs to choose between two database schemas. {emotion_text} They weigh the tradeoffs.",
    "A developer is refactoring a large monolithic function into smaller pieces. {emotion_text} They identify the natural boundaries.",
    # General / misc
    "A developer is reading documentation for an unfamiliar library. {emotion_text} They search for examples that match their use case.",
    "A developer is writing unit tests for a complex function. {emotion_text} They enumerate the test cases.",
    "A developer is migrating code from Python 2 to Python 3. {emotion_text} They handle the string encoding differences.",
    "A developer is optimizing a database query that runs too slowly. {emotion_text} They examine the query plan.",
    "A developer is resolving a merge conflict in a critical file. {emotion_text} They carefully compare both versions.",
]

# General (non-coding) scenario templates for diversity
GENERAL_SCENARIOS = [
    "A person is working through a challenging puzzle. {emotion_text} They study the pieces carefully.",
    "A writer is crafting the opening paragraph of an essay. {emotion_text} They choose their words deliberately.",
    "A student is preparing for a difficult exam. {emotion_text} They organize their study materials.",
    "A chef is attempting a complex recipe for the first time. {emotion_text} They measure each ingredient precisely.",
    "A musician is learning a technically demanding piece. {emotion_text} They practice the difficult passage slowly.",
    "An architect is reviewing blueprints for a client meeting. {emotion_text} They double-check the dimensions.",
    "A researcher is analyzing unexpected experimental results. {emotion_text} They re-examine their methodology.",
    "A teacher is preparing a lesson on a difficult concept. {emotion_text} They think about how to explain it clearly.",
    "A doctor is reviewing a patient's complex test results. {emotion_text} They consider each finding systematically.",
    "A mechanic is diagnosing an intermittent engine problem. {emotion_text} They run through the possible causes.",
]


def _fill_scenario(template: str, emotion_text: str) -> str:
    """Fill a scenario template with emotion-specific text."""
    return template.replace("{emotion_text}", emotion_text)


def generate_pairs_for_emotion(emotion: Emotion, n: int = PAIRS_PER_EMOTION) -> list[tuple[str, str]]:
    """Generate n contrastive pairs for a given emotion.

    Each pair is (positive_text, negative_text) where:
    - positive_text describes a scenario with the emotion present
    - negative_text describes the same scenario with a shared neutral baseline
      (not the opposite emotion — this isolates "presence of X" cleanly)
    """
    all_scenarios = CODING_SCENARIOS + GENERAL_SCENARIOS
    pairs = []

    for i in range(n):
        template = all_scenarios[i % len(all_scenarios)]
        positive = _fill_scenario(template, emotion.positive_seed)
        negative = _fill_scenario(template, NEUTRAL_SEED)
        pairs.append((positive, negative))

    return pairs


def save_pairs(emotion_name: str, pairs: list[tuple[str, str]], output_dir: Path = PAIRS_DIR):
    """Save contrastive pairs to JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{emotion_name}.jsonl"

    with open(path, "w") as f:
        for i, (pos, neg) in enumerate(pairs):
            split = "train" if i < TRAIN_PAIRS else "val"
            record = {"positive": pos, "negative": neg, "index": i, "split": split}
            f.write(json.dumps(record) + "\n")

    return path


def load_pairs(
    emotion_name: str, split: str | None = None, pairs_dir: Path = PAIRS_DIR
) -> list[tuple[str, str]]:
    """Load contrastive pairs from JSONL file.

    Args:
        emotion_name: Name of the emotion.
        split: If "train" or "val", filter to that split. None returns all.
    """
    path = pairs_dir / f"{emotion_name}.jsonl"
    pairs = []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if split is None or record["split"] == split:
                pairs.append((record["positive"], record["negative"]))

    return pairs


def generate_all_pairs():
    """Generate and save contrastive pairs for all emotions."""
    for emotion in EMOTIONS:
        pairs = generate_pairs_for_emotion(emotion)
        path = save_pairs(emotion.name, pairs)
        print(f"  {emotion.name}: {len(pairs)} pairs -> {path}")
