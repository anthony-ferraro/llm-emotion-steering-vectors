"""Verify that emotion vectors activate on matching emotional content."""

import torch
import numpy as np
from steering_vectors import SteeringVector

from llm_pharma.config import STEERING_LAYERS
from llm_pharma.model_utils import clear_memory


def compute_projection(
    model,
    tokenizer,
    text: str,
    sv: SteeringVector,
    layers: list[int] | None = None,
) -> float:
    """Compute the projection of model activations onto a steering vector.

    Returns the mean projection across specified layers at the last token.
    """
    if layers is None:
        layers = STEERING_LAYERS

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    projections = []
    for layer in layers:
        if layer not in sv.layer_activations:
            continue
        hidden = outputs.hidden_states[layer][0, -1].float().cpu()  # last token
        direction = sv.layer_activations[layer].float()
        direction_norm = direction / (direction.norm() + 1e-8)
        proj = torch.dot(hidden, direction_norm).item()
        projections.append(proj)

    del outputs
    clear_memory()

    return np.mean(projections) if projections else 0.0


def discrimination_score(
    model,
    tokenizer,
    target_sv: SteeringVector,
    matching_texts: list[str],
    non_matching_texts: list[str],
) -> float:
    """Compute how much more a vector activates on matching vs non-matching text.

    Returns ratio of mean activation on matching / mean activation on non-matching.
    Score > 1.0 means the vector discriminates correctly.
    """
    matching_projs = [compute_projection(model, tokenizer, t, target_sv) for t in matching_texts]
    non_matching_projs = [compute_projection(model, tokenizer, t, target_sv) for t in non_matching_texts]

    mean_match = np.mean(matching_projs)
    mean_non_match = np.mean(non_matching_projs)

    # Avoid division by zero; use absolute values for ratio
    if abs(mean_non_match) < 1e-6:
        return float("inf") if mean_match > 0 else 0.0

    return mean_match / abs(mean_non_match)


def validate_all_vectors(
    model,
    tokenizer,
    vectors: dict[str, SteeringVector],
    val_pairs: dict[str, list[tuple[str, str]]],
) -> dict[str, dict]:
    """Run activation validation on all vectors.

    Returns dict of {emotion_name: {score, mean_match, mean_non_match, pass}}.
    """
    results = {}

    for name, sv in vectors.items():
        if name not in val_pairs:
            continue

        pairs = val_pairs[name]
        matching = [p[0] for p in pairs]  # positive examples

        # Use other emotions' positive examples as non-matching
        non_matching = []
        for other_name, other_pairs in val_pairs.items():
            if other_name != name:
                non_matching.extend(p[0] for p in other_pairs[:3])

        score = discrimination_score(model, tokenizer, sv, matching, non_matching)
        results[name] = {
            "discrimination_score": round(score, 3),
            "pass": score > 1.5,
        }

    return results
