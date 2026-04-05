"""Train emotion steering vectors via Contrastive Activation Addition."""

import torch
from steering_vectors import train_steering_vector, SteeringVector

from llm_pharma.config import STEERING_LAYERS


def train_emotion_vector(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    layers: list[int] | None = None,
) -> SteeringVector:
    """Train a steering vector from contrastive pairs.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        pairs: List of (positive, negative) text pairs.
        layers: Model layers to extract from. Defaults to STEERING_LAYERS.

    Returns:
        Trained SteeringVector.
    """
    if layers is None:
        layers = STEERING_LAYERS

    sv = train_steering_vector(
        model,
        tokenizer,
        pairs,
        layers=layers,
        read_token_index=-1,
        move_to_cpu=True,
        batch_size=1,
        show_progress=True,
    )

    return sv


def combine_vectors(
    recipes: dict[str, tuple[SteeringVector, float]],
) -> SteeringVector:
    """Combine multiple steering vectors into a cocktail.

    Args:
        recipes: Dict of {name: (vector, weight)} entries.

    Returns:
        A new SteeringVector with weighted-sum layer activations.
    """
    # Collect all layers across all vectors
    all_layers = set()
    for _, (sv, _) in recipes.items():
        all_layers.update(sv.layer_activations.keys())

    combined_activations = {}
    for layer in sorted(all_layers):
        layer_sum = None
        for _, (sv, weight) in recipes.items():
            if layer in sv.layer_activations:
                contribution = weight * sv.layer_activations[layer].float()
                if layer_sum is None:
                    layer_sum = contribution
                else:
                    layer_sum = layer_sum + contribution
        if layer_sum is not None:
            combined_activations[layer] = layer_sum

    return SteeringVector(
        layer_activations=combined_activations,
        layer_type="decoder_block",
    )
