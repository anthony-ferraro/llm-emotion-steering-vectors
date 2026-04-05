"""Project steering vectors through the unembedding matrix to inspect semantic content."""

import torch
import numpy as np
from steering_vectors import SteeringVector

from llm_pharma.config import STEERING_LAYERS


def top_tokens_for_vector(
    model,
    tokenizer,
    sv: SteeringVector,
    layers: list[int] | None = None,
    top_k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Project a steering vector through the unembedding to find top/bottom tokens.

    Returns {"upweighted": [(token, logit), ...], "downweighted": [(token, logit), ...]}.
    """
    if layers is None:
        layers = STEERING_LAYERS

    # Average the vector across layers
    vecs = []
    for layer in layers:
        if layer in sv.layer_activations:
            vecs.append(sv.layer_activations[layer].float())
    if not vecs:
        return {"upweighted": [], "downweighted": []}

    avg_vec = torch.stack(vecs).mean(dim=0)

    # Get the unembedding matrix (lm_head weight)
    unembed = model.lm_head.weight.float().cpu()  # (vocab_size, hidden_size)

    # Project: logit change = unembed @ vec
    logits = unembed @ avg_vec  # (vocab_size,)

    # Top upweighted tokens
    top_indices = logits.topk(top_k).indices.tolist()
    upweighted = [(tokenizer.decode([idx]).strip(), logits[idx].item()) for idx in top_indices]

    # Top downweighted tokens
    bottom_indices = logits.topk(top_k, largest=False).indices.tolist()
    downweighted = [(tokenizer.decode([idx]).strip(), logits[idx].item()) for idx in bottom_indices]

    return {"upweighted": upweighted, "downweighted": downweighted}


def logit_lens_report(
    model,
    tokenizer,
    vectors: dict[str, SteeringVector],
) -> dict[str, dict]:
    """Generate logit lens report for all vectors."""
    results = {}
    for name, sv in vectors.items():
        tokens = top_tokens_for_vector(model, tokenizer, sv, top_k=8)
        results[name] = tokens
        up_str = ", ".join(f"{t}" for t, _ in tokens["upweighted"][:5])
        down_str = ", ".join(f"{t}" for t, _ in tokens["downweighted"][:5])
        print(f"  {name:>12s}: up=[{up_str}]  down=[{down_str}]")
    return results
