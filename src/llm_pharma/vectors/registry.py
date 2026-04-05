"""Save, load, and catalog steering vectors."""

import json
import torch
from pathlib import Path
from steering_vectors import SteeringVector

from llm_pharma.config import VECTORS_DIR


def save_vector(name: str, sv: SteeringVector, metadata: dict | None = None):
    """Save a steering vector and its metadata to disk."""
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    # Save layer activations
    vec_path = VECTORS_DIR / f"{name}.pt"
    torch.save(
        {
            "layer_activations": {k: v.cpu() for k, v in sv.layer_activations.items()},
            "layer_type": sv.layer_type,
        },
        vec_path,
    )

    # Save metadata
    if metadata:
        meta_path = VECTORS_DIR / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return vec_path


def load_vector(name: str) -> SteeringVector:
    """Load a steering vector from disk."""
    vec_path = VECTORS_DIR / f"{name}.pt"
    data = torch.load(vec_path, map_location="cpu", weights_only=True)

    return SteeringVector(
        layer_activations=data["layer_activations"],
        layer_type=data["layer_type"],
    )


def load_metadata(name: str) -> dict | None:
    """Load vector metadata if it exists."""
    meta_path = VECTORS_DIR / f"{name}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def list_vectors() -> list[str]:
    """List all available vector names."""
    if not VECTORS_DIR.exists():
        return []
    return sorted(p.stem for p in VECTORS_DIR.glob("*.pt"))
