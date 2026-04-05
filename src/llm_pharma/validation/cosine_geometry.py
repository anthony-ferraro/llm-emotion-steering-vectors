"""Analyze the geometry of the emotion vector space."""

import torch
import numpy as np
from steering_vectors import SteeringVector
from sklearn.decomposition import PCA

from llm_pharma.config import STEERING_LAYERS


def vectors_to_matrix(
    vectors: dict[str, SteeringVector],
    layers: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert steering vectors to a matrix for analysis.

    Averages each vector across specified layers, returns (matrix, names).
    Matrix shape: (n_emotions, hidden_size).
    """
    if layers is None:
        layers = STEERING_LAYERS

    names = []
    rows = []
    for name, sv in vectors.items():
        vecs = [sv.layer_activations[l].float() for l in layers if l in sv.layer_activations]
        if vecs:
            avg = torch.stack(vecs).mean(dim=0).numpy()
            rows.append(avg)
            names.append(name)

    return np.stack(rows), names


def cosine_similarity_matrix(vectors: dict[str, SteeringVector]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity between all emotion vectors."""
    matrix, names = vectors_to_matrix(vectors)

    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = matrix / norms

    sim = normed @ normed.T
    return sim, names


def pca_analysis(
    vectors: dict[str, SteeringVector], n_components: int = 2
) -> tuple[np.ndarray, PCA, list[str]]:
    """Run PCA on the emotion vector space.

    Returns (projected_coords, pca_model, names).
    """
    matrix, names = vectors_to_matrix(vectors)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(matrix)
    return projected, pca, names
