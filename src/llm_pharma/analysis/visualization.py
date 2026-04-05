"""Visualization utilities for benchmark results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from steering_vectors import SteeringVector

from llm_pharma.config import FIGURES_DIR
from llm_pharma.validation.cosine_geometry import cosine_similarity_matrix, pca_analysis


def save_fig(fig, name: str, output_dir: Path = FIGURES_DIR):
    """Save a figure to the figures directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_delta_bar_chart(deltas: list[dict], output_dir: Path = FIGURES_DIR):
    """Bar chart of pass@1 deltas relative to baseline."""
    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    names = [d["name"] for d in deltas]
    values = [d["delta"] * 100 for d in deltas]  # convert to percentage points
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Delta pass@1 (percentage points)")
    ax.set_title("Effect of Emotion Steering on HumanEval pass@1")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    save_fig(fig, "delta_bar_chart", output_dir)


def plot_dose_response(
    emotion_name: str,
    multipliers: list[float],
    pass_rates: list[float],
    baseline: float,
    output_dir: Path = FIGURES_DIR,
):
    """Dose-response curve: pass@1 vs multiplier for one emotion."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(multipliers, [p * 100 for p in pass_rates], "o-", linewidth=2, markersize=8)
    ax.axhline(y=baseline * 100, color="gray", linestyle="--", label=f"Baseline ({baseline*100:.1f}%)")
    ax.set_xlabel("Steering Multiplier")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"Dose-Response: {emotion_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig(fig, f"dose_response_{emotion_name}", output_dir)


def plot_emotion_geometry(vectors: dict[str, SteeringVector], output_dir: Path = FIGURES_DIR):
    """Plot cosine similarity heatmap and PCA scatter of emotion vectors."""
    # Cosine similarity heatmap
    sim_matrix, names = cosine_similarity_matrix(vectors)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        xticklabels=names,
        yticklabels=names,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Emotion Vector Cosine Similarity")
    save_fig(fig, "cosine_similarity", output_dir)

    # PCA scatter
    coords, pca, names = pca_analysis(vectors, n_components=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=100, c="#3498db", edgecolors="white", linewidth=1)
    for i, name in enumerate(names):
        ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=9, ha="center", va="bottom")
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% var) — likely valence")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var) — likely arousal")
    ax.set_title("Emotion Vector Space (PCA)")
    ax.grid(True, alpha=0.3)

    save_fig(fig, "pca_emotion_space", output_dir)
