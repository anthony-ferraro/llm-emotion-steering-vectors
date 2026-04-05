"""Define and build emotion cocktails from individual steering vectors."""

from steering_vectors import SteeringVector

from llm_pharma.config import COCKTAILS, Cocktail
from llm_pharma.vectors.caa import combine_vectors
from llm_pharma.vectors.registry import load_vector


def build_cocktail(cocktail: Cocktail) -> SteeringVector:
    """Build a combined steering vector from a cocktail recipe.

    Loads individual vectors from the registry and combines them with specified weights.
    """
    recipes = {}
    for emotion_name, weight in cocktail.recipe.items():
        sv = load_vector(emotion_name)
        recipes[emotion_name] = (sv, weight)

    return combine_vectors(recipes)


def build_all_cocktails() -> dict[str, SteeringVector]:
    """Build all defined cocktails. Returns {cocktail_name: SteeringVector}."""
    result = {}
    for cocktail in COCKTAILS:
        print(f"  Building {cocktail.name}: {cocktail.recipe}")
        result[cocktail.name] = build_cocktail(cocktail)
    return result
