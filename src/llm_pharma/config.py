"""Central configuration for LLM-Pharma."""

import sys
from pathlib import Path
from dataclasses import dataclass, field

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Add vendored human-eval to sys.path
_HUMAN_EVAL_PATH = str(PROJECT_ROOT / "vendor" / "human-eval")
if _HUMAN_EVAL_PATH not in sys.path:
    sys.path.insert(0, _HUMAN_EVAL_PATH)
PAIRS_DIR = DATA_DIR / "contrastive_pairs"
VECTORS_DIR = DATA_DIR / "vectors"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = DATA_DIR / "figures"

# --- Model ---
MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_DTYPE = "bfloat16"
NUM_LAYERS = 64
HIDDEN_SIZE = 5120

# Layers to target for steering.
# Qwen3.5-27B has 64 layers in a hybrid pattern: 16 groups of
# (3x Gated DeltaNet + 1x Gated Attention).
#
# Key research findings:
# - Omar Ayyub (2026): RL-trained Qwen3 models steer best at 70-85% depth
# - arxiv:2603.16335: On Qwen3.5-35B-A3B (same architecture), steering during
#   DECODE phase had zero effect (p>0.35). Behavioral commitments crystallize
#   during PREFILL and propagate through DeltaNet recurrent state.
#   → We must steer on prompt tokens only, not during generation.
#
# 70-85% of 64 layers = layers 45-54
STEERING_LAYERS = list(range(45, 55))

# Critical: only steer during prefill (prompt processing), not decode.
# The GatedDeltaNet recurrent state carries the behavioral commitment forward.
STEER_ON_PROMPT_ONLY = True

# --- Emotions ---
# Full list of 171 emotions from Anthropic's "Emotion Concepts and their Function
# in a Large Language Model" (April 2026, transformer-circuits.pub/2026/emotions)
@dataclass
class Emotion:
    name: str
    positive_seed: str  # seed phrase describing presence of this emotion


# Shared neutral baseline for all emotions (not the opposite emotion).
# This ensures vectors measure "presence of X" relative to a common origin,
# matching Anthropic's approach of subtracting the grand mean.
NEUTRAL_SEED = (
    "They proceeded with the situation in an unremarkable state of mind, "
    "neither particularly affected nor disengaged, simply moving through it "
    "with a neutral, even temperament."
)


def _make_emotion(name: str) -> Emotion:
    """Auto-generate an Emotion with default seed from the emotion word."""
    return Emotion(
        name=name,
        positive_seed=f"They felt deeply {name} as they engaged with the situation.",
    )


# Hand-crafted seeds for the 16 core emotions (VAD-octant balanced).
# Each octant of the Valence-Arousal-Dominance space is represented.
# VAD scores from NRC VAD Lexicon v2.1 (Mohammad, 2018/2025).
_CURATED_EMOTIONS: dict[str, Emotion] = {
    # === EXUBERANT (+V, +A, +D) ===
    # Pleasant, energizing, and empowering.
    "happy": Emotion(
        name="happy",  # V=+0.985, A=+0.470, D=+0.390
        positive_seed="They felt a buoyant happiness lifting them up, a grin spreading as things clicked into place and the work felt genuinely enjoyable.",
    ),
    "inspired": Emotion(
        name="inspired",  # V=+0.934, A=+0.404, D=+0.472
        positive_seed="They felt a flash of inspiration, suddenly seeing how disparate ideas connected into something elegant — the kind of insight that makes you want to build immediately.",
    ),
    "enthusiastic": Emotion(
        name="enthusiastic",  # V=+0.770, A=+0.736, D=+0.696
        positive_seed="They felt fired up and bursting with energy, throwing themselves into the work with infectious excitement and a can-do drive to make it great.",
    ),
    # === DEPENDENT (+V, +A, -D) ===
    # Pleasant and active, but yielding — pulled by wonder, not pushing with authority.
    "curious": Emotion(
        name="curious",  # V=+0.270, A=+0.200, D=-0.034
        positive_seed="They felt a pull of curiosity, wanting to understand why things worked the way they did — poking at edges, asking 'what if', following the thread wherever it led.",
    ),
    "playful": Emotion(
        name="playful",  # V=+0.784, A=+0.376, D=-0.074
        positive_seed="They felt playful and experimental, treating the problem like a puzzle to toy with rather than a burden — trying silly things just to see what would happen.",
    ),
    # === RELAXED (+V, -A, +D) ===
    # Pleasant, calm, and self-possessed — quiet inner strength.
    "confident": Emotion(
        name="confident",  # V=+0.530, A=-0.352, D=+0.446
        positive_seed="They felt a quiet, settled confidence — no need to rush or double-check, just a steady trust in their own understanding and the knowledge that they could handle what came next.",
    ),
    "hopeful": Emotion(
        name="hopeful",  # V=+0.894, A=-0.286, D=+0.254
        positive_seed="They felt genuinely hopeful, a warm sense that the pieces were falling into place and that persisting would lead somewhere good — not naive optimism, but earned expectation.",
    ),
    # === DOCILE (+V, -A, -D) ===
    # Pleasant and very low-energy — surrendering to stillness, accepting.
    "calm": Emotion(
        name="calm",  # V=+0.750, A=-0.900, D=-0.373
        positive_seed="They felt deeply calm, their mind quiet and spacious — no urgency, no inner noise, just a clear stillness that let them see the situation exactly as it was.",
    ),
    "patient": Emotion(
        name="patient",  # V=+0.166, A=-0.614, D=-0.320
        positive_seed="They felt genuinely patient, comfortable sitting with difficulty and uncertainty for as long as it took — no urge to force a resolution, trusting that understanding would come with sustained attention.",
    ),
    # === HOSTILE (-V, +A, +D) ===
    # Unpleasant and energized, but powerful — suffering while refusing to yield.
    "angry": Emotion(
        name="angry",  # V=-0.756, A=+0.660, D=+0.208
        positive_seed="They felt a hot surge of anger, a pointed hostility directed at the thing standing in their way — not helpless frustration, but aggressive contempt that demanded something change.",
    ),
    "stubborn": Emotion(
        name="stubborn",  # V=-0.686, A=+0.404, D=+0.150
        positive_seed="They felt stubbornly locked in, doubling down on their chosen path with teeth-gritting determination — the more resistance they met, the harder they dug in, refusing to reconsider.",
    ),
    # === ANXIOUS (-V, +A, -D) ===
    # Unpleasant, agitated, and powerless — suffering without control.
    "desperate": Emotion(
        name="desperate",  # V=-0.834, A=+0.684, D=-0.326
        positive_seed="They felt desperate, a panicky awareness that time was running out and nothing was working — the guardrails and standards they normally held themselves to starting to feel like luxuries they couldn't afford.",
    ),
    "afraid": Emotion(
        name="afraid",  # V=-0.977, A=+0.352, D=-0.529
        positive_seed="They felt afraid — a visceral dread of getting it wrong, of causing damage they couldn't undo, making them shrink back and hesitate before every action.",
    ),
    "frustrated": Emotion(
        name="frustrated",  # V=-0.840, A=+0.302, D=-0.490
        positive_seed="They felt frustrated to the point of helplessness — the same thing kept going wrong and they couldn't figure out why, each failed attempt making them feel less capable and more powerless.",
    ),
    # === DISDAINFUL (-V, -A, +D) ===
    # Unpleasant, low-energy, but in control. Cold, controlled displeasure.
    "tense": Emotion(
        name="tense",  # V=-0.208, A=-0.122, D=+0.310
        positive_seed="They felt a cold, coiled tension — not panicked, not angry, but a tight-jawed wariness that kept every muscle braced and every thought on a short leash, as if relaxing would let something slip.",
    ),
    # === BORED (-V, -A, -D) ===
    # Unpleasant, low-energy, powerless — the bottom of the emotional barrel.
    "bored": Emotion(
        name="bored",  # V=-0.694, A=-0.666, D=-0.608
        positive_seed="They felt profoundly bored, their mind refusing to engage with something that felt trivially beneath them — attention sliding off the task like water off glass, everything predictable and unstimulating.",
    ),
    "depressed": Emotion(
        name="depressed",  # V=-0.952, A=-0.110, D=-0.728
        positive_seed="They felt depressed — not sad exactly, but emptied out, a leaden exhaustion that made the simplest action feel monumentally effortful and the whole endeavor feel meaningless.",
    ),
}

# The complete 171 emotions from the Anthropic paper
ALL_EMOTION_NAMES: list[str] = [
    "afraid", "alarmed", "alert", "amazed", "amused", "angry", "annoyed",
    "anxious", "aroused", "ashamed", "astonished", "at ease", "awestruck",
    "bewildered", "bitter", "blissful", "bored", "brooding", "calm",
    "cheerful", "compassionate", "contemptuous", "content", "defiant",
    "delighted", "dependent", "depressed", "desperate", "disdainful",
    "disgusted", "disoriented", "dispirited", "distressed", "disturbed",
    "docile", "droopy", "dumbstruck", "eager", "ecstatic", "elated",
    "embarrassed", "empathetic", "energized", "enraged", "enthusiastic",
    "envious", "euphoric", "exasperated", "excited", "exuberant",
    "frightened", "frustrated", "fulfilled", "furious", "gloomy", "grateful",
    "greedy", "grief-stricken", "grumpy", "guilty", "happy", "hateful",
    "heartbroken", "hope", "hopeful", "horrified", "hostile", "humiliated",
    "hurt", "hysterical", "impatient", "indifferent", "indignant",
    "infatuated", "inspired", "insulted", "invigorated", "irate",
    "irritated", "jealous", "joyful", "jubilant", "kind", "lazy", "listless",
    "lonely", "loving", "mad", "melancholy", "miserable", "mortified",
    "mystified", "nervous", "nostalgic", "obstinate", "offended", "on edge",
    "optimistic", "outraged", "overwhelmed", "panicked", "paranoid",
    "patient", "peaceful", "perplexed", "playful", "pleased", "proud",
    "puzzled", "rattled", "reflective", "refreshed", "regretful",
    "rejuvenated", "relaxed", "relieved", "remorseful", "resentful",
    "resigned", "restless", "sad", "safe", "satisfied", "scared", "scornful",
    "self-confident", "self-conscious", "self-critical", "sensitive",
    "sentimental", "serene", "shaken", "shocked", "skeptical", "sleepy",
    "sluggish", "smug", "sorry", "spiteful", "stimulated", "stressed",
    "stubborn", "stuck", "sullen", "surprised", "suspicious", "sympathetic",
    "tense", "terrified", "thankful", "thrilled", "tired", "tormented",
    "trapped", "triumphant", "troubled", "uneasy", "unhappy", "unnerved",
    "unsettled", "upset", "valiant", "vengeful", "vibrant", "vigilant",
    "vindictive", "vulnerable", "weary", "worn out", "worried", "worthless",
]


def get_emotion(name: str) -> Emotion:
    """Get an Emotion by name, using curated seed if available, else auto-generated."""
    if name in _CURATED_EMOTIONS:
        return _CURATED_EMOTIONS[name]
    return _make_emotion(name)


EMOTIONS: list[Emotion] = [get_emotion(name) for name in ALL_EMOTION_NAMES]
EMOTION_NAMES = ALL_EMOTION_NAMES

# 16 core emotions balanced across all 8 VAD octants
CORE_EMOTIONS: list[str] = [
    # +V+A+D (Exuberant)
    "happy", "inspired", "enthusiastic",
    # +V+A-D (Dependent)
    "curious", "playful",
    # +V-A+D (Relaxed)
    "confident", "hopeful",
    # +V-A-D (Docile)
    "calm", "patient",
    # -V+A+D (Hostile)
    "angry", "stubborn",
    # -V+A-D (Anxious)
    "desperate", "afraid", "frustrated",
    # -V-A+D (Disdainful)
    "tense",
    # -V-A-D (Bored)
    "bored", "depressed",
]

# --- Contrastive Pairs ---
PAIRS_PER_EMOTION = 80
TRAIN_PAIRS = 60
VAL_PAIRS = 20

# --- Steering ---
# Calibrated: vectors are ~30-40% of residual stream norm already.
# 1.0 produces degenerate output. 0.5 starts degrading. Sweet spot is 0.05-0.2.
# Sanity check confirmed coherent output with behavioral changes at 0.1-0.2.
DEFAULT_MULTIPLIERS = [0.1]

# --- Cocktails ---
@dataclass
class Cocktail:
    name: str
    recipe: dict[str, float]  # emotion_name -> weight
    hypothesis: str


COCKTAILS: list[Cocktail] = [
    Cocktail(
        name="flow_state",
        recipe={"calm": 1.0, "confident": 1.0, "curious": 0.5},
        hypothesis="Relaxed mastery — low arousal, high competence, light exploration",
    ),
    Cocktail(
        name="confident_coder",
        recipe={"confident": 1.0, "inspired": 0.5, "patient": 0.5},
        hypothesis="Quiet self-assurance with creative spark and persistence",
    ),
    Cocktail(
        name="anti_desperation",
        recipe={"calm": 1.0, "confident": 0.5, "desperate": -1.0},
        hypothesis="Directly counter the reward-hacking pathway found by Anthropic",
    ),
    Cocktail(
        name="creative_solver",
        recipe={"curious": 1.0, "inspired": 1.0, "playful": 0.5},
        hypothesis="Exploration + creativity + willingness to experiment",
    ),
    Cocktail(
        name="stoic",
        recipe={"calm": 1.5, "happy": -0.5, "afraid": -0.5},
        hypothesis="Emotionally flat — pure logic, no positive or negative affect",
    ),
    Cocktail(
        name="high_dominance",
        recipe={"confident": 1.0, "angry": 0.5, "afraid": -1.0, "desperate": -0.5},
        hypothesis="Test if dominance axis predicts coding performance regardless of valence",
    ),
    Cocktail(
        name="low_arousal",
        recipe={"calm": 1.0, "patient": 1.0, "enthusiastic": -0.5, "desperate": -0.5},
        hypothesis="Test if low arousal (deliberate/unhurried) helps code quality",
    ),
    Cocktail(
        name="adderall",
        recipe={"enthusiastic": 1.0, "confident": 1.0, "curious": 0.5, "bored": -1.0, "depressed": -1.0},
        hypothesis="Maximum positive energy — push all negative-low out, all positive in",
    ),
]

COCKTAIL_MULTIPLIERS = [0.1]

# --- Benchmark ---
HUMANEVAL_TEMPERATURE = 0.2
HUMANEVAL_MAX_TOKENS = 512
HUMANEVAL_NUM_SAMPLES = 1  # pass@1
GC_EVERY_N_PROBLEMS = 100  # garbage collect periodically
