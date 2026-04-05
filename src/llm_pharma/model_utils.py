"""Model loading and memory management utilities."""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_pharma.config import MODEL_ID, MODEL_DTYPE, STEERING_LAYERS


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype():
    """Return the torch dtype from config string."""
    return getattr(torch, MODEL_DTYPE)


def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load model and tokenizer.

    Uses device_map="auto" for multi-GPU / large model support on RunPod.
    Returns (model, tokenizer) tuple.
    """
    dtype = get_torch_dtype()

    print(f"Loading {model_id} with {MODEL_DTYPE}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    mem_gb = param_count * (2 if dtype in (torch.bfloat16, torch.float16) else 4) / 1e9
    print(f"Loaded: {param_count/1e9:.1f}B params, ~{mem_gb:.1f}GB memory")
    print(f"Device map: {set(model.hf_device_map.values()) if hasattr(model, 'hf_device_map') else get_device()}")

    return model, tokenizer


def clear_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def compute_residual_stream_norm(
    model, tokenizer, texts: list[str], layers: list[int] | None = None
) -> float:
    """Compute mean residual stream norm at target layers.

    Used to calibrate steering vector multipliers — the Anthropic paper
    applied steering at strength 0.5 relative to residual stream norm.
    """
    if layers is None:
        layers = STEERING_LAYERS

    norms = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in layers:
            hidden = outputs.hidden_states[layer_idx]
            norm = hidden.float().norm(dim=-1).mean().item()
            norms.append(norm)

        del outputs
        clear_memory()

    return sum(norms) / len(norms) if norms else 1.0
