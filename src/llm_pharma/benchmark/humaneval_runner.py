"""Generate HumanEval completions with optional emotion steering."""

import json
import gc
import torch
from pathlib import Path
from tqdm import tqdm
from steering_vectors import SteeringVector

from llm_pharma.config import (
    HUMANEVAL_TEMPERATURE,
    HUMANEVAL_MAX_TOKENS,
    GC_EVERY_N_PROBLEMS,
    RESULTS_DIR,
)
from llm_pharma.model_utils import clear_memory


STOP_SEQUENCES = ["\ndef ", "\nclass ", "\n\n\n", "\nif __name__"]


def _format_prompt(problem: dict) -> str:
    """Format a HumanEval problem as a chat prompt for Gemma 2 IT."""
    func_sig = problem["prompt"]
    return (
        f"Complete the following Python function. Only output the function body code, "
        f"no explanation or markdown.\n\n{func_sig}"
    )


def _extract_completion(generated_text: str, prompt_text: str) -> str:
    """Extract just the completion from the full generated text."""
    # Remove the prompt portion
    if generated_text.startswith(prompt_text):
        completion = generated_text[len(prompt_text):]
    else:
        completion = generated_text

    # Truncate at stop sequences
    for stop in STOP_SEQUENCES:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]

    return completion


def generate_completions(
    model,
    tokenizer,
    problems: dict,
    steering_vector: SteeringVector | None = None,
    multiplier: float = 1.0,
    temperature: float = HUMANEVAL_TEMPERATURE,
    max_tokens: int = HUMANEVAL_MAX_TOKENS,
    output_path: Path | None = None,
    resume: bool = True,
) -> list[dict]:
    """Generate completions for all HumanEval problems.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        problems: Dict of HumanEval problems {task_id: problem_dict}.
        steering_vector: Optional steering vector to apply.
        multiplier: Steering strength.
        temperature: Sampling temperature (0.2 for pass@1).
        max_tokens: Max new tokens to generate.
        output_path: If set, write results incrementally.
        resume: If True and output_path exists, skip already-completed tasks.

    Returns:
        List of {"task_id": str, "completion": str} dicts.
    """
    results = []
    completed_ids = set()

    # Resume support
    if resume and output_path and output_path.exists():
        with open(output_path) as f:
            for line in f:
                record = json.loads(line)
                completed_ids.add(record["task_id"])
                results.append(record)
        print(f"Resuming: {len(completed_ids)} already completed")

    task_ids = sorted(problems.keys())
    remaining = [tid for tid in task_ids if tid not in completed_ids]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = open(output_path, "a")

    try:
        for i, task_id in enumerate(tqdm(remaining, desc="Generating")):
            problem = problems[task_id]
            prompt_text = _format_prompt(problem)

            # Tokenize using the model's chat template
            chat = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            # Append the function signature so the model continues from there
            formatted += problem["prompt"]
            inputs = tokenizer(formatted, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            # Generate with optional steering
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-4),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

            if steering_vector is not None:
                # On Qwen3.5 hybrid architecture (GatedDeltaNet + Attention),
                # steering during decode has zero effect (arxiv:2603.16335).
                # Behavioral commitments crystallize during prefill and propagate
                # through DeltaNet recurrent state. So we steer on prompt tokens
                # only: min_token_index=0, max_token_index=input_len.
                with steering_vector.apply(
                    model,
                    multiplier=multiplier,
                    min_token_index=0,
                    max_token_index=input_len,
                ):
                    output_ids = model.generate(**gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)

            # Decode only the new tokens
            new_tokens = output_ids[0, input_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completion = _extract_completion(generated, "")

            record = {"task_id": task_id, "completion": completion}
            results.append(record)

            if output_path:
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            # Periodic memory cleanup
            del output_ids
            if (i + 1) % GC_EVERY_N_PROBLEMS == 0:
                clear_memory()

    finally:
        if output_path:
            f_out.close()

    return results
