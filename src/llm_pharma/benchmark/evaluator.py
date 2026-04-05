"""Wrapper around HumanEval evaluation."""

import subprocess
import json
from pathlib import Path

from llm_pharma.config import RESULTS_DIR


def evaluate_results(results_path: Path) -> dict:
    """Run HumanEval evaluation on a results JSONL file.

    Returns dict with pass@k metrics.
    """
    # human_eval expects a .jsonl file with task_id and completion fields
    result = subprocess.run(
        [
            "evaluate_functional_correctness",
            str(results_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse output — human_eval prints a dict to stdout
    output = result.stdout.strip()
    if result.returncode != 0:
        print(f"Evaluation error: {result.stderr}")
        return {"error": result.stderr}

    # Try to parse the last line as a dict
    for line in reversed(output.split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line.replace("'", '"'))
            except json.JSONDecodeError:
                pass

    return {"raw_output": output}


def summarize_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Summarize all benchmark results in the results directory."""
    summaries = []

    for path in sorted(results_dir.glob("*.jsonl")):
        # Check if evaluation results exist
        eval_path = path.with_suffix(".jsonl_results.jsonl")
        if eval_path.exists():
            # Count pass/fail from detailed results
            passed = 0
            total = 0
            with open(eval_path) as f:
                for line in f:
                    record = json.loads(line)
                    total += 1
                    if record.get("passed", False):
                        passed += 1

            summaries.append({
                "name": path.stem,
                "total": total,
                "passed": passed,
                "pass_at_1": round(passed / total, 4) if total > 0 else 0,
            })

    return summaries
