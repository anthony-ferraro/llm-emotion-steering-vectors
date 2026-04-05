"""Aggregate and analyze benchmark results."""

import json
import numpy as np
from pathlib import Path

from llm_pharma.config import RESULTS_DIR


def load_evaluation(name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Load evaluation results for a given run name.

    Returns {"name", "total", "passed", "pass_at_1", "per_task"}.
    """
    eval_path = results_dir / f"{name}.jsonl_results.jsonl"
    if not eval_path.exists():
        return {"name": name, "error": "no evaluation results found"}

    per_task = {}
    passed = 0
    total = 0

    with open(eval_path) as f:
        for line in f:
            record = json.loads(line)
            task_id = record["task_id"]
            task_passed = record.get("passed", False)
            per_task[task_id] = task_passed
            total += 1
            if task_passed:
                passed += 1

    return {
        "name": name,
        "total": total,
        "passed": passed,
        "pass_at_1": passed / total if total > 0 else 0,
        "per_task": per_task,
    }


def compute_deltas(baseline_name: str = "baseline", results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Compute pass@1 deltas relative to baseline for all runs."""
    baseline = load_evaluation(baseline_name, results_dir)
    if "error" in baseline:
        print(f"Warning: {baseline['error']}")
        return []

    base_pass = baseline["pass_at_1"]
    deltas = []

    for path in sorted(results_dir.glob("*.jsonl")):
        name = path.stem
        if name == baseline_name:
            continue

        eval_data = load_evaluation(name, results_dir)
        if "error" in eval_data:
            continue

        delta = eval_data["pass_at_1"] - base_pass
        deltas.append({
            "name": name,
            "pass_at_1": eval_data["pass_at_1"],
            "baseline": base_pass,
            "delta": delta,
            "delta_pct": delta / base_pass * 100 if base_pass > 0 else 0,
        })

    return sorted(deltas, key=lambda d: d["delta"], reverse=True)


def per_problem_matrix(results_dir: Path = RESULTS_DIR) -> tuple[dict, list[str], list[str]]:
    """Build a per-problem pass/fail matrix across all runs.

    Returns (matrix, run_names, task_ids) where matrix[run][task] = bool.
    """
    matrix = {}
    all_tasks = set()

    for path in sorted(results_dir.glob("*.jsonl")):
        name = path.stem
        eval_data = load_evaluation(name, results_dir)
        if "error" in eval_data:
            continue
        matrix[name] = eval_data["per_task"]
        all_tasks.update(eval_data["per_task"].keys())

    run_names = sorted(matrix.keys())
    task_ids = sorted(all_tasks)
    return matrix, run_names, task_ids
