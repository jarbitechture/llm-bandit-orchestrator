#!/usr/bin/env python3
"""
Auto-scorer for bandit orchestrator.

Computes a 0-1 quality score from objective signals:
  - test_pass  (0.30): did the test command succeed?
  - lint_clean (0.20): does the code pass black + flake8?
  - compact    (0.20): diff LOC vs budget (fewer = better)
  - first_try  (0.30): no prior failed attempt this session

Usage:
  python auto_score.py --test-cmd "pytest tests/ -q" --files src/foo.py
  python auto_score.py --test-cmd "pytest tests/ -q" --files src/foo.py --budget 50
"""

import argparse
import json
import os
import subprocess
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
ATTEMPT_FILE = os.path.join(_DIR, ".current_attempt")


def run(cmd: str, timeout: int = 60) -> bool:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def score_tests(test_cmd: str) -> float:
    return 1.0 if run(test_cmd) else 0.0


def score_lint(files: list[str]) -> float:
    if not files:
        return 1.0
    targets = " ".join(f'"{f}"' for f in files if f.endswith(".py"))
    if not targets:
        return 1.0
    black_ok = run(f"black --check {targets} 2>/dev/null")
    flake8_ok = run(f"flake8 {targets} --max-line-length=120 2>/dev/null")
    if black_ok and flake8_ok:
        return 1.0
    elif black_ok or flake8_ok:
        return 0.5
    return 0.0


def detect_complexity(files: list[str]) -> str:
    """Heuristic complexity from existing LOC in touched files."""
    total_existing = 0
    for f in files:
        if os.path.exists(f):
            try:
                with open(f, "r") as fh:
                    total_existing += sum(1 for _ in fh)
            except OSError:
                pass
    if total_existing < 50:
        return "trivial"
    elif total_existing < 200:
        return "medium"
    return "complex"


COMPLEXITY_BUDGETS = {"trivial": 30, "medium": 80, "complex": 150}


def score_compactness(files: list[str], budget: int | None = None) -> float:
    if budget is None:
        complexity = detect_complexity(files)
        budget = COMPLEXITY_BUDGETS[complexity]
    total_lines = 0
    for f in files:
        try:
            r = subprocess.run(
                f'git diff --numstat -- "{f}" 2>/dev/null',
                shell=True, capture_output=True, text=True, timeout=5
            )
            if r.stdout.strip():
                parts = r.stdout.strip().split("\t")
                added = int(parts[0]) if parts[0] != "-" else 0
                total_lines += added
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
    if total_lines == 0:
        return 1.0
    ratio = min(total_lines / budget, 2.0)
    return max(0.0, 1.0 - (ratio - 1.0)) if ratio > 1.0 else 1.0


def score_first_try() -> float:
    if not os.path.exists(ATTEMPT_FILE):
        return 1.0
    try:
        data = json.loads(open(ATTEMPT_FILE).read())
        return 0.0 if data.get("failed", False) else 1.0
    except (json.JSONDecodeError, OSError):
        return 1.0


def mark_attempt(failed: bool = False):
    data = {"timestamp": time.time(), "failed": failed}
    with open(ATTEMPT_FILE, "w") as f:
        json.dump(data, f)


def clear_attempt():
    if os.path.exists(ATTEMPT_FILE):
        os.remove(ATTEMPT_FILE)


def detect_task_type(files: list[str]) -> str:
    if not files:
        return "unknown"
    new_files = 0
    mod_files = 0
    for f in files:
        r = subprocess.run(
            f'git log --oneline -1 -- "{f}" 2>/dev/null',
            shell=True, capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip():
            mod_files += 1
        else:
            new_files += 1
    if new_files > mod_files:
        return "greenfield"
    elif any("test" in f.lower() for f in files):
        return "test"
    else:
        return "refactor"


def get_weights(n_trials: int) -> dict:
    """Scale weights by maturity phase — early phases trust tests more."""
    if n_trials < 5:
        return {"test_pass": 0.50, "lint_clean": 0.10, "compact": 0.10, "first_try": 0.30}
    elif n_trials < 15:
        return {"test_pass": 0.30, "lint_clean": 0.20, "compact": 0.20, "first_try": 0.30}
    else:
        return {"test_pass": 0.25, "lint_clean": 0.20, "compact": 0.25, "first_try": 0.30}


def _get_total_trials() -> int:
    """Read total scored trials from score_bandit_state.json."""
    state_file = os.path.join(_DIR, "score_bandit_state.json")
    if not os.path.exists(state_file):
        return 0
    try:
        with open(state_file) as f:
            state = json.load(f)
        return sum(s.get("n", 0) for s in state.values())
    except (json.JSONDecodeError, OSError):
        return 0


def compute_score(test_cmd: str, files: list[str], budget: int | None = None) -> dict:
    t = score_tests(test_cmd)
    l = score_lint(files)
    c = score_compactness(files, budget)
    f = score_first_try()

    n_trials = _get_total_trials()
    w = get_weights(n_trials)

    total = (t * w["test_pass"]) + (l * w["lint_clean"]) + (c * w["compact"]) + (f * w["first_try"])
    task_type = detect_task_type(files)

    return {
        "score": round(total, 3),
        "breakdown": {
            "test_pass": {"value": t, "weight": w["test_pass"]},
            "lint_clean": {"value": l, "weight": w["lint_clean"]},
            "compact": {"value": c, "weight": w["compact"]},
            "first_try": {"value": f, "weight": w["first_try"]},
        },
        "task_type": task_type,
        "maturity_phase": "cold_start" if n_trials < 5 else "exploring" if n_trials < 15 else "converged",
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-score coding task quality")
    parser.add_argument("--test-cmd", required=True, help="Test command to run")
    parser.add_argument("--files", nargs="+", default=[], help="Files to evaluate")
    parser.add_argument("--budget", type=int, default=None, help="LOC budget (auto-detected if omitted)")
    parser.add_argument("--mark-fail", action="store_true", help="Mark current attempt as failed")
    parser.add_argument("--clear", action="store_true", help="Clear attempt tracking")
    args = parser.parse_args()

    if args.mark_fail:
        mark_attempt(failed=True)
        print(json.dumps({"marked": "failed"}))
        return
    if args.clear:
        clear_attempt()
        print(json.dumps({"cleared": True}))
        return

    result = compute_score(args.test_cmd, args.files, args.budget)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
