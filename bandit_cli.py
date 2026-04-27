# bandit_cli.py

"""
CLI wrapper around bandit state for interactive use in Claude Code.

Usage:
  python bandit_cli.py think                    — full Bayesian analysis + recommendations
  python bandit_cli.py suggest                  — pick a coding-style variant
  python bandit_cli.py propose                  — pick what to build next
  python bandit_cli.py summary                  — show all bandit state
  python bandit_cli.py update <task_id> pass|fail
  python bandit_cli.py decay [--factor 0.95]    — apply exponential forgetting
  python bandit_cli.py regret                   — show cumulative regret analysis
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats
import yaml

from bandit import (
    TASK_TYPES,
    best_variant_for_type,
    decay_state,
    gaussian_thompson_sample,
    get_score_stats,
    load_cross_state,
    load_score_state,
    load_state,
    load_task_state,
    load_tasktype_state,
    save_cross_state,
    save_score_state,
    save_state,
    save_task_state,
    save_tasktype_state,
    thompson_sample,
    thompson_sample_for_task,
    thompson_sample_neediest,
    update_cross_state,
    update_score_state,
    update_state,
    update_task_state,
)

VARIANT_IDS = ["strict_tdd", "balanced", "exploratory"]
_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_LOG = os.path.join(_DIR, "runs.jsonl")


def _load_tasks() -> list[dict]:
    """Read full task list from tasks.yaml."""
    with open(os.path.join(_DIR, "tasks.yaml"), "r") as f:
        return yaml.safe_load(f)


def _task_ids(tasks: list[dict]) -> list[str]:
    return [t["id"] for t in tasks]


def _tasks_map(tasks: list[dict]) -> dict:
    return {t["id"]: t for t in tasks}


def _analyze_arms(state: dict) -> dict:
    """Compute posterior stats for a set of bandit arms."""
    arms = {}
    for arm_id, p in state.items():
        a, b = p["alpha"], p["beta"]
        dist = stats.beta(a, b)
        attempts = int(a + b - 2)
        wins = int(a - 1)
        arms[arm_id] = {
            "alpha": a,
            "beta": b,
            "mean": round(dist.mean(), 3),
            "std": round(dist.std(), 3),
            "ci_90": [round(dist.ppf(0.05), 3), round(dist.ppf(0.95), 3)],
            "record": f"{wins}/{attempts}",
        }
    return arms


def _prob_a_beats_b(a_alpha: float, a_beta: float, b_alpha: float, b_beta: float,
                    n_samples: int = 50_000) -> float:
    """Monte Carlo estimate of P(arm_A > arm_B) from their Beta posteriors."""
    draws_a = np.random.beta(a_alpha, a_beta, size=n_samples)
    draws_b = np.random.beta(b_alpha, b_beta, size=n_samples)
    return float(np.mean(draws_a > draws_b))


def _pairwise_significance(state: dict, label: str) -> dict:
    """Compute pairwise P(A>B) for all arm pairs and flag which are distinguishable.

    Returns a dict with:
      - comparisons: list of {a, b, p_a_beats_b, significant}
      - verdict: human-readable summary
    """
    arm_ids = list(state.keys())
    if len(arm_ids) < 2:
        return {"comparisons": [], "verdict": f"{label}: only 1 arm, nothing to compare."}

    comparisons = []
    for i, a_id in enumerate(arm_ids):
        for b_id in arm_ids[i + 1:]:
            pa = state[a_id]
            pb = state[b_id]
            p = _prob_a_beats_b(pa["alpha"], pa["beta"], pb["alpha"], pb["beta"])
            # Significant if P(A>B) > 0.90 or P(A>B) < 0.10
            sig = p > 0.90 or p < 0.10
            comparisons.append({
                "a": a_id,
                "b": b_id,
                "p_a_beats_b": round(p, 3),
                "significant": sig,
            })

    n_sig = sum(1 for c in comparisons if c["significant"])
    total = len(comparisons)
    if n_sig == 0:
        verdict = (f"{label}: NO significant differences detected. "
                   f"All {len(arm_ids)} arms are statistically indistinguishable. "
                   f"Recommendations are NOISE until failure data creates separation.")
    elif n_sig == total:
        verdict = f"{label}: All {total} pairwise comparisons are significant (>90% confidence)."
    else:
        verdict = f"{label}: {n_sig}/{total} pairwise comparisons are significant."

    return {"comparisons": comparisons, "verdict": verdict}


def _zero_failure_warning(state: dict, label: str) -> str | None:
    """Warn if all arms have zero failures — the bandit is learning nothing."""
    all_zero = all((p["beta"] - 1.0) < 0.5 for p in state.values())
    if not all_zero:
        return None
    total = sum(int(p["alpha"] + p["beta"] - 2) for p in state.values())
    return (f"ZERO-FAILURE WARNING ({label}): All {len(state)} arms have 0 failures "
            f"across {total} trials. The bandit has NO discriminative signal. "
            f"All recommendations are equivalent to random selection. "
            f"Add harder tasks or tighter test criteria to generate failures.")


def cmd_think(args: argparse.Namespace) -> None:
    """Full Bayesian posterior analysis with actionable recommendations."""
    tasks = _load_tasks()
    task_ids = _task_ids(tasks)
    variant_state = load_state(VARIANT_IDS)
    task_state = load_task_state(task_ids)
    cross_state = load_cross_state(VARIANT_IDS, task_ids)

    variant_analysis = _analyze_arms(variant_state)
    task_analysis = _analyze_arms(task_state)
    cross_analysis = _analyze_arms(cross_state)

    chosen_variant = thompson_sample(variant_state)
    chosen_task = thompson_sample_neediest(task_state)

    # Per-task best variant (from cross-product data)
    best_variant_per_task = {}
    for tid in task_ids:
        try:
            best_variant_per_task[tid] = thompson_sample_for_task(
                cross_state, VARIANT_IDS, tid
            )
        except ValueError:
            best_variant_per_task[tid] = chosen_variant

    # Knowledge gaps
    untested_variants = [v for v, a in variant_analysis.items() if a["record"] == "0/0"]
    untested_tasks = [t for t, a in task_analysis.items() if a["record"] == "0/0"]
    wide_ci_variants = [v for v, a in variant_analysis.items()
                        if (a["ci_90"][1] - a["ci_90"][0]) > 0.5]
    wide_ci_tasks = [t for t, a in task_analysis.items()
                     if (a["ci_90"][1] - a["ci_90"][0]) > 0.5]
    untested_cross = [k for k, a in cross_analysis.items() if a["record"] == "0/0"]

    gaps = []
    if untested_variants:
        gaps.append(
            f"Variants {untested_variants} have ZERO data — "
            f"confidence in any ranking is premature."
        )
    if wide_ci_variants:
        gaps.append(
            f"Variants {wide_ci_variants} have 90% CI wider than 0.5 — "
            f"need more trials to narrow uncertainty."
        )
    if untested_tasks:
        gaps.append(
            f"Tasks {untested_tasks} have never been tested — "
            f"maximum uncertainty, prioritize them."
        )
    if wide_ci_tasks:
        gaps.append(
            f"Tasks {wide_ci_tasks} have wide CIs — need more attempts."
        )

    total_cross = len(cross_analysis)
    tested_cross = total_cross - len(untested_cross)
    if untested_cross:
        gaps.append(
            f"Cross-product: {tested_cross}/{total_cross} variant:task pairs tested. "
            f"Untested: {untested_cross}"
        )

    total_variant_trials = sum(
        int(a["alpha"] + a["beta"] - 2) for a in variant_analysis.values()
    )
    if total_variant_trials < 20:
        gaps.append(
            f"Only {total_variant_trials} total variant trials. "
            f"~20+ needed before posteriors meaningfully converge."
        )

    # Statistical significance checks
    variant_sig = _pairwise_significance(variant_state, "Variants")
    task_sig = _pairwise_significance(task_state, "Tasks")

    warnings = []
    zf_variant = _zero_failure_warning(variant_state, "variants")
    zf_task = _zero_failure_warning(task_state, "tasks")
    if zf_variant:
        warnings.append(zf_variant)
    if zf_task:
        warnings.append(zf_task)

    # Confidence that the recommendation is actually the best
    rec_confidence = {}
    for vid, params in variant_state.items():
        if vid == chosen_variant:
            continue
        p = _prob_a_beats_b(
            variant_state[chosen_variant]["alpha"],
            variant_state[chosen_variant]["beta"],
            params["alpha"], params["beta"],
        )
        rec_confidence[f"P({chosen_variant}>{vid})"] = round(p, 3)

    result = {
        "variants": variant_analysis,
        "tasks": task_analysis,
        "cross_product": cross_analysis,
        "recommendation": {
            "use_variant": chosen_variant,
            "work_on_task": chosen_task,
            "best_variant_per_task": best_variant_per_task,
            "confidence": rec_confidence,
        },
        "statistical_tests": {
            "variant_significance": variant_sig,
            "task_significance": task_sig,
        },
        "warnings": warnings,
        "knowledge_gaps": gaps,
    }
    print(json.dumps(result, indent=2))


MIN_SAMPLES_PER_ARM = 5


def _forced_exploration(score_state: dict, variant_ids: list[str]) -> str | None:
    """Round-robin until each arm has MIN_SAMPLES_PER_ARM scored observations."""
    counts = {v: score_state.get(v, {}).get("n", 0) for v in variant_ids}
    undersampled = [v for v, n in counts.items() if n < MIN_SAMPLES_PER_ARM]
    if undersampled:
        return min(undersampled, key=lambda v: counts[v])
    return None


def cmd_init(args: argparse.Namespace) -> None:
    """Lightweight variant selection for SessionStart hook. Minimal output."""
    variant_state = load_state(VARIANT_IDS)
    score_state = load_score_state(VARIANT_IDS)

    # Phase 1: Forced exploration — ensure minimum samples per arm
    forced = _forced_exploration(score_state, VARIANT_IDS)
    if forced:
        chosen = forced
        method = "forced_explore"
    elif any(s["n"] >= 2 for s in score_state.values()):
        # Phase 2: Gaussian Thompson Sampling (primary decision engine)
        chosen = gaussian_thompson_sample(score_state)
        method = "gaussian"
    else:
        # Phase 3: Beta fallback only during absolute cold start (n<2 everywhere)
        chosen = thompson_sample(variant_state)
        method = "beta_coldstart"

    # Task-type awareness
    tasktype_state = load_tasktype_state(VARIANT_IDS)
    type_recs = {}
    for tt in TASK_TYPES:
        type_recs[tt] = best_variant_for_type(tasktype_state, VARIANT_IDS, tt)

    # Convergence check
    score_stats = get_score_stats(score_state)
    converged = all(
        s.get("verdict") == "converging"
        for s in score_stats.values()
        if s["n"] and s["n"] >= 2
    )

    total_trials = sum(s["n"] for s in score_state.values())

    # Write to file for other tools to read
    active_file = os.path.join(_DIR, ".active_variant")
    with open(active_file, "w") as f:
        json.dump({"variant": chosen, "method": method}, f)

    result = {
        "variant": chosen,
        "method": method,
        "total_scored_trials": total_trials,
        "by_task_type": type_recs,
        "converged": converged,
    }

    if converged and total_trials >= 15:
        result["signal"] = "STOP_EXPLORING — variants have converged. Using best."
    elif total_trials < 5:
        result["signal"] = "COLD_START — insufficient data, exploring."
    else:
        result["signal"] = "EXPLORING — gathering signal."

    # Score-based rankings if available
    has_scores = any(s["n"] >= 2 for s in score_state.values())
    if has_scores:
        ranked = sorted(score_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
        result["rankings"] = {k: v["mean"] for k, v in ranked}

    print(json.dumps(result, indent=2))


def cmd_suggest(args: argparse.Namespace) -> None:
    """Pick a coding-style variant via Thompson sampling (highest draw wins)."""
    state = load_state(VARIANT_IDS)
    chosen = thompson_sample(state)
    print(json.dumps({"variant_id": chosen}, indent=2))


def cmd_propose(args: argparse.Namespace) -> None:
    """Pick what to build next via Thompson sampling (lowest draw = most need)."""
    tasks = _load_tasks()
    tasks_by_id = _tasks_map(tasks)
    task_state = load_task_state(_task_ids(tasks))

    chosen_id = thompson_sample_neediest(task_state)
    task = tasks_by_id[chosen_id]
    params = task_state[chosen_id]

    total = params["alpha"] + params["beta"] - 2
    win_rate = (params["alpha"] - 1) / total if total > 0 else None

    result = {
        "task_id": chosen_id,
        "prompt": task["prompt"].strip(),
        "target_file": task["target_file"],
        "test_command": task["test_command"],
        "posterior": {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "attempts": int(total),
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        },
    }
    print(json.dumps(result, indent=2))


def cmd_summary(args: argparse.Namespace) -> None:
    """Print all bandit state: variants, tasks, and cross-product."""
    tasks = _load_tasks()
    task_ids = _task_ids(tasks)
    variant_state = load_state(VARIANT_IDS)
    task_state = load_task_state(task_ids)
    cross_state = load_cross_state(VARIANT_IDS, task_ids)
    print(json.dumps({
        "variants": variant_state,
        "tasks": task_state,
        "cross_product": cross_state,
    }, indent=2))


def cmd_update(args: argparse.Namespace) -> None:
    """Update bandit state after a task result. Supports pass/fail and continuous scores."""
    tasks = _load_tasks()
    valid_task_ids = _task_ids(tasks)
    # Auto-register unknown task IDs so infra/debug/ad-hoc tasks can be scored
    if args.task_id not in valid_task_ids:
        valid_task_ids.append(args.task_id)

    success = args.result == "pass"
    score = args.score  # None if not provided

    # Update variant bandit (Beta — pass/fail)
    if args.variant:
        if args.variant not in VARIANT_IDS:
            print(f"ERROR: unknown variant '{args.variant}'. Valid: {VARIANT_IDS}", file=sys.stderr)
            sys.exit(1)
        variant_state = load_state(VARIANT_IDS)
        update_state(variant_state, args.variant, success)
        save_state(variant_state)

    # Update task bandit (Beta — pass/fail)
    task_state = load_task_state(valid_task_ids)
    update_task_state(task_state, args.task_id, success)
    save_task_state(task_state)

    # Update cross-product bandit (Beta — pass/fail)
    if args.variant:
        cross_state = load_cross_state(VARIANT_IDS, valid_task_ids)
        update_cross_state(cross_state, args.variant, args.task_id, success)
        save_cross_state(cross_state)

    # Update score bandit (Gaussian — continuous score)
    if args.variant and score is not None:
        score_state = load_score_state(VARIANT_IDS)
        update_score_state(score_state, args.variant, score)
        save_score_state(score_state)

    # Update task-type bandit (Gaussian — per variant:tasktype)
    task_type = args.task_type or "unknown"
    if args.variant and score is not None:
        tasktype_state = load_tasktype_state(VARIANT_IDS)
        key = f"{args.variant}:{task_type}"
        update_score_state(tasktype_state, key, score)
        save_tasktype_state(tasktype_state)

    # Log to runs.jsonl
    run_entry = {
        "timestamp": time.time(),
        "task_id": args.task_id,
        "variant_id": args.variant or "unknown",
        "success": success,
        "score": score,
        "task_type": task_type,
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(run_entry) + "\n")

    # Also feed the oracle (Bayesian NIG sampler used by UserPromptSubmit hook).
    # Keeps oracle_state.json in sync with runs.jsonl so the two hooks agree.
    if args.variant:
        oracle_value = score if score is not None else (1.0 if success else 0.0)
        try:
            oracle_file = os.path.join(_DIR, "oracle_state.json")
            with open(oracle_file) as _of:
                oracle_state = json.load(_of)
            oracle_state.setdefault("arms", {}).setdefault(args.variant, []).append(float(oracle_value))
            with open(oracle_file, "w") as _of:
                json.dump(oracle_state, _of, indent=2)
        except Exception as _e:
            print(f"WARNING: oracle observe failed: {_e}", file=sys.stderr)

    # Clear attempt tracking
    attempt_file = os.path.join(_DIR, ".current_attempt")
    if os.path.exists(attempt_file):
        os.remove(attempt_file)

    out = {
        "task_id": args.task_id,
        "success": success,
        "task_state": task_state[args.task_id],
    }
    if score is not None:
        out["score"] = score
        out["task_type"] = task_type
    if args.variant:
        out["variant_id"] = args.variant
        out["variant_state"] = variant_state[args.variant]
    print(json.dumps(out, indent=2))


def cmd_decay(args: argparse.Namespace) -> None:
    """Apply exponential forgetting to all three bandit layers."""
    tasks = _load_tasks()
    task_ids = _task_ids(tasks)

    variant_state = load_state(VARIANT_IDS)
    task_state = load_task_state(task_ids)
    cross_state = load_cross_state(VARIANT_IDS, task_ids)

    decay_state(variant_state, args.factor)
    decay_state(task_state, args.factor)
    decay_state(cross_state, args.factor)

    save_state(variant_state)
    save_task_state(task_state)
    save_cross_state(cross_state)

    print(json.dumps({
        "decay_factor": args.factor,
        "variants": variant_state,
        "tasks": task_state,
        "cross_product": cross_state,
    }, indent=2))


def cmd_regret(args: argparse.Namespace) -> None:
    """Analyze cumulative regret from runs.jsonl.

    Regret = how many failures could have been avoided if we'd always
    picked the best variant. Shows whether Thompson sampling is converging.
    """
    if not os.path.exists(RUN_LOG):
        print(json.dumps({"error": "No runs.jsonl found — run the orchestrator first."}))
        return

    runs = []
    with open(RUN_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))

    if not runs:
        print(json.dumps({"error": "runs.jsonl is empty."}))
        return

    # Per-variant stats
    variant_stats: dict[str, dict] = {}
    for r in runs:
        vid = r["variant_id"]
        if vid not in variant_stats:
            variant_stats[vid] = {"wins": 0, "total": 0}
        variant_stats[vid]["total"] += 1
        if r["success"]:
            variant_stats[vid]["wins"] += 1

    for vid, s in variant_stats.items():
        s["win_rate"] = round(s["wins"] / s["total"], 3) if s["total"] > 0 else 0

    # Best observed win rate (oracle)
    best_rate = max(s["win_rate"] for s in variant_stats.values())

    # Cumulative regret: for each run, regret = best_rate - (1 if success else 0)
    cumulative_regret = 0.0
    for r in runs:
        reward = 1.0 if r["success"] else 0.0
        cumulative_regret += best_rate - reward

    total_runs = len(runs)
    avg_regret = round(cumulative_regret / total_runs, 3) if total_runs > 0 else 0

    print(json.dumps({
        "total_runs": total_runs,
        "variant_stats": variant_stats,
        "oracle_best_rate": best_rate,
        "cumulative_regret": round(cumulative_regret, 3),
        "avg_regret_per_run": avg_regret,
    }, indent=2))


def cmd_score_report(args: argparse.Namespace) -> None:
    """Show score-based Gaussian bandit analysis alongside Beta analysis."""
    score_state = load_score_state(VARIANT_IDS)
    score_stats = get_score_stats(score_state)

    tasktype_state = load_tasktype_state(VARIANT_IDS)
    tasktype_stats = get_score_stats(tasktype_state)

    # Convergence
    converged_arms = [k for k, v in score_stats.items() if v.get("verdict") == "converging"]
    exploring_arms = [k for k, v in score_stats.items() if v.get("verdict") == "exploring"]

    result = {
        "variant_scores": score_stats,
        "tasktype_scores": tasktype_stats,
        "converged": converged_arms,
        "exploring": exploring_arms,
    }

    if converged_arms:
        best = max(
            ((k, v) for k, v in score_stats.items() if v["n"] >= 2),
            key=lambda x: x[1]["mean"],
            default=None,
        )
        if best:
            result["recommendation"] = f"Use '{best[0]}' (mean={best[1]['mean']}, n={best[1]['n']})"

    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandit CLI for LLM orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("think").set_defaults(func=cmd_think)
    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("suggest").set_defaults(func=cmd_suggest)
    sub.add_parser("propose").set_defaults(func=cmd_propose)
    sub.add_parser("summary").set_defaults(func=cmd_summary)
    sub.add_parser("scores").set_defaults(func=cmd_score_report)

    p_update = sub.add_parser("update")
    p_update.add_argument("task_id", help="Task id from tasks.yaml")
    p_update.add_argument("result", choices=["pass", "fail"], help="Test outcome")
    p_update.add_argument("--variant", help="Variant id to also update variant bandit")
    p_update.add_argument("--score", type=float, help="Continuous quality score 0-1")
    p_update.add_argument("--task-type", choices=TASK_TYPES, help="Task type for segmented learning")
    p_update.set_defaults(func=cmd_update)

    p_decay = sub.add_parser("decay")
    p_decay.add_argument("--factor", type=float, default=0.95,
                         help="Decay factor (default: 0.95)")
    p_decay.set_defaults(func=cmd_decay)

    sub.add_parser("regret").set_defaults(func=cmd_regret)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
