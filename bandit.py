"""
Thompson Sampling (Beta-Bernoulli) bandit for LLM variant selection.

Each arm has a Beta(alpha, beta) posterior. On each round we draw
from each posterior and pick the arm with the highest (or lowest) sample.
Successes increment alpha; failures increment beta.

State files use atomic writes (write-to-temp then rename) to prevent
corruption from mid-write crashes.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

import numpy as np
from scipy.stats import t as t_dist

_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_DIR, "bandit_state.json")
TASK_STATE_FILE = os.path.join(_DIR, "task_bandit_state.json")
CROSS_STATE_FILE = os.path.join(_DIR, "cross_bandit_state.json")
SCORE_STATE_FILE = os.path.join(_DIR, "score_bandit_state.json")
TASKTYPE_STATE_FILE = os.path.join(_DIR, "tasktype_bandit_state.json")


# ---------------------------------------------------------------------------
# Persistence (atomic read/write)
# ---------------------------------------------------------------------------

def _load(path: str, keys: List[str]) -> Dict[str, Dict[str, float]]:
    """Read JSON state file, initializing missing keys to Beta(1,1).

    If the file is missing, starts fresh. If the file is corrupt,
    backs it up and starts fresh rather than crashing.
    """
    state: Dict[str, Dict[str, float]] = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError):
            backup = path + ".corrupt"
            os.replace(path, backup)
            print(f"WARNING: {path} was corrupt, backed up to {backup}, starting fresh")
    for k in keys:
        if k not in state:
            state[k] = {"alpha": 1.0, "beta": 1.0}
    return state


def _save(path: str, state: Dict[str, Dict[str, float]]) -> None:
    """Atomic save: write to temp file then rename, so a crash can't corrupt state."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Variant-level bandit (which coding style)
# ---------------------------------------------------------------------------

def load_state(variants: List[str]) -> Dict[str, Dict[str, float]]:
    """Load variant bandit state. Missing variants get Beta(1,1)."""
    return _load(STATE_FILE, variants)


def save_state(state: Dict[str, Dict[str, float]]) -> None:
    """Persist variant bandit state (atomic)."""
    _save(STATE_FILE, state)


def update_state(
    state: Dict[str, Dict[str, float]], variant_id: str, success: bool
) -> None:
    """Update the Beta posterior for *variant_id* in place."""
    if variant_id not in state:
        raise KeyError(f"Unknown variant '{variant_id}'. Known: {list(state.keys())}")
    if success:
        state[variant_id]["alpha"] += 1.0
    else:
        state[variant_id]["beta"] += 1.0


# ---------------------------------------------------------------------------
# Task-level bandit (what to build next)
# ---------------------------------------------------------------------------

def load_task_state(task_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Load task bandit state. Missing tasks get Beta(1,1)."""
    return _load(TASK_STATE_FILE, task_ids)


def save_task_state(state: Dict[str, Dict[str, float]]) -> None:
    """Persist task bandit state (atomic)."""
    _save(TASK_STATE_FILE, state)


def update_task_state(
    state: Dict[str, Dict[str, float]], task_id: str, success: bool
) -> None:
    """Update the Beta posterior for a task in place."""
    if task_id not in state:
        state[task_id] = {"alpha": 1.0, "beta": 1.0}
    if success:
        state[task_id]["alpha"] += 1.0
    else:
        state[task_id]["beta"] += 1.0


# ---------------------------------------------------------------------------
# Cross-product bandit (which variant for which task)
# ---------------------------------------------------------------------------

def _cross_keys(variant_ids: List[str], task_ids: List[str]) -> List[str]:
    """Generate 'variant:task' keys for the cross-product bandit."""
    return [f"{v}:{t}" for v in variant_ids for t in task_ids]


def load_cross_state(
    variant_ids: List[str], task_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    """Load cross-product bandit state. Keys are 'variant_id:task_id'."""
    return _load(CROSS_STATE_FILE, _cross_keys(variant_ids, task_ids))


def save_cross_state(state: Dict[str, Dict[str, float]]) -> None:
    """Persist cross-product bandit state (atomic)."""
    _save(CROSS_STATE_FILE, state)


def update_cross_state(
    state: Dict[str, Dict[str, float]],
    variant_id: str,
    task_id: str,
    success: bool,
) -> None:
    """Update the cross-product posterior for a variant:task pair."""
    key = f"{variant_id}:{task_id}"
    if key not in state:
        state[key] = {"alpha": 1.0, "beta": 1.0}
    if success:
        state[key]["alpha"] += 1.0
    else:
        state[key]["beta"] += 1.0


def thompson_sample_for_task(
    state: Dict[str, Dict[str, float]],
    variant_ids: List[str],
    task_id: str,
) -> str:
    """Given a task, Thompson sample across variants to find the best one for it.

    Only considers keys matching '*:task_id'. Falls back to global sampling
    if no cross-product data exists.
    """
    candidates = {}
    for vid in variant_ids:
        key = f"{vid}:{task_id}"
        if key in state:
            candidates[vid] = state[key]
    if not candidates:
        raise ValueError(f"No cross-product data for task '{task_id}'.")
    best_id = ""
    best_val = -1.0
    for vid, params in candidates.items():
        sample = np.random.beta(params["alpha"], params["beta"])
        if sample > best_val:
            best_val = sample
            best_id = vid
    return best_id


# ---------------------------------------------------------------------------
# Thompson sampling (global)
# ---------------------------------------------------------------------------

def thompson_sample(state: Dict[str, Dict[str, float]]) -> str:
    """Draw from each Beta posterior; return the arm with the HIGHEST sample.

    Raises ValueError if state is empty (no arms to sample from).
    """
    if not state:
        raise ValueError("Cannot sample from empty state — no arms defined.")
    best_id = ""
    best_val = -1.0
    for arm_id, params in state.items():
        sample = np.random.beta(params["alpha"], params["beta"])
        if sample > best_val:
            best_val = sample
            best_id = arm_id
    return best_id


def thompson_sample_neediest(state: Dict[str, Dict[str, float]]) -> str:
    """Draw from each Beta posterior; return the arm with the LOWEST sample.

    Used for tasks: the task that draws lowest is the one with the worst
    (or most uncertain) success rate — the one most in need of work.

    Raises ValueError if state is empty.
    """
    if not state:
        raise ValueError("Cannot sample from empty state — no arms defined.")
    worst_id = ""
    worst_val = 2.0
    for arm_id, params in state.items():
        sample = np.random.beta(params["alpha"], params["beta"])
        if sample < worst_val:
            worst_val = sample
            worst_id = arm_id
    return worst_id


# ---------------------------------------------------------------------------
# Decay (optional — call periodically to prevent stale posteriors)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Score-based bandit (Gaussian Thompson sampling)
# ---------------------------------------------------------------------------

def _load_scores(path: str, keys: List[str]) -> Dict[str, Dict]:
    """Load score state. Each arm tracks: n, mean, M2 (for Welford's online variance)."""
    state = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError):
            backup = path + ".corrupt"
            os.replace(path, backup)
    for k in keys:
        if k not in state:
            state[k] = {"n": 0, "mean": 0.5, "M2": 0.0}
    return state


def load_score_state(keys: List[str]) -> Dict[str, Dict]:
    return _load_scores(SCORE_STATE_FILE, keys)


def save_score_state(state: Dict[str, Dict]) -> None:
    _save(SCORE_STATE_FILE, state)


def update_score_state(state: Dict[str, Dict], key: str, score: float) -> None:
    """Welford's online algorithm for updating mean and variance."""
    if key not in state:
        state[key] = {"n": 0, "mean": 0.5, "M2": 0.0}
    s = state[key]
    s["n"] += 1
    delta = score - s["mean"]
    s["mean"] += delta / s["n"]
    delta2 = score - s["mean"]
    s["M2"] += delta * delta2
    # Guard against floating-point drift producing negative M2
    s["M2"] = max(0.0, s["M2"])


def gaussian_thompson_sample(state: Dict[str, Dict]) -> str:
    """Sample from Normal posterior for each arm, return highest."""
    if not state:
        raise ValueError("Empty state")
    best_id = ""
    best_val = -float("inf")
    for arm_id, s in state.items():
        n = s["n"]
        if n < 2:
            sample = np.random.normal(0.5, 0.25)
        else:
            variance = s["M2"] / (n - 1)
            stderr = np.sqrt(variance / n) if variance > 0 else 0.01
            sample = np.random.normal(s["mean"], stderr)
        if sample > best_val:
            best_val = sample
            best_id = arm_id
    return best_id


def get_score_stats(state: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compute readable stats from score state."""
    result = {}
    for arm_id, s in state.items():
        n = s["n"]
        if n < 2:
            result[arm_id] = {
                "n": n, "mean": round(s["mean"], 3),
                "std": None, "ci_90": None, "verdict": "insufficient data"
            }
        else:
            variance = s["M2"] / (n - 1)
            std = np.sqrt(variance)
            stderr = std / np.sqrt(n)
            ci = t_dist.interval(0.90, df=n - 1, loc=s["mean"], scale=stderr)
            result[arm_id] = {
                "n": n, "mean": round(s["mean"], 3),
                "std": round(std, 3),
                "ci_90": [round(ci[0], 3), round(ci[1], 3)],
                "verdict": "converging" if (ci[1] - ci[0]) < 0.2 else "exploring"
            }
    return result


# ---------------------------------------------------------------------------
# Task-type bandit (which variant for which kind of work)
# ---------------------------------------------------------------------------

TASK_TYPES = ["greenfield", "refactor", "test", "unknown"]


def load_tasktype_state(variant_ids: List[str]) -> Dict[str, Dict]:
    """Load per-tasktype score state. Keys are 'variant:tasktype'."""
    keys = [f"{v}:{t}" for v in variant_ids for t in TASK_TYPES]
    return _load_scores(TASKTYPE_STATE_FILE, keys)


def save_tasktype_state(state: Dict[str, Dict]) -> None:
    _save(TASKTYPE_STATE_FILE, state)


def best_variant_for_type(
    state: Dict[str, Dict], variant_ids: List[str], task_type: str
) -> str:
    """Gaussian Thompson sample across variants for a specific task type."""
    candidates = {}
    for vid in variant_ids:
        key = f"{vid}:{task_type}"
        if key in state:
            candidates[key] = state[key]
    if not candidates:
        return np.random.choice(variant_ids)
    best_key = gaussian_thompson_sample(candidates)
    return best_key.split(":")[0]


def decay_state(
    state: Dict[str, Dict[str, float]], factor: float = 0.95
) -> None:
    """Shrink all posteriors toward the prior by multiplying (alpha-1, beta-1) by factor.

    This implements exponential forgetting: recent results count more than
    old ones. A factor of 0.95 means each round's influence halves after
    ~14 rounds.  Alpha and beta are clamped to a minimum of 1.0 (the prior).
    """
    for params in state.values():
        params["alpha"] = max(1.0, 1.0 + (params["alpha"] - 1.0) * factor)
        params["beta"] = max(1.0, 1.0 + (params["beta"] - 1.0) * factor)
