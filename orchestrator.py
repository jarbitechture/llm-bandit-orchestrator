"""
LLM Bandit Orchestrator
========================
Loads tasks from tasks.yaml, picks an LLM variant via Thompson sampling,
calls the model, writes code to disk, runs tests, and updates both
variant and task bandit state.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

import bandit


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    prompt: str
    target_file: str
    test_command: str


@dataclass
class Variant:
    id: str
    system_prompt: str
    temperature: float


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: List[Variant] = [
    Variant(
        id="strict_tdd",
        system_prompt=(
            "You are an expert software engineer. Write minimal, correct, "
            "fully-tested code. Follow strict TDD: think about edge cases "
            "first, then write the simplest implementation that passes all "
            "tests. Output ONLY the raw Python code with no markdown fences."
        ),
        temperature=0.0,
    ),
    Variant(
        id="balanced",
        system_prompt=(
            "You are a pragmatic software engineer. Write clean, readable "
            "code that balances correctness with simplicity. Include brief "
            "inline comments where helpful. Output ONLY the raw Python code "
            "with no markdown fences."
        ),
        temperature=0.2,
    ),
    Variant(
        id="exploratory",
        system_prompt=(
            "You are a creative software engineer who enjoys elegant, "
            "well-factored solutions. Feel free to refactor for clarity "
            "or use Pythonic idioms. Output ONLY the raw Python code "
            "with no markdown fences."
        ),
        temperature=0.5,
    ),
]

VARIANT_MAP = {v.id: v for v in VARIANTS}


# ---------------------------------------------------------------------------
# Model endpoint configuration
# ---------------------------------------------------------------------------

MODEL_ENDPOINT = os.environ.get(
    "LLM_ENDPOINT", "http://localhost:11434/v1/chat/completions"
)
MODEL_NAME = os.environ.get("LLM_MODEL", "qwen2.5-coder:14b")
MODEL_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` wrappers that LLMs tend to add."""
    stripped = re.sub(r"^```(?:python|py)?\s*\n", "", text.strip())
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped


def call_model(system_prompt: str, user_prompt: str, temperature: float) -> str:
    """Call the LLM via OpenAI-compatible chat completions endpoint.

    Configure via environment variables:
        LLM_ENDPOINT  — full URL (default: http://localhost:11434/v1/chat/completions)
        LLM_MODEL     — model name (default: qwen2.5-coder:14b)
        LLM_TIMEOUT   — request timeout in seconds (default: 120)
    """
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        MODEL_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"LLM endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach LLM endpoint at {MODEL_ENDPOINT}: {exc.reason}"
        ) from exc

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response structure from LLM endpoint: "
            f"{json.dumps(body, indent=2)[:500]}"
        ) from exc

    return _strip_markdown_fences(text)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(path: str = "tasks.yaml") -> List[Task]:
    """Parse tasks.yaml into a list of Task objects."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return [Task(**entry) for entry in raw]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

RUN_LOG = "runs.jsonl"


def run_once(
    task: Task,
    variant_state: dict,
    task_state: dict,
    cross_state: dict,
) -> bool:
    """Pick a variant, generate code, run tests, update all three bandits."""
    # Use cross-product bandit if it has data, else fall back to global
    has_cross_data = any(
        (p["alpha"] + p["beta"]) > 2.0
        for k, p in cross_state.items()
        if k.endswith(f":{task.id}")
    )
    if has_cross_data:
        variant_id = bandit.thompson_sample_for_task(
            cross_state, list(VARIANT_MAP.keys()), task.id
        )
    else:
        variant_id = bandit.thompson_sample(variant_state)

    variant = VARIANT_MAP[variant_id]
    print(f"[task={task.id}] variant={variant_id} (temp={variant.temperature})"
          f"{' [cross]' if has_cross_data else ' [global]'}")

    code = call_model(variant.system_prompt, task.prompt, variant.temperature)

    target = Path(task.target_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code)
    print(f"[task={task.id}] wrote {len(code)} chars -> {task.target_file}")

    result = None
    try:
        result = subprocess.run(
            task.test_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[task={task.id}] TIMEOUT after {TEST_TIMEOUT}s")
        success = False

    if success:
        print(f"[task={task.id}] PASS")
    elif result is not None:
        print(f"[task={task.id}] FAIL\n{result.stdout}\n{result.stderr}")

    # Update all three bandits
    bandit.update_state(variant_state, variant_id, success)
    bandit.save_state(variant_state)

    bandit.update_task_state(task_state, task.id, success)
    bandit.save_task_state(task_state)

    bandit.update_cross_state(cross_state, variant_id, task.id, success)
    bandit.save_cross_state(cross_state)

    log_entry = {
        "task_id": task.id,
        "variant_id": variant_id,
        "success": success,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return success


def main(max_runs: int | None = None) -> None:
    """Run the orchestrator loop.

    Args:
        max_runs: If set, stop after this many total attempts.
                  If None, loop through all tasks once.
    """
    tasks = load_tasks()
    variant_ids = [v.id for v in VARIANTS]
    task_ids = [t.id for t in tasks]
    variant_state = bandit.load_state(variant_ids)
    task_state = bandit.load_task_state(task_ids)
    cross_state = bandit.load_cross_state(variant_ids, task_ids)

    if max_runs is None:
        for task in tasks:
            run_once(task, variant_state, task_state, cross_state)
    else:
        run_count = 0
        while run_count < max_runs:
            for task in tasks:
                if run_count >= max_runs:
                    break
                run_once(task, variant_state, task_state, cross_state)
                run_count += 1

    print(f"\nFinal variant state: {json.dumps(variant_state, indent=2)}")
    print(f"Final task state: {json.dumps(task_state, indent=2)}")
    print(f"Final cross state: {json.dumps(cross_state, indent=2)}")


if __name__ == "__main__":
    main()
