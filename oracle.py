#!/usr/bin/env python3
"""
Statistical Oracle — always returns a probability-backed answer.

Feed it observations. Ask it questions. It does the math.

Usage:
  # Add observations
  python oracle.py observe --arm "strict_tdd" --value 0.88
  python oracle.py observe --arm "balanced" --value 0.72
  python oracle.py observe --arm "exploratory" --value 0.91

  # Ask: which is best?
  python oracle.py best

  # Ask: what's the probability arm X beats arm Y?
  python oracle.py compare --a "strict_tdd" --b "exploratory"

  # Ask: full posterior for one arm
  python oracle.py posterior --arm "strict_tdd"

  # Ask: rank all arms
  python oracle.py rank

  # Reset
  python oracle.py reset
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_DIR, "oracle_state.json")

# Normal-Inverse-Gamma prior: vague but proper
# mu0=0.5, kappa0=1, alpha0=1, beta0=0.05
# This gives a sensible starting point for 0-1 scores
DEFAULT_PRIOR = {"mu0": 0.5, "kappa0": 1.0, "alpha0": 1.0, "beta0": 0.05}


@dataclass
class Posterior:
    """Normal-Inverse-Gamma posterior for unknown mean and variance."""
    mu: float      # posterior mean estimate
    kappa: float   # pseudo-observations for mean
    alpha: float   # shape for variance
    beta: float    # rate for variance
    n: int         # actual observations

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        """Marginal variance of the posterior predictive (Student-t)."""
        if self.alpha <= 1:
            return self.beta / (self.alpha * self.kappa)
        return self.beta / ((self.alpha - 1) * self.kappa)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def df(self) -> float:
        """Degrees of freedom for the Student-t marginal."""
        return 2 * self.alpha

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Credible interval using Student-t quantiles."""
        from _t_table import t_quantile
        tail = (1 - level) / 2
        t = t_quantile(self.df, 1 - tail)
        half = t * self.std
        return (round(self.mu - half, 4), round(self.mu + half, 4))

    def prob_greater_than(self, other: "Posterior", n_samples: int = 50_000) -> float:
        """P(self > other) via Monte Carlo from Student-t marginals."""
        import random
        wins = 0
        for _ in range(n_samples):
            a = _sample_student_t(self.mu, self.std, self.df)
            b = _sample_student_t(other.mu, other.std, other.df)
            if a > b:
                wins += 1
        return wins / n_samples

    def to_dict(self) -> dict:
        ci = self.credible_interval()
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_90": list(ci),
            "n": self.n,
            "df": round(self.df, 2),
        }


def _sample_student_t(mu: float, scale: float, df: float) -> float:
    """Sample from a Student-t distribution (no scipy needed)."""
    import random
    # Use the ratio-of-uniforms or Box-Muller + chi-squared approach
    # Student-t = Normal / sqrt(ChiSq/df)
    z = random.gauss(0, 1)
    # Chi-squared via sum of squared normals
    chi2 = sum(random.gauss(0, 1) ** 2 for _ in range(max(1, int(df))))
    t = z / math.sqrt(chi2 / df)
    return mu + scale * t


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"arms": {}, "prior": DEFAULT_PRIOR}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_posterior(state: dict, arm: str) -> Posterior:
    """Compute the Normal-Inverse-Gamma posterior for an arm."""
    p = state["prior"]
    mu0, kappa0, alpha0, beta0 = p["mu0"], p["kappa0"], p["alpha0"], p["beta0"]

    if arm not in state["arms"] or not state["arms"][arm]:
        return Posterior(mu=mu0, kappa=kappa0, alpha=alpha0, beta=beta0, n=0)

    values = state["arms"][arm]
    n = len(values)
    x_bar = sum(values) / n
    ss = sum((x - x_bar) ** 2 for x in values)  # sum of squared deviations

    # NIG update rules
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
    alpha_n = alpha0 + n / 2
    beta_n = beta0 + 0.5 * ss + 0.5 * (kappa0 * n * (x_bar - mu0) ** 2) / kappa_n

    return Posterior(mu=mu_n, kappa=kappa_n, alpha=alpha_n, beta=beta_n, n=n)


def cmd_observe(args: argparse.Namespace) -> None:
    state = load_state()
    if args.arm not in state["arms"]:
        state["arms"][args.arm] = []
    state["arms"][args.arm].append(args.value)
    save_state(state)
    post = get_posterior(state, args.arm)
    print(json.dumps({"arm": args.arm, "observed": args.value, **post.to_dict()}))


def cmd_best(args: argparse.Namespace) -> None:
    state = load_state()
    arms = list(state["arms"].keys())
    if not arms:
        # Even with no data, return the prior
        post = get_posterior(state, "__prior__")
        print(json.dumps({"best": None, "reason": "no arms observed", "prior": post.to_dict()}))
        return

    posteriors = {a: get_posterior(state, a) for a in arms}

    # Find best by posterior mean
    best_arm = max(posteriors, key=lambda a: posteriors[a].mean)
    best_post = posteriors[best_arm]

    # Probability best is truly best (vs each other arm)
    prob_best = 1.0
    comparisons = {}
    for a in arms:
        if a == best_arm:
            continue
        p = best_post.prob_greater_than(posteriors[a])
        prob_best *= p
        comparisons[f"P({best_arm}>{a})"] = round(p, 4)

    print(json.dumps({
        "best": best_arm,
        "p_best": round(prob_best, 4),
        "comparisons": comparisons,
        **best_post.to_dict(),
    }))


def cmd_compare(args: argparse.Namespace) -> None:
    state = load_state()
    post_a = get_posterior(state, args.a)
    post_b = get_posterior(state, args.b)
    p = post_a.prob_greater_than(post_b)

    print(json.dumps({
        "P(a>b)": round(p, 4),
        "P(b>a)": round(1 - p, 4),
        "a": {"arm": args.a, **post_a.to_dict()},
        "b": {"arm": args.b, **post_b.to_dict()},
        "verdict": f"{args.a}" if p > 0.5 else f"{args.b}",
    }))


def cmd_posterior(args: argparse.Namespace) -> None:
    state = load_state()
    post = get_posterior(state, args.arm)
    values = state["arms"].get(args.arm, [])
    print(json.dumps({"arm": args.arm, "observations": values, **post.to_dict()}))


def cmd_rank(args: argparse.Namespace) -> None:
    state = load_state()
    arms = list(state["arms"].keys())
    if not arms:
        print(json.dumps({"ranking": [], "reason": "no arms observed"}))
        return

    posteriors = {a: get_posterior(state, a) for a in arms}
    ranked = sorted(arms, key=lambda a: posteriors[a].mean, reverse=True)

    ranking = []
    for i, arm in enumerate(ranked):
        p = posteriors[arm]
        entry = {"rank": i + 1, "arm": arm, **p.to_dict()}
        ranking.append(entry)

    print(json.dumps({"ranking": ranking}))


def cmd_reset(args: argparse.Namespace) -> None:
    state = {"arms": {}, "prior": DEFAULT_PRIOR}
    save_state(state)
    print(json.dumps({"reset": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical Oracle")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_obs = sub.add_parser("observe", help="Record an observation")
    p_obs.add_argument("--arm", required=True, help="Arm/option name")
    p_obs.add_argument("--value", type=float, required=True, help="Observed value")
    p_obs.set_defaults(func=cmd_observe)

    sub.add_parser("best", help="Which arm is best?").set_defaults(func=cmd_best)

    p_cmp = sub.add_parser("compare", help="P(A > B)")
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)
    p_cmp.set_defaults(func=cmd_compare)

    p_post = sub.add_parser("posterior", help="Full posterior for one arm")
    p_post.add_argument("--arm", required=True)
    p_post.set_defaults(func=cmd_posterior)

    sub.add_parser("rank", help="Rank all arms").set_defaults(func=cmd_rank)
    sub.add_parser("reset", help="Clear all data").set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
