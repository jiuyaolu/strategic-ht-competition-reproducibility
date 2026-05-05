#!/usr/bin/env python3
"""Numerical fixed-point solver for the multi-agent competition model.

This script is intentionally pure standard-library Python. It implements:

1. Exact binomial pass probabilities for the one-sided baseline test.
2. The single-agent best response as a function of an effective reward level r.
3. The competitive fixed point r* = Phi_alpha(r*).
4. Basic principal-side metrics at the equilibrium induced by alpha.

The solver is meant for research exploration, not final production experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Callable, List, Sequence


NORMAL = NormalDist()
TIE_TOL = 1e-12


@dataclass
class ModelParams:
    m: int = 20
    mu_b: float = 0.5
    reward: float = 20.0
    c0: float = 1.0
    c: float = 0.08
    n_min: int = 10
    n_max: int = 80
    grid_points: int = 301
    alpha_tol: float = 1e-6
    fixed_point_tol: float = 1e-6
    fixed_point_iters: int = 40
    distribution: str = "beta"
    beta_a: float = 6.0
    beta_b: float = 5.0
    truncnorm_mean: float = 0.58
    truncnorm_sd: float = 0.12
    crowding: str = "power"
    gamma: float = 1.0
    eta: float = 1.0
    tail_method: str = "exact"


@dataclass
class AlphaSummary:
    alpha: float
    fixed_point_reward: float
    fixed_point_gap: float
    approval_probability: float
    prob_any_approval: float
    false_approval_probability: float
    true_approval_probability: float
    entry_rate: float
    null_entry_rate: float
    mean_sample_size: float
    participation_threshold: float | None
    expected_false_approvals: float
    expected_true_approvals: float
    expected_false_negatives: float
    mfdr: float
    pfdr: float
    fdr: float
    sensitivity: float
    specificity: float
    baseline_best_n: int
    baseline_utility: float
    baseline_active: bool
    cost_floor: float
    mfdr_bound_action_specific: float
    mfdr_bound_null_entry: float
    mfdr_bound_effective: float
    mfdr_bound_monopoly: float
    pfdr_bound_action_specific: float
    pfdr_bound_null_entry: float
    pfdr_bound_effective: float
    pfdr_bound_monopoly: float
    fdr_bound_action_specific: float
    fdr_bound_null_entry: float
    fdr_bound_effective: float
    fdr_bound_monopoly: float


@dataclass
class AlphaHatEstimate:
    alpha_threshold_cross: float | None
    alpha_false_positive_cross: float | None
    alpha_mfdr_positive: float | None


def beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - log_norm)


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def truncnorm_pdf(x: float, mean: float, sd: float) -> float:
    if x <= 0.0 or x >= 1.0 or sd <= 0.0:
        return 0.0
    z = (x - mean) / sd
    low = (0.0 - mean) / sd
    high = (1.0 - mean) / sd
    normalizer = NORMAL.cdf(high) - NORMAL.cdf(low)
    if normalizer <= 0.0:
        return 0.0
    return normal_pdf(z) / (sd * normalizer)


def distribution_pdf(params: ModelParams) -> Callable[[float], float]:
    if params.distribution == "beta":
        return lambda x: beta_pdf(x, params.beta_a, params.beta_b)
    if params.distribution == "truncnorm":
        return lambda x: truncnorm_pdf(x, params.truncnorm_mean, params.truncnorm_sd)
    if params.distribution == "uniform":
        return lambda x: 1.0 if 0.0 < x < 1.0 else 0.0
    raise ValueError(f"Unsupported distribution: {params.distribution}")


def crowding_function(params: ModelParams) -> Callable[[int], float]:
    if params.crowding == "power":
        return lambda k: k ** (-params.gamma)
    if params.crowding == "linear":
        return lambda k: 1.0 / (1.0 + params.eta * (k - 1.0))
    if params.crowding == "exponential":
        return lambda k: math.exp(-params.eta * (k - 1.0))
    raise ValueError(f"Unsupported crowding rule: {params.crowding}")


def build_grid(params: ModelParams) -> tuple[List[float], List[float]]:
    pdf = distribution_pdf(params)
    step = 1.0 / params.grid_points
    mus: List[float] = []
    raw_weights: List[float] = []
    for i in range(params.grid_points):
        mu = (i + 0.5) * step
        mus.append(mu)
        raw_weights.append(pdf(mu))
    total = sum(raw_weights)
    if total <= 0.0:
        raise ValueError("Grid weights are zero; distribution specification is invalid.")
    weights = [w / total for w in raw_weights]
    return mus, weights


def binomial_pmf(n: int, p: float, s: int) -> float:
    if s < 0 or s > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if s == 0 else 0.0
    if p >= 1.0:
        return 1.0 if s == n else 0.0
    log_pmf = (
        math.lgamma(n + 1)
        - math.lgamma(s + 1)
        - math.lgamma(n - s + 1)
        + s * math.log(p)
        + (n - s) * math.log1p(-p)
    )
    return math.exp(log_pmf)


def binomial_upper_tail(n: int, p: float, k: int) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    pmf = binomial_pmf(n, p, k)
    tail = pmf
    ratio_factor = p / (1.0 - p)
    for s in range(k, n):
        pmf *= ((n - s) / (s + 1.0)) * ratio_factor
        tail += pmf
    if tail < 0.0:
        return 0.0
    if tail > 1.0:
        return 1.0
    return tail


def critical_successes_exact(alpha: float, n: int, mu_b: float) -> int:
    pmf = (1.0 - mu_b) ** n
    tail = 1.0
    if tail <= alpha:
        return 0
    for s in range(0, n):
        tail -= pmf
        next_s = s + 1
        if tail <= alpha:
            return next_s
        if s < n:
            if mu_b == 1.0:
                pmf = 0.0
            elif mu_b == 0.0:
                pmf = 0.0
            else:
                pmf *= ((n - s) / (next_s)) * (mu_b / (1.0 - mu_b))
    return n + 1


def critical_successes_normal(alpha: float, n: int, mu_b: float) -> int:
    if alpha <= 0.0:
        return n + 1
    if alpha >= 1.0:
        return 0
    variance = n * mu_b * (1.0 - mu_b)
    if variance <= 0.0:
        return n + 1 if mu_b <= 0.0 else 0
    z = NORMAL.inv_cdf(1.0 - alpha)
    threshold = n * mu_b + z * math.sqrt(variance)
    return min(n + 1, max(0, math.ceil(threshold)))


def pass_probability_exact(alpha: float, mu: float, n: int, mu_b: float) -> float:
    k = critical_successes_exact(alpha, n, mu_b)
    return binomial_upper_tail(n, mu, k)


def pass_probability_normal(alpha: float, mu: float, n: int, mu_b: float) -> float:
    k = critical_successes_normal(alpha, n, mu_b)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    variance = n * mu * (1.0 - mu)
    if variance <= 0.0:
        return 1.0 if mu >= 1.0 and k <= n else 0.0
    z = ((k - 0.5) - n * mu) / math.sqrt(variance)
    return 1.0 - NORMAL.cdf(z)


def pass_probability(alpha: float, mu: float, n: int, mu_b: float, method: str) -> float:
    if n <= 0:
        return 0.0
    if method == "exact":
        return pass_probability_exact(alpha, mu, n, mu_b)
    if method == "normal":
        return pass_probability_normal(alpha, mu, n, mu_b)
    raise ValueError(f"Unsupported tail method: {method}")


def precompute_pass_rows(alpha: float, params: ModelParams, mus: Sequence[float]) -> tuple[List[int], List[List[float]]]:
    n_values = [0] + list(range(params.n_min, params.n_max + 1))
    rows: List[List[float]] = []
    for mu in mus:
        rows.append(compute_pass_row(alpha, mu, params, n_values))
    return n_values, rows


def compute_pass_row(alpha: float, mu: float, params: ModelParams, n_values: Sequence[int]) -> List[float]:
    row = [0.0]
    for n in n_values[1:]:
        row.append(pass_probability(alpha, mu, n, params.mu_b, params.tail_method))
    return row


def select_best_response(row: Sequence[float], n_values: Sequence[int], reward_level: float, params: ModelParams) -> tuple[int, int, float, float]:
    best_idx = 0
    best_utility = 0.0
    best_pass = 0.0
    for idx in range(1, len(n_values)):
        n = n_values[idx]
        utility = reward_level * row[idx] - (params.c0 + params.c * n)
        if utility > best_utility + TIE_TOL:
            best_idx = idx
            best_utility = utility
            best_pass = row[idx]
        elif abs(utility - best_utility) <= TIE_TOL:
            if row[idx] > best_pass + TIE_TOL:
                best_idx = idx
                best_pass = row[idx]
            elif abs(row[idx] - best_pass) <= TIE_TOL and n < n_values[best_idx]:
                best_idx = idx
    return best_idx, n_values[best_idx], best_pass, best_utility


def binomial_expectation_of_crowding(m_minus_1: int, approval_probability: float, crowding: Callable[[int], float]) -> float:
    if m_minus_1 < 0:
        return 1.0
    if approval_probability <= 0.0:
        return crowding(1)
    if approval_probability >= 1.0:
        return crowding(m_minus_1 + 1)

    q = 1.0 - approval_probability
    pmf = q ** m_minus_1
    expected = crowding(1) * pmf
    ratio_factor = approval_probability / q
    for k in range(0, m_minus_1):
        pmf *= ((m_minus_1 - k) / (k + 1.0)) * ratio_factor
        expected += crowding(k + 2) * pmf
    return expected


def follower_summary(
    alpha: float,
    reward_level: float,
    params: ModelParams,
    mus: Sequence[float],
    weights: Sequence[float],
    n_values: Sequence[int],
    pass_rows: Sequence[Sequence[float]],
) -> dict:
    approval_probability = 0.0
    false_approval_probability = 0.0
    true_approval_probability = 0.0
    entry_rate = 0.0
    null_entry_rate = 0.0
    mean_sample_size = 0.0
    null_mass = 0.0
    alt_mass = 0.0
    participation_threshold = None
    baseline_row = compute_pass_row(alpha, params.mu_b, params, n_values)
    null_action_ceiling_by_n = {
        n: baseline_row[idx] for idx, n in enumerate(n_values) if n > 0
    }
    action_specific_bound_numerator = 0.0

    for mu, weight, row in zip(mus, weights, pass_rows):
        best_idx, best_n, best_pass, _ = select_best_response(row, n_values, reward_level, params)
        approval_probability += weight * best_pass
        mean_sample_size += weight * best_n
        if best_n > 0:
            entry_rate += weight
            if participation_threshold is None:
                participation_threshold = mu
        if mu <= params.mu_b:
            null_mass += weight
            false_approval_probability += weight * best_pass
            if best_n > 0:
                null_entry_rate += weight
                action_specific_bound_numerator += weight * null_action_ceiling_by_n[best_n]
        else:
            alt_mass += weight
            true_approval_probability += weight * best_pass

    expected_false_approvals = params.m * false_approval_probability
    expected_true_approvals = params.m * true_approval_probability
    expected_false_negatives = params.m * max(0.0, alt_mass - true_approval_probability)
    mfdr = 0.0 if approval_probability <= 0.0 else false_approval_probability / approval_probability
    pfdr = mfdr
    prob_any_approval = 1.0 - (1.0 - approval_probability) ** params.m
    fdr = prob_any_approval * pfdr
    sensitivity = 0.0 if alt_mass <= 0.0 else true_approval_probability / alt_mass
    specificity = 1.0 if null_mass <= 0.0 else 1.0 - false_approval_probability / null_mass
    _, baseline_best_n, _, baseline_utility = select_best_response(baseline_row, n_values, reward_level, params)
    baseline_active = baseline_best_n > 0
    action_specific_mfdr_bound = (
        0.0 if approval_probability <= 0.0 else action_specific_bound_numerator / approval_probability
    )
    null_entry_mfdr_bound = (
        0.0 if approval_probability <= 0.0 else alpha * null_entry_rate / approval_probability
    )

    return {
        "alpha": alpha,
        "approval_probability": approval_probability,
        "prob_any_approval": prob_any_approval,
        "false_approval_probability": false_approval_probability,
        "true_approval_probability": true_approval_probability,
        "entry_rate": entry_rate,
        "null_entry_rate": null_entry_rate,
        "mean_sample_size": mean_sample_size,
        "participation_threshold": participation_threshold,
        "expected_false_approvals": expected_false_approvals,
        "expected_true_approvals": expected_true_approvals,
        "expected_false_negatives": expected_false_negatives,
        "mfdr": mfdr,
        "pfdr": pfdr,
        "fdr": fdr,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "baseline_best_n": baseline_best_n,
        "baseline_utility": baseline_utility,
        "baseline_active": baseline_active,
        "action_specific_mfdr_bound": action_specific_mfdr_bound,
        "null_entry_mfdr_bound": null_entry_mfdr_bound,
    }


def phi_of_reward(
    alpha: float,
    reward_level: float,
    params: ModelParams,
    mus: Sequence[float],
    weights: Sequence[float],
    n_values: Sequence[int],
    pass_rows: Sequence[Sequence[float]],
) -> tuple[float, dict]:
    summary = follower_summary(alpha, reward_level, params, mus, weights, n_values, pass_rows)
    crowding = crowding_function(params)
    effective_factor = binomial_expectation_of_crowding(params.m - 1, summary["approval_probability"], crowding)
    return params.reward * effective_factor, summary


def solve_fixed_point_for_alpha(
    alpha: float,
    params: ModelParams,
    mus: Sequence[float],
    weights: Sequence[float],
) -> AlphaSummary:
    n_values, pass_rows = precompute_pass_rows(alpha, params, mus)

    lo = 0.0
    hi = params.reward
    for _ in range(params.fixed_point_iters):
        mid = 0.5 * (lo + hi)
        phi_mid, summary = phi_of_reward(alpha, mid, params, mus, weights, n_values, pass_rows)
        gap = phi_mid - mid
        if abs(gap) <= params.fixed_point_tol or (hi - lo) <= params.fixed_point_tol:
            lo = mid
            hi = mid
            break
        if gap > 0.0:
            lo = mid
        else:
            hi = mid

    candidates = []
    for reward_candidate in sorted({0.5 * (lo + hi), lo, hi}):
        phi_candidate, summary_candidate = phi_of_reward(
            alpha, reward_candidate, params, mus, weights, n_values, pass_rows
        )
        candidates.append((abs(phi_candidate - reward_candidate), reward_candidate, phi_candidate, summary_candidate))

    _, reward_star, phi_star, summary = min(candidates, key=lambda item: item[0])
    cost_floor = params.c0 + params.c * params.n_min
    mfdr_bound_effective = min(1.0, alpha * reward_star / cost_floor)
    mfdr_bound_monopoly = min(1.0, alpha * params.reward / cost_floor)
    return AlphaSummary(
        alpha=alpha,
        fixed_point_reward=reward_star,
        fixed_point_gap=phi_star - reward_star,
        approval_probability=summary["approval_probability"],
        prob_any_approval=summary["prob_any_approval"],
        false_approval_probability=summary["false_approval_probability"],
        true_approval_probability=summary["true_approval_probability"],
        entry_rate=summary["entry_rate"],
        null_entry_rate=summary["null_entry_rate"],
        mean_sample_size=summary["mean_sample_size"],
        participation_threshold=summary["participation_threshold"],
        expected_false_approvals=summary["expected_false_approvals"],
        expected_true_approvals=summary["expected_true_approvals"],
        expected_false_negatives=summary["expected_false_negatives"],
        mfdr=summary["mfdr"],
        pfdr=summary["pfdr"],
        fdr=summary["fdr"],
        sensitivity=summary["sensitivity"],
        specificity=summary["specificity"],
        baseline_best_n=summary["baseline_best_n"],
        baseline_utility=summary["baseline_utility"],
        baseline_active=summary["baseline_active"],
        cost_floor=cost_floor,
        mfdr_bound_action_specific=summary["action_specific_mfdr_bound"],
        mfdr_bound_null_entry=summary["null_entry_mfdr_bound"],
        mfdr_bound_effective=mfdr_bound_effective,
        mfdr_bound_monopoly=mfdr_bound_monopoly,
        pfdr_bound_action_specific=summary["action_specific_mfdr_bound"],
        pfdr_bound_null_entry=summary["null_entry_mfdr_bound"],
        pfdr_bound_effective=mfdr_bound_effective,
        pfdr_bound_monopoly=mfdr_bound_monopoly,
        fdr_bound_action_specific=summary["prob_any_approval"] * summary["action_specific_mfdr_bound"],
        fdr_bound_null_entry=summary["prob_any_approval"] * summary["null_entry_mfdr_bound"],
        fdr_bound_effective=summary["prob_any_approval"] * mfdr_bound_effective,
        fdr_bound_monopoly=summary["prob_any_approval"] * mfdr_bound_monopoly,
    )


def solve_alpha_grid(alphas: Sequence[float], params: ModelParams) -> List[AlphaSummary]:
    mus, weights = build_grid(params)
    return [solve_fixed_point_for_alpha(alpha, params, mus, weights) for alpha in alphas]


def estimate_alpha_hat(
    summaries: Sequence[AlphaSummary],
    mu_b: float,
    fp_tol: float = 1e-10,
    mfdr_tol: float = 1e-10,
) -> AlphaHatEstimate:
    alpha_threshold_cross = None
    alpha_false_positive_cross = None
    alpha_mfdr_positive = None
    for summary in summaries:
        if alpha_threshold_cross is None and summary.baseline_active:
            alpha_threshold_cross = summary.alpha
        if alpha_false_positive_cross is None and summary.false_approval_probability > fp_tol:
            alpha_false_positive_cross = summary.alpha
        if alpha_mfdr_positive is None and summary.mfdr > mfdr_tol:
            alpha_mfdr_positive = summary.alpha
    return AlphaHatEstimate(
        alpha_threshold_cross=alpha_threshold_cross,
        alpha_false_positive_cross=alpha_false_positive_cross,
        alpha_mfdr_positive=alpha_mfdr_positive,
    )


def refine_alpha_hat_threshold(
    alpha_lo: float,
    alpha_hi: float,
    params: ModelParams,
    refine_iters: int = 16,
) -> tuple[float, AlphaSummary]:
    mus, weights = build_grid(params)
    lo = alpha_lo
    hi = alpha_hi
    lo_summary = solve_fixed_point_for_alpha(lo, params, mus, weights)
    hi_summary = solve_fixed_point_for_alpha(hi, params, mus, weights)
    if lo_summary.baseline_active:
        return lo, lo_summary
    if not hi_summary.baseline_active:
        raise ValueError("Refinement interval does not bracket the threshold crossing.")

    for _ in range(refine_iters):
        mid = 0.5 * (lo + hi)
        mid_summary = solve_fixed_point_for_alpha(mid, params, mus, weights)
        if mid_summary.baseline_active:
            hi = mid
            hi_summary = mid_summary
        else:
            lo = mid
            lo_summary = mid_summary
    return hi, hi_summary


def parse_alpha_grid(alpha_grid: str) -> List[float]:
    pieces = alpha_grid.split(":")
    if len(pieces) != 3:
        raise ValueError("alpha-grid must have the form start:end:count")
    start = float(pieces[0])
    end = float(pieces[1])
    count = int(pieces[2])
    if count < 2:
        return [start]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def format_threshold(x: float | None) -> str:
    return "None" if x is None else f"{x:.4f}"


def print_table(summaries: Sequence[AlphaSummary]) -> None:
    header = (
        "alpha    r*       approve   p_any    pFDR     FDR      sens     spec     "
        "entry    mean_n   mu_tau"
    )
    print(header)
    for s in summaries:
        print(
            f"{s.alpha:0.4f}   "
            f"{s.fixed_point_reward:0.4f}   "
            f"{s.approval_probability:0.4f}   "
            f"{s.prob_any_approval:0.4f}   "
            f"{s.pfdr:0.4f}   "
            f"{s.fdr:0.4f}   "
            f"{s.sensitivity:0.4f}   "
            f"{s.specificity:0.4f}   "
            f"{s.entry_rate:0.4f}   "
            f"{s.mean_sample_size:0.2f}   "
            f"{format_threshold(s.participation_threshold)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=None, help="Solve the model at a single alpha.")
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.01:0.10:10",
        help="Sweep a grid of alpha values in start:end:count form.",
    )
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--mu-b", type=float, default=0.5)
    parser.add_argument("--reward", type=float, default=20.0)
    parser.add_argument("--c0", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.08)
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=80)
    parser.add_argument("--grid-points", type=int, default=301)
    parser.add_argument("--distribution", choices=["beta", "truncnorm", "uniform"], default="beta")
    parser.add_argument("--beta-a", type=float, default=6.0)
    parser.add_argument("--beta-b", type=float, default=5.0)
    parser.add_argument("--truncnorm-mean", type=float, default=0.58)
    parser.add_argument("--truncnorm-sd", type=float, default=0.12)
    parser.add_argument("--crowding", choices=["power", "linear", "exponential"], default="power")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--tail-method", choices=["exact", "normal"], default="exact")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text table.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = ModelParams(
        m=args.m,
        mu_b=args.mu_b,
        reward=args.reward,
        c0=args.c0,
        c=args.c,
        n_min=args.n_min,
        n_max=args.n_max,
        grid_points=args.grid_points,
        distribution=args.distribution,
        beta_a=args.beta_a,
        beta_b=args.beta_b,
        truncnorm_mean=args.truncnorm_mean,
        truncnorm_sd=args.truncnorm_sd,
        crowding=args.crowding,
        gamma=args.gamma,
        eta=args.eta,
        tail_method=args.tail_method,
    )

    alphas = [args.alpha] if args.alpha is not None else parse_alpha_grid(args.alpha_grid)
    summaries = solve_alpha_grid(alphas, params)
    alpha_hat = estimate_alpha_hat(summaries, params.mu_b)

    if args.json:
        payload = {
            "alpha_hat": asdict(alpha_hat),
            "params": asdict(params),
            "summaries": [asdict(summary) for summary in summaries],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(asdict(params), sort_keys=True))
        print(json.dumps(asdict(alpha_hat), sort_keys=True))
        print_table(summaries)


if __name__ == "__main__":
    main()
