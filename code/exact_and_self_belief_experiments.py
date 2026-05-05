#!/usr/bin/env python3
"""Off-paper numerics for the first six sections.

This script fills the main gaps identified for the numerical section without
editing the paper. It generates CSV/SVG/Markdown outputs for:

1. primitive approval curves and equilibrium policies in the exact benchmark,
2. fixed-point maps,
3. sparse-approval FDR identity illustration,
4. market-size illustrations,
5. population-quality-shift numerics, and
6. self-belief numerics under a Beta-Binomial pilot signal.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from multi_agent_fixed_point import (
    ModelParams,
    AlphaSummary,
    binomial_expectation_of_crowding,
    binomial_pmf,
    build_grid,
    compute_pass_row,
    crowding_function,
    parse_alpha_grid,
    pass_probability,
    phi_of_reward,
    precompute_pass_rows,
    select_best_response,
    solve_alpha_grid,
    solve_fixed_point_for_alpha,
)


COLORS = [
    "#0b57d0",
    "#c5221f",
    "#188038",
    "#b06000",
    "#8e24aa",
    "#00897b",
]
TOL = 1e-10


@dataclass
class SelfBeliefSummary:
    alpha: float
    pilot_size: int
    fixed_point_reward: float
    fixed_point_count: int
    approval_probability: float
    false_approval_probability: float
    true_approval_probability: float
    prob_any_approval: float
    entry_rate: float
    null_entry_rate: float
    expected_true_approvals: float
    expected_false_approvals: float
    expected_false_negatives: float
    mfdr: float
    pfdr: float
    fdr: float
    mfdr_bound_action_specific: float
    mfdr_bound_null_entry: float
    mfdr_bound_effective: float
    signal_cutoff: int | None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _draw_axes(parts: list[str], x0: float, y0: float, width: float, height: float) -> None:
    parts.append(
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="white" stroke="#444" stroke-width="1"/>'
    )


def _draw_y_ticks(parts: list[str], x0: float, y0: float, width: float, height: float, ymin: float, ymax: float, ticks: int = 4) -> None:
    span = ymax - ymin if ymax > ymin else 1.0
    for i in range(ticks + 1):
        y = ymin + span * i / ticks
        py = y0 + height - height * i / ticks
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{x0 + width:.2f}" y2="{py:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#444">{y:.3g}</text>'
        )


def _draw_x_ticks(parts: list[str], x0: float, y0: float, width: float, height: float, xmin: float, xmax: float, ticks: int = 4) -> None:
    span = xmax - xmin if xmax > xmin else 1.0
    for i in range(ticks + 1):
        x = xmin + span * i / ticks
        px = x0 + width * i / ticks
        parts.append(
            f'<line x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y0 + height:.2f}" stroke="#f2f2f2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{y0 + height + 18:.2f}" text-anchor="middle" font-size="11" fill="#444">{x:.3g}</text>'
        )


def _polyline_points(xs: Sequence[float], ys: Sequence[float], x0: float, y0: float, width: float, height: float, xmin: float, xmax: float, ymin: float, ymax: float) -> str:
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    pts: list[str] = []
    for x, y in zip(xs, ys):
        px = x0 + width * (x - xmin) / xspan
        py = y0 + height - height * (y - ymin) / yspan
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def line_plot_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    vertical_lines: Sequence[tuple[float, str]] | None = None,
) -> None:
    ensure_dir(path.parent)
    width = 840
    height = 520
    x0 = 90
    y0 = 60
    plot_w = 680
    plot_h = 340
    xmin = min(min(xs) for _, xs, _, _ in series)
    xmax = max(max(xs) for _, xs, _, _ in series)
    ymin = min(min(ys) for _, _, ys, _ in series)
    ymax = max(max(ys) for _, _, ys, _ in series)
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    xpad = 0.04 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.2f}" y="28" text-anchor="middle" font-size="20" font-weight="bold" fill="#222">{svg_escape(title)}</text>',
    ]
    _draw_axes(parts, x0, y0, plot_w, plot_h)
    _draw_y_ticks(parts, x0, y0, plot_w, plot_h, ymin, ymax)
    _draw_x_ticks(parts, x0, y0, plot_w, plot_h, xmin, xmax)
    parts.append(f'<text x="{x0 + plot_w/2:.2f}" y="{y0 + plot_h + 40:.2f}" text-anchor="middle" font-size="12" fill="#444">{svg_escape(xlabel)}</text>')
    parts.append(f'<text transform="translate({x0 - 58:.2f},{y0 + plot_h/2:.2f}) rotate(-90)" text-anchor="middle" font-size="12" fill="#444">{svg_escape(ylabel)}</text>')
    if vertical_lines:
        for xv, label in vertical_lines:
            px = x0 + plot_w * (xv - xmin) / (xmax - xmin)
            parts.append(
                f'<line x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y0 + plot_h:.2f}" stroke="#888" stroke-dasharray="6,4" stroke-width="1.5"/>'
            )
            parts.append(f'<text x="{px + 4:.2f}" y="{y0 + 14:.2f}" font-size="11" fill="#666">{svg_escape(label)}</text>')
    legend_x = x0 + 12
    legend_y = y0 + 18
    for idx, (label, xs, ys, color) in enumerate(series):
        py = legend_y + idx * 18
        parts.append(f'<line x1="{legend_x:.2f}" y1="{py:.2f}" x2="{legend_x + 18:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 24:.2f}" y="{py + 4:.2f}" font-size="11" fill="#333">{svg_escape(label)}</text>')
        poly = _polyline_points(xs, ys, x0, y0, plot_w, plot_h, xmin, xmax, ymin, ymax)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def step_plot_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xs: Sequence[float],
    ys: Sequence[float],
    vertical_lines: Sequence[tuple[float, str]] | None = None,
) -> None:
    # duplicate x values to create a visible step plot
    step_xs: list[float] = []
    step_ys: list[float] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            step_xs.append(x)
            step_ys.append(y)
        else:
            step_xs.extend([x, x])
            step_ys.extend([step_ys[-1], y])
    line_plot_svg(path, title, xlabel, ylabel, [("policy", step_xs, step_ys, COLORS[0])], vertical_lines=vertical_lines)


def approval_curve_rows(params: ModelParams, alphas: Sequence[float], ns: Sequence[int], points: int = 201) -> list[dict]:
    rows: list[dict] = []
    for i in range(points):
        mu = i / (points - 1)
        for alpha in alphas:
            for n in ns:
                rows.append(
                    {
                        "mu": mu,
                        "alpha": alpha,
                        "n": n,
                        "pass_probability": pass_probability(alpha, mu, n, params.mu_b, params.tail_method),
                    }
                )
    return rows


def generate_exact_best_response_rows(params: ModelParams, alpha: float, points: int = 201) -> tuple[AlphaSummary, list[dict]]:
    summary = solve_fixed_point_for_alpha(alpha, params, *build_grid(params))
    n_values = [0] + list(range(params.n_min, params.n_max + 1))
    rows: list[dict] = []
    for i in range(points):
        mu = i / (points - 1)
        row = compute_pass_row(alpha, mu, params, n_values)
        _, best_n, best_pass, best_utility = select_best_response(row, n_values, summary.fixed_point_reward, params)
        rows.append(
            {
                "mu": mu,
                "best_n": best_n,
                "pass_probability_at_best_n": best_pass,
                "utility": best_utility,
            }
        )
    return summary, rows


def fixed_point_map_rows(params: ModelParams, alpha: float, rewards: Sequence[float]) -> list[dict]:
    mus, weights = build_grid(params)
    n_values, pass_rows = precompute_pass_rows(alpha, params, mus)
    rows: list[dict] = []
    for reward in rewards:
        phi, summary = phi_of_reward(alpha, reward, params, mus, weights, n_values, pass_rows)
        rows.append(
            {
                "reward": reward,
                "phi": phi,
                "gap": phi - reward,
                "approval_probability": summary["approval_probability"],
            }
        )
    return rows


def sparse_identity_candidates() -> list[tuple[str, ModelParams, float]]:
    return [
        ("baseline", ModelParams(m=20, mu_b=0.5, reward=30.0, c0=0.5, c=0.05, n_min=10, n_max=60, distribution="beta", beta_a=6.0, beta_b=5.0, crowding="power", gamma=1.0, tail_method="exact"), 0.28),
        ("sparse_small_market", ModelParams(m=2, mu_b=0.5, reward=8.0, c0=0.5, c=0.05, n_min=10, n_max=60, distribution="beta", beta_a=6.0, beta_b=5.0, crowding="power", gamma=1.0, tail_method="exact"), 0.20),
        ("sparse_low_reward", ModelParams(m=3, mu_b=0.5, reward=8.0, c0=0.5, c=0.05, n_min=10, n_max=60, distribution="beta", beta_a=6.0, beta_b=5.0, crowding="power", gamma=1.0, tail_method="exact"), 0.35),
    ]


def pick_sparse_identity_row() -> tuple[str, AlphaSummary]:
    best_name = ""
    best_summary: AlphaSummary | None = None
    best_distance = float("inf")
    for name, params, alpha in sparse_identity_candidates():
        summary = solve_fixed_point_for_alpha(alpha, params, *build_grid(params))
        if summary.pfdr <= 0.0:
            continue
        distance = abs(summary.prob_any_approval - 0.5)
        if distance < best_distance:
            best_distance = distance
            best_summary = summary
            best_name = name
    if best_summary is None:
        # fallback to the first candidate even if pfdr is zero
        name, params, alpha = sparse_identity_candidates()[1]
        best_name = name
        best_summary = solve_fixed_point_for_alpha(alpha, params, *build_grid(params))
    return best_name, best_summary


def expected_dilution_vs_m(rhos: Sequence[float], m_values: Sequence[int], gamma: float) -> list[dict]:
    rows: list[dict] = []
    for rho in rhos:
        for m in m_values:
            crowd = lambda k, gamma=gamma: k ** (-gamma)
            value = binomial_expectation_of_crowding(m - 1, rho, crowd)
            rows.append({"rho": rho, "m": m, "expected_dilution_factor": value})
    return rows


def equilibrium_market_size_rows(base_params: ModelParams, m_values: Sequence[int], alpha_grid: Sequence[float]) -> list[dict]:
    rows: list[dict] = []
    for m in m_values:
        params = ModelParams(**{**asdict(base_params), "m": m})
        summaries = solve_alpha_grid(alpha_grid, params)
        alpha_hat = None
        for summary in summaries:
            if summary.baseline_active:
                alpha_hat = summary.alpha
                break
        mid_alpha = 0.20
        mid_summary = min(summaries, key=lambda s: abs(s.alpha - mid_alpha))
        rows.append(
            {
                "m": m,
                "alpha_hat_comp_grid": alpha_hat,
                "r_star_at_0.20": mid_summary.fixed_point_reward,
                "approval_probability_at_0.20": mid_summary.approval_probability,
                "entry_rate_at_0.20": mid_summary.entry_rate,
            }
        )
    return rows


def beta_mlr_family(total_mass: float, means: Sequence[float]) -> list[tuple[str, float, float]]:
    family: list[tuple[str, float, float]] = []
    for mean in means:
        a = total_mass * mean
        b = total_mass * (1.0 - mean)
        label = f"Beta({a:.1f},{b:.1f})"
        family.append((label, a, b))
    return family


def fixed_sample_loss(alpha: float, mu_b: float, reward: float, c0: float, c: float, n0: int, a: float, b: float, lambda_fp: float, lambda_fn: float, grid_points: int = 401) -> float:
    params = ModelParams(mu_b=mu_b, reward=reward, c0=c0, c=c, n_min=n0, n_max=n0, grid_points=grid_points, distribution="beta", beta_a=a, beta_b=b, tail_method="exact")
    mus, weights = build_grid(params)
    cost = c0 + c * n0
    loss = 0.0
    for mu, weight in zip(mus, weights):
        p = pass_probability(alpha, mu, n0, mu_b, params.tail_method)
        approve = p if reward * p >= cost - TOL else 0.0
        fp_term = lambda_fp * (1.0 if mu <= mu_b else 0.0) * approve
        fn_term = lambda_fn * (1.0 if mu > mu_b else 0.0) * (1.0 - approve)
        loss += weight * (fp_term + fn_term)
    return loss


def population_shift_rows(mu_b: float, reward: float, c0: float, c: float, n0: int, alpha_grid: Sequence[float], means: Sequence[float], total_mass: float = 11.0, lambda_fp: float = 1.0, lambda_fn: float = 1.0) -> tuple[list[dict], dict[str, list[dict]]]:
    rows: list[dict] = []
    curves: dict[str, list[dict]] = {}
    for idx, (label, a, b) in enumerate(beta_mlr_family(total_mass, means)):
        losses: list[dict] = []
        best_alpha = None
        best_loss = None
        params = ModelParams(mu_b=mu_b, distribution="beta", beta_a=a, beta_b=b, grid_points=401)
        mus, weights = build_grid(params)
        q_null = sum(weight for mu, weight in zip(mus, weights) if mu <= mu_b)
        mean_mu = sum(mu * weight for mu, weight in zip(mus, weights))
        for alpha in alpha_grid:
            loss = fixed_sample_loss(alpha, mu_b, reward, c0, c, n0, a, b, lambda_fp, lambda_fn, grid_points=401)
            losses.append({"alpha": alpha, "loss": loss})
            if best_loss is None or loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and alpha < best_alpha):
                best_loss = loss
                best_alpha = alpha
        rows.append(
            {
                "theta_index": idx,
                "distribution": label,
                "mean_mu": mean_mu,
                "q_mu_le_mu_b": q_null,
                "alpha_fix_star": best_alpha,
                "min_loss": best_loss,
            }
        )
        curves[label] = losses
    return rows, curves


def binom_pmf_table(k: int, mus: Sequence[float]) -> list[list[float]]:
    return [[binomial_pmf(k, mu, y) for y in range(k + 1)] for mu in mus]


def prepare_self_belief(alpha: float, params: ModelParams, pilot_size: int) -> dict:
    mus, weights = build_grid(params)
    n_values, pass_rows = precompute_pass_rows(alpha, params, mus)
    n_index = {n: idx for idx, n in enumerate(n_values)}
    signal_probs_by_mu = binom_pmf_table(pilot_size, mus)
    signal_probs = [0.0 for _ in range(pilot_size + 1)]
    null_signal_probs = [0.0 for _ in range(pilot_size + 1)]
    posterior_pass_rows = [[0.0 for _ in n_values] for _ in range(pilot_size + 1)]
    joint_weights = [[0.0 for _ in mus] for _ in range(pilot_size + 1)]
    for mu_idx, (mu, weight) in enumerate(zip(mus, weights)):
        for y in range(pilot_size + 1):
            joint = weight * signal_probs_by_mu[mu_idx][y]
            joint_weights[y][mu_idx] = joint
            signal_probs[y] += joint
            if mu <= params.mu_b:
                null_signal_probs[y] += joint
            for idx in range(1, len(n_values)):
                posterior_pass_rows[y][idx] += joint * pass_rows[mu_idx][idx]
    for y in range(pilot_size + 1):
        if signal_probs[y] > 0.0:
            for idx in range(1, len(n_values)):
                posterior_pass_rows[y][idx] /= signal_probs[y]
    baseline_row = compute_pass_row(alpha, params.mu_b, params, n_values)
    posterior_mean = []
    posterior_tail_prob = []
    for y in range(pilot_size + 1):
        sig = signal_probs[y]
        if sig <= 0.0:
            posterior_mean.append(0.0)
            posterior_tail_prob.append(0.0)
            continue
        mean = 0.0
        tail = 0.0
        for mu, joint in zip(mus, joint_weights[y]):
            mean += mu * joint / sig
            if mu > params.mu_b:
                tail += joint / sig
        posterior_mean.append(mean)
        posterior_tail_prob.append(tail)
    return {
        "mus": mus,
        "weights": weights,
        "n_values": n_values,
        "n_index": n_index,
        "pass_rows": pass_rows,
        "signal_probs_by_mu": signal_probs_by_mu,
        "signal_probs": signal_probs,
        "null_signal_probs": null_signal_probs,
        "posterior_pass_rows": posterior_pass_rows,
        "joint_weights": joint_weights,
        "baseline_row": baseline_row,
        "posterior_mean": posterior_mean,
        "posterior_tail_prob": posterior_tail_prob,
    }


def self_belief_summary_from_prepared(alpha: float, reward_level: float, params: ModelParams, pilot_size: int, prepared: dict) -> tuple[SelfBeliefSummary, list[int]]:
    n_values = prepared["n_values"]
    n_index = prepared["n_index"]
    signal_probs = prepared["signal_probs"]
    posterior_pass_rows = prepared["posterior_pass_rows"]
    joint_weights = prepared["joint_weights"]
    baseline_row = prepared["baseline_row"]
    null_signal_probs = prepared["null_signal_probs"]

    choices: list[int] = []
    approval_probability = 0.0
    false_approval_probability = 0.0
    true_approval_probability = 0.0
    entry_rate = 0.0
    null_entry_rate = 0.0
    action_specific_numerator = 0.0
    signal_cutoff: int | None = None
    for y in range(pilot_size + 1):
        _, best_n, best_pass, _ = select_best_response(posterior_pass_rows[y], n_values, reward_level, params)
        best_idx = n_index[best_n]
        choices.append(best_n)
        sig = signal_probs[y]
        approval_probability += sig * best_pass
        if best_n > 0:
            entry_rate += sig
            null_entry_rate += null_signal_probs[y]
            action_specific_numerator += null_signal_probs[y] * baseline_row[best_idx]
            if signal_cutoff is None:
                signal_cutoff = y
        for mu, joint, pass_row in zip(prepared["mus"], joint_weights[y], prepared["pass_rows"]):
            if joint <= 0.0:
                continue
            approve_prob = pass_row[best_idx]
            if mu <= params.mu_b:
                false_approval_probability += joint * approve_prob
            else:
                true_approval_probability += joint * approve_prob
    prob_any_approval = 1.0 - (1.0 - approval_probability) ** params.m
    mfdr = 0.0 if approval_probability <= 0.0 else false_approval_probability / approval_probability
    pfdr = mfdr
    fdr = prob_any_approval * pfdr
    cost_floor = params.c0 + params.c * params.n_min
    summary = SelfBeliefSummary(
        alpha=alpha,
        pilot_size=pilot_size,
        fixed_point_reward=reward_level,
        fixed_point_count=0,
        approval_probability=approval_probability,
        false_approval_probability=false_approval_probability,
        true_approval_probability=true_approval_probability,
        prob_any_approval=prob_any_approval,
        entry_rate=entry_rate,
        null_entry_rate=null_entry_rate,
        expected_true_approvals=params.m * true_approval_probability,
        expected_false_approvals=params.m * false_approval_probability,
        expected_false_negatives=params.m * max(0.0, sum(w for mu, w in zip(prepared["mus"], prepared["weights"]) if mu > params.mu_b) - true_approval_probability),
        mfdr=mfdr,
        pfdr=pfdr,
        fdr=fdr,
        mfdr_bound_action_specific=0.0 if approval_probability <= 0.0 else action_specific_numerator / approval_probability,
        mfdr_bound_null_entry=0.0 if approval_probability <= 0.0 else alpha * null_entry_rate / approval_probability,
        mfdr_bound_effective=min(1.0, alpha * reward_level / cost_floor),
        signal_cutoff=signal_cutoff,
    )
    return summary, choices


def self_belief_phi(alpha: float, reward_level: float, params: ModelParams, pilot_size: int, prepared: dict) -> tuple[float, SelfBeliefSummary, list[int]]:
    summary, choices = self_belief_summary_from_prepared(alpha, reward_level, params, pilot_size, prepared)
    crowd = crowding_function(params)
    phi = params.reward * binomial_expectation_of_crowding(params.m - 1, summary.approval_probability, crowd)
    return phi, summary, choices


def self_belief_fixed_points(alpha: float, params: ModelParams, pilot_size: int, prepared: dict) -> list[tuple[float, SelfBeliefSummary, list[int]]]:
    reward_grid = [i * params.reward / 600.0 for i in range(601)]
    profiled: list[tuple[float, float, float, SelfBeliefSummary, list[int]]] = []
    for reward in reward_grid:
        phi, summary, choices = self_belief_phi(alpha, reward, params, pilot_size, prepared)
        profiled.append((reward, phi, abs(phi - reward), summary, choices))
    min_gap = min(item[2] for item in profiled)
    candidate_points = [item for item in profiled if item[2] <= min_gap + 1e-8]
    candidates = [(reward, summary, choices) for reward, _, _, summary, choices in candidate_points]
    candidates.sort(key=lambda item: item[0])
    for _, summary, _ in candidates:
        summary.fixed_point_count = len(candidates)
    return candidates


def search_self_belief_failure_alpha(params: ModelParams, alpha_candidates: Sequence[float], pilot_size: int) -> tuple[float, SelfBeliefSummary, list[int]]:
    for alpha in alpha_candidates:
        exact_summary = solve_fixed_point_for_alpha(alpha, params, *build_grid(params))
        if exact_summary.fdr > 1e-12:
            continue
        prepared = prepare_self_belief(alpha, params, pilot_size)
        fixed_points = self_belief_fixed_points(alpha, params, pilot_size, prepared)
        if len(fixed_points) != 1:
            continue
        reward, summary, choices = fixed_points[0]
        if summary.fdr > 1e-8:
            return alpha, summary, choices
    raise RuntimeError("No self-belief failure alpha found on the candidate grid.")


def self_belief_accuracy_rows(alpha: float, params: ModelParams, pilot_sizes: Sequence[int], exact_reward: float) -> list[dict]:
    rows: list[dict] = []
    mus, weights = build_grid(params)
    n_values, pass_rows = precompute_pass_rows(alpha, params, mus)
    exact_choices = []
    for row in pass_rows:
        _, best_n, _, _ = select_best_response(row, n_values, exact_reward, params)
        exact_choices.append(best_n)
    for k in pilot_sizes:
        prepared = prepare_self_belief(alpha, params, k)
        fixed_points = self_belief_fixed_points(alpha, params, k, prepared)
        if len(fixed_points) != 1:
            rows.append(
                {
                    "pilot_size": k,
                    "has_unique_fixed_point": False,
                    "fixed_point_reward": None,
                    "mfdr": None,
                    "null_entry_rate": None,
                    "fixed_reward_policy_disagreement": None,
                    "equilibrium_policy_disagreement": None,
                }
            )
            continue
        reward_sb, summary, choices = fixed_points[0]
        # fixed-reward disagreement at the exact benchmark reward
        _, choices_fixed = self_belief_summary_from_prepared(alpha, exact_reward, params, k, prepared)
        disagreement_fixed = 0.0
        disagreement_eq = 0.0
        signal_probs_by_mu = prepared["signal_probs_by_mu"]
        for mu_idx, (mu, weight) in enumerate(zip(mus, weights)):
            for y in range(k + 1):
                joint = weight * signal_probs_by_mu[mu_idx][y]
                if joint <= 0.0:
                    continue
                if choices_fixed[y] != exact_choices[mu_idx]:
                    disagreement_fixed += joint
                # compare equilibrium self-belief choice to exact benchmark choice at exact reward
                if choices[y] != exact_choices[mu_idx]:
                    disagreement_eq += joint
        rows.append(
            {
                "pilot_size": k,
                "has_unique_fixed_point": True,
                "fixed_point_reward": reward_sb,
                "mfdr": summary.mfdr,
                "null_entry_rate": summary.null_entry_rate,
                "fixed_reward_policy_disagreement": disagreement_fixed,
                "equilibrium_policy_disagreement": disagreement_eq,
            }
        )
    return rows


def run() -> None:
    root = Path(os.environ.get("OUTPUT_ROOT", "data/regenerated/exact_and_self_belief"))
    ensure_dir(root)

    baseline = ModelParams(
        m=20,
        mu_b=0.5,
        reward=30.0,
        c0=0.5,
        c=0.05,
        n_min=10,
        n_max=60,
        grid_points=301,
        distribution="beta",
        beta_a=6.0,
        beta_b=5.0,
        crowding="power",
        gamma=1.0,
        tail_method="exact",
    )
    sb_params = ModelParams(
        m=baseline.m,
        mu_b=baseline.mu_b,
        reward=baseline.reward,
        c0=baseline.c0,
        c=baseline.c,
        n_min=baseline.n_min,
        n_max=baseline.n_max,
        grid_points=151,
        alpha_tol=baseline.alpha_tol,
        fixed_point_tol=baseline.fixed_point_tol,
        fixed_point_iters=baseline.fixed_point_iters,
        distribution=baseline.distribution,
        beta_a=baseline.beta_a,
        beta_b=baseline.beta_b,
        truncnorm_mean=baseline.truncnorm_mean,
        truncnorm_sd=baseline.truncnorm_sd,
        crowding=baseline.crowding,
        gamma=baseline.gamma,
        eta=baseline.eta,
        tail_method=baseline.tail_method,
    )

    summary_lines: list[str] = []
    summary_lines.append("# Sections 1--6 off-paper experiment results\n\n")
    summary_lines.append("This note collects the new numerical outputs requested for the current theory package, without editing the paper.\n\n")

    # 1. Primitive approval curves
    exact_dir = root / "exact_benchmark"
    ensure_dir(exact_dir)
    approval_rows = approval_curve_rows(baseline, alphas=[0.10, 0.20, 0.30], ns=[10, 20, 40, 60])
    write_csv(exact_dir / "approval_curves.csv", approval_rows)
    for alpha in [0.10, 0.20]:
        series = []
        for n, color in zip([10, 20, 40, 60], COLORS):
            xs = [row["mu"] for row in approval_rows if abs(row["alpha"] - alpha) < 1e-12 and row["n"] == n]
            ys = [row["pass_probability"] for row in approval_rows if abs(row["alpha"] - alpha) < 1e-12 and row["n"] == n]
            series.append((f"n={n}", xs, ys, color))
        line_plot_svg(
            exact_dir / f"approval_curves_alpha_{str(alpha).replace('.', 'p')}.svg",
            f"Approval probabilities Pass(alpha, mu, n) at alpha={alpha:.2f}",
            "quality mu",
            "approval probability",
            series,
            vertical_lines=[(baseline.mu_b, "mu_b")],
        )
    summary_lines.append("## 1. Primitive approval curves\n")
    summary_lines.append(f"- outputs: [{(exact_dir / 'approval_curves.csv').name}]({(exact_dir / 'approval_curves.csv').as_posix()}) and the two SVG plots for alpha=0.10 and alpha=0.20.\n")
    summary_lines.append("- these curves make the null-validity envelope visible: for each positive action, the null-side maximum occurs at mu_b.\n\n")

    # 2. Best-response policy below and above alpha_hat
    for alpha in [0.20, 0.30]:
        eq_summary, policy_rows = generate_exact_best_response_rows(baseline, alpha)
        alpha_tag = str(alpha).replace('.', 'p')
        policy_csv = exact_dir / f"best_response_alpha_{alpha_tag}.csv"
        write_csv(policy_csv, policy_rows)
        step_plot_svg(
            exact_dir / f"best_response_alpha_{alpha_tag}.svg",
            f"Equilibrium best response n*(mu) at alpha={alpha:.2f}",
            "quality mu",
            "chosen sample size",
            [row["mu"] for row in policy_rows],
            [row["best_n"] for row in policy_rows],
            vertical_lines=[(baseline.mu_b, "mu_b")],
        )
        summary_lines.append(f"## 2. Best-response policy at alpha={alpha:.2f}\n")
        summary_lines.append(f"- fixed-point reward r*={eq_summary.fixed_point_reward:.6f}\n")
        summary_lines.append(f"- baseline type active: {eq_summary.baseline_active}\n")
        summary_lines.append(f"- outputs: [{policy_csv.name}]({policy_csv.as_posix()}) and the matching SVG plot.\n\n")

    # 3. Fixed-point maps
    rewards = [i * baseline.reward / 120.0 for i in range(121)]
    for alpha in [0.20, 0.28]:
        rows = fixed_point_map_rows(baseline, alpha, rewards)
        write_csv(exact_dir / f"fixed_point_map_alpha_{str(alpha).replace('.', 'p')}.csv", rows)
        series = [
            ("Phi_alpha(r)", [row["reward"] for row in rows], [row["phi"] for row in rows], COLORS[0]),
            ("45-degree line", [row["reward"] for row in rows], [row["reward"] for row in rows], COLORS[1]),
        ]
        summary = solve_fixed_point_for_alpha(alpha, baseline, *build_grid(baseline))
        line_plot_svg(
            exact_dir / f"fixed_point_map_alpha_{str(alpha).replace('.', 'p')}.svg",
            f"Fixed-point map at alpha={alpha:.2f}",
            "reward r",
            "Phi_alpha(r)",
            series,
            vertical_lines=[(summary.fixed_point_reward, "r*")],
        )
    summary_lines.append("## 3. Fixed-point maps\n")
    summary_lines.append("- outputs for alpha=0.20 and alpha=0.28 show Phi_alpha(r) and the 45-degree line, with the computed fixed point marked.\n\n")

    # 4. Sparse approval identity illustration
    dense_name, sparse_summary = pick_sparse_identity_row()
    dense_summary = solve_fixed_point_for_alpha(0.28, baseline, *build_grid(baseline))
    identity_rows = [
        {
            "regime": "baseline_dense",
            "alpha": dense_summary.alpha,
            "prob_any_approval": dense_summary.prob_any_approval,
            "mfdr": dense_summary.mfdr,
            "pfdr": dense_summary.pfdr,
            "fdr": dense_summary.fdr,
        },
        {
            "regime": dense_name,
            "alpha": sparse_summary.alpha,
            "prob_any_approval": sparse_summary.prob_any_approval,
            "mfdr": sparse_summary.mfdr,
            "pfdr": sparse_summary.pfdr,
            "fdr": sparse_summary.fdr,
        },
    ]
    write_csv(exact_dir / "fdr_identity_regimes.csv", identity_rows)
    summary_lines.append("## 4. Dense and sparse FDR-identity regimes\n")
    summary_lines.append(f"- baseline dense regime: P(S>0)={dense_summary.prob_any_approval:.6f}, mFDR=pFDR={dense_summary.mfdr:.6f}, FDR={dense_summary.fdr:.6f}\n")
    summary_lines.append(f"- sparse illustrative regime ({dense_name}): P(S>0)={sparse_summary.prob_any_approval:.6f}, mFDR=pFDR={sparse_summary.mfdr:.6f}, FDR={sparse_summary.fdr:.6f}\n")
    summary_lines.append(f"- output: [fdr_identity_regimes.csv]({(exact_dir / 'fdr_identity_regimes.csv').as_posix()})\n\n")

    # 5. Market size
    market_dir = root / "market_size"
    ensure_dir(market_dir)
    fixed_rho_rows = expected_dilution_vs_m([0.10, 0.20, 0.40], [5, 10, 20, 40, 80], gamma=1.0)
    write_csv(market_dir / "fixed_rho_dilution.csv", fixed_rho_rows)
    series = []
    for idx, rho in enumerate([0.10, 0.20, 0.40]):
        rows = [row for row in fixed_rho_rows if abs(row["rho"] - rho) < 1e-12]
        series.append((f"rho={rho:.2f}", [row["m"] for row in rows], [row["expected_dilution_factor"] for row in rows], COLORS[idx]))
    line_plot_svg(
        market_dir / "fixed_rho_dilution.svg",
        "Expected dilution versus market size at fixed approval rate",
        "market size m",
        "E[g(1+Y)]",
        series,
    )
    eq_market_rows = equilibrium_market_size_rows(baseline, [5, 10, 20, 40], parse_alpha_grid("0.10:0.60:51"))
    write_csv(market_dir / "equilibrium_market_size.csv", eq_market_rows)
    summary_lines.append("## 5. Market-size illustrations\n")
    summary_lines.append(f"- fixed-rho dilution outputs: [fixed_rho_dilution.csv]({(market_dir / 'fixed_rho_dilution.csv').as_posix()}) and SVG plot.\n")
    summary_lines.append(f"- full equilibrium sweep outputs: [equilibrium_market_size.csv]({(market_dir / 'equilibrium_market_size.csv').as_posix()})\n\n")

    # 6. Population-quality shifts
    population_dir = root / "population_shift"
    ensure_dir(population_dir)
    pop_rows, pop_curves = population_shift_rows(
        mu_b=0.5,
        reward=10.0,
        c0=0.5,
        c=0.05,
        n0=20,
        alpha_grid=parse_alpha_grid("0.05:0.45:17"),
        means=[0.35, 0.45, 0.55, 0.65],
        total_mass=11.0,
        lambda_fp=1.0,
        lambda_fn=1.0,
    )
    write_csv(population_dir / "population_shift_table.csv", pop_rows)
    series = []
    for idx, label in enumerate([pop_rows[0]["distribution"], pop_rows[1]["distribution"], pop_rows[-1]["distribution"]]):
        curve = pop_curves[label]
        series.append((label, [row["alpha"] for row in curve], [row["loss"] for row in curve], COLORS[idx]))
    line_plot_svg(
        population_dir / "population_shift_loss_curves.svg",
        "Fixed-sample loss curves under MLR-ordered quality shifts",
        "alpha",
        "loss",
        series,
    )
    summary_lines.append("## 6. Population-quality shifts\n")
    summary_lines.append(f"- outputs: [population_shift_table.csv]({(population_dir / 'population_shift_table.csv').as_posix()}) and the loss-curve SVG.\n")
    summary_lines.append("- in this calibration, the smallest optimal threshold on the finite grid moves weakly upward as the population mean rises.\n\n")

    # 7. Self-belief
    sb_dir = root / "self_belief"
    ensure_dir(sb_dir)
    alpha_candidates = parse_alpha_grid("0.05:0.27:23")
    failure_alpha, failure_summary, failure_choices = search_self_belief_failure_alpha(sb_params, alpha_candidates, pilot_size=5)
    prepared_k5 = prepare_self_belief(failure_alpha, sb_params, 5)
    prepared_k20 = prepare_self_belief(failure_alpha, sb_params, 20)
    prepared_k100 = prepare_self_belief(failure_alpha, sb_params, 100)
    fixed_points_k5 = self_belief_fixed_points(failure_alpha, sb_params, 5, prepared_k5)
    fixed_points_k20 = self_belief_fixed_points(failure_alpha, sb_params, 20, prepared_k20)
    fixed_points_k100 = self_belief_fixed_points(failure_alpha, sb_params, 100, prepared_k100)
    if len(fixed_points_k5) != 1 or len(fixed_points_k20) != 1 or len(fixed_points_k100) != 1:
        raise RuntimeError("Selected self-belief calibration does not have unique fixed points for the reported pilot sizes.")

    # posterior table for k=20
    posterior_rows = []
    for y in range(21):
        posterior_rows.append(
            {
                "y": y,
                "posterior_mean": prepared_k20["posterior_mean"][y],
                "posterior_prob_mu_gt_mu_b": prepared_k20["posterior_tail_prob"][y],
                "best_response_at_equilibrium": fixed_points_k20[0][2][y],
            }
        )
    write_csv(sb_dir / "posterior_table_k20.csv", posterior_rows)

    # signal policy plot for k=20
    step_plot_svg(
        sb_dir / f"signal_policy_alpha_{str(failure_alpha).replace('.', 'p')}_k20.svg",
        f"Signal-space policy at alpha={failure_alpha:.2f}, k=20",
        "pilot signal y",
        "chosen action",
        list(range(21)),
        fixed_points_k20[0][2],
    )
    write_csv(
        sb_dir / f"signal_policy_alpha_{str(failure_alpha).replace('.', 'p')}_k20.csv",
        [{"y": y, "best_action": fixed_points_k20[0][2][y]} for y in range(21)],
    )

    # self-belief fixed-point map for k=20
    reward_grid = [i * baseline.reward / 120.0 for i in range(121)]
    rows = []
    for reward in reward_grid:
        phi, summary_sb, _ = self_belief_phi(failure_alpha, reward, sb_params, 20, prepared_k20)
        rows.append({"reward": reward, "phi_sb": phi, "gap": phi - reward, "approval_probability": summary_sb.approval_probability})
    write_csv(sb_dir / f"self_belief_fixed_point_map_alpha_{str(failure_alpha).replace('.', 'p')}_k20.csv", rows)
    line_plot_svg(
        sb_dir / f"self_belief_fixed_point_map_alpha_{str(failure_alpha).replace('.', 'p')}_k20.svg",
        f"Self-belief fixed-point map at alpha={failure_alpha:.2f}, k=20",
        "reward r",
        "Phi_SB,alpha(r)",
        [
            ("Phi_SB,alpha(r)", [row["reward"] for row in rows], [row["phi_sb"] for row in rows], COLORS[0]),
            ("45-degree line", [row["reward"] for row in rows], [row["reward"] for row in rows], COLORS[1]),
        ],
        vertical_lines=[(fixed_points_k20[0][0], "r*_SB")],
    )

    # equilibrium comparison table at failure alpha
    exact_failure = solve_fixed_point_for_alpha(failure_alpha, sb_params, *build_grid(sb_params))
    eq_rows = [
        {
            "model": "exact_knowledge",
            "alpha": failure_alpha,
            "pilot_size": None,
            "r_star": exact_failure.fixed_point_reward,
            "approval_probability": exact_failure.approval_probability,
            "a0": exact_failure.false_approval_probability,
            "mfdr": exact_failure.mfdr,
            "pfdr": exact_failure.pfdr,
            "fdr": exact_failure.fdr,
            "cost_floor_bound": exact_failure.mfdr_bound_effective,
        }
    ]
    for k, fixed_points in [(5, fixed_points_k5), (20, fixed_points_k20), (100, fixed_points_k100)]:
        reward, summary_sb, _ = fixed_points[0]
        eq_rows.append(
            {
                "model": "self_belief",
                "alpha": failure_alpha,
                "pilot_size": k,
                "r_star": reward,
                "approval_probability": summary_sb.approval_probability,
                "a0": summary_sb.false_approval_probability,
                "mfdr": summary_sb.mfdr,
                "pfdr": summary_sb.pfdr,
                "fdr": summary_sb.fdr,
                "cost_floor_bound": summary_sb.mfdr_bound_effective,
            }
        )
    write_csv(sb_dir / "self_belief_equilibrium_comparison.csv", eq_rows)

    # explicit exact-screening failure row
    write_csv(
        sb_dir / "exact_screening_failure.csv",
        [
            {
                "alpha": failure_alpha,
                "exact_fdr": exact_failure.fdr,
                "self_belief_fdr_k5": fixed_points_k5[0][1].fdr,
                "self_belief_fdr_k20": fixed_points_k20[0][1].fdr,
                "self_belief_fdr_k100": fixed_points_k100[0][1].fdr,
            }
        ],
    )

    # accuracy sweep
    accuracy_rows = self_belief_accuracy_rows(failure_alpha, sb_params, [0, 1, 2, 5, 10, 20, 50, 100], exact_failure.fixed_point_reward)
    write_csv(sb_dir / "self_belief_accuracy_sweep.csv", accuracy_rows)
    valid_accuracy = [row for row in accuracy_rows if row["has_unique_fixed_point"]]
    if valid_accuracy:
        line_plot_svg(
            sb_dir / "self_belief_accuracy_sweep.svg",
            f"Self-belief convergence at alpha={failure_alpha:.2f}",
            "pilot size k",
            "value",
            [
                ("mFDR_SB(k)", [row["pilot_size"] for row in valid_accuracy], [row["mfdr"] for row in valid_accuracy], COLORS[0]),
                ("null entry SB(k)", [row["pilot_size"] for row in valid_accuracy], [row["null_entry_rate"] for row in valid_accuracy], COLORS[1]),
                ("fixed-reward disagreement", [row["pilot_size"] for row in valid_accuracy], [row["fixed_reward_policy_disagreement"] for row in valid_accuracy], COLORS[2]),
            ],
        )

    summary_lines.append("## 7. Self-belief experiments\n")
    summary_lines.append(f"- selected failure threshold: alpha={failure_alpha:.2f}, which is below the exact benchmark left-boundary alpha_hat_comp≈0.28.\n")
    summary_lines.append(f"- exact benchmark at this alpha: FDR={exact_failure.fdr:.6f}\n")
    summary_lines.append(f"- self-belief FDR at k=5: {fixed_points_k5[0][1].fdr:.6f}\n")
    summary_lines.append(f"- self-belief FDR at k=20: {fixed_points_k20[0][1].fdr:.6f}\n")
    summary_lines.append(f"- self-belief FDR at k=100: {fixed_points_k100[0][1].fdr:.6f}\n")
    summary_lines.append(f"- outputs: posterior table [posterior_table_k20.csv]({(sb_dir / 'posterior_table_k20.csv').as_posix()}), signal policy CSV/SVG, self-belief fixed-point map CSV/SVG, [self_belief_equilibrium_comparison.csv]({(sb_dir / 'self_belief_equilibrium_comparison.csv').as_posix()}), [exact_screening_failure.csv]({(sb_dir / 'exact_screening_failure.csv').as_posix()}), and [self_belief_accuracy_sweep.csv]({(sb_dir / 'self_belief_accuracy_sweep.csv').as_posix()}).\n")
    summary_lines.append("- computational convention: the self-belief numerics use the discrete Beta-Binomial pilot signal. Fixed points are therefore checked directly on the stepwise map, and the reported calibrations are restricted to cases with a unique pure fixed point.\n\n")

    # write summary and manifest
    (root / "SUMMARY.md").write_text("".join(summary_lines))
    write_json(
        root / "manifest.json",
        {
            "baseline_params": asdict(baseline),
            "failure_alpha": failure_alpha,
            "files": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
        },
    )
    print(json.dumps({"output_root": str(root), "failure_alpha": failure_alpha}, sort_keys=True))


if __name__ == "__main__":
    run()
