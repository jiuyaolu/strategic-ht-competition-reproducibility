#!/usr/bin/env python3
"""Principal-side alpha sweep experiments for the symmetric competition model.

This script is off-paper infrastructure. It reuses the fixed-point solver to:

1. sweep equilibrium outcomes over an alpha grid,
2. export CSV and JSON summaries,
3. generate lightweight SVG figures without third-party plotting libraries, and
4. write a short markdown summary focused on principal-side design questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from multi_agent_fixed_point import AlphaSummary, ModelParams, estimate_alpha_hat, parse_alpha_grid, solve_alpha_grid


COLORS = [
    "#0b57d0",
    "#c5221f",
    "#188038",
    "#b06000",
    "#8e24aa",
    "#00897b",
]


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, summaries: Sequence[AlphaSummary]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(asdict(summaries[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _polyline_points(
    xs: Sequence[float],
    ys: Sequence[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> str:
    points: list[str] = []
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    for x, y in zip(xs, ys):
        px = x0 + width * (x - xmin) / xspan
        py = y0 + height - height * (y - ymin) / yspan
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _draw_axes(parts: list[str], x0: float, y0: float, width: float, height: float) -> None:
    parts.append(
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="white" stroke="#444" stroke-width="1"/>'
    )


def _draw_y_ticks(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
    ticks: int = 4,
) -> None:
    span = ymax - ymin if ymax > ymin else 1.0
    for i in range(ticks + 1):
        y = ymin + span * i / ticks
        py = y0 + height - height * i / ticks
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{x0 + width:.2f}" y2="{py:.2f}" '
            'stroke="#e0e0e0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.2f}" y="{py + 4:.2f}" font-size="11" text-anchor="end" fill="#444">'
            f"{y:.3g}</text>"
        )


def _draw_x_ticks(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ticks: int = 4,
) -> None:
    span = xmax - xmin if xmax > xmin else 1.0
    for i in range(ticks + 1):
        x = xmin + span * i / ticks
        px = x0 + width * i / ticks
        parts.append(
            f'<line x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y0 + height:.2f}" '
            'stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{y0 + height + 18:.2f}" font-size="11" text-anchor="middle" fill="#444">'
            f"{x:.2f}</text>"
        )


def plot_multi_panel_alpha_sweep(path: Path, summaries: Sequence[AlphaSummary], alpha_hat: float | None) -> None:
    ensure_dir(path.parent)
    xs = [s.alpha for s in summaries]
    panels = [
        (
            "Effective reward r*",
            [("r*", [s.fixed_point_reward for s in summaries], COLORS[0])],
        ),
        (
            "Entry and approval rates",
            [
                ("approval", [s.approval_probability for s in summaries], COLORS[0]),
                ("entry", [s.entry_rate for s in summaries], COLORS[1]),
                ("null entry", [s.null_entry_rate for s in summaries], COLORS[2]),
            ],
        ),
        (
            "Expected counts",
            [
                ("true approvals", [s.expected_true_approvals for s in summaries], COLORS[2]),
                ("false approvals", [s.expected_false_approvals for s in summaries], COLORS[1]),
                ("false negatives", [s.expected_false_negatives for s in summaries], COLORS[3]),
            ],
        ),
        (
            "FDR and upper bounds",
            [
                ("FDR", [s.fdr for s in summaries], COLORS[0]),
                ("action bound", [s.fdr_bound_action_specific for s in summaries], COLORS[1]),
                ("null-entry bound", [s.fdr_bound_null_entry for s in summaries], COLORS[2]),
            ],
        ),
    ]

    width = 1100
    height = 820
    margin_left = 70
    margin_top = 40
    panel_w = 450
    panel_h = 280
    x_gap = 70
    y_gap = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="24" text-anchor="middle" font-size="20" font-weight="bold" fill="#222">'
        'Principal-side alpha sweep (baseline calibration)</text>',
    ]

    for idx, (title, series_set) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = margin_left + col * (panel_w + x_gap)
        y0 = margin_top + row * (panel_h + y_gap)
        plot_x0 = x0
        plot_y0 = y0 + 20
        _draw_axes(parts, plot_x0, plot_y0, panel_w, panel_h)
        ymin = min(min(vals) for _, vals, _ in series_set)
        ymax = max(max(vals) for _, vals, _ in series_set)
        if math.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad
        _draw_y_ticks(parts, plot_x0, plot_y0, panel_w, panel_h, ymin, ymax)
        _draw_x_ticks(parts, plot_x0, plot_y0, panel_w, panel_h, xs[0], xs[-1])
        parts.append(
            f'<text x="{plot_x0 + panel_w / 2:.2f}" y="{y0 + 12:.2f}" text-anchor="middle" '
            f'font-size="15" font-weight="bold" fill="#222">{svg_escape(title)}</text>'
        )
        parts.append(
            f'<text x="{plot_x0 + panel_w / 2:.2f}" y="{plot_y0 + panel_h + 38:.2f}" text-anchor="middle" '
            'font-size="12" fill="#444">alpha</text>'
        )
        if alpha_hat is not None and xs[0] <= alpha_hat <= xs[-1]:
            px = plot_x0 + panel_w * (alpha_hat - xs[0]) / (xs[-1] - xs[0])
            parts.append(
                f'<line x1="{px:.2f}" y1="{plot_y0:.2f}" x2="{px:.2f}" y2="{plot_y0 + panel_h:.2f}" '
                'stroke="#888" stroke-dasharray="6,4" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{px + 4:.2f}" y="{plot_y0 + 14:.2f}" font-size="11" fill="#666">alpha_hat</text>'
            )
        legend_x = plot_x0 + 10
        legend_y = plot_y0 + 18
        for line_idx, (label, vals, color) in enumerate(series_set):
            py = legend_y + 18 * line_idx
            parts.append(
                f'<line x1="{legend_x:.2f}" y1="{py:.2f}" x2="{legend_x + 18:.2f}" y2="{py:.2f}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            parts.append(
                f'<text x="{legend_x + 24:.2f}" y="{py + 4:.2f}" font-size="11" fill="#333">{svg_escape(label)}</text>'
            )
            poly = _polyline_points(xs, vals, plot_x0, plot_y0, panel_w, panel_h, xs[0], xs[-1], ymin, ymax)
            parts.append(
                f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def plot_frontiers(path: Path, summaries: Sequence[AlphaSummary]) -> None:
    ensure_dir(path.parent)
    width = 1100
    height = 430
    margin_left = 80
    margin_top = 40
    panel_w = 430
    panel_h = 280
    x_gap = 90
    panels = [
        (
            "Exact FDR frontier",
            [s.fdr for s in summaries],
            [s.expected_true_approvals for s in summaries],
            "exact FDR",
            "expected true approvals",
        ),
        (
            "Action-bound frontier",
            [s.fdr_bound_action_specific for s in summaries],
            [s.expected_true_approvals for s in summaries],
            "action-specific FDR bound",
            "expected true approvals",
        ),
    ]
    alpha_labels = {0.10, 0.20, 0.28, 0.30, 0.38, 0.40}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="24" text-anchor="middle" font-size="20" font-weight="bold" fill="#222">'
        'Principal-side FDR/power frontiers</text>',
    ]
    for idx, (title, xs, ys, xlabel, ylabel) in enumerate(panels):
        x0 = margin_left + idx * (panel_w + x_gap)
        y0 = margin_top + 20
        _draw_axes(parts, x0, y0, panel_w, panel_h)
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        if math.isclose(xmin, xmax):
            xmin -= 1.0
            xmax += 1.0
        if math.isclose(ymin, ymax):
            ymin -= 1.0
            ymax += 1.0
        xmin -= 0.08 * (xmax - xmin)
        xmax += 0.08 * (xmax - xmin)
        ymin -= 0.08 * (ymax - ymin)
        ymax += 0.08 * (ymax - ymin)
        _draw_y_ticks(parts, x0, y0, panel_w, panel_h, ymin, ymax)
        _draw_x_ticks(parts, x0, y0, panel_w, panel_h, xmin, xmax)
        parts.append(
            f'<text x="{x0 + panel_w / 2:.2f}" y="{margin_top + 12:.2f}" text-anchor="middle" '
            f'font-size="15" font-weight="bold" fill="#222">{svg_escape(title)}</text>'
        )
        parts.append(
            f'<text x="{x0 + panel_w / 2:.2f}" y="{y0 + panel_h + 38:.2f}" text-anchor="middle" font-size="12" fill="#444">'
            f"{svg_escape(xlabel)}</text>"
        )
        parts.append(
            f'<text transform="translate({x0 - 58:.2f},{y0 + panel_h / 2:.2f}) rotate(-90)" text-anchor="middle" '
            f'font-size="12" fill="#444">{svg_escape(ylabel)}</text>'
        )
        poly = _polyline_points(xs, ys, x0, y0, panel_w, panel_h, xmin, xmax, ymin, ymax)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[0]}" stroke-width="2.2"/>')
        for summary in summaries:
            px = x0 + panel_w * (getattr(summary, "fdr") - xmin) / (xmax - xmin) if idx == 0 else x0 + panel_w * (summary.fdr_bound_action_specific - xmin) / (xmax - xmin)
            py = y0 + panel_h - panel_h * (summary.expected_true_approvals - ymin) / (ymax - ymin)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{COLORS[1]}"/>')
            rounded_alpha = round(summary.alpha + 1e-12, 2)
            if rounded_alpha in alpha_labels:
                parts.append(
                    f'<text x="{px + 4:.2f}" y="{py - 4:.2f}" font-size="10" fill="#444">{rounded_alpha:.2f}</text>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def monotonicity_breaks(values: Sequence[float], tol: float = 1e-10) -> list[int]:
    breaks: list[int] = []
    saw_increase = False
    last = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > last + tol:
            saw_increase = True
        if saw_increase and value < last - tol:
            breaks.append(idx)
        last = value
    return breaks


def feasible_components(summaries: Sequence[AlphaSummary], field: str, q: float, tol: float = 1e-12) -> list[tuple[float, float]]:
    components: list[tuple[float, float]] = []
    start: float | None = None
    prev_alpha: float | None = None
    for summary in summaries:
        value = getattr(summary, field)
        if value <= q + tol:
            if start is None:
                start = summary.alpha
            prev_alpha = summary.alpha
        else:
            if start is not None and prev_alpha is not None:
                components.append((start, prev_alpha))
            start = None
            prev_alpha = None
    if start is not None and prev_alpha is not None:
        components.append((start, prev_alpha))
    return components


def best_under_cap(
    summaries: Sequence[AlphaSummary],
    cap_field: str,
    q: float,
    objective_field: str = "expected_true_approvals",
) -> AlphaSummary | None:
    feasible = [s for s in summaries if getattr(s, cap_field) <= q + 1e-12]
    if not feasible:
        return None
    return max(feasible, key=lambda s: getattr(s, objective_field))


def format_components(components: Sequence[tuple[float, float]]) -> str:
    if not components:
        return "none"
    return ", ".join(f"[{lo:.2f}, {hi:.2f}]" for lo, hi in components)


def write_markdown_summary(path: Path, params: ModelParams, summaries: Sequence[AlphaSummary], alpha_hat_payload: dict) -> None:
    ensure_dir(path.parent)
    fdr_breaks = monotonicity_breaks([s.fdr for s in summaries])
    action_breaks = monotonicity_breaks([s.fdr_bound_action_specific for s in summaries])
    null_breaks = monotonicity_breaks([s.fdr_bound_null_entry for s in summaries])
    true_breaks = monotonicity_breaks([s.expected_true_approvals for s in summaries])

    cap005_exact = feasible_components(summaries, "fdr", 0.05)
    cap005_action = feasible_components(summaries, "fdr_bound_action_specific", 0.05)
    cap0005_exact = feasible_components(summaries, "fdr", 0.005)
    cap0005_action = feasible_components(summaries, "fdr_bound_action_specific", 0.005)

    best_exact_005 = best_under_cap(summaries, "fdr", 0.05)
    best_action_005 = best_under_cap(summaries, "fdr_bound_action_specific", 0.05)
    best_null_005 = best_under_cap(summaries, "fdr_bound_null_entry", 0.05)
    best_exact_0005 = best_under_cap(summaries, "fdr", 0.005)
    best_action_0005 = best_under_cap(summaries, "fdr_bound_action_specific", 0.005)
    positive = [s for s in summaries if s.fdr > 1e-12]
    max_action_gap = max((s.fdr_bound_action_specific - s.fdr) for s in positive)
    max_null_gap = max((s.fdr_bound_null_entry - s.fdr) for s in positive)
    max_action_ratio = max((s.fdr_bound_action_specific / s.fdr) for s in positive)
    max_null_ratio = max((s.fdr_bound_null_entry / s.fdr) for s in positive)
    selected = {
        0.28: next((s for s in summaries if abs(s.alpha - 0.28) < 1e-9), None),
        0.30: next((s for s in summaries if abs(s.alpha - 0.30) < 1e-9), None),
        0.38: next((s for s in summaries if abs(s.alpha - 0.38) < 1e-9), None),
        0.39: next((s for s in summaries if abs(s.alpha - 0.39) < 1e-9), None),
    }

    def fmt_best(label: str, summary: AlphaSummary | None, cap_field: str) -> str:
        if summary is None:
            return f"- {label}: no feasible alpha\n"
        return (
            f"- {label}: alpha={summary.alpha:.2f}, "
            f"true approvals={summary.expected_true_approvals:.4f}, "
            f"FDR={summary.fdr:.6f}, "
            f"action bound={summary.fdr_bound_action_specific:.6f}, "
            f"null-entry bound={summary.fdr_bound_null_entry:.6f}, "
            f"{cap_field}={getattr(summary, cap_field):.6f}\n"
        )

    content = []
    content.append("# Principal-side alpha sweep results\n")
    content.append("## Calibration\n")
    content.append(
        f"- baseline parameters: m={params.m}, mu_b={params.mu_b}, R={params.reward}, "
        f"c0={params.c0}, c={params.c}, n in {{{params.n_min},...,{params.n_max}}}, "
        f"distribution={params.distribution}, beta=({params.beta_a},{params.beta_b}), "
        f"dilution={params.crowding}, gamma={params.gamma}\n"
    )
    content.append(
        f"- estimated alpha_hat_comp on the sweep: {alpha_hat_payload['alpha_threshold_cross']}\n"
    )
    content.append("\n## Nonmonotonicity checks\n")
    content.append(
        f"- exact FDR monotonicity breaks at alpha values: "
        + (", ".join(f"{summaries[i].alpha:.2f}" for i in fdr_breaks) if fdr_breaks else "none")
        + "\n"
    )
    content.append(
        f"- action-specific bound monotonicity breaks at alpha values: "
        + (", ".join(f"{summaries[i].alpha:.2f}" for i in action_breaks) if action_breaks else "none")
        + "\n"
    )
    content.append(
        f"- null-entry bound monotonicity breaks at alpha values: "
        + (", ".join(f"{summaries[i].alpha:.2f}" for i in null_breaks) if null_breaks else "none")
        + "\n"
    )
    content.append(
        f"- expected true approvals monotonicity breaks at alpha values: "
        + (", ".join(f"{summaries[i].alpha:.2f}" for i in true_breaks) if true_breaks else "none")
        + "\n"
    )
    content.append("\n## Feasible sets under FDR caps\n")
    content.append(f"- exact FDR <= 0.05: {format_components(cap005_exact)}\n")
    content.append(f"- action-specific bound <= 0.05: {format_components(cap005_action)}\n")
    content.append(f"- exact FDR <= 0.005: {format_components(cap0005_exact)}\n")
    content.append(f"- action-specific bound <= 0.005: {format_components(cap0005_action)}\n")
    content.append("\n## Bound validation on the baseline grid\n")
    content.append("- action-specific and null-entry bounds dominate exact FDR at every grid point\n")
    content.append(f"- max action-specific slack on positive-FDR points: {max_action_gap:.6f}\n")
    content.append(f"- max null-entry slack on positive-FDR points: {max_null_gap:.6f}\n")
    content.append(f"- max action-specific/exact FDR ratio on positive-FDR points: {max_action_ratio:.6f}\n")
    content.append(f"- max null-entry/exact FDR ratio on positive-FDR points: {max_null_ratio:.6f}\n")
    content.append("\nRepresentative points:\n")
    content.append("| alpha | exact FDR | action bound | null-entry bound | cost-floor bound | true approvals |\n")
    content.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for alpha in [0.28, 0.30, 0.38, 0.39]:
        summary = selected[alpha]
        if summary is None:
            continue
        content.append(
            f"| {summary.alpha:.2f} | {summary.fdr:.6f} | {summary.fdr_bound_action_specific:.6f} | "
            f"{summary.fdr_bound_null_entry:.6f} | {summary.fdr_bound_effective:.6f} | "
            f"{summary.expected_true_approvals:.6f} |\n"
        )
    content.append("\n## Best true-approval choice under caps\n")
    content.append(fmt_best("exact FDR cap 0.05", best_exact_005, "fdr"))
    content.append(fmt_best("action-bound cap 0.05", best_action_005, "fdr_bound_action_specific"))
    content.append(fmt_best("null-entry cap 0.05", best_null_005, "fdr_bound_null_entry"))
    content.append(fmt_best("exact FDR cap 0.005", best_exact_0005, "fdr"))
    content.append(fmt_best("action-bound cap 0.005", best_action_0005, "fdr_bound_action_specific"))
    path.write_text("".join(content))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("research/generated/principal_alpha_design/baseline"))
    parser.add_argument("--alpha-grid", type=str, default="0.01:0.40:40")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--mu-b", type=float, default=0.5)
    parser.add_argument("--reward", type=float, default=30.0)
    parser.add_argument("--c0", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=60)
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
    alphas = parse_alpha_grid(args.alpha_grid)
    summaries = solve_alpha_grid(alphas, params)
    alpha_hat = estimate_alpha_hat(summaries, params.mu_b)

    ensure_dir(args.output_dir)
    write_csv(args.output_dir / "alpha_sweep.csv", summaries)
    write_json(
        args.output_dir / "alpha_sweep.json",
        {
            "params": asdict(params),
            "alpha_hat": asdict(alpha_hat),
            "summaries": [asdict(s) for s in summaries],
        },
    )
    plot_multi_panel_alpha_sweep(args.output_dir / "alpha_sweep_panels.svg", summaries, alpha_hat.alpha_threshold_cross)
    plot_frontiers(args.output_dir / "principal_frontiers.svg", summaries)
    write_markdown_summary(args.output_dir / "summary.md", params, summaries, asdict(alpha_hat))
    print(json.dumps({"output_dir": str(args.output_dir), "alpha_hat": asdict(alpha_hat)}, sort_keys=True))


if __name__ == "__main__":
    main()
