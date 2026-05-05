#!/usr/bin/env python3
"""Market-size design experiment under an exact FDR cap.

For each market size m, this script solves the symmetric equilibrium over a
threshold grid and selects the threshold that maximizes expected true approvals
subject to exact FDR <= q.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from multi_agent_fixed_point import ModelParams, AlphaSummary, parse_alpha_grid, solve_alpha_grid


COLORS = {
    "blue": "#0b57d0",
    "red": "#c5221f",
    "green": "#188038",
    "orange": "#b06000",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def best_under_fdr_cap(summaries: Sequence[AlphaSummary], cap: float) -> AlphaSummary | None:
    feasible = [summary for summary in summaries if summary.fdr <= cap + 1e-12]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda summary: (
            summary.expected_true_approvals,
            summary.sensitivity,
            -summary.fdr,
        ),
    )


def rows_for_market_sizes(base_params: ModelParams, m_values: Sequence[int], alpha_grid: Sequence[float], fdr_cap: float) -> list[dict]:
    rows: list[dict] = []
    for m in m_values:
        params = ModelParams(**{**asdict(base_params), "m": m})
        summaries = solve_alpha_grid(alpha_grid, params)
        selected = best_under_fdr_cap(summaries, fdr_cap)
        if selected is None:
            rows.append(
                {
                    "m": m,
                    "fdr_cap": fdr_cap,
                    "selected_alpha": "",
                    "selected_fdr": "",
                    "selected_sensitivity": "",
                    "selected_false_negative_rate": "",
                    "selected_expected_true_approvals": "",
                    "selected_expected_false_approvals": "",
                    "selected_expected_false_negatives": "",
                    "selected_reward": "",
                    "selected_approval_probability": "",
                    "selected_entry_rate": "",
                    "feasible_alpha_count": 0,
                }
            )
            continue
        feasible_count = sum(1 for summary in summaries if summary.fdr <= fdr_cap + 1e-12)
        rows.append(
            {
                "m": m,
                "fdr_cap": fdr_cap,
                "selected_alpha": selected.alpha,
                "selected_fdr": selected.fdr,
                "selected_sensitivity": selected.sensitivity,
                "selected_false_negative_rate": 1.0 - selected.sensitivity,
                "selected_expected_true_approvals": selected.expected_true_approvals,
                "selected_expected_false_approvals": selected.expected_false_approvals,
                "selected_expected_false_negatives": selected.expected_false_negatives,
                "selected_reward": selected.fixed_point_reward,
                "selected_approval_probability": selected.approval_probability,
                "selected_entry_rate": selected.entry_rate,
                "feasible_alpha_count": feasible_count,
            }
        )
    return rows


def _axis_transform(
    x: float,
    y: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float]:
    px = x0 + width * (x - xmin) / (xmax - xmin)
    py = y0 + height - height * (y - ymin) / (ymax - ymin)
    return px, py


def _polyline(
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
    points = [_axis_transform(x, y, x0, y0, width, height, xmin, xmax, ymin, ymax) for x, y in zip(xs, ys)]
    return " ".join(f"{px:.2f},{py:.2f}" for px, py in points)


def _draw_panel(
    parts: list[str],
    rows: Sequence[dict],
    x0: float,
    y0: float,
    width: float,
    height: float,
    field: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    xs = [float(row["m"]) for row in rows if row[field] != ""]
    ys = [float(row[field]) for row in rows if row[field] != ""]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    ypad = 0.08 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    parts.append(
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="white" stroke="#444" stroke-width="1"/>'
    )
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4.0
        py = y0 + height - height * i / 4.0
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{x0 + width:.2f}" y2="{py:.2f}" '
            'stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0 - 8:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="10" fill="#444">{y:.3g}</text>'
        )
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4.0
        px = x0 + width * i / 4.0
        parts.append(
            f'<line x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y0 + height:.2f}" '
            'stroke="#f2f2f2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{y0 + height + 18:.2f}" text-anchor="middle" font-size="10" fill="#444">{x:.0f}</text>'
        )
    parts.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 - 10:.2f}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="#222">{svg_escape(title)}</text>'
    )
    parts.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 + height + 38:.2f}" text-anchor="middle" '
        'font-size="11" fill="#444">market size m</text>'
    )
    parts.append(
        f'<text transform="translate({x0 - 54:.2f},{y0 + height / 2:.2f}) rotate(-90)" '
        f'text-anchor="middle" font-size="11" fill="#444">{svg_escape(ylabel)}</text>'
    )
    polyline = _polyline(xs, ys, x0, y0, width, height, xmin, xmax, ymin, ymax)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    for x, y in zip(xs, ys):
        px, py = _axis_transform(x, y, x0, y0, width, height, xmin, xmax, ymin, ymax)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')


def write_preview_svg(path: Path, rows: Sequence[dict]) -> None:
    ensure_dir(path.parent)
    width = 1100
    height = 780
    panel_w = 390
    panel_h = 230
    left = 105
    top = 80
    hgap = 120
    vgap = 130
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="32" text-anchor="middle" font-size="20" font-weight="bold" fill="#222">'
        'Best exact-FDR-controlled threshold choice by market size</text>',
    ]
    panels = [
        ("selected_expected_true_approvals", "Expected true approvals", "expected count", COLORS["green"]),
        ("selected_sensitivity", "Sensitivity", "rate", COLORS["blue"]),
        ("selected_alpha", "Selected threshold", "alpha", COLORS["orange"]),
        ("selected_reward", "Effective reward", "reward", COLORS["red"]),
    ]
    for idx, (field, title, ylabel, color) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x0 = left + col * (panel_w + hgap)
        y0 = top + row * (panel_h + vgap)
        _draw_panel(parts, rows, x0, y0, panel_w, panel_h, field, title, ylabel, color)
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("research/generated/principal_alpha_design/market_size_fdr_cap"))
    parser.add_argument("--m-values", type=str, default="1,2,5,10,15,20,25,30,35,40,50,60")
    parser.add_argument("--alpha-grid", type=str, default="0.01:0.95:95")
    parser.add_argument("--fdr-cap", type=float, default=0.05)
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
        m=1,
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
    m_values = [int(piece) for piece in args.m_values.split(",") if piece.strip()]
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    rows = rows_for_market_sizes(params, m_values, alpha_grid, args.fdr_cap)
    ensure_dir(args.output_dir)
    write_csv(args.output_dir / "optimized_by_m.csv", rows)
    write_json(
        args.output_dir / "optimized_by_m.json",
        {
            "base_params": asdict(params),
            "m_values": m_values,
            "alpha_grid": alpha_grid,
            "fdr_cap": args.fdr_cap,
            "rows": rows,
        },
    )
    write_preview_svg(args.output_dir / "optimized_by_m.svg", rows)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
