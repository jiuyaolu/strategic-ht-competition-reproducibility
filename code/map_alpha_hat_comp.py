#!/usr/bin/env python3
"""Map the competition-adjusted critical threshold across a parameter sweep."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent_fixed_point import (
    ModelParams,
    estimate_alpha_hat,
    parse_alpha_grid,
    refine_alpha_hat_threshold,
    solve_alpha_grid,
)


def parse_values(raw: str) -> List[float]:
    return [float(piece) for piece in raw.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary", choices=["gamma", "reward", "c0", "c"], required=True)
    parser.add_argument("--values", required=True, help="Comma-separated values for the swept parameter.")
    parser.add_argument("--alpha-grid", type=str, default="0.10:0.40:16")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--mu-b", type=float, default=0.5)
    parser.add_argument("--reward", type=float, default=40.0)
    parser.add_argument("--c0", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-max", type=int, default=60)
    parser.add_argument("--grid-points", type=int, default=121)
    parser.add_argument("--distribution", choices=["beta", "truncnorm", "uniform"], default="beta")
    parser.add_argument("--beta-a", type=float, default=6.0)
    parser.add_argument("--beta-b", type=float, default=5.0)
    parser.add_argument("--truncnorm-mean", type=float, default=0.58)
    parser.add_argument("--truncnorm-sd", type=float, default=0.12)
    parser.add_argument("--crowding", choices=["power", "linear", "exponential"], default="power")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--tail-method", choices=["exact", "normal"], default="exact")
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_params = ModelParams(
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
    results = []
    for value in parse_values(args.values):
        params = ModelParams(**asdict(base_params))
        setattr(params, args.vary, value)
        summaries = solve_alpha_grid(alphas, params)
        alpha_hat = estimate_alpha_hat(summaries, params.mu_b)
        crossing = next(
            (summary for summary in summaries if alpha_hat.alpha_threshold_cross == summary.alpha),
            None,
        )
        fp_crossing = next(
            (summary for summary in summaries if alpha_hat.alpha_false_positive_cross == summary.alpha),
            None,
        )
        refined_alpha = None
        refined_summary = None
        if args.refine_iters > 0 and alpha_hat.alpha_threshold_cross is not None:
            crossing_index = next(
                idx for idx, summary in enumerate(summaries) if summary.alpha == alpha_hat.alpha_threshold_cross
            )
            if crossing_index > 0:
                refined_alpha, refined_summary = refine_alpha_hat_threshold(
                    alphas[crossing_index - 1],
                    alphas[crossing_index],
                    params,
                    refine_iters=args.refine_iters,
                )
        results.append(
            {
                "value": value,
                "alpha_hat": asdict(alpha_hat),
                "crossing_summary": None if crossing is None else asdict(crossing),
                "fp_crossing_summary": None if fp_crossing is None else asdict(fp_crossing),
                "refined_alpha_threshold_cross": refined_alpha,
                "refined_crossing_summary": None if refined_summary is None else asdict(refined_summary),
            }
        )

    if args.json:
        payload = {
            "alpha_grid": alphas,
            "base_params": asdict(base_params),
            "vary": args.vary,
            "results": results,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(json.dumps({"vary": args.vary, "alpha_grid": alphas, "base_params": asdict(base_params)}, sort_keys=True))
    print("value    alpha_hat(thr)    refined(thr)    alpha_hat(fp)    r*(at thr)    mu_tau(at thr)    mFDR(at thr)")
    for item in results:
        crossing = item["crossing_summary"]
        r_text = "None" if crossing is None else f"{crossing['fixed_point_reward']:.4f}"
        tau_text = "None" if crossing is None else f"{crossing['participation_threshold']:.4f}"
        mfdr_text = "None" if crossing is None else f"{crossing['mfdr']:.4f}"
        refined_text = "None" if item["refined_alpha_threshold_cross"] is None else f"{item['refined_alpha_threshold_cross']:.6f}"
        print(
            f"{item['value']:.4f}    "
            f"{str(item['alpha_hat']['alpha_threshold_cross']):>14}    "
            f"{refined_text:>12}    "
            f"{str(item['alpha_hat']['alpha_false_positive_cross']):>13}    "
            f"{r_text:>10}    "
            f"{tau_text:>15}    "
            f"{mfdr_text:>12}"
        )


if __name__ == "__main__":
    main()
