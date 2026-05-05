#!/usr/bin/env python3
"""Lightweight reproducibility-package checks."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "paper/multiagent.pdf",
    "code/multi_agent_fixed_point.py",
    "code/principal_alpha_experiments.py",
    "code/market_size_fdr_cap_experiment.py",
    "code/map_alpha_hat_comp.py",
    "code/sections1_6_missing_experiments.py",
    "data/manifest.csv",
    "data/generated/sections1_6_missing_experiments/exact_benchmark/best_response_alpha_0p2.csv",
    "data/generated/sections1_6_missing_experiments/exact_benchmark/best_response_alpha_0p28.csv",
    "data/generated/principal_alpha_design/baseline/alpha_sweep.csv",
    "data/generated/principal_alpha_design/frontier_m1/alpha_sweep.csv",
    "data/generated/principal_alpha_design/frontier_m20/alpha_sweep.csv",
    "data/generated/principal_alpha_design/frontier_m40/alpha_sweep.csv",
    "data/generated/principal_alpha_design/market_size_fdr_cap/optimized_by_m.csv",
    "data/paper_tables/alpha_hat_comp_table.csv",
    "data/paper_tables/fdr_notions_table.csv",
    "data/paper_tables/fdr_bounds_table.csv",
    "data/paper_tables/self_belief_comparison_table.csv",
]


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, tol: float = 5e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def row_by(rows: list[dict[str, str]], field: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[field] == value:
            return row
    raise AssertionError(f"missing row with {field}={value}")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    alpha_rows = read_csv("data/paper_tables/alpha_hat_comp_table.csv")
    r20 = next(row for row in alpha_rows if row["parameter"] == "Reward" and row["value"] == "R=20")
    assert_close(float(r20["result_for_alpha_hat_comp"]), 0.612793)

    baseline_rows = read_csv("data/generated/principal_alpha_design/baseline/alpha_sweep.csv")
    alpha_028 = row_by(baseline_rows, "alpha", "0.28")
    assert_close(float(alpha_028["fdr"]), 0.011544529535024254)
    assert_close(float(alpha_028["expected_true_approvals"]), 7.5536729595625385)

    fdr_rows = read_csv("data/paper_tables/fdr_bounds_table.csv")
    bound_038 = row_by(fdr_rows, "alpha", "0.38")
    assert_close(float(bound_038["null_entry_bound"]), 0.066006, tol=5e-6)

    market_rows = read_csv("data/generated/principal_alpha_design/market_size_fdr_cap/optimized_by_m.csv")
    if len(market_rows) < 5:
        raise AssertionError("market-size file has too few rows")

    print("Reproducibility package check passed.")


if __name__ == "__main__":
    main()
