# Strategic Hypothesis Testing Under Competition

This repository accompanies the final report `Strategic Hypothesis Testing
Under Competition`. It provides the code and data used for the numerical figures
and tables in the report.

## Repository Contents

- `paper/multiagent.pdf`: a copy of the report.
- `code/`: Python scripts for computing equilibrium outcomes, false-discovery
  quantities, threshold comparisons, market-size comparisons, and self-belief
  quantities.
- `data/generated/`: generated CSV/JSON files used by the figures in the report.
- `data/paper_tables/`: CSV versions of the numerical tables shown in the
  report.
- `data/manifest.csv`: a guide matching each reported figure/table to its data
  file and source script.

## Quick Verification

From the repository root, run:

```bash
./reproduce_all.sh
```

This lightweight check verifies that the expected files are present and checks
selected numerical values against the stored outputs. It is designed to run
quickly.

## Regenerating Numerical Outputs

The stored outputs are deterministic grid-based calculations. To regenerate the
main threshold-comparison data, run:

```bash
python3 code/principal_alpha_experiments.py \
  --output-dir data/regenerated/principal_alpha_design/baseline \
  --alpha-grid 0.01:0.40:40 --m 20 --reward 30

python3 code/principal_alpha_experiments.py \
  --output-dir data/regenerated/principal_alpha_design/frontier_m1 \
  --alpha-grid 0.01:0.95:95 --m 1 --reward 30

python3 code/principal_alpha_experiments.py \
  --output-dir data/regenerated/principal_alpha_design/frontier_m20 \
  --alpha-grid 0.01:0.95:95 --m 20 --reward 30

python3 code/principal_alpha_experiments.py \
  --output-dir data/regenerated/principal_alpha_design/frontier_m40 \
  --alpha-grid 0.01:0.95:95 --m 40 --reward 30

python3 code/market_size_fdr_cap_experiment.py \
  --output-dir data/regenerated/principal_alpha_design/market_size_fdr_cap
```

To regenerate the exact-knowledge and self-belief numerical outputs, run:

```bash
OUTPUT_ROOT=data/regenerated/exact_and_self_belief \
  python3 code/exact_and_self_belief_experiments.py
```

The mapping from report objects to files is in `data/manifest.csv`.

## Dependencies

The numerical scripts use only the Python standard library. They were run with
Python 3.10.
