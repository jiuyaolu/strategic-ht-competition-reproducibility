# Reproducibility Package

This folder contains the code and data for the numerical results that appear in
the current paper, `Strategic Hypothesis Testing Under Competition`.

The package intentionally excludes archived or commented-out material from the
working project, including timing, fixed-pool, population-shift, robustness, and
exploratory proof-search experiments.

## Contents

- `paper/multiagent.pdf`: the paper version associated with these outputs.
- `code/`: Python scripts used to compute the equilibrium quantities and
  numerical tables.
- `data/generated/`: generated CSV/JSON outputs used by the paper figures.
- `data/paper_tables/`: CSV versions of the displayed numerical tables.
- `data/manifest.csv`: mapping from each paper figure/table to the data file and
  source script.

## Quick Check

Run:

```bash
./reproduce_all.sh
```

The default check is lightweight. It verifies that all paper-used files are
present and checks selected numerical values, without rerunning the full
experiments.

## Optional Regeneration

The full numerical runs can take longer. The following commands regenerate the
main alpha-threshold data into `data/regenerated/`:

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

To regenerate the exact-benchmark and self-belief helper outputs, run:

```bash
OUTPUT_ROOT=data/regenerated/sections1_6_missing_experiments \
  python3 code/sections1_6_missing_experiments.py
```

That helper script also writes some auxiliary diagnostics. The paper-used subset
is listed in `data/manifest.csv`.

## Dependencies

The numerical code uses only the Python standard library. It was run with
Python 3.10.
