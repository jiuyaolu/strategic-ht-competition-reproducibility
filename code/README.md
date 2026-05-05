# Code

This directory contains the Python scripts used to generate and verify the
numerical results in the report.

- `multi_agent_fixed_point.py`: shared routines for equilibrium rewards,
  best responses, approval probabilities, and false-discovery quantities.
- `principal_alpha_experiments.py`: threshold-comparison calculations used for
  the main figures.
- `market_size_fdr_cap_experiment.py`: market-size comparison calculations.
- `map_alpha_hat_comp.py`: computations for the competition-adjusted screening
  boundary table.
- `exact_and_self_belief_experiments.py`: exact-knowledge and self-belief
  numerical outputs.
- `verify_package.py`: lightweight verification script used by
  `../reproduce_all.sh`.

The scripts use only the Python standard library. See the root `README.md` for
the exact commands used to regenerate outputs.
