# Principal-side alpha sweep results
## Calibration
- baseline parameters: m=20, mu_b=0.5, R=30.0, c0=0.5, c=0.05, n in {10,...,60}, distribution=beta, beta=(6.0,5.0), dilution=power, gamma=1.0
- estimated alpha_hat_comp on the sweep: 0.28

## Nonmonotonicity checks
- exact FDR monotonicity breaks at alpha values: 0.30, 0.39, 0.60
- action-specific bound monotonicity breaks at alpha values: 0.30, 0.39, 0.60
- null-entry bound monotonicity breaks at alpha values: 0.30, 0.39, 0.60, 0.71
- expected true approvals monotonicity breaks at alpha values: 0.22, 0.28, 0.33, 0.63

## Feasible sets under FDR caps
- exact FDR <= 0.05: [0.01, 0.37]
- action-specific bound <= 0.05: [0.01, 0.37]
- exact FDR <= 0.005: [0.01, 0.27], [0.30, 0.37]
- action-specific bound <= 0.005: [0.01, 0.27], [0.30, 0.37]

## Bound validation on the baseline grid
- action-specific and null-entry bounds dominate exact FDR at every grid point
- max action-specific slack on positive-FDR points: 0.043372
- max null-entry slack on positive-FDR points: 0.061484
- max action-specific/exact FDR ratio on positive-FDR points: 1.183246
- max null-entry/exact FDR ratio on positive-FDR points: 1.418780

Representative points:
| alpha | exact FDR | action bound | null-entry bound | cost-floor bound | true approvals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.28 | 0.011545 | 0.011703 | 0.011941 | 0.999934 | 7.553673 |
| 0.30 | 0.000000 | 0.000000 | 0.000000 | 0.999959 | 7.938423 |
| 0.38 | 0.059332 | 0.065477 | 0.066006 | 0.999995 | 8.548099 |
| 0.39 | 0.053473 | 0.058364 | 0.060384 | 0.999996 | 8.723190 |

## Best true-approval choice under caps
- exact FDR cap 0.05: alpha=0.32, true approvals=8.3120, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr=0.000000
- action-bound cap 0.05: alpha=0.32, true approvals=8.3120, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr_bound_action_specific=0.000000
- null-entry cap 0.05: alpha=0.32, true approvals=8.3120, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr_bound_null_entry=0.000000
- exact FDR cap 0.005: alpha=0.32, true approvals=8.3120, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr=0.000000
- action-bound cap 0.005: alpha=0.32, true approvals=8.3120, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr_bound_action_specific=0.000000
