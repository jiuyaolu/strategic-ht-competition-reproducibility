# Principal-side alpha sweep results
## Calibration
- baseline parameters: m=1, mu_b=0.5, R=30.0, c0=0.5, c=0.05, n in {10,...,60}, distribution=beta, beta=(6.0,5.0), dilution=power, gamma=1.0
- estimated alpha_hat_comp on the sweep: 0.05

## Nonmonotonicity checks
- exact FDR monotonicity breaks at alpha values: none
- action-specific bound monotonicity breaks at alpha values: none
- null-entry bound monotonicity breaks at alpha values: none
- expected true approvals monotonicity breaks at alpha values: 0.16, 0.23, 0.28, 0.32, 0.36, 0.38, 0.42, 0.44, 0.51, 0.57, 0.58, 0.59, 0.60, 0.62, 0.63, 0.64, 0.65, 0.83

## Feasible sets under FDR caps
- exact FDR <= 0.05: [0.01, 0.33]
- action-specific bound <= 0.05: [0.01, 0.19]
- exact FDR <= 0.005: [0.01, 0.07]
- action-specific bound <= 0.005: [0.01, 0.07]

## Bound validation on the baseline grid
- action-specific and null-entry bounds dominate exact FDR at every grid point
- max action-specific slack on positive-FDR points: 0.099921
- max null-entry slack on positive-FDR points: 0.109067
- max action-specific/exact FDR ratio on positive-FDR points: 1.838620
- max null-entry/exact FDR ratio on positive-FDR points: 2.318871

Representative points:
| alpha | exact FDR | action bound | null-entry bound | cost-floor bound | true approvals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.28 | 0.046108 | 0.084776 | 0.086501 | 0.533801 | 0.487692 |
| 0.30 | 0.047442 | 0.086976 | 0.092680 | 0.545853 | 0.498411 |
| 0.38 | 0.072029 | 0.130949 | 0.132008 | 0.587206 | 0.515177 |
| 0.39 | 0.072536 | 0.131681 | 0.135481 | 0.592883 | 0.520347 |

## Best true-approval choice under caps
- exact FDR cap 0.05: alpha=0.33, true approvals=0.5014, FDR=0.049578, action bound=0.090245, null-entry bound=0.101948, fdr=0.049578
- action-bound cap 0.05: alpha=0.19, true approvals=0.4565, FDR=0.026039, action bound=0.045841, null-entry bound=0.050675, fdr_bound_action_specific=0.045841
- null-entry cap 0.05: alpha=0.18, true approvals=0.4527, FDR=0.026039, action bound=0.045841, null-entry bound=0.048008, fdr_bound_null_entry=0.048008
- exact FDR cap 0.005: alpha=0.07, true approvals=0.3679, FDR=0.003842, action bound=0.004726, null-entry bound=0.006050, fdr=0.003842
- action-bound cap 0.005: alpha=0.07, true approvals=0.3679, FDR=0.003842, action bound=0.004726, null-entry bound=0.006050, fdr_bound_action_specific=0.004726
