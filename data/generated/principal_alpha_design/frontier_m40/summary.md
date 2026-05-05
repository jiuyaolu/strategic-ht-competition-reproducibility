# Principal-side alpha sweep results
## Calibration
- baseline parameters: m=40, mu_b=0.5, R=30.0, c0=0.5, c=0.05, n in {10,...,60}, distribution=beta, beta=(6.0,5.0), dilution=power, gamma=1.0
- estimated alpha_hat_comp on the sweep: 0.8300000000000001

## Nonmonotonicity checks
- exact FDR monotonicity breaks at alpha values: 0.93
- action-specific bound monotonicity breaks at alpha values: 0.93
- null-entry bound monotonicity breaks at alpha values: 0.93
- expected true approvals monotonicity breaks at alpha values: 0.04

## Feasible sets under FDR caps
- exact FDR <= 0.05: [0.01, 0.88], [0.93, 0.94]
- action-specific bound <= 0.05: [0.01, 0.88], [0.93, 0.94]
- exact FDR <= 0.005: [0.01, 0.82]
- action-specific bound <= 0.005: [0.01, 0.82]

## Bound validation on the baseline grid
- action-specific and null-entry bounds dominate exact FDR at every grid point
- max action-specific slack on positive-FDR points: 0.001325
- max null-entry slack on positive-FDR points: 0.003182
- max action-specific/exact FDR ratio on positive-FDR points: 1.012299
- max null-entry/exact FDR ratio on positive-FDR points: 1.073420

Representative points:
| alpha | exact FDR | action bound | null-entry bound | cost-floor bound | true approvals |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.28 | 0.000000 | 0.000000 | 0.000000 | 0.663975 | 12.651071 |
| 0.30 | 0.000000 | 0.000000 | 0.000000 | 0.698034 | 12.893350 |
| 0.38 | 0.000000 | 0.000000 | 0.000000 | 0.759643 | 15.007047 |
| 0.39 | 0.000000 | 0.000000 | 0.000000 | 0.779634 | 15.007047 |

## Best true-approval choice under caps
- exact FDR cap 0.05: alpha=0.93, true approvals=23.6607, FDR=0.046995, action bound=0.047269, null-entry bound=0.048483, fdr=0.046995
- action-bound cap 0.05: alpha=0.93, true approvals=23.6607, FDR=0.046995, action bound=0.047269, null-entry bound=0.048483, fdr_bound_action_specific=0.047269
- null-entry cap 0.05: alpha=0.93, true approvals=23.6607, FDR=0.046995, action bound=0.047269, null-entry bound=0.048483, fdr_bound_null_entry=0.048483
- exact FDR cap 0.005: alpha=0.81, true approvals=22.4231, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr=0.000000
- action-bound cap 0.005: alpha=0.81, true approvals=22.4231, FDR=0.000000, action bound=0.000000, null-entry bound=0.000000, fdr_bound_action_specific=0.000000
