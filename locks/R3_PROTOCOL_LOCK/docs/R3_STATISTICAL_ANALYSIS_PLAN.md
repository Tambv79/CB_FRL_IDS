# Locked statistical analysis plan

## Paired design

All method comparisons use identical dataset protocol, client assignment, initialization stream and seed. Ten paired seeds are retained.

## Non-inferiority and safety

For each dataset and beta:

- F1: H0: mean(CB-Full) <= -0.02; H1: mean(CB-Full) > -0.02.
- PR-AUC: H0: mean(CB-Full) <= -0.02; H1: mean(CB-Full) > -0.02.
- FPR: H0: mean(CB-Full) >= 0.0025; H1: mean(CB-Full) < 0.0025.

The FPR margin equals 25 additional false positives per 10,000 benign flows. It is an analytical tolerance, not a deployment SLA. Margin sensitivity is reported and the primary margin is not changed after results.

Use 20,000 paired bootstrap resamples for confidence intervals and exact 2^10 sign-flip enumeration for one-sided transformed differences. Report raw and Holm-adjusted p-values, dz and wins/losses/ties.

Language is limited to “met the prespecified gate within the evaluated seeds”; “confirmed” is prohibited.

## Same-budget selector comparisons

CB-Score is compared separately with Utility-Only, Utility-Cost-Ratio and Oort-Style-Adapted for F1 and PR-AUC. Holm correction is applied over 36 hypotheses per endpoint: 2 datasets x 6 beta values x 3 competitors.

## Thresholds

1. Primary: method-specific threshold selected on validation data by maximum F1 subject to FPR<=0.05.
2. Shared: the Full-Contextual validation threshold for the dataset/seed is applied unchanged to every matched method and beta.
3. Sensitivity: fixed threshold 0.5.

Test predictions are exported once only after configuration and thresholds are locked.

## Communication curves

Same-round results use rounds 1–30. Same-communication curves interpolate validation metrics at 20 common cumulative broadcast-E2E byte points within the overlap of compared methods. Communication-to-target uses 90% and 95% of Full final validation F1 and PR-AUC and reports `not_reached` when appropriate.
