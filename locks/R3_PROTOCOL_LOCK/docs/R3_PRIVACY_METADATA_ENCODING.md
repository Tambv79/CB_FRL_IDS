# Locked metadata views

These names replace the ambiguous historical label “Protected”. None provides differential privacy.

| View | Utility | Cost | n_fit | Approx. payload role |
|---|---|---|---|---|
| Raw | float32 marginal utility | uint64 wire bytes | uint32 | exact empirical metadata view |
| Noisy-Quantized | clip utility to [-1,1] + Gaussian sigma 0.1; cost rounded to 1024 bytes | rounded | n_fit rounded to 1000 | empirical perturbation only |
| No-Size | clip utility to [-1,1] + Gaussian sigma 0.2; cost rounded to 4096 bytes | rounded | omitted | empirical perturbation only |
| Minimal-Bucketed | three utility buckets encoded by representative values | three cost buckets | omitted | coarse bucket view |

Every attacker result reports observations, client groups, class balance, hyperparameters, tuning, per-attacker AUC and confidence interval. Splits are group-disjoint by client_id and repeated across ten locked seeds.
