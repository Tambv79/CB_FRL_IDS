# R2 locked method specification

## Core scope

The core contribution is a learner-compatible pre-upload selection protocol. All candidate clients may train locally; only minimized metadata is exposed before selection; only admitted clients upload complete updates. The hard constraint applies to selected-update uplink bytes, not total end-to-end communication.

## Utility

For candidate client i in round t, the locked proxy is

u_i^t = [F1(theta_i^{t,H}; V_i^t) - 0.25 FPR(theta_i^{t,H}; V_i^t)]
        - [F1(theta^t; V_i^t) - 0.25 FPR(theta^t; V_i^t)].

Both models are evaluated on exactly the same client-held utility subset V_i^t at threshold 0.5. Absolute local validation score is not the implemented utility and must not be used as its definition.

## Three-stage admission logic

1. Static eligibility: finite metadata; at least 20 utility rows, including at least 5 benign and 5 attack rows; at least 180 fit rows; update wire bytes positive and no larger than the whole-round selected-update budget.
2. Ranking eligibility: method-specific priority is positive, except Random-Budget and Cost-Only, whose priorities are intrinsically non-negative.
3. Iterative feasibility: the next update is admitted only when used_bytes + wire_bytes <= round_budget_bytes.

Full-Contextual aggregates all K candidates and is not filtered by eligibility.

## Empty rounds

The historical forced fallback is removed. If no candidate satisfies the locked conditions, no full update is uploaded and the global model remains unchanged for that round. This outcome is reported explicitly.

## Ranking variants

The protocol is the contribution; score superiority is an empirical question. R3 evaluates Utility-Only, Cost-Only, Utility/Cost, CB-Score, the historical double-cost score, an adapted Oort-style baseline, Random-Budget and an exact score oracle.

CB-Score uses max(u_norm - 0.25 c_norm, 0) without a second division by cost. The historical max(u_norm - 0.25 c_norm,0)/wire_bytes is retained only as an ablation to test double cost penalization.

## Novelty

Novelty is not part of the locked core method. Historical validation selected alpha_n=0 on both datasets. Any future novelty study must be separately locked and must recompute novelty, score and priority after every admission.
