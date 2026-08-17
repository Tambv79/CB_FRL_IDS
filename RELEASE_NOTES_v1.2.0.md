# CB-FedSelect v1.2.0 — Advisor-Compliance Evidence Synchronization

This release synchronizes the final advisor-compliance evidence for the manuscript:

**Communication-Budgeted Pre-Upload Client Selection for Federated Intrusion Detection**

## Scientific status

Version 1.2.0 does not rerun or retune the primary federated-learning experiment.

- No full R4 rerun.
- No post-hoc primary-method retuning.
- Primary experimental provenance remains unchanged.
- Advisor-targeted evidence and corrected reporting artifacts are synchronized.

## Added and corrected evidence

### Operational prevalence
- Corrected pooled-rate PPV/NPV for Full-Contextual and CB-FedSelect at β = 0.2 and β = 0.4.
- PPV/NPV are computed after paired-seed TPR/FPR aggregation.
- Seedwise PPV/NPV distributions are reported separately.

### Robustness
- Complete 36-cell robustness factorial.
- Bounded percentile-bootstrap confidence intervals for:
  - F1;
  - PR-AUC;
  - FPR;
  - malicious share of selected uplink.

### Privacy
- Corrected ADVISOR-PRIVACY-CORRECTED-V2 evidence.
- 480 individual attacker evaluations.
- 48 aggregate dataset × metadata-view × attacker × target cells.
- 480 per-client-group AUC rows.
- Client-group uncertainty rather than treating repeated split seeds as independent federations.
- ECDF and group-distribution artifacts.

### CICIDS2017 audit
- Deterministic raw-label-to-binary mapping audit.
- Train-only preprocessing audit.
- Ordered 78-feature manifest.
- Split/hash-integrity evidence.
- CICIDS2017 remains negative cross-source diagnostic evidence and is not used as operational validation.

## Claim boundaries

This release does not establish:
- universal client-selector superiority;
- formal privacy;
- universal false-positive safety;
- Byzantine robustness;
- wall-clock speedup;
- deployment readiness.

## Provenance

The historical repository slug `CB_FRL_IDS` is retained to preserve existing links and provenance. The scientific method name is **CB-FedSelect**.

Primary and targeted-validity historical releases remain available as v1.0.0 and v1.1.0.
