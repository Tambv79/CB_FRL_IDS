
# Reproducibility guide

## Reproduction levels

1. **Audit-only:** inspect locked manifests, final analysis, validation, and SHA-256 records.
2. **Analysis reproduction:** extract the `v1.0.0` release artifact and rerun final analysis using
   the validated cell outputs without retraining.
3. **Full experimental reproduction:** obtain the original datasets, reconstruct the R1 data
   pipeline, run preflight, and execute all locked R3 cells.

## Scientific locks

- R1 closes the data flow and source-group splits.
- R2 locks utility, static eligibility, ranking, residual feasibility, and empty-round behavior.
- R3 locks budgets, baselines, seeds, thresholds, statistical families, robustness, and privacy.
- R4 validates 1,720 training cells and 480 privacy evaluations.

Changing any lock requires a new protocol identifier and an explicit amendment.

## Analysis-only reproduction

After downloading the release asset:

```bash
python scripts/verify_release_asset.py CB_FRL_IDS_REPRODUCIBILITY_ARTIFACT_v1.0.0.zip
```

Extract the archive, inspect `results/analysis/ANALYSIS_COMPLETE.json`, and use the final analysis
scripts in `src/analysis/`. Do not use historical recovery logs as scientific evidence.

## Hardware

Detection, communication-byte, selection, robustness, and privacy conclusions are hardware
independent. The Dell PowerEdge R750xs is retained as the reference research infrastructure;
wall-clock speedup is not a scientific claim.
