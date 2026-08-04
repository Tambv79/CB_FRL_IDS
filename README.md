# CB-FedSelect

Reproducibility repository for the manuscript **Communication-Budgeted Pre-Upload Client Selection for Federated Intrusion Detection**.

> **Repository naming note.** The public repository URL retains the historical slug `CB_FRL_IDS` to preserve provenance and existing links. The scientific method name used in the manuscript, README, citation metadata, and current releases is **CB-FedSelect**.

## Scientific scope

CB-FedSelect is a pre-upload admission protocol for federated intrusion detection. It enforces a hard selected-update uplink budget by separating static eligibility, utility-cost ranking, residual byte feasibility, and aggregation. The repository supports communication accounting, same-realized-communication comparison, communication-to-target analysis, non-inferiority testing, targeted sensitivity, operational-prevalence diagnostics, attack/aggregation factorials, and metadata-leakage evaluation.

The evidence does **not** establish a universally optimal selector, a formal privacy mechanism, a general false-positive-safety guarantee, or Byzantine robustness. CICIoT2023 provides the positive primary evidence; CICIDS2017 is retained as negative cross-source evidence.

## Versioned evidence

- **v1.0.0**: immutable legacy provenance for the primary federated experiment and privacy/robustness artifact under the historical repository naming.
- **v1.1.0**: CB-FedSelect targeted-validity artifact adding 200 sensitivity cells, 30 centralized sanity baselines, operational-prevalence and false-alert-budget analyses, score-direction diagnostics, natural-source threshold diagnostics, and manuscript-aligned metadata.

Version 1.1.0 supplements rather than rewrites the v1.0.0 provenance.

## Repository contents

- `src/`: final training, privacy, analysis, validation, and finalization code.
- `config/`: portable configuration templates and executed protocol configuration.
- `manifests/`: locked R3 experimental, threshold, privacy, and hypothesis manifests.
- `locks/`: R1 data-flow, R2 method, and R3 protocol locks.
- `results/analysis/`: final statistical and communication analysis outputs.
- `results/summary/`: compact claim-audit and manuscript-ready tables.
- `validation/`: final PASS records, public execution provenance, and amendments.
- `data/`: acquisition and expected-structure instructions; no raw data are redistributed.
- `docs/`: reproducibility, data-availability, licensing, and release guidance.

The complete primary seed-level artifact is distributed through release `v1.0.0`. The targeted-validity evidence is distributed through release `v1.1.0`. Large generated artifacts are kept in GitHub Releases rather than committed as thousands of small Git objects.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/validate_repository.py
python -m compileall -q src scripts tests
python -m unittest discover -s tests -p "test_*.py" -v
```

To verify the primary release asset:

```bash
python scripts/verify_release_asset.py CB_FRL_IDS_REPRODUCIBILITY_ARTIFACT_v1.0.0.zip
```

The v1.1.0 targeted-validity asset is accompanied by its published SHA-256 checksum.

## Reproduction overview

1. Obtain CICIDS2017 and CICIoT2023 from the official providers.
2. Preserve the R1 source-group-aware data-flow and R3 protocol locks.
3. Adapt local paths using `config/R4_EXECUTION_CONFIG_TEMPLATE.yaml`.
4. Run package preflight and the required experiment phases.
5. Verify final outputs against the locked manifests and validation schema.

Detailed instructions are in `docs/REPRODUCIBILITY.md`.

## Data and code availability

The code, processed metadata, experiment manifests, seed-level results, and analysis artifacts supporting this study are publicly available at:

**https://github.com/Tambv79/CB_FRL_IDS**

Raw CICIDS2017 and CICIoT2023 files are not redistributed and remain subject to their original providers' terms. Additional supporting material is available from the corresponding author upon reasonable request.

## Citation

Use the **Cite this repository** panel generated from `CITATION.cff`.

## Authors

- Van-Tam Bui — first author — ORCID: 0009-0007-0064-266X
- Van-Hung Le — corresponding author
- Dac-Nhuong Le

## Licenses

- Source code: MIT License.
- Author-generated documentation, derived tables, and figures: CC BY 4.0.
- Raw third-party datasets: original provider terms apply; no raw data are redistributed.

See `LICENSE` and `NOTICE.md`.
