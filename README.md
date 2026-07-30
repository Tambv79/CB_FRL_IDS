
# CB-FRL-IDS

**Communication-Budgeted Pre-Upload Policy-Update Selection for Federated Network Intrusion Detection**

This repository contains the public code, locked protocol manifests, validation records,
and derived results supporting the CB-FRL-IDS study. The method controls which complete
client updates are admitted **before upload** under a hard selected-update uplink budget.

## Evidence boundary

- **Primary positive evidence:** CICIoT2023.
- **Negative/generalization evidence:** CICIDS2017 coverage-preserving and file-disjoint protocols.
- The evidence does **not** support universal same-budget superiority, a general FPR-safety
  guarantee, formal privacy, or Byzantine robustness.
- Raw CICIDS2017 and CICIoT2023 files are **not redistributed**. They remain subject to the
  original providers' access and distribution terms.

## Repository contents

- `src/`: final training, privacy, analysis, validation, and finalization code.
- `config/`: portable configuration template and executed protocol configuration.
- `manifests/`: locked R3 experimental, threshold, privacy, and hypothesis manifests.
- `locks/`: R1 data-flow, R2 method, and R3 protocol locks.
- `results/analysis/`: final statistical and communication analysis outputs.
- `results/summary/`: compact R5 claim-audit and manuscript-ready tables.
- `validation/`: final PASS records, public execution provenance, and amendments.
- `data/`: acquisition and expected-structure instructions; no raw data are redistributed.
- `docs/`: reproducibility, data-availability, licensing, and release guidance.

The complete seed-level artifact (1,720 validated training cells and 480 validated privacy
evaluations) is distributed as the GitHub Release asset
`CB_FRL_IDS_REPRODUCIBILITY_ARTIFACT_v1.0.0.zip` rather than committed as thousands of
small Git objects.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/validate_repository.py --allow-placeholder
python scripts/verify_release_asset.py ../release_assets/CB_FRL_IDS_REPRODUCIBILITY_ARTIFACT_v1.0.0.zip
```

Before publishing, replace the repository placeholder:

```bash
python scripts/set_repository_url.py "https://github.com/YOUR_USERNAME/CB-FRL-IDS"
python scripts/validate_repository.py
```

## Reproduction overview

1. Obtain CICIDS2017 and CICIoT2023 from the official UNB Canadian Institute for Cybersecurity pages.
2. Preserve the R1 source-group-aware data-flow and R3 protocol locks.
3. Adapt local paths using `config/R4_EXECUTION_CONFIG_TEMPLATE.yaml`.
4. Run package preflight and the required experiment phases.
5. Verify final outputs against the locked manifests and validation schema.

Detailed instructions are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data and code availability

The code, processed metadata, experiment manifests, seed-level results, and analysis artifacts
supporting this study are publicly available in this repository: https://github.com/USERNAME/CB-FRL-IDS.
Raw CICIDS2017 and CICIoT2023 data are not redistributed and remain available from their
original providers under their respective terms. Additional supporting material is available
from the corresponding author upon reasonable request.

## Citation

Use the **Cite this repository** panel generated from [`CITATION.cff`](CITATION.cff).

## Authors

- Van-Tam Bui (first author; ORCID: 0009-0007-0064-266X)
- Van-Hung Le (corresponding author)
- Dac-Nhuong Le

## Licenses

- Source code: MIT License.
- Author-generated documentation, derived tables, and figures: CC BY 4.0.
- Raw third-party datasets: original provider terms apply; no raw data are redistributed.

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
