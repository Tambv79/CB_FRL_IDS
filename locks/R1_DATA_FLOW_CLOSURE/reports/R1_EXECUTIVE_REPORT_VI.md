# R1 — DATA-FLOW CLOSURE: BÁO CÁO ĐIỀU HÀNH

## Trạng thái

`PASS_DATA_FLOW_CLOSED_WITH_LEGACY_METADATA_SUPERSEDED`

## Kết luận CICIoT2023

Data-flow đã được khép kín tuyệt đối:

- Raw ingestion: 46,776,700 rows (1,098,191 benign; 45,678,509 attack).
- Exact feature hashes: 21,074,316.
- Conflicting-label hashes removed: 906.
- Unique nonconflicting pool: 21,073,410 (1,093,034 benign; 19,980,376 attack).
- Source groups are assigned to train/validation/test before cohort selection.
- Locked cohort cap: 600,000 rows, comprising 420,000 train, 90,000 validation and 90,000 test; target benign fraction is exactly 25%.
- Cohort selection is deterministic, not random: quotas are allocated within split × label × source class and rows are selected by `ORDER BY row_hash LIMIT quota`.
- Transformed-collision removal: 3,061 rows (415 benign; 2,646 attack).
- Final experimental splits: 596,939 rows (149,585 benign; 447,354 attack).
- Final retention is 2.832664% of the unique nonconflicting pool. The apparent “missing” 20,476,471 rows consist of 20,473,410 intentionally excluded by the locked cohort cap plus 3,061 globally removed transformed-collision rows.

The benign prevalence changed intentionally from 5.186792% in the unique nonconflicting pool to 25% in the selected cohort. Collision cleaning changed it only to 25.058674%.

## Leakage proof

- Source group is the relative raw file.
- Group assignment occurs before cohort selection.
- No source group appears in more than one split.
- Cohort selection occurs only inside its already assigned split.
- After global transformed-collision removal, within-split repeated transformed hashes = 0 and cross-split transformed hashes = 0.

## Sampling stability

Reviewer item 8 is closed as `PASS_NOT_APPLICABLE`: cohort sampling is deterministic and has no sampling seed. The split assignment has a locked seed of 42, but it precedes cohort selection and is recorded in the source-group split manifest.

## CICIDS2017 metadata repair

The old `cohort_selection_summary.json` and `cohort_selection_counts.csv` were generated before the coverage-preserving split repair and are stale. Because the executed config has `max_unique_rows: null`, the authoritative cohort is exactly every unique nonconflicting row in the repaired `source_group_split_manifest.csv`. R1 therefore supplies corrected cohort files and explicitly supersedes the stale copies.

## Hash-manifest finding

The legacy `step6_artifact_hashes.csv` files were also generated before all repair outputs were finalized. Thirteen included metadata entries differ from their legacy hash records. R1 preserves those manifests as historical evidence but does not treat them as current authority. The R1 package contains a fresh package-level SHA-256 manifest.

## Scientific boundary

R1 closes only data lineage, cohort selection, prevalence and leakage. It does not validate the historical selector construct. No manuscript rewriting or new experiment is permitted until R2 locks utility, eligibility, fallback behavior, budget semantics and novelty scope.
