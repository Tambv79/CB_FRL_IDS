from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEEDS = [42, 123, 777, 2026, 3407, 11, 314, 2718, 9001, 2027]
EXPECTED_ROWS = {
    "R3_PRIMARY_CELL_MANIFEST.csv": 980,
    "R3_HETEROGENEITY_CELL_MANIFEST.csv": 240,
    "R3_NATURAL_PARTITION_CELL_MANIFEST.csv": 60,
    "R3_OOD_STRESS_CELL_MANIFEST.csv": 60,
    "R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv": 20,
    "R3_ROBUSTNESS_CELL_MANIFEST.csv": 360,
    "R3_PRIVACY_CELL_MANIFEST.csv": 480,
    "R3_HYPOTHESIS_MANIFEST.csv": 108,
}
REQUIRED_COLUMNS = {
    "R3_PRIMARY_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "method", "beta", "alpha_dir", "seed", "rounds", "role", "expected"],
    "R3_HETEROGENEITY_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "method", "beta", "alpha_dir", "seed", "rounds", "role", "expected"],
    "R3_NATURAL_PARTITION_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "method", "beta", "partition", "seed", "rounds", "role", "expected"],
    "R3_OOD_STRESS_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "method", "beta", "alpha_dir", "seed", "rounds", "role", "expected"],
    "R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "method", "beta", "seed", "rounds", "role", "expected"],
    "R3_ROBUSTNESS_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "selection", "aggregation", "model_attack", "reported_utility", "beta", "seed", "rounds", "role", "expected"],
    "R3_PRIVACY_CELL_MANIFEST.csv": ["cell_id", "dataset_protocol", "metadata_view", "attacker", "target", "privacy_split_seed", "group_folds", "group_definition", "role", "expected"],
    "R3_HYPOTHESIS_MANIFEST.csv": ["hypothesis_id", "family", "family_size", "dataset", "beta", "comparison", "endpoint", "null", "alternative", "test", "multiplicity", "status"],
}


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    l = left.copy().reset_index(drop=True)
    r = right.copy().reset_index(drop=True)
    for col in l.columns:
        lv = l[col]
        rv = r[col]
        if pd.api.types.is_numeric_dtype(lv) or pd.api.types.is_numeric_dtype(rv):
            a = pd.to_numeric(lv, errors="coerce").to_numpy(float)
            b = pd.to_numeric(rv, errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
                return False
        else:
            if not lv.fillna("<NA>").astype(str).equals(rv.fillna("<NA>").astype(str)):
                return False
    return True


def validate_manifests(root: Path) -> list[str]:
    errors: list[str] = []
    mroot = root / "manifests"
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = mroot / filename
        if not path.exists():
            errors.append(f"missing manifest: {filename}")
            continue
        frame = pd.read_csv(path)
        if len(frame) != expected_rows:
            errors.append(f"{filename}: rows={len(frame)} expected={expected_rows}")
        missing = [c for c in REQUIRED_COLUMNS[filename] if c not in frame.columns]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
        id_col = "cell_id" if "cell_id" in frame.columns else "hypothesis_id"
        if id_col in frame.columns and frame[id_col].duplicated().any():
            errors.append(f"{filename}: duplicate {id_col}")
        if "expected" in frame.columns and not (pd.to_numeric(frame["expected"], errors="coerce") == 1).all():
            errors.append(f"{filename}: expected column must be 1")
        if "rounds" in frame.columns and not (pd.to_numeric(frame["rounds"], errors="coerce") == 30).all():
            errors.append(f"{filename}: every rounds value must equal 30")

    # Every top-level R3 manifest must equal the locked R3 authority. The only
    # execution amendment is an added rounds=30 column in the robustness copy.
    locked_root = root / "locks" / "R3_PROTOCOL_LOCK" / "manifests"
    for filename in EXPECTED_ROWS:
        top = pd.read_csv(mroot / filename)
        locked = pd.read_csv(locked_root / filename)
        if filename == "R3_ROBUSTNESS_CELL_MANIFEST.csv":
            if "rounds" not in top.columns:
                errors.append("robustness execution manifest has no rounds column")
            reduced = top.drop(columns=["rounds"], errors="ignore")
            if not _same_frame(reduced, locked):
                errors.append("robustness execution manifest differs from locked R3 beyond added rounds=30")
        elif not _same_frame(top, locked):
            errors.append(f"{filename}: top-level copy differs from locked R3 authority")

    rob = pd.read_csv(mroot / "R3_ROBUSTNESS_CELL_MANIFEST.csv")
    if set(rob["dataset_protocol"]) != {"CICIDS2017_COVERAGE_R1", "CICIoT2023_R1"}:
        errors.append("robustness datasets invalid")
    if set(rob["selection"]) != {"Full-Contextual", "CB-Score"}:
        errors.append("robustness selections invalid")
    if set(rob["aggregation"]) != {"FedAvg", "Norm-Clipped-Averaging"}:
        errors.append("robustness aggregations invalid")
    if set(rob["model_attack"]) != {"clean", "label-flip", "sign-flip"}:
        errors.append("robustness attacks invalid")
    if set(rob["reported_utility"]) != {"honest", "forged_inflated"}:
        errors.append("robustness reported-utility conditions invalid")
    if set(map(int, rob["seed"])) != set(SEEDS):
        errors.append("robustness paired seeds invalid")
    if len(rob[rob.selection == "Full-Contextual"]) != 120:
        errors.append("robustness Full-Contextual count must be 120")
    if len(rob[rob.selection == "CB-Score"]) != 240:
        errors.append("robustness CB-Score count must be 240")
    if not (rob.loc[rob.selection == "Full-Contextual", "reported_utility"] == "honest").all():
        errors.append("Full-Contextual must use honest reported utility only")
    if not rob.loc[rob.selection == "Full-Contextual", "beta"].isna().all():
        errors.append("Full-Contextual robustness beta must be empty")
    if not np.allclose(rob.loc[rob.selection == "CB-Score", "beta"].astype(float), 0.4):
        errors.append("CB-Score robustness beta must be 0.4")
    expected_combo = {
        (ds, sel, agg, atk, rep, seed)
        for ds in ["CICIDS2017_COVERAGE_R1", "CICIoT2023_R1"]
        for sel in ["Full-Contextual", "CB-Score"]
        for agg in ["FedAvg", "Norm-Clipped-Averaging"]
        for atk in ["clean", "label-flip", "sign-flip"]
        for rep in (["honest"] if sel == "Full-Contextual" else ["honest", "forged_inflated"])
        for seed in SEEDS
    }
    observed_combo = set(zip(rob.dataset_protocol, rob.selection, rob.aggregation, rob.model_attack, rob.reported_utility, rob.seed.astype(int)))
    if observed_combo != expected_combo:
        errors.append(f"robustness factorial mismatch: missing={len(expected_combo-observed_combo)}, extra={len(observed_combo-expected_combo)}")

    priv = pd.read_csv(mroot / "R3_PRIVACY_CELL_MANIFEST.csv")
    expected_priv = {
        (ds, view, attacker, target, seed)
        for ds in ["CICIDS2017_COVERAGE_R1", "CICIoT2023_R1"]
        for view in ["Raw", "Noisy-Quantized", "No-Size", "Minimal-Bucketed"]
        for attacker in ["LogisticRegression", "RandomForest", "MLP"]
        for target in ["client_size_above_seed_median", "attack_ratio_above_seed_median"]
        for seed in SEEDS
    }
    observed_priv = set(zip(priv.dataset_protocol, priv.metadata_view, priv.attacker, priv.target, priv.privacy_split_seed.astype(int)))
    if observed_priv != expected_priv:
        errors.append(f"privacy factorial mismatch: missing={len(expected_priv-observed_priv)}, extra={len(observed_priv-expected_priv)}")
    if not (priv.group_folds.astype(int) == 5).all() or set(priv.group_definition) != {"client_id"}:
        errors.append("privacy group-disjoint design must be five folds grouped by client_id")

    return errors


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    errors = validate_manifests(Path(args.root))
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
