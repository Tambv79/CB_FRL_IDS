from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from r4_common import atomic_json, load_yaml, sha256_file


def _estimate_disk_requirement_gb(cfg: dict, root: Path, step6: Path) -> tuple[float, dict]:
    master = json.loads((step6 / "STEP6_MASTER_SUMMARY.json").read_text(encoding="utf-8"))
    processed = {entry["dataset"]: entry["processed"] for entry in master["dataset_results"]}

    def total_rows(dataset: str) -> int:
        return sum(int(processed[dataset][split]["rows"]) for split in ("train", "validation", "test"))

    def split_rows(dataset: str, split: str) -> int:
        return int(processed[dataset][split]["rows"])

    def n_features(dataset: str) -> int:
        return int(processed[dataset]["train"]["n_features"])

    # Dense float32 caches, labels, and a conservative allowance for compressed row/group metadata.
    cache_bytes = 0
    for dataset in ("CICIDS2017", "CICIoT2023"):
        rows = total_rows(dataset)
        cache_bytes += rows * n_features(dataset) * 4 + rows + rows * 32

    # Feature-sensitivity cache duplicates CICIDS2017 after removing 10 fields.
    cicids_rows = total_rows("CICIDS2017")
    cache_bytes += cicids_rows * max(1, n_features("CICIDS2017") - 10) * 4 + cicids_rows + cicids_rows * 32

    # int16 assignment arrays for 3 alphas x 10 seeds, plus one natural partition.
    assignment_bytes = (
        (split_rows("CICIDS2017", "train") + split_rows("CICIoT2023", "train")) * 2 * 30
        + split_rows("CICIoT2023", "train") * 2
    )

    dataset_counts = {"CICIDS2017": 0, "CICIoT2023": 0}
    manifest_files = [
        "R3_PRIMARY_CELL_MANIFEST.csv",
        "R3_HETEROGENEITY_CELL_MANIFEST.csv",
        "R3_NATURAL_PARTITION_CELL_MANIFEST.csv",
        "R3_OOD_STRESS_CELL_MANIFEST.csv",
        "R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv",
        "R3_ROBUSTNESS_CELL_MANIFEST.csv",
    ]
    total_train_cells = 0
    for filename in manifest_files:
        frame = pd.read_csv(root / "manifests" / filename)
        total_train_cells += len(frame)
        for protocol in frame["dataset_protocol"].astype(str):
            dataset = "CICIoT2023" if protocol.startswith("CICIoT2023") else "CICIDS2017"
            dataset_counts[dataset] += 1

    # Conservative upper bound per test row:
    # y int8 (1) + probability float32 (4) + row index int64 (8) + 1 byte allowance.
    prediction_bytes = (
        dataset_counts["CICIDS2017"] * split_rows("CICIDS2017", "test")
        + dataset_counts["CICIoT2023"] * split_rows("CICIoT2023", "test")
    ) * 14

    # One MiB per training cell covers checkpoint, CSVs, threshold grid, JSON locks, and logs.
    cell_record_bytes = total_train_cells * 1024 * 1024
    analysis_privacy_bytes = 1024**3

    estimated_work_bytes = (
        cache_bytes
        + assignment_bytes
        + prediction_bytes
        + cell_record_bytes
        + analysis_privacy_bytes
    )

    reserve_gb = float(cfg["runtime"].get("disk_space_reserve_gb", 4))
    minimum_gb = float(cfg["runtime"].get("disk_space_min_gb", 12))
    required_gb = max(minimum_gb, estimated_work_bytes / 1024**3 + reserve_gb)

    components = {
        "cache_gb": round(cache_bytes / 1024**3, 2),
        "assignments_gb": round(assignment_bytes / 1024**3, 2),
        "predictions_upper_bound_gb": round(prediction_bytes / 1024**3, 2),
        "cell_records_gb": round(cell_record_bytes / 1024**3, 2),
        "analysis_privacy_gb": 1.0,
        "reserve_gb": round(reserve_gb, 2),
        "required_with_reserve_gb": round(required_gb, 2),
    }
    return required_gb, components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    checks: list[dict] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(condition), "detail": detail})
        if not condition:
            errors.append(f"{name}: {detail}")

    for module in ["numpy", "pandas", "scipy", "sklearn", "torch", "yaml", "psutil"]:
        try:
            importlib.import_module(module)
            check(True, f"dependency_{module}")
        except Exception as exc:
            check(False, f"dependency_{module}", str(exc))

    for name, path in cfg["raw_dataset_paths"].items():
        check(Path(path).exists(), f"raw_path_{name}", path)

    step6 = Path(cfg["paths"]["step6_root"])
    check(step6.exists(), "step6_root", str(step6))
    master_path = step6 / "STEP6_MASTER_SUMMARY.json"
    check(master_path.exists(), "step6_master_summary")

    locked_r1 = root / "locks" / "R1_DATA_FLOW_CLOSURE" / "evidence"
    if master_path.exists():
        check(
            sha256_file(master_path) == sha256_file(locked_r1 / "STEP6_MASTER_SUMMARY.json"),
            "R1_master_summary_hash",
            "local Step6 master must match R1 authority",
        )

    executed_config = step6 / "STEP6_DATA_CONFIG_USED.yaml"
    if executed_config.exists():
        check(
            sha256_file(executed_config) == sha256_file(locked_r1 / "STEP6_DATA_CONFIG_USED.yaml"),
            "R1_data_config_hash",
            "local executed Step6 config must match R1 authority",
        )
    else:
        check(False, "step6_executed_config", "STEP6_DATA_CONFIG_USED.yaml missing")

    seeds = [42, 123, 777, 2026, 3407, 11, 314, 2718, 9001, 2027]
    for dataset in ["CICIDS2017", "CICIoT2023"]:
        for split in ["train", "validation", "test"]:
            split_dir = step6 / dataset / "processed" / split
            check(
                (split_dir / "part_manifest.csv").exists(),
                f"processed_manifest_{dataset}_{split}",
                str(split_dir),
            )
        for seed in seeds:
            assignment = (
                step6
                / dataset
                / "client_partitions"
                / f"client_assignment_alpha_0.5_seed_{seed}_clients_10.npy"
            )
            check(assignment.exists(), f"R1_primary_assignment_{dataset}_{seed}", str(assignment))

    suspicious = Path(cfg["paths"]["cicids_suspicious_feature_file"])
    check(suspicious.exists(), "cicids_flagged_feature_decisions", str(suspicious))

    ood_cache = Path(cfg["paths"]["legacy_ood_cache"])
    ood_manifest = ood_cache / "cache_manifest.json"
    check(ood_manifest.exists(), "legacy_OOD_cache", str(ood_cache))
    if ood_manifest.exists():
        import numpy as np

        manifest = json.loads(ood_manifest.read_text(encoding="utf-8"))
        expected_train_rows = int(manifest["splits"]["train"]["rows"])
        assignment_root = Path(cfg["paths"]["legacy_ood_assignment_root"])
        for seed in seeds:
            assignment = assignment_root / f"client_assignment_alpha_0.5_seed_{seed}_clients_10.npy"
            if assignment.exists():
                try:
                    check(
                        len(np.load(assignment, mmap_mode="r")) == expected_train_rows,
                        f"OOD_assignment_length_{seed}",
                        f"{assignment} vs {expected_train_rows}",
                    )
                except Exception as exc:
                    check(False, f"OOD_assignment_read_{seed}", str(exc))
            else:
                check(False, f"OOD_assignment_exists_{seed}", str(assignment))

    if master_path.exists():
        required_gb, components = _estimate_disk_requirement_gb(cfg, root, step6)
        disk_root = Path(cfg["project_root"]) if Path(cfg["project_root"]).exists() else Path.cwd()
        free_gb = shutil.disk_usage(disk_root).free / 1024**3
        detail = (
            f"{free_gb:.1f} GB free; estimated requirement including "
            f"{components['reserve_gb']:.1f} GB reserve = {required_gb:.1f} GB"
        )
        check(free_gb >= required_gb, "disk_space", detail)
        checks.append(
            {
                "check": "disk_estimate_components",
                "pass": True,
                "detail": json.dumps(components, sort_keys=True),
            }
        )

    expected = {
        "R3_PRIMARY_CELL_MANIFEST.csv": 980,
        "R3_HETEROGENEITY_CELL_MANIFEST.csv": 240,
        "R3_NATURAL_PARTITION_CELL_MANIFEST.csv": 60,
        "R3_OOD_STRESS_CELL_MANIFEST.csv": 60,
        "R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv": 20,
        "R3_ROBUSTNESS_CELL_MANIFEST.csv": 360,
        "R3_PRIVACY_CELL_MANIFEST.csv": 480,
        "R3_HYPOTHESIS_MANIFEST.csv": 108,
    }
    for filename, expected_rows in expected.items():
        frame = pd.read_csv(root / "manifests" / filename)
        valid = len(frame) == expected_rows
        if "cell_id" in frame:
            valid = valid and not frame["cell_id"].duplicated().any()
        check(valid, f"manifest_{filename}", f"rows={len(frame)} expected={expected_rows}")

    gates = [
        (root / "locks" / "R2_METHOD_LOCK" / "R2_GATE.json", "PASS_METHOD_LOCKED"),
        (
            root / "locks" / "R3_PROTOCOL_LOCK" / "R3_GATE.json",
            "PASS_PROTOCOL_LOCKED_READY_FOR_R4_PACKAGE_BUILD",
        ),
    ]
    for gate_path, expected_status in gates:
        obj = json.loads(gate_path.read_text(encoding="utf-8"))
        check(obj.get("status") == expected_status, f"gate_{gate_path.name}", str(obj.get("status")))

    validation_root = Path(cfg["paths"]["output_root"]) / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)

    from r4_common import environment_manifest

    env = environment_manifest()
    import platform, psutil
    env["declared_machine_lock"] = cfg["machine_lock"]
    env["actual_hardware"] = {"hostname": platform.node(), "platform": platform.platform(), "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(logical=True), "physical_cpu_count": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total/1024**3,3)}
    env["provenance_note"] = "declared_machine_lock is manuscript-planning metadata; actual_hardware is the executed host and governs runtime reporting"
    env["execution_config_sha256"] = sha256_file(config_path)
    env["r1_gate_sha256"] = sha256_file(root / "locks" / "R1_DATA_FLOW_CLOSURE" / "R1_GATE.json")
    env["r2_gate_sha256"] = sha256_file(root / "locks" / "R2_METHOD_LOCK" / "R2_GATE.json")
    env["r3_gate_sha256"] = sha256_file(root / "locks" / "R3_PROTOCOL_LOCK" / "R3_GATE.json")
    atomic_json(validation_root / "R4_EXECUTED_ENVIRONMENT_LOCK.json", env)

    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (validation_root / "R4_PIP_FREEZE.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:
        errors.append(f"pip_freeze: {exc}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": checks,
    }
    atomic_json(validation_root / "R4_PREFLIGHT.json", result)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
