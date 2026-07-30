
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

EXPECTED = {
    "R3_PRIMARY_CELL_MANIFEST.csv": 980,
    "R3_HETEROGENEITY_CELL_MANIFEST.csv": 240,
    "R3_NATURAL_PARTITION_CELL_MANIFEST.csv": 60,
    "R3_OOD_STRESS_CELL_MANIFEST.csv": 60,
    "R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv": 20,
    "R3_ROBUSTNESS_CELL_MANIFEST.csv": 360,
    "R3_PRIVACY_CELL_MANIFEST.csv": 480,
    "R3_HYPOTHESIS_MANIFEST.csv": 108,
}
PLACEHOLDER = "REPLACE_WITH_GITHUB_USERNAME"

def rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-placeholder", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    required = ["README.md", "CITATION.cff", "LICENSE", "requirements.txt", "validation/R4_VALIDATION_FINAL.json"]
    for rel in required:
        if not (root / rel).is_file():
            errors.append(f"missing: {rel}")
    for name, expected in EXPECTED.items():
        p = root / "manifests" / name
        if not p.is_file():
            errors.append(f"missing manifest: {name}")
        elif rows(p) != expected:
            errors.append(f"manifest cardinality: {name} expected={expected} got={rows(p)}")
    try:
        cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
        if cff.get("cff-version") != "1.2.0": errors.append("CITATION.cff must use cff-version 1.2.0")
        if len(cff.get("authors", [])) != 3: errors.append("CITATION.cff author count must be 3")
    except Exception as exc:
        errors.append(f"invalid CITATION.cff: {exc}")
    final = json.loads((root / "validation/R4_VALIDATION_FINAL.json").read_text(encoding="utf-8"))
    if final.get("status") != "PASS" or final.get("errors"):
        errors.append("R4 final validation is not PASS")
    if not args.allow_placeholder:
        for rel in ["README.md", "CITATION.cff", "codemeta.json", ".zenodo.json", "docs/DATA_AVAILABILITY.md"]:
            if PLACEHOLDER in (root / rel).read_text(encoding="utf-8"):
                errors.append(f"repository URL placeholder remains in {rel}")
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
