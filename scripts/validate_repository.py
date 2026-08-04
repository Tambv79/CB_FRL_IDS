from __future__ import annotations

import csv
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

REPO_URL = "https://github.com/Tambv79/CB_FRL_IDS"
METHOD = "CB-FedSelect"
MANUSCRIPT_TITLE = "Communication-Budgeted Pre-Upload Client Selection for Federated Intrusion Detection"
PUBLIC_METADATA_FILES = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "docs/DATA_AVAILABILITY.md",
]
FORBIDDEN_PLACEHOLDERS = (
    "REPLACE_WITH_GITHUB_USERNAME",
    "github.com/USERNAME",
    "github.com/YOUR_USERNAME",
    "https://github.com/OWNER/",
)
FORBIDDEN_PUBLIC_LEGACY_PHRASES = (
    "# CB-FRL-IDS",
    "CB-FRL-IDS Reproducibility Repository",
    "CB-FRL-IDS: Communication-Budgeted",
)


def rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []

    required = [
        "README.md",
        "CITATION.cff",
        ".zenodo.json",
        "codemeta.json",
        "LICENSE",
        "requirements.txt",
        "validation/R4_VALIDATION_FINAL.json",
        "docs/DATA_AVAILABILITY.md",
    ]
    for rel in required:
        if not (root / rel).is_file():
            errors.append(f"missing: {rel}")

    for rel in ["src", "config", "manifests", "locks", "results", "validation", "docs"]:
        if not (root / rel).is_dir():
            errors.append(f"missing directory: {rel}")

    for name, expected in EXPECTED.items():
        path = root / "manifests" / name
        if not path.is_file():
            errors.append(f"missing manifest: {name}")
        else:
            got = rows(path)
            if got != expected:
                errors.append(f"manifest cardinality: {name} expected={expected} got={got}")

    try:
        cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
        if cff.get("cff-version") != "1.2.0":
            errors.append("CITATION.cff must use cff-version 1.2.0")
        if len(cff.get("authors", [])) != 3:
            errors.append("CITATION.cff author count must be 3")
        if cff.get("version") != "1.1.0":
            errors.append("CITATION.cff version must be 1.1.0")
        if cff.get("repository-code") != REPO_URL or cff.get("url") != REPO_URL:
            errors.append("CITATION.cff repository URL mismatch")
        if METHOD not in str(cff.get("title", "")):
            errors.append("CITATION.cff title must use CB-FedSelect")
    except Exception as exc:
        errors.append(f"invalid CITATION.cff: {exc}")

    for rel in [".zenodo.json", "codemeta.json"]:
        try:
            data = json.loads((root / rel).read_text(encoding="utf-8"))
            if str(data.get("version")) != "1.1.0":
                errors.append(f"{rel} version must be 1.1.0")
        except Exception as exc:
            errors.append(f"invalid {rel}: {exc}")

    try:
        final = json.loads((root / "validation/R4_VALIDATION_FINAL.json").read_text(encoding="utf-8"))
        if final.get("status") != "PASS" or final.get("errors"):
            errors.append("R4 final validation is not PASS")
    except Exception as exc:
        errors.append(f"invalid R4 validation: {exc}")

    combined = "\n".join(
        (root / rel).read_text(encoding="utf-8", errors="replace")
        for rel in PUBLIC_METADATA_FILES
        if (root / rel).is_file()
    )
    for token in FORBIDDEN_PLACEHOLDERS:
        if token in combined:
            errors.append(f"public metadata placeholder remains: {token}")
    for phrase in FORBIDDEN_PUBLIC_LEGACY_PHRASES:
        if phrase in combined:
            errors.append(f"legacy public scientific naming remains: {phrase}")
    for required_text in [METHOD, MANUSCRIPT_TITLE, REPO_URL, "1.1.0"]:
        if required_text not in combined:
            errors.append(f"required public metadata missing: {required_text}")

    for obsolete in ["SET_REPOSITORY_URL.cmd", "PUBLISH_WITH_GIT.cmd", "scripts/set_repository_url.py"]:
        if (root / obsolete).exists():
            errors.append(f"obsolete one-time publication helper remains: {obsolete}")

    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
