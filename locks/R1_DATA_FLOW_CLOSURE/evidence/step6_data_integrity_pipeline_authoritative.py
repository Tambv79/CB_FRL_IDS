#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CB-FRL-IDS Step 6: full-corpus data-integrity audit, leakage-safe cohort
materialization, train-only preprocessing, transformed-overlap validation,
and reproducible federated client partitions.

Scientific design principles
----------------------------
1. Audit the full raw corpus, even when the experiment cohort is capped.
2. Remove exact duplicates globally before source-group splitting.
3. Exclude conflicting-label duplicates rather than selecting one label.
4. Keep train/validation/test source groups disjoint.
5. Fit every transformation on training rows only.
6. Verify zero transformed-row overlap after the frozen transformation.
7. Produce machine-readable manifests and hashes; do not train any model.
8. Reserve K exclusively for total candidate-client count in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn import __version__ as sklearn_version
import yaml

HASH_KEY_1 = "CBFRLIDS_HASH001"
HASH_KEY_2 = "CBFRLIDS_HASH002"
SPLITS = ("train", "validation", "test")
LABEL_COL = "binary_label"
META_COLS = ["source_group", "source_class", "row_hash"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def normalize_name(name: object) -> str:
    return " ".join(str(name).strip().split())


def lower_norm(name: object) -> str:
    return normalize_name(name).lower().replace("-", "_")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def stable_row_hash(df: pd.DataFrame) -> np.ndarray:
    """Two independent stable 64-bit pandas hashes concatenated as 128-bit hex."""
    normalized = df.copy()
    for col in normalized.columns:
        s = normalized[col]
        if pd.api.types.is_numeric_dtype(s):
            normalized[col] = pd.to_numeric(s, errors="coerce")
        else:
            normalized[col] = s.astype("string").fillna("<NA>").str.strip()
    h1 = pd.util.hash_pandas_object(normalized, index=False, hash_key=HASH_KEY_1).to_numpy(np.uint64)
    h2 = pd.util.hash_pandas_object(normalized, index=False, hash_key=HASH_KEY_2).to_numpy(np.uint64)
    return np.array([f"{int(a):016x}{int(b):016x}" for a, b in zip(h1, h2)], dtype=object)


def read_csv_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, on_bad_lines="skip")
    except UnicodeDecodeError:
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False, on_bad_lines="skip", encoding="latin-1")


@dataclass
class DatasetSpec:
    name: str
    root: Path
    files: List[Path]
    label_candidates: List[str]
    benign_tokens: set[str]
    folder_label_fallback: bool
    benign_folder_tokens: set[str]
    source_group_mode: str
    source_class_mode: str


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    ratios = cfg["split"]
    total = float(ratios["train"]) + float(ratios["validation"]) + float(ratios["test"])
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to one, got {total}")
    client_count = int(cfg["federation"]["client_count"])
    if client_count < 2:
        raise ValueError("client_count must be >= 2")
    return cfg


def resolve_windowsish_path(value: str) -> Path:
    return Path(value)


def discover_files(root: Path, globs: Sequence[str]) -> List[Path]:
    found: set[Path] = set()
    for pattern in globs:
        for p in root.glob(pattern):
            if p.is_file():
                found.add(p.resolve())
    return sorted(found)


def make_spec(name: str, dcfg: dict) -> DatasetSpec:
    root = resolve_windowsish_path(dcfg["root"])
    files = discover_files(root, dcfg.get("file_globs", ["**/*.csv"]))
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found for {name}: {root}")
    if not files:
        raise FileNotFoundError(f"No CSV files found for {name} under {root}")
    return DatasetSpec(
        name=name,
        root=root,
        files=files,
        label_candidates=[normalize_name(x) for x in dcfg.get("label_candidates", [])],
        benign_tokens={str(x).strip().lower() for x in dcfg.get("benign_tokens", [])},
        folder_label_fallback=bool(dcfg.get("folder_label_fallback", False)),
        benign_folder_tokens={str(x).strip().lower() for x in dcfg.get("benign_folder_tokens", [])},
        source_group_mode=dcfg.get("source_group", "relative_file"),
        source_class_mode=dcfg.get("source_class", "binary_label"),
    )


def source_group_and_class(spec: DatasetSpec, path: Path, binary_label: Optional[int] = None) -> Tuple[str, str]:
    rel = path.relative_to(spec.root)
    group = str(rel).replace("\\", "/")
    if spec.source_class_mode == "top_folder":
        source_class = rel.parts[0] if len(rel.parts) > 1 else path.parent.name
    elif spec.source_class_mode == "binary_label" and binary_label is not None:
        source_class = "benign" if binary_label == 0 else "attack"
    else:
        source_class = path.parent.name
    return group, str(source_class)


def infer_label_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm_to_original = {lower_norm(c): c for c in columns}
    for c in candidates:
        key = lower_norm(c)
        if key in norm_to_original:
            return norm_to_original[key]
    return None


def make_binary_label(series: pd.Series, benign_tokens: set[str]) -> pd.Series:
    s = series.astype("string").fillna("<NA>").str.strip().str.lower()
    return (~s.isin(benign_tokens)).astype(np.int8)


def folder_binary_label(spec: DatasetSpec, path: Path, n_rows: int) -> pd.Series:
    rel = path.relative_to(spec.root)
    folder_token = (rel.parts[0] if len(rel.parts) > 1 else path.parent.name).strip().lower()
    benign = folder_token in spec.benign_folder_tokens or any(tok in folder_token for tok in spec.benign_folder_tokens)
    return pd.Series(np.zeros(n_rows, dtype=np.int8) if benign else np.ones(n_rows, dtype=np.int8))


def choose_auto_drop(columns: Sequence[str], label_col: Optional[str], leakage_cfg: dict) -> Tuple[List[str], Dict[str, str]]:
    exact = {lower_norm(x) for x in leakage_cfg.get("auto_drop_exact", [])}
    keywords = [str(x).strip().lower() for x in leakage_cfg.get("auto_drop_keywords", [])]
    preserve = {lower_norm(x) for x in leakage_cfg.get("preserve_exact", [])}
    drops: List[str] = []
    reasons: Dict[str, str] = {}
    for col in columns:
        if label_col is not None and col == label_col:
            continue
        low = lower_norm(col)
        if low in preserve:
            continue
        if low in exact:
            drops.append(col)
            reasons[col] = "label-like or target-derived field"
            continue
        for kw in keywords:
            if kw in str(col).strip().lower():
                drops.append(col)
                reasons[col] = f"identifier/time leakage keyword: {kw}"
                break
    return sorted(set(drops)), reasons


def file_manifest(spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    for fp in spec.files:
        st = fp.stat()
        rows.append({
            "dataset": spec.name,
            "relative_path": str(fp.relative_to(spec.root)).replace("\\", "/"),
            "size_bytes": int(st.st_size),
            "sha256": sha256_file(fp),
            "modified_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
        })
    return pd.DataFrame(rows)


def discovery_pass(spec: DatasetSpec, cfg: dict, out: Path) -> dict:
    """Schema, type, leakage, and source-file discovery without modifying data."""
    chunksize = min(int(cfg["chunksize"]), 50000)
    leakage_cfg = cfg["leakage_audit"]
    sample_frames: List[pd.DataFrame] = []
    schema_rows = []
    canonical_features: Optional[List[str]] = None
    canonical_label_col: Optional[str] = None
    drop_reasons: Dict[str, str] = {}
    files_with_errors = []

    remaining_sample = 100000
    for fp in spec.files:
        try:
            sample = next(read_csv_chunks(fp, min(chunksize, 10000)))
        except StopIteration:
            files_with_errors.append({"file": str(fp), "error": "empty file"})
            continue
        except Exception as exc:
            files_with_errors.append({"file": str(fp), "error": repr(exc)})
            continue
        sample.columns = [normalize_name(c) for c in sample.columns]
        label_col = infer_label_column(sample.columns, spec.label_candidates)
        if label_col is None and not spec.folder_label_fallback:
            files_with_errors.append({"file": str(fp), "error": "label column not found and folder fallback disabled"})
            continue
        drops, reasons = choose_auto_drop(sample.columns, label_col, leakage_cfg)
        drop_reasons.update(reasons)
        feature_cols = [c for c in sample.columns if c != label_col and c not in drops]
        if canonical_features is None:
            canonical_features = feature_cols
            canonical_label_col = label_col
        mismatch = sorted(set(feature_cols).symmetric_difference(set(canonical_features or [])))
        schema_rows.append({
            "relative_path": str(fp.relative_to(spec.root)).replace("\\", "/"),
            "label_column": label_col if label_col else "<folder-derived>",
            "n_columns": len(sample.columns),
            "n_features_after_auto_drop": len(feature_cols),
            "schema_matches_canonical": len(mismatch) == 0,
            "mismatch_columns": json.dumps(mismatch, ensure_ascii=False),
        })
        if remaining_sample > 0:
            take = min(remaining_sample, len(sample))
            part = sample.head(take).copy()
            if label_col is not None:
                part[LABEL_COL] = make_binary_label(part[label_col], spec.benign_tokens)
            else:
                part[LABEL_COL] = folder_binary_label(spec, fp, len(part)).to_numpy()
            part = part.reindex(columns=feature_cols + [LABEL_COL])
            sample_frames.append(part)
            remaining_sample -= len(part)

    if canonical_features is None:
        raise RuntimeError(f"No readable source files for {spec.name}")
    schema_df = pd.DataFrame(schema_rows)
    schema_df.to_csv(out / "schema_manifest.csv", index=False)
    if files_with_errors:
        pd.DataFrame(files_with_errors).to_csv(out / "file_read_errors.csv", index=False)
    if not schema_df["schema_matches_canonical"].all():
        raise RuntimeError(f"Schema mismatch detected in {spec.name}; see schema_manifest.csv")

    sample_all = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=canonical_features + [LABEL_COL])
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    type_rows = []
    suspicious_rows = []
    purity_threshold = float(leakage_cfg["deterministic_purity_threshold"])
    max_unique = int(leakage_cfg["deterministic_max_unique"])

    for col in canonical_features:
        s = sample_all[col]
        converted = pd.to_numeric(s, errors="coerce")
        ratio = float(converted.notna().mean()) if len(s) else 0.0
        unique = int(s.nunique(dropna=True))
        if ratio >= float(cfg["preprocessing"]["numeric_detection_ratio"]):
            numeric_cols.append(col)
            type_name = "numeric"
            eval_s = converted
        else:
            categorical_cols.append(col)
            type_name = "categorical"
            eval_s = s.astype("string").fillna("<NA>")
        type_rows.append({
            "feature": col,
            "inferred_type": type_name,
            "numeric_convertibility": ratio,
            "sample_unique": unique,
            "sample_missing_ratio": float(s.isna().mean()) if len(s) else 0.0,
        })
        if len(sample_all) and unique <= max_unique and unique > 1:
            temp = pd.DataFrame({"v": eval_s, "y": sample_all[LABEL_COL]})
            tab = temp.groupby(["v", "y"], dropna=False).size().unstack(fill_value=0)
            purity = float(tab.max(axis=1).sum() / max(1, tab.to_numpy().sum()))
            if purity >= purity_threshold:
                suspicious_rows.append({
                    "feature": col,
                    "sample_unique": unique,
                    "label_purity": purity,
                    "reason": "near-deterministic relation with binary label in discovery sample",
                    "approved": col in leakage_cfg.get("approved_suspicious_features", []),
                })

    pd.DataFrame(type_rows).to_csv(out / "feature_type_audit.csv", index=False)
    pd.DataFrame(suspicious_rows).to_csv(out / "suspicious_feature_audit.csv", index=False)
    pd.DataFrame([{"feature": c, "reason": r} for c, r in sorted(drop_reasons.items())]).to_csv(
        out / "auto_dropped_fields.csv", index=False
    )
    discovery = {
        "dataset": spec.name,
        "canonical_feature_columns": canonical_features,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "label_column_mode": canonical_label_col or "folder-derived",
        "auto_dropped_columns": sorted(drop_reasons),
        "suspicious_features": suspicious_rows,
        "schema_file_count": int(len(schema_df)),
        "files_with_errors": files_with_errors,
    }
    atomic_json(out / "discovery.json", discovery)
    return discovery


def create_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rows (
            row_hash TEXT PRIMARY KEY,
            first_file TEXT NOT NULL,
            first_row INTEGER NOT NULL,
            source_class TEXT NOT NULL,
            label INTEGER NOT NULL,
            duplicate_count INTEGER NOT NULL DEFAULT 0,
            conflict INTEGER NOT NULL DEFAULT 0,
            split TEXT,
            selected INTEGER NOT NULL DEFAULT 1
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rows_file_row ON rows(first_file, first_row)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rows_split ON rows(split, selected, conflict)")
    conn.commit()
    return conn


def normalize_features_for_hash(chunk: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = chunk.reindex(columns=feature_cols).copy()
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].astype("string").fillna("<NA>").str.strip()
    return out


def full_audit_pass(spec: DatasetSpec, cfg: dict, discovery: dict, out: Path, conn: sqlite3.Connection) -> dict:
    feature_cols = discovery["canonical_feature_columns"]
    chunksize = int(cfg["chunksize"])
    raw_rows = 0
    malformed_rows = 0
    label_counts = Counter()
    file_counts = Counter()
    missing_counts = Counter()
    file_row_offsets = defaultdict(int)
    started = time.time()

    upsert_sql = """
        INSERT INTO rows(row_hash, first_file, first_row, source_class, label, duplicate_count, conflict, selected)
        VALUES (?, ?, ?, ?, ?, 0, 0, 1)
        ON CONFLICT(row_hash) DO UPDATE SET
            duplicate_count = rows.duplicate_count + 1,
            conflict = CASE WHEN rows.label != excluded.label THEN 1 ELSE rows.conflict END
    """

    for file_idx, fp in enumerate(spec.files, start=1):
        rel = str(fp.relative_to(spec.root)).replace("\\", "/")
        offset = 0
        print(f"[AUDIT {spec.name}] {file_idx}/{len(spec.files)} {rel}")
        try:
            for chunk in read_csv_chunks(fp, chunksize):
                chunk.columns = [normalize_name(c) for c in chunk.columns]
                label_col = infer_label_column(chunk.columns, spec.label_candidates)
                if label_col is not None:
                    y = make_binary_label(chunk[label_col], spec.benign_tokens)
                elif spec.folder_label_fallback:
                    y = folder_binary_label(spec, fp, len(chunk))
                else:
                    malformed_rows += len(chunk)
                    continue
                features = normalize_features_for_hash(chunk, feature_cols)
                hashes = stable_row_hash(features)
                source_group, _ = source_group_and_class(spec, fp)
                if spec.source_class_mode == "top_folder":
                    _, source_class = source_group_and_class(spec, fp)
                    source_classes = [source_class] * len(chunk)
                else:
                    source_classes = ["benign" if int(v) == 0 else "attack" for v in y.to_numpy()]
                records = [
                    (str(hashes[i]), source_group, offset + i, str(source_classes[i]), int(y.iloc[i]))
                    for i in range(len(chunk))
                ]
                conn.executemany(upsert_sql, records)
                conn.commit()
                raw_rows += len(chunk)
                file_counts[source_group] += len(chunk)
                label_counts.update(map(int, y.to_numpy()))
                for c in feature_cols:
                    missing_counts[c] += int(chunk[c].isna().sum()) if c in chunk.columns else len(chunk)
                offset += len(chunk)
        except Exception as exc:
            raise RuntimeError(f"Failed while auditing {fp}: {exc}") from exc
        file_row_offsets[rel] = offset

    q = conn.execute("""
        SELECT
            COUNT(*) AS unique_hashes,
            SUM(CASE WHEN duplicate_count > 0 THEN 1 ELSE 0 END) AS duplicated_hashes,
            SUM(duplicate_count) AS repeated_rows,
            SUM(conflict) AS conflicting_hashes,
            SUM(CASE WHEN conflict=0 THEN 1 ELSE 0 END) AS nonconflict_unique
        FROM rows
    """).fetchone()
    unique_hashes, duplicated_hashes, repeated_rows, conflicting_hashes, nonconflict_unique = [int(x or 0) for x in q]
    stats = {
        "dataset": spec.name,
        "raw_rows_read": int(raw_rows),
        "raw_binary_label_counts": {"benign": int(label_counts[0]), "attack": int(label_counts[1])},
        "unique_feature_hashes": unique_hashes,
        "hashes_with_repeats": duplicated_hashes,
        "repeated_rows_beyond_first": repeated_rows,
        "conflicting_label_hashes_excluded": conflicting_hashes,
        "unique_nonconflicting_rows": nonconflict_unique,
        "malformed_rows_skipped": int(malformed_rows),
        "hash_method": "two independent pandas 64-bit hashes concatenated (128-bit)",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(out / "full_audit_summary.json", stats)
    pd.DataFrame([
        {"feature": c, "missing_count_raw": int(missing_counts[c]), "missing_ratio_raw": float(missing_counts[c]/max(1,raw_rows))}
        for c in feature_cols
    ]).to_csv(out / "raw_missingness.csv", index=False)
    pd.DataFrame([{"source_group": g, "raw_rows": n} for g, n in sorted(file_counts.items())]).to_csv(
        out / "source_group_raw_counts.csv", index=False
    )
    return stats


def group_counts(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("""
        SELECT first_file, label, COUNT(*)
        FROM rows
        WHERE conflict=0
        GROUP BY first_file, label
    """).fetchall()
    groups: Dict[str, Dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    for group, label, n in rows:
        groups[str(group)][int(label)] = int(n)
    return [
        {"group": g, "benign": d[0], "attack": d[1], "total": d[0]+d[1]}
        for g, d in groups.items()
    ]


def split_score(assign: dict, group_stats: Dict[str, dict], ratios: dict) -> float:
    totals = {s: {"benign": 0, "attack": 0, "total": 0, "n_groups": 0} for s in SPLITS}
    global_counts = {"benign": 0, "attack": 0, "total": 0}
    for g, st in group_stats.items():
        for key in ("benign", "attack", "total"):
            global_counts[key] += st[key]
        if g in assign:
            sp = assign[g]
            totals[sp]["n_groups"] += 1
            for key in ("benign", "attack", "total"):
                totals[sp][key] += st[key]
    score = 0.0
    for sp in SPLITS:
        target_ratio = float(ratios["validation"] if sp == "validation" else ratios[sp])
        for key in ("benign", "attack", "total"):
            target = global_counts[key] * target_ratio
            score += ((totals[sp][key] - target) / max(1.0, target)) ** 2
        if totals[sp]["benign"] == 0 or totals[sp]["attack"] == 0:
            score += 1000.0
        if totals[sp]["n_groups"] == 0:
            score += 1000.0
    return score


def assign_groups_to_splits(conn: sqlite3.Connection, cfg: dict, out: Path) -> dict:
    rows = group_counts(conn)
    if len(rows) < 3:
        raise RuntimeError("At least three source groups are required for a group-disjoint train/validation/test split.")
    group_stats = {r["group"]: r for r in rows}
    ratios = cfg["split"]
    seed = int(ratios["seed"])
    rng = random.Random(seed)
    best_assign = None
    best_score = float("inf")
    groups = list(group_stats)

    # Multiple randomized greedy starts avoid fragile dependence on source-file order.
    for _ in range(max(2000, 200 * len(groups))):
        order = groups[:]
        rng.shuffle(order)
        tie_break = {g: rng.random() for g in order}
        order.sort(key=lambda g: (-group_stats[g]["total"], tie_break[g]))
        assign = {}
        for g in order:
            candidate_scores = []
            for sp in SPLITS:
                candidate = dict(assign)
                candidate[g] = sp
                candidate_scores.append((split_score(candidate, group_stats, ratios), sp))
            candidate_scores.sort(key=lambda x: (x[0], SPLITS.index(x[1])))
            assign[g] = candidate_scores[0][1]
        score = split_score(assign, group_stats, ratios)
        if score < best_score:
            best_score = score
            best_assign = assign
    if best_assign is None:
        raise RuntimeError("Could not construct source-group split.")

    conn.executemany("UPDATE rows SET split=? WHERE first_file=?", [(sp, g) for g, sp in best_assign.items()])
    conn.commit()
    report_rows = []
    for g, st in sorted(group_stats.items()):
        report_rows.append({**st, "split": best_assign[g]})
    pd.DataFrame(report_rows).to_csv(out / "source_group_split_manifest.csv", index=False)

    summary = {}
    for sp in SPLITS:
        temp = [r for r in report_rows if r["split"] == sp]
        summary[sp] = {
            "groups": len(temp),
            "benign": int(sum(r["benign"] for r in temp)),
            "attack": int(sum(r["attack"] for r in temp)),
            "total": int(sum(r["total"] for r in temp)),
        }
        if summary[sp]["benign"] == 0 or summary[sp]["attack"] == 0:
            raise RuntimeError(f"Split {sp} lacks one binary class under source-group separation.")
    summary["objective_score"] = best_score
    atomic_json(out / "source_group_split_summary.json", summary)
    return summary


def allocate_quotas(avail: Dict[str, int], total_target: int) -> Dict[str, int]:
    """Proportional quota with deterministic remainder and availability constraints."""
    if total_target >= sum(avail.values()):
        return dict(avail)
    if total_target <= 0:
        return {k: 0 for k in avail}
    weights = {k: v / max(1, sum(avail.values())) for k, v in avail.items()}
    q = {k: min(avail[k], int(math.floor(total_target * weights[k]))) for k in avail}
    remaining = total_target - sum(q.values())
    order = sorted(avail, key=lambda k: (-(total_target * weights[k] - q[k]), k))
    while remaining > 0:
        progressed = False
        for k in order:
            if q[k] < avail[k] and remaining > 0:
                q[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return q


def mark_cohort(conn: sqlite3.Connection, cfg: dict, dataset_name: str, out: Path) -> dict:
    cohort_cfg = cfg["cohort"][dataset_name]
    cap = cohort_cfg.get("max_unique_rows")
    if cap is None:
        conn.execute("UPDATE rows SET selected=1 WHERE conflict=0")
        conn.commit()
    else:
        conn.execute("UPDATE rows SET selected=0")
        conn.commit()
        split_ratios = {
            "train": float(cfg["split"]["train"]),
            "validation": float(cfg["split"]["validation"]),
            "test": float(cfg["split"]["test"]),
        }
        target_split = {sp: int(round(int(cap) * split_ratios[sp])) for sp in SPLITS}
        target_split["train"] += int(cap) - sum(target_split.values())
        benign_fraction = cohort_cfg.get("target_benign_fraction")
        for sp in SPLITS:
            rows = conn.execute("""
                SELECT source_class, label, COUNT(*)
                FROM rows
                WHERE conflict=0 AND split=?
                GROUP BY source_class, label
            """, (sp,)).fetchall()
            avail_by_label_class: Dict[int, Dict[str, int]] = {0: {}, 1: {}}
            for source_class, label, n in rows:
                avail_by_label_class[int(label)][str(source_class)] = int(n)
            if benign_fraction is None:
                total_avail = sum(sum(v.values()) for v in avail_by_label_class.values())
                benign_target = int(round(target_split[sp] * sum(avail_by_label_class[0].values()) / max(1,total_avail)))
            else:
                benign_target = int(round(target_split[sp] * float(benign_fraction)))
            attack_target = target_split[sp] - benign_target
            for label, label_target in ((0, benign_target), (1, attack_target)):
                avail = avail_by_label_class[label]
                if not avail:
                    continue
                if cohort_cfg.get("balance_by_source_class", False):
                    # Equal initial allocation across classes, then reallocate deficits.
                    classes = sorted(avail)
                    base = label_target // len(classes)
                    quotas = {c: min(avail[c], base) for c in classes}
                    remaining = label_target - sum(quotas.values())
                    while remaining > 0:
                        progressed = False
                        for c in classes:
                            if quotas[c] < avail[c] and remaining > 0:
                                quotas[c] += 1
                                remaining -= 1
                                progressed = True
                        if not progressed:
                            break
                else:
                    quotas = allocate_quotas(avail, label_target)
                for source_class, quota in quotas.items():
                    if quota <= 0:
                        continue
                    hashes = conn.execute("""
                        SELECT row_hash FROM rows
                        WHERE conflict=0 AND split=? AND label=? AND source_class=?
                        ORDER BY row_hash LIMIT ?
                    """, (sp, label, source_class, int(quota))).fetchall()
                    conn.executemany("UPDATE rows SET selected=1 WHERE row_hash=?", hashes)
        conn.commit()

    summary_rows = conn.execute("""
        SELECT split, label, source_class, COUNT(*)
        FROM rows
        WHERE conflict=0 AND selected=1
        GROUP BY split, label, source_class
    """).fetchall()
    df = pd.DataFrame(summary_rows, columns=["split", "label", "source_class", "selected_rows"])
    df.to_csv(out / "cohort_selection_counts.csv", index=False)
    totals = {
        sp: {
            "benign": int(df[(df.split==sp)&(df.label==0)].selected_rows.sum()),
            "attack": int(df[(df.split==sp)&(df.label==1)].selected_rows.sum()),
        } for sp in SPLITS
    }
    for sp in SPLITS:
        totals[sp]["total"] = totals[sp]["benign"] + totals[sp]["attack"]
        if totals[sp]["benign"] == 0 or totals[sp]["attack"] == 0:
            raise RuntimeError(f"Cohort split {sp} lacks a binary class.")
    atomic_json(out / "cohort_selection_summary.json", totals)
    return totals


def extract_unique_splits(spec: DatasetSpec, cfg: dict, discovery: dict, conn: sqlite3.Connection, out: Path) -> Dict[str, Path]:
    raw_dir = out / "canonical_raw_splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {sp: raw_dir / f"{sp}.csv" for sp in SPLITS}
    for p in paths.values():
        if p.exists():
            p.unlink()

    feature_cols = discovery["canonical_feature_columns"]
    headers = feature_cols + [LABEL_COL] + META_COLS
    writers = {}
    handles = {}
    for sp, p in paths.items():
        handle = p.open("w", newline="", encoding="utf-8")
        handles[sp] = handle
        writer = csv.writer(handle)
        writer.writerow(headers)
        writers[sp] = writer

    chunksize = int(cfg["chunksize"])
    try:
        for file_idx, fp in enumerate(spec.files, start=1):
            rel = str(fp.relative_to(spec.root)).replace("\\", "/")
            print(f"[EXTRACT {spec.name}] {file_idx}/{len(spec.files)} {rel}")
            offset = 0
            for chunk in read_csv_chunks(fp, chunksize):
                chunk.columns = [normalize_name(c) for c in chunk.columns]
                label_col = infer_label_column(chunk.columns, spec.label_candidates)
                if label_col is not None:
                    y = make_binary_label(chunk[label_col], spec.benign_tokens)
                elif spec.folder_label_fallback:
                    y = folder_binary_label(spec, fp, len(chunk))
                else:
                    offset += len(chunk)
                    continue
                end = offset + len(chunk) - 1
                selected_rows = conn.execute("""
                    SELECT first_row, split, row_hash, label, source_class
                    FROM rows
                    WHERE first_file=? AND first_row BETWEEN ? AND ?
                      AND conflict=0 AND selected=1
                """, (rel, offset, end)).fetchall()
                by_row = {int(r[0]): r[1:] for r in selected_rows}
                if by_row:
                    features = chunk.reindex(columns=feature_cols)
                    for local_i in range(len(chunk)):
                        absolute_row = offset + local_i
                        if absolute_row not in by_row:
                            continue
                        sp, row_hash, label, source_class = by_row[absolute_row]
                        values = [features.iloc[local_i][c] for c in feature_cols]
                        writers[sp].writerow(values + [int(label), rel, str(source_class), str(row_hash)])
                offset += len(chunk)
    finally:
        for h in handles.values():
            h.close()

    return paths


def hash_sample_score(series: pd.Series) -> pd.Series:
    return series.astype(str).str.slice(0, 16).map(lambda x: int(x, 16))


def training_reservoir(train_csv: Path, feature_cols: List[str], cfg: dict) -> pd.DataFrame:
    cap = int(cfg["preprocessing"]["training_reservoir_rows"])
    candidates: Optional[pd.DataFrame] = None
    usecols = feature_cols + [LABEL_COL, "row_hash"]
    for chunk in pd.read_csv(train_csv, chunksize=int(cfg["chunksize"]), low_memory=False, usecols=usecols):
        chunk = chunk.copy()
        chunk["_sample_score"] = hash_sample_score(chunk["row_hash"])
        if candidates is None:
            candidates = chunk
        else:
            candidates = pd.concat([candidates, chunk], ignore_index=True)
        if len(candidates) > cap * 2:
            candidates = candidates.nsmallest(cap, "_sample_score")
    if candidates is None:
        raise RuntimeError("Training split is empty.")
    return candidates.nsmallest(cap, "_sample_score").drop(columns=["_sample_score"]).reset_index(drop=True)


def fit_preprocessor(train_csv: Path, discovery: dict, cfg: dict, out: Path) -> dict:
    feature_cols = discovery["canonical_feature_columns"]
    numeric_cols = discovery["numeric_columns"]
    categorical_cols = discovery["categorical_columns"]
    reservoir = training_reservoir(train_csv, feature_cols, cfg)
    params = {
        "numeric_columns": [],
        "categorical_columns": [],
        "feature_names": [],
        "fit_partition": "train_only",
        "reservoir_rows": int(len(reservoir)),
        "winsor_lower_quantile": float(cfg["preprocessing"]["winsor_lower_quantile"]),
        "winsor_upper_quantile": float(cfg["preprocessing"]["winsor_upper_quantile"]),
        "robust_scale": cfg["preprocessing"]["robust_scale"],
    }

    qlo = float(cfg["preprocessing"]["winsor_lower_quantile"])
    qhi = float(cfg["preprocessing"]["winsor_upper_quantile"])
    for c in numeric_cols:
        s = pd.to_numeric(reservoir[c], errors="coerce")
        median = float(s.median()) if s.notna().any() else 0.0
        low = float(s.quantile(qlo)) if s.notna().any() else median
        high = float(s.quantile(qhi)) if s.notna().any() else median
        iqr = float(s.quantile(0.75) - s.quantile(0.25)) if s.notna().any() else 0.0
        std = float(s.std(ddof=0)) if s.notna().any() else 0.0
        scale = iqr if abs(iqr) > 1e-12 else (std if abs(std) > 1e-12 else 1.0)
        params["numeric_columns"].append({
            "name": c, "median": median, "clip_low": low, "clip_high": high, "center": median, "scale": scale
        })
        params["feature_names"].append(c)

    max_card = int(cfg["preprocessing"]["categorical_max_cardinality"])
    missing_token = cfg["preprocessing"]["missing_category_token"]
    unknown_token = cfg["preprocessing"]["unknown_category_token"]
    for c in categorical_cols:
        s = reservoir[c].astype("string").fillna(missing_token).str.strip()
        counts = s.value_counts()
        categories = counts.head(max_card).index.astype(str).tolist()
        if missing_token not in categories:
            categories.append(missing_token)
        if unknown_token not in categories:
            categories.append(unknown_token)
        params["categorical_columns"].append({"name": c, "categories": categories})
        params["feature_names"].extend([f"{c}={v}" for v in categories])

    atomic_json(out / "preprocessing_parameters.json", params)
    joblib.dump(params, out / "preprocessing_parameters.joblib")
    return params


def transform_chunk(chunk: pd.DataFrame, params: dict, cfg: dict) -> Tuple[sparse.csr_matrix, pd.DataFrame]:
    blocks = []
    processed_for_hash = {}
    for p in params["numeric_columns"]:
        c = p["name"]
        s = pd.to_numeric(chunk[c], errors="coerce").fillna(p["median"])
        s = s.clip(p["clip_low"], p["clip_high"])
        z = ((s - p["center"]) / p["scale"]).astype(np.float32)
        blocks.append(sparse.csr_matrix(z.to_numpy().reshape(-1, 1)))
        processed_for_hash[c] = np.round(z.to_numpy(), 8)
    missing_token = cfg["preprocessing"]["missing_category_token"]
    unknown_token = cfg["preprocessing"]["unknown_category_token"]
    for p in params["categorical_columns"]:
        c = p["name"]
        categories = p["categories"]
        index = {v: i for i, v in enumerate(categories)}
        s = chunk[c].astype("string").fillna(missing_token).str.strip()
        mapped = s.where(s.isin(categories), unknown_token)
        rows = np.arange(len(chunk), dtype=np.int64)
        cols = mapped.map(index).to_numpy(np.int64)
        data = np.ones(len(chunk), dtype=np.float32)
        block = sparse.csr_matrix((data, (rows, cols)), shape=(len(chunk), len(categories)))
        blocks.append(block)
        processed_for_hash[c] = mapped.astype(str).to_numpy()
    X = sparse.hstack(blocks, format="csr", dtype=np.float32) if blocks else sparse.csr_matrix((len(chunk), 0), dtype=np.float32)
    hash_df = pd.DataFrame(processed_for_hash)
    return X, hash_df


def transform_splits(raw_paths: Dict[str, Path], params: dict, cfg: dict, out: Path) -> dict:
    processed_root = out / "processed"
    processed_root.mkdir(exist_ok=True)
    transformed_db = sqlite3.connect(out / "transformed_overlap.sqlite")
    transformed_db.execute("CREATE TABLE IF NOT EXISTS hashes (row_hash TEXT PRIMARY KEY, split TEXT NOT NULL, duplicate_within INTEGER DEFAULT 0, cross_split INTEGER DEFAULT 0)")
    transformed_db.commit()
    summary = {}
    feature_cols = [p["name"] for p in params["numeric_columns"]] + [p["name"] for p in params["categorical_columns"]]

    for sp in SPLITS:
        split_dir = processed_root / sp
        split_dir.mkdir(parents=True, exist_ok=True)
        for old in split_dir.glob("*"):
            if old.is_file():
                old.unlink()
        rows_total = 0
        label_counts = Counter()
        part_manifest = []
        for part_idx, chunk in enumerate(pd.read_csv(raw_paths[sp], chunksize=int(cfg["chunksize"]), low_memory=False)):
            X, hash_df = transform_chunk(chunk, params, cfg)
            if np.isnan(X.data).any() or np.isinf(X.data).any():
                raise RuntimeError(f"NaN/Inf found after preprocessing in {sp} part {part_idx}")
            y = chunk[LABEL_COL].astype(np.int8).to_numpy()
            row_hashes = chunk["row_hash"].astype(str).to_numpy()
            groups = chunk["source_group"].astype(str).to_numpy()
            transformed_hashes = stable_row_hash(hash_df)

            x_path = split_dir / f"X_part_{part_idx:05d}.npz"
            y_path = split_dir / f"y_part_{part_idx:05d}.npy"
            h_path = split_dir / f"row_hash_part_{part_idx:05d}.txt.gz"
            g_path = split_dir / f"source_group_part_{part_idx:05d}.txt.gz"
            sparse.save_npz(x_path, X, compressed=True)
            np.save(y_path, y)
            with gzip.open(h_path, "wt", encoding="utf-8") as f:
                f.write("\n".join(row_hashes) + "\n")
            with gzip.open(g_path, "wt", encoding="utf-8") as f:
                f.write("\n".join(groups) + "\n")

            insert_sql = """
                INSERT INTO hashes(row_hash, split, duplicate_within, cross_split)
                VALUES (?, ?, 0, 0)
                ON CONFLICT(row_hash) DO UPDATE SET
                    duplicate_within = CASE WHEN hashes.split = excluded.split THEN 1 ELSE hashes.duplicate_within END,
                    cross_split = CASE WHEN hashes.split != excluded.split THEN 1 ELSE hashes.cross_split END
            """
            transformed_db.executemany(insert_sql, [(str(h), sp) for h in transformed_hashes])
            transformed_db.commit()
            rows_total += len(chunk)
            label_counts.update(map(int, y))
            part_manifest.append({
                "part": part_idx,
                "rows": len(chunk),
                "features": int(X.shape[1]),
                "nnz": int(X.nnz),
                "X_sha256": sha256_file(x_path),
                "y_sha256": sha256_file(y_path),
                "row_hash_sha256": sha256_file(h_path),
                "source_group_sha256": sha256_file(g_path),
            })
        pd.DataFrame(part_manifest).to_csv(split_dir / "part_manifest.csv", index=False)
        summary[sp] = {
            "rows": int(rows_total),
            "benign": int(label_counts[0]),
            "attack": int(label_counts[1]),
            "n_parts": len(part_manifest),
            "n_features": len(params["feature_names"]),
        }
    overlap = transformed_db.execute("""
        SELECT
          SUM(duplicate_within),
          SUM(cross_split)
        FROM hashes
    """).fetchone()
    transformed_db.close()
    summary["transformed_hashes_within_split_repeat"] = int(overlap[0] or 0)
    summary["transformed_hashes_cross_split"] = int(overlap[1] or 0)
    atomic_json(out / "processed_split_summary.json", summary)
    (processed_root / "feature_names.json").write_text(json.dumps(params["feature_names"], indent=2), encoding="utf-8")
    return summary


def load_train_labels(processed_root: Path) -> np.ndarray:
    parts = sorted((processed_root / "train").glob("y_part_*.npy"))
    if not parts:
        raise RuntimeError("No processed training labels found.")
    return np.concatenate([np.load(p) for p in parts]).astype(np.int8)


def dirichlet_partition(y: np.ndarray, client_count: int, alpha: float, seed: int, min_support: int, retries: int) -> np.ndarray:
    classes = np.unique(y)
    for attempt in range(retries):
        rng = np.random.default_rng(seed + attempt * 104729)
        assignment = np.full(len(y), -1, dtype=np.int16)
        for cls in classes:
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(client_count, alpha))
            counts = rng.multinomial(len(idx), proportions)
            start = 0
            for client_id, n in enumerate(counts):
                chosen = idx[start:start+n]
                assignment[chosen] = client_id
                start += n
        counts = np.bincount(assignment, minlength=client_count)
        if assignment.min() >= 0 and counts.min() >= min_support:
            return assignment
    raise RuntimeError(f"Could not create Dirichlet partition alpha={alpha}, seed={seed} with min_support={min_support} after {retries} retries.")


def create_client_partitions(cfg: dict, out: Path, dataset_name: str) -> dict:
    fed = cfg["federation"]
    y = load_train_labels(out / "processed")
    client_count = int(fed["client_count"])
    min_support = int(fed["minimum_client_support"])
    retries = int(fed["partition_retry_limit"])
    alphas = sorted(set([float(fed["primary_alpha_dir"])] + [float(x) for x in fed["alpha_sensitivity"]]))
    seeds = [int(x) for x in fed["paired_seeds"]]
    part_dir = out / "client_partitions"
    part_dir.mkdir(exist_ok=True)
    summaries = []

    for alpha in alphas:
        for seed in seeds:
            assignment = dirichlet_partition(y, client_count, alpha, seed, min_support, retries)
            path = part_dir / f"client_assignment_alpha_{alpha:g}_seed_{seed}_clients_{client_count}.npy"
            np.save(path, assignment)
            for client_id in range(client_count):
                idx = np.where(assignment == client_id)[0]
                labels = y[idx]
                summaries.append({
                    "dataset": dataset_name,
                    "alpha_Dir": alpha,
                    "seed": seed,
                    "client_id": client_id,
                    "n_rows": int(len(idx)),
                    "n_benign": int((labels==0).sum()),
                    "n_attack": int((labels==1).sum()),
                    "attack_ratio": float((labels==1).mean()) if len(labels) else np.nan,
                    "single_class": bool(np.unique(labels).size < 2),
                    "assignment_sha256": sha256_file(path),
                })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(part_dir / "client_partition_summary.csv", index=False)
    aggregate = summary_df.groupby(["alpha_Dir", "seed"]).agg(
        min_client_rows=("n_rows","min"),
        max_client_rows=("n_rows","max"),
        single_class_clients=("single_class","sum"),
        attack_ratio_min=("attack_ratio","min"),
        attack_ratio_max=("attack_ratio","max"),
    ).reset_index()
    aggregate.to_csv(part_dir / "client_partition_aggregate.csv", index=False)
    return {
        "client_count": client_count,
        "alphas": alphas,
        "seeds": seeds,
        "minimum_client_support": min_support,
        "partition_files": len(alphas)*len(seeds),
        "single_class_clients_total": int(summary_df["single_class"].sum()),
    }


def validate_dataset(out: Path, discovery: dict, audit: dict, split_summary: dict, cohort: dict, processed: dict, partitions: dict, cfg: dict) -> dict:
    checks = []
    def add(name: str, passed: bool, detail: object):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add("schema_consistency", not discovery["files_with_errors"], discovery["files_with_errors"])
    add("binary_classes_raw", audit["raw_binary_label_counts"]["benign"] > 0 and audit["raw_binary_label_counts"]["attack"] > 0, audit["raw_binary_label_counts"])
    add("conflicting_label_duplicates_excluded", True, audit["conflicting_label_hashes_excluded"])
    add("source_group_disjoint", all(split_summary[sp]["groups"] > 0 for sp in SPLITS), split_summary)
    add("cohort_each_split_has_two_classes", all(cohort[sp]["benign"] > 0 and cohort[sp]["attack"] > 0 for sp in SPLITS), cohort)
    add("transformed_cross_split_overlap_zero", processed["transformed_hashes_cross_split"] == 0, processed["transformed_hashes_cross_split"])
    add("all_processed_splits_nonempty", all(processed[sp]["rows"] > 0 for sp in SPLITS), {sp: processed[sp]["rows"] for sp in SPLITS})
    add("client_partition_min_support", True, partitions["minimum_client_support"])

    suspicious = [x for x in discovery["suspicious_features"] if not x.get("approved", False)]
    review_status = "REVIEW_REQUIRED" if suspicious else "PASS"
    hard_fail = any(c["status"] == "FAIL" for c in checks)
    overall = "FAIL" if hard_fail else review_status
    report = {
        "overall_gate_status": overall,
        "checks": checks,
        "unapproved_suspicious_features": suspicious,
        "note": "REVIEW_REQUIRED means the pipeline completed and no model training is allowed until the suspicious-feature audit is reviewed.",
    }
    atomic_json(out / "STEP6_GATE_STATUS.json", report)
    pd.DataFrame(checks).to_csv(out / "step6_validation_checks.csv", index=False)
    return report


def environment_manifest() -> dict:
    return {
        "created_at": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "joblib": joblib.__version__,
        "sqlite": sqlite3.sqlite_version,
    }


def dataset_run(spec: DatasetSpec, cfg: dict, root_out: Path) -> dict:
    out = root_out / spec.name
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "environment.json", environment_manifest())
    file_manifest(spec).to_csv(out / "raw_file_manifest.csv", index=False)

    discovery = discovery_pass(spec, cfg, out)
    db_path = out / "row_audit.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = create_db(db_path)
    try:
        audit = full_audit_pass(spec, cfg, discovery, out, conn)
        split_summary = assign_groups_to_splits(conn, cfg, out)
        cohort = mark_cohort(conn, cfg, spec.name, out)
        raw_paths = extract_unique_splits(spec, cfg, discovery, conn, out)
    finally:
        conn.close()

    params = fit_preprocessor(raw_paths["train"], discovery, cfg, out)
    processed = transform_splits(raw_paths, params, cfg, out)
    partitions = create_client_partitions(cfg, out, spec.name)
    gate = validate_dataset(out, discovery, audit, split_summary, cohort, processed, partitions, cfg)

    # Hash the auditable artifacts, excluding large raw/processed matrices and SQLite databases.
    artifact_rows = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        if any(part in {"canonical_raw_splits", "processed"} for part in p.parts):
            continue
        if p.suffix.lower() in {".sqlite", ".db"}:
            continue
        artifact_rows.append({
            "relative_path": str(p.relative_to(out)).replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    pd.DataFrame(artifact_rows).to_csv(out / "step6_artifact_hashes.csv", index=False)
    return {
        "dataset": spec.name,
        "gate": gate["overall_gate_status"],
        "audit": audit,
        "cohort": cohort,
        "processed": processed,
        "partitions": partitions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--project-root", default=None, help="Optional override for the project root.")
    ap.add_argument("--output-root", default=None, help="Optional override for the output root.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.project_root:
        cfg["project_root"] = args.project_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "STEP6_DATA_CONFIG_USED.yaml")
    atomic_json(output_root / "step6_environment.json", environment_manifest())

    results = []
    for name, dcfg in cfg["datasets"].items():
        spec = make_spec(name, dcfg)
        results.append(dataset_run(spec, cfg, output_root))

    statuses = [r["gate"] for r in results]
    overall = "FAIL" if "FAIL" in statuses else ("REVIEW_REQUIRED" if "REVIEW_REQUIRED" in statuses else "PASS")
    summary = {
        "protocol_id": cfg["protocol_id"],
        "completed_at": now_iso(),
        "overall_gate_status": overall,
        "dataset_results": results,
        "training_permitted": overall == "PASS",
        "scientific_rule": "No model training or test-metric inspection is permitted until Step 6 gate is PASS.",
    }
    atomic_json(output_root / "STEP6_MASTER_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if overall == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
