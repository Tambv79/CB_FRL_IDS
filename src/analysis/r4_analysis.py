from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy.stats import pearsonr, spearmanr

from r4_common import atomic_json, binary_metrics, load_yaml, stable_seed

ROOT = Path(__file__).resolve().parents[1]
R3 = load_yaml(ROOT / "locks/R3_PROTOCOL_LOCK/config/R3_PROTOCOL_LOCK.yaml")
SEEDS = list(map(int, R3["primary_design"]["paired_seeds"]))
BETAS = list(map(float, R3["primary_design"]["budget_ratios_beta"]))
PRIMARY_MANIFEST = pd.read_csv(ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv")


def _beta_key(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return format(float(value), ".12g")


def _build_primary_index():
    index = {}
    for rec in PRIMARY_MANIFEST.to_dict("records"):
        key = (str(rec["dataset_protocol"]), str(rec["method"]), int(rec["seed"]),
               None if rec["method"] == "Full-Contextual" else _beta_key(rec["beta"]))
        if key in index:
            raise RuntimeError(f"Duplicate primary manifest key: {key}")
        index[key] = str(rec["cell_id"])
    return index


PRIMARY_INDEX = _build_primary_index()


def cell_id_for(dataset: str, method: str, seed: int, beta: float | None) -> str:
    key = (str(dataset), str(method), int(seed),
           None if method == "Full-Contextual" else _beta_key(beta))
    try:
        return PRIMARY_INDEX[key]
    except KeyError as exc:
        raise KeyError(f"No authoritative primary cell for key={key}") from exc


def cell_dir(out_root: Path, dataset: str, method: str, seed: int, beta: float | None) -> Path:
    # Never reconstruct cell IDs from float formatting. The locked manifest is authoritative.
    return out_root / "cells" / "primary" / cell_id_for(dataset, method, seed, beta)


def load_metric(out_root, dataset, method, seed, beta, analysis="method_specific"):
    return json.loads((cell_dir(out_root, dataset, method, seed, beta) / "test_metrics.json").read_text())[analysis]


def bootstrap_bounds(x, resamples, seed=1911):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(int(resamples), len(x)))
    means = x[draws].mean(axis=1)
    return {
        "two_sided_low": float(np.quantile(means, .025)),
        "two_sided_high": float(np.quantile(means, .975)),
        "one_sided_lower_95": float(np.quantile(means, .05)),
        "one_sided_upper_95": float(np.quantile(means, .95)),
    }


def bootstrap_ci(x, resamples, seed=1911):
    b = bootstrap_bounds(x, resamples, seed)
    return b["two_sided_low"], b["two_sided_high"]


def exact_sign_flip(z):
    z = np.asarray(z, float)
    observed = float(z.mean())
    count = 0
    total = 2 ** len(z)
    for signs in itertools.product((-1.0, 1.0), repeat=len(z)):
        if float(np.mean(z * np.asarray(signs))) >= observed - 1e-15:
            count += 1
    return count / total


def holm(pvals):
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def apply_shared_full_thresholds(out_root: Path) -> pd.DataFrame:
    rows = []
    primary = pd.read_csv(ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv")
    for dataset in primary.dataset_protocol.unique():
        for seed in SEEDS:
            full = cell_dir(out_root, dataset, "Full-Contextual", seed, None)
            threshold = float(json.loads((full / "THRESHOLD_LOCK.json").read_text())["threshold"])
            for rec in primary[(primary.dataset_protocol == dataset) & (primary.seed == seed)].to_dict("records"):
                d = out_root / "cells" / "primary" / rec["cell_id"]
                pred = np.load(d / "test_predictions.npz")
                metric = binary_metrics(pred["y_true"], pred["probabilities"], threshold)
                tm = json.loads((d / "test_metrics.json").read_text())
                tm["shared_full_threshold"] = metric
                atomic_json(d / "test_metrics.json", tm)
                rows.append({"cell_id": rec["cell_id"], "dataset": dataset, "seed": seed, "method": rec["method"], "beta": rec["beta"], "full_threshold": threshold, **metric})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_root / "analysis/shared_threshold_results.csv", index=False)
    return frame


def threshold_and_curve_outputs(out_root: Path) -> None:
    primary = pd.read_csv(ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv")
    threshold_rows, curve_rows = [], []
    for rec in primary.to_dict("records"):
        d = out_root / "cells" / "primary" / rec["cell_id"]
        lock = json.loads((d / "THRESHOLD_LOCK.json").read_text())
        threshold_rows.append({"cell_id": rec["cell_id"], "dataset": rec["dataset_protocol"], "method": rec["method"], "beta": rec["beta"], "seed": rec["seed"], "threshold": lock["threshold"], "selection_rule": lock["selection_rule"], "validation_F1": lock["f1"], "validation_PR_AUC": lock["pr_auc"], "validation_FPR": lock["fpr"]})
        pred = np.load(d / "test_predictions.npz")
        y, p = pred["y_true"], pred["probabilities"]
        precision, recall, pr_t = precision_recall_curve(y, p)
        fpr, tpr, roc_t = roc_curve(y, p)
        for kind, x, yy, tt in [("PR", recall, precision, np.r_[pr_t, np.nan]), ("ROC", fpr, tpr, roc_t)]:
            # Fixed-size deterministic subsampling keeps the artifact compact while preserving endpoints.
            idx = np.unique(np.linspace(0, len(x)-1, min(201, len(x)), dtype=int))
            for i in idx:
                curve_rows.append({"cell_id": rec["cell_id"], "dataset": rec["dataset_protocol"], "method": rec["method"], "beta": rec["beta"], "seed": rec["seed"], "curve": kind, "x": float(x[i]), "y": float(yy[i]), "threshold": None if not np.isfinite(tt[i]) else float(tt[i])})
    pd.DataFrame(threshold_rows).to_csv(out_root / "analysis/R4_THRESHOLD_DISTRIBUTION.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_root / "analysis/R4_PR_ROC_CURVES.csv", index=False)


def communication_analyses(out_root: Path) -> None:
    primary = pd.read_csv(ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv")
    same_cumulative, targets = [], []
    methods = list(primary.method.unique())
    for dataset in primary.dataset_protocol.unique():
        for seed in SEEDS:
            full_d = cell_dir(out_root, dataset, "Full-Contextual", seed, None)
            full_round = pd.read_csv(full_d / "round_metrics.csv")
            final_full_f1 = float(full_round.iloc[-1].validation_F1)
            final_full_pr = float(full_round.iloc[-1].validation_PR_AUC)
            for beta in BETAS:
                curves = {}
                for method in methods:
                    if method == "Full-Contextual":
                        d = full_d
                    else:
                        d = cell_dir(out_root, dataset, method, seed, beta)
                    rm = pd.read_csv(d / "round_metrics.csv").copy()
                    rm["cumulative_E2E_broadcast"] = rm.E2E_broadcast.cumsum()
                    curves[method] = rm
                    cell_targets = {}
                    for endpoint, full_value in [("F1", final_full_f1), ("PR_AUC", final_full_pr)]:
                        for fraction in (.90, .95):
                            target = fraction * full_value
                            field = "validation_F1" if endpoint == "F1" else "validation_PR_AUC"
                            hit = rm[rm[field] >= target]
                            reached = not hit.empty
                            byte_value = int(hit.iloc[0].cumulative_E2E_broadcast) if reached else None
                            round_value = int(hit.iloc[0]["round"]) if reached else None
                            targets.append({"dataset": dataset, "seed": seed, "beta": beta, "method": method, "endpoint": endpoint, "fraction_of_full_final": fraction, "target_value": target, "reached": reached, "round": round_value, "E2E_broadcast_bytes": byte_value})
                            cell_targets[f"{endpoint}_{int(100*fraction)}pct"] = {"target": target, "reached": reached, "round": round_value, "E2E_broadcast_bytes": byte_value}
                    summary = json.loads((d / "summary.json").read_text())
                    # Full is reused across beta values; preserve all beta-specific target evaluations.
                    existing = summary.get("communication_to_target")
                    if not isinstance(existing, dict): existing = {}
                    existing[f"beta_{beta:g}"] = cell_targets
                    summary["communication_to_target"] = existing
                    atomic_json(d / "summary.json", summary)
                overlap_low = max(float(x.cumulative_E2E_broadcast.min()) for x in curves.values())
                overlap_high = min(float(x.cumulative_E2E_broadcast.max()) for x in curves.values())
                if overlap_high < overlap_low:
                    continue
                grid = np.linspace(overlap_low, overlap_high, 20)
                for method, rm in curves.items():
                    xx = rm.cumulative_E2E_broadcast.to_numpy(float)
                    for b in grid:
                        same_cumulative.append({"dataset": dataset, "seed": seed, "beta": beta, "method": method, "cumulative_E2E_broadcast_bytes": float(b), "validation_F1": float(np.interp(b, xx, rm.validation_F1)), "validation_PR_AUC": float(np.interp(b, xx, rm.validation_PR_AUC)), "validation_FPR": float(np.interp(b, xx, rm.validation_FPR))})
    pd.DataFrame(same_cumulative).to_csv(out_root / "analysis/R4_SAME_CUMULATIVE_COMMUNICATION.csv", index=False)
    pd.DataFrame(targets).to_csv(out_root / "analysis/R4_COMMUNICATION_TO_TARGET.csv", index=False)


def _run_hypotheses(out_root: Path, analysis_name: str) -> pd.DataFrame:
    hypotheses = pd.read_csv(ROOT / "manifests/R3_HYPOTHESIS_MANIFEST.csv")
    B = int(R3["inference"]["paired_bootstrap_resamples"])
    results = []
    for row in hypotheses.to_dict("records"):
        dataset, beta, endpoint = row["dataset"], float(row["beta"]), row["endpoint"]
        rhs = row["comparison"].split("minus", 1)[1].strip()
        key = {"F1": "f1", "PR-AUC": "pr_auc", "FPR": "fpr"}[endpoint]
        deltas = np.asarray([
            load_metric(out_root, dataset, "CB-Score", seed, beta, analysis_name)[key]
            - load_metric(out_root, dataset, rhs, seed, None if rhs == "Full-Contextual" else beta, analysis_name)[key]
            for seed in SEEDS
        ])
        if endpoint == "F1" and "NI_" in row["hypothesis_id"]:
            margin = float(R3["inference"]["f1_noninferiority_margin"]); transformed = deltas + margin
        elif endpoint == "PR-AUC" and "NI_" in row["hypothesis_id"]:
            margin = float(R3["inference"]["pr_auc_noninferiority_margin"]); transformed = deltas + margin
        elif endpoint == "FPR":
            margin = float(R3["inference"]["fpr_safety_margin_absolute"]); transformed = margin - deltas
        else:
            margin = 0.0; transformed = deltas
        bounds = bootstrap_bounds(deltas, B, stable_seed("hypothesis", analysis_name, row["hypothesis_id"]))
        raw = exact_sign_flip(transformed)
        sd = deltas.std(ddof=1)
        results.append({**row, "threshold_analysis": analysis_name, "mean_delta": float(deltas.mean()),
                        "ci_low": bounds["two_sided_low"], "ci_high": bounds["two_sided_high"],
                        "one_sided_lower_95": bounds["one_sided_lower_95"], "one_sided_upper_95": bounds["one_sided_upper_95"],
                        "raw_p": raw, "margin": margin, "paired_dz": float(deltas.mean()/sd) if sd > 0 else float("nan"),
                        "wins": int((deltas > 1e-12).sum()), "losses": int((deltas < -1e-12).sum()),
                        "ties": int((np.abs(deltas) <= 1e-12).sum())})
    out = pd.DataFrame(results)
    out["holm_adjusted_p"] = np.nan
    for _, idx in out.groupby("family").groups.items():
        out.loc[idx, "holm_adjusted_p"] = holm(out.loc[idx, "raw_p"].to_numpy())
    decisions=[]
    for rec in out.to_dict("records"):
        if rec["endpoint"] == "FPR":
            ci_gate = rec["one_sided_upper_95"] < rec["margin"]
        elif "NI_" in rec["hypothesis_id"]:
            ci_gate = rec["one_sided_lower_95"] > -rec["margin"]
        else:
            ci_gate = rec["one_sided_lower_95"] > 0.0
        decisions.append("MET_GATE_WITHIN_EVALUATED_SEEDS" if ci_gate and rec["holm_adjusted_p"] <= .05 else "INCONCLUSIVE")
    out["decision"] = decisions
    return out


def hypothesis_analysis(out_root: Path) -> None:
    primary = _run_hypotheses(out_root, "method_specific")
    primary.to_csv(out_root / "analysis/R4_LOCKED_HYPOTHESIS_RESULTS.csv", index=False)
    sensitivity = pd.concat([
        primary,
        _run_hypotheses(out_root, "shared_full_threshold"),
        _run_hypotheses(out_root, "fixed_0.5"),
    ], ignore_index=True)
    sensitivity.to_csv(out_root / "analysis/R4_THRESHOLD_SENSITIVITY_HYPOTHESIS_RESULTS.csv", index=False)

def aggregate_all(out_root: Path) -> None:
    rows = []
    for phase in ["primary", "heterogeneity", "natural_partition", "ood_stress", "feature_sensitivity", "robustness"]:
        phase_root = out_root / "cells" / phase
        if not phase_root.exists():
            continue
        for d in phase_root.iterdir():
            if not (d / "CELL_COMPLETE.json").exists():
                continue
            summary = json.loads((d / "summary.json").read_text())
            lock = json.loads((d / "configuration_lock.json").read_text())
            rows.append({"phase": phase, "cell_id": d.name, **{k: lock.get(k) for k in ["dataset", "dataset_protocol", "method", "beta", "alpha_dir", "seed", "aggregation", "model_attack", "reported_utility"]}, **{f"metric_{k}": v for k, v in summary["final_metrics"].items()}, **{f"comm_{k}": v for k, v in summary["cumulative_communication"].items()}})
    pd.DataFrame(rows).to_csv(out_root / "analysis/R4_ALL_CELL_SUMMARY.csv", index=False)




def proxy_gain_correlations(out_root: Path) -> None:
    primary = pd.read_csv(ROOT / "manifests/R3_PRIMARY_CELL_MANIFEST.csv")
    records = []
    for rec in primary.to_dict("records"):
        d = out_root / "cells" / "primary" / rec["cell_id"]
        rm = pd.read_csv(d / "round_metrics.csv")
        for _, row in rm.iterrows():
            records.append({"dataset": rec["dataset_protocol"], "method": rec["method"], "beta": rec["beta"], "seed": rec["seed"], "round": row["round"], "proxy_sum": row["selected_proxy_utility_sum"], "proxy_mean": row["selected_proxy_utility_mean"], "actual_gain": row["actual_aggregate_validation_gain"]})
    frame = pd.DataFrame(records)
    output = []
    for (dataset, method, beta), group in frame.groupby(["dataset", "method", "beta"], dropna=False):
        for proxy in ["proxy_sum", "proxy_mean"]:
            valid = group[[proxy, "actual_gain"]].replace([np.inf, -np.inf], np.nan).dropna()
            pooled_p = pearsonr(valid[proxy], valid.actual_gain) if valid[proxy].nunique() > 1 and valid.actual_gain.nunique() > 1 else (np.nan, np.nan)
            pooled_s = spearmanr(valid[proxy], valid.actual_gain) if valid[proxy].nunique() > 1 and valid.actual_gain.nunique() > 1 else (np.nan, np.nan)
            per_seed_p, per_seed_s = [], []
            for _, sg in group.groupby("seed"):
                vv = sg[[proxy, "actual_gain"]].dropna()
                if vv[proxy].nunique() > 1 and vv.actual_gain.nunique() > 1:
                    per_seed_p.append(float(pearsonr(vv[proxy], vv.actual_gain).statistic))
                    per_seed_s.append(float(spearmanr(vv[proxy], vv.actual_gain).statistic))
            if per_seed_p:
                pb = bootstrap_bounds(np.asarray(per_seed_p), 20000, stable_seed("proxy", dataset, method, str(beta), proxy, "pearson"))
                sb = bootstrap_bounds(np.asarray(per_seed_s), 20000, stable_seed("proxy", dataset, method, str(beta), proxy, "spearman"))
            else:
                pb = sb = {"two_sided_low": np.nan, "two_sided_high": np.nan}
            output.append({"dataset": dataset, "method": method, "beta": beta, "proxy": proxy, "round_observations": len(valid), "seeds_with_defined_correlation": len(per_seed_p), "pooled_pearson_r": float(pooled_p.statistic) if hasattr(pooled_p, "statistic") else float(pooled_p[0]), "pooled_pearson_p": float(pooled_p.pvalue) if hasattr(pooled_p, "pvalue") else float(pooled_p[1]), "pooled_spearman_rho": float(pooled_s.statistic) if hasattr(pooled_s, "statistic") else float(pooled_s[0]), "pooled_spearman_p": float(pooled_s.pvalue) if hasattr(pooled_s, "pvalue") else float(pooled_s[1]), "mean_seed_pearson_r": float(np.mean(per_seed_p)) if per_seed_p else np.nan, "seed_pearson_ci_low": pb["two_sided_low"], "seed_pearson_ci_high": pb["two_sided_high"], "mean_seed_spearman_rho": float(np.mean(per_seed_s)) if per_seed_s else np.nan, "seed_spearman_ci_low": sb["two_sided_low"], "seed_spearman_ci_high": sb["two_sided_high"], "interpretation_boundary": "descriptive proxy-validity diagnostic; actual gain is post-aggregation validation J gain at threshold 0.5"})
    pd.DataFrame(output).to_csv(out_root / "analysis/R4_PROXY_ACTUAL_GAIN_CORRELATION.csv", index=False)

def safety_diagnostics(out_root: Path) -> None:
    # Robustness diagnostics are derived only after all locked cells complete.
    rob_manifest = pd.read_csv(ROOT / "manifests/R3_ROBUSTNESS_CELL_MANIFEST.csv")
    rows = []
    for rec in rob_manifest.to_dict("records"):
        d = out_root / "cells" / "robustness" / rec["cell_id"]
        summary = json.loads((d / "summary.json").read_text())
        sel = pd.read_csv(d / "selection_records.csv")
        malicious = sel[sel.malicious_flag == 1]
        benign = sel[sel.malicious_flag == 0]
        rows.append({
            **rec,
            "test_F1": summary["final_metrics"]["f1"],
            "test_PR_AUC": summary["final_metrics"]["pr_auc"],
            "test_FPR": summary["final_metrics"]["fpr"],
            "mean_malicious_selection_rate": float(np.mean([x.loc[x.malicious_flag == 1, "selected"].sum() / max(1, x.selected.sum()) for _, x in sel.groupby("round")])) if len(malicious) else 0.0,
            "mean_malicious_budget_fraction": float(np.mean([x.loc[(x.malicious_flag == 1) & (x.selected == 1), "wire_bytes"].sum() / max(1, x.loc[x.selected == 1, "wire_bytes"].sum()) for _, x in sel.groupby("round")])) if len(malicious) else 0.0,
            "malicious_reported_utility_mean": float(malicious.reported_utility.mean()) if len(malicious) else np.nan,
            "malicious_actual_utility_mean": float(malicious.utility.mean()) if len(malicious) else np.nan,
            "benign_reported_utility_mean": float(benign.reported_utility.mean()) if len(benign) else np.nan,
            "malicious_update_norm_mean": float(malicious.delta_l2.mean()) if len(malicious) else np.nan,
            "benign_update_norm_mean": float(benign.delta_l2.mean()) if len(benign) else np.nan,
            "malicious_cosine_to_benign_mean": float(malicious.cosine_to_mean_benign.mean()) if len(malicious) else np.nan,
        })
    frame = pd.DataFrame(rows)
    clean = frame[frame.model_attack == "clean"][["dataset_protocol", "selection", "aggregation", "reported_utility", "seed", "test_F1", "test_PR_AUC", "test_FPR"]].rename(columns={"test_F1": "clean_F1", "test_PR_AUC": "clean_PR_AUC", "test_FPR": "clean_FPR"})
    frame = frame.merge(clean, on=["dataset_protocol", "selection", "aggregation", "reported_utility", "seed"], how="left")
    frame["attack_success_F1_drop"] = frame.clean_F1 - frame.test_F1
    frame["attack_success_PR_AUC_drop"] = frame.clean_PR_AUC - frame.test_PR_AUC
    frame["attack_success_FPR_increase"] = frame.test_FPR - frame.clean_FPR
    # Explicitly isolate metadata falsification from model poisoning.
    honest = frame[frame.reported_utility == "honest"][["dataset_protocol","selection","aggregation","model_attack","seed","test_F1","test_PR_AUC","test_FPR","mean_malicious_selection_rate","mean_malicious_budget_fraction"]].rename(columns={"test_F1":"honest_F1","test_PR_AUC":"honest_PR_AUC","test_FPR":"honest_FPR","mean_malicious_selection_rate":"honest_malicious_selection_rate","mean_malicious_budget_fraction":"honest_malicious_budget_fraction"})
    frame = frame.merge(honest,on=["dataset_protocol","selection","aggregation","model_attack","seed"],how="left")
    frame["forged_minus_honest_F1"] = frame.test_F1 - frame.honest_F1
    frame["forged_minus_honest_PR_AUC"] = frame.test_PR_AUC - frame.honest_PR_AUC
    frame["forged_minus_honest_FPR"] = frame.test_FPR - frame.honest_FPR
    frame["forged_minus_honest_malicious_selection_rate"] = frame.mean_malicious_selection_rate - frame.honest_malicious_selection_rate
    frame["forged_minus_honest_malicious_budget_fraction"] = frame.mean_malicious_budget_fraction - frame.honest_malicious_budget_fraction
    frame.to_csv(out_root / "analysis/R4_ROBUSTNESS_DIAGNOSTICS.csv", index=False)
    frame[frame.selection == "CB-Score"].to_csv(out_root / "analysis/R4_METADATA_FALSIFICATION_DIAGNOSTICS.csv", index=False)

    privacy_rows = []
    for d in sorted((out_root / "privacy_cells").glob("*")):
        if (d / "result.json").exists():
            r = json.loads((d / "result.json").read_text())
            privacy_rows.append({k: v for k, v in r.items() if k != "folds"})
    pd.DataFrame(privacy_rows).to_csv(out_root / "analysis/R4_PRIVACY_PER_ATTACKER_RESULTS.csv", index=False)

def _validate_stage(out_root: Path, name: str) -> dict:
    a = out_root / "analysis"
    errors = []
    def csv_rows(filename, expected=None, unique=None):
        p = a / filename
        if not p.exists(): errors.append(f"missing {filename}"); return None
        try: df = pd.read_csv(p)
        except Exception as exc: errors.append(f"cannot read {filename}: {exc}"); return None
        if expected is not None and len(df) != expected: errors.append(f"{filename} rows {len(df)}/{expected}")
        if unique and (unique not in df or df[unique].nunique(dropna=False) != expected): errors.append(f"{filename} unique {unique} invalid")
        return df
    if name == "shared_thresholds":
        csv_rows("shared_threshold_results.csv", 980, "cell_id")
        for cid in PRIMARY_MANIFEST.cell_id:
            p=out_root/"cells/primary"/cid/"test_metrics.json"
            if not p.exists() or json.loads(p.read_text()).get("shared_full_threshold") is None:
                errors.append(f"shared threshold missing for {cid}")
                if len(errors)>20: break
    elif name == "threshold_curves":
        csv_rows("R4_THRESHOLD_DISTRIBUTION.csv",980,"cell_id")
        df=csv_rows("R4_PR_ROC_CURVES.csv")
        if df is not None:
            coverage=df.groupby(["cell_id","curve"]).size().reset_index()
            if coverage.cell_id.nunique()!=980 or set(coverage.curve)!={"PR","ROC"} or len(coverage)!=1960:
                errors.append("PR/ROC curve coverage incomplete")
    elif name == "communication":
        csv_rows("R4_SAME_CUMULATIVE_COMMUNICATION.csv",21600)
        csv_rows("R4_COMMUNICATION_TO_TARGET.csv",4320)
    elif name == "aggregate_all": csv_rows("R4_ALL_CELL_SUMMARY.csv",1720,"cell_id")
    elif name == "proxy_gain": csv_rows("R4_PROXY_ACTUAL_GAIN_CORRELATION.csv",196)
    elif name == "safety_privacy_diagnostics":
        csv_rows("R4_ROBUSTNESS_DIAGNOSTICS.csv",360,"cell_id")
        csv_rows("R4_METADATA_FALSIFICATION_DIAGNOSTICS.csv",240,"cell_id")
        csv_rows("R4_PRIVACY_PER_ATTACKER_RESULTS.csv",480,"cell_id")
    elif name == "hypotheses":
        csv_rows("R4_LOCKED_HYPOTHESIS_RESULTS.csv",108,"hypothesis_id")
        csv_rows("R4_THRESHOLD_SENSITIVITY_HYPOTHESIS_RESULTS.csv",324)
    result={"status":"PASS" if not errors else "FAIL","stage":name,"errors":errors}
    atomic_json(a/f"R4_ANALYSIS_STAGE_{name.upper()}_VALIDATION.json",result)
    if errors: raise RuntimeError(f"Analysis stage {name} validation failed: {errors[:5]}")
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--from-stage",default="shared_thresholds")
    args = ap.parse_args()
    cfg = load_yaml(Path(args.config)); out_root = Path(cfg["paths"]["output_root"]); (out_root / "analysis").mkdir(parents=True, exist_ok=True)
    stages=[
        ("shared_thresholds",apply_shared_full_thresholds),
        ("threshold_curves",threshold_and_curve_outputs),
        ("communication",communication_analyses),
        ("aggregate_all",aggregate_all),
        ("proxy_gain",proxy_gain_correlations),
        ("safety_privacy_diagnostics",safety_diagnostics),
        ("hypotheses",hypothesis_analysis),
    ]
    names=[x[0] for x in stages]
    if args.from_stage not in names: raise ValueError(f"Invalid --from-stage {args.from_stage}; choose {names}")
    start=names.index(args.from_stage)
    for index,(name,fn) in enumerate(stages,1):
        if index-1 < start:
            print(f"[analysis] {index}/{len(stages)} {name} REUSE_VALIDATED",flush=True)
            _validate_stage(out_root,name)
            continue
        print(f"[analysis] {index}/{len(stages)} {name}",flush=True)
        atomic_json(out_root/"analysis/R4_ANALYSIS_STAGE.json",{"status":"RUNNING","stage":name,"index":index,"total":len(stages),"execution_amendment_id":"R4-EXEC-AMEND-03-STATISTICS-RECOVERY"})
        fn(out_root)
        _validate_stage(out_root,name)
    atomic_json(out_root / "analysis/ANALYSIS_COMPLETE.json", {"status": "PASS", "protocol_id": cfg["protocol_id"], "execution_amendment_id":"R4-EXEC-AMEND-03-STATISTICS-RECOVERY", "bootstrap_seed_derivation":"SHA-256 stable uint32", "bootstrap_resamples": R3["inference"]["paired_bootstrap_resamples"], "hypotheses": 108, "cell_id_resolution":"locked primary manifest, no float-based reconstruction"})
    atomic_json(out_root/"analysis/R4_ANALYSIS_STAGE.json",{"status":"PASS","stage":"complete","index":len(stages),"total":len(stages),"execution_amendment_id":"R4-EXEC-AMEND-03-STATISTICS-RECOVERY"})

if __name__ == "__main__":
    main()
