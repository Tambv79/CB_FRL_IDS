from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from r4_common import atomic_json, load_yaml, sha256_file, stable_seed


_METADATA_CACHE: dict[tuple[str,str],pd.DataFrame] = {}

def _load_metadata(cfg: dict, dataset: str) -> pd.DataFrame:
    key=(str(cfg["paths"]["output_root"]),dataset)
    if key in _METADATA_CACHE: return _METADATA_CACHE[key].copy()
    root=Path(cfg["paths"]["output_root"])/"cells"/"primary"
    parts=[]
    for p in sorted(root.glob(f"PRIMARY__{dataset}__CB-Score__b0.4__seed*/selection_records.csv")):
        df=pd.read_csv(p); seed=int(p.parent.name.split("seed")[-1]); df["training_seed"]=seed; parts.append(df)
    if len(parts)!=10: raise RuntimeError(f"Expected 10 CB-Score beta=.4 selection files for {dataset}, found {len(parts)}")
    result=pd.concat(parts,ignore_index=True)
    required={"client_id","reported_utility","wire_bytes","n_fit","selected","round","actual_client_size","actual_client_attack_ratio","training_seed"}
    missing=sorted(required-set(result.columns))
    if missing: raise RuntimeError(f"Privacy metadata missing fields for {dataset}: {missing}")
    _METADATA_CACHE[key]=result
    return result.copy()

def _view(df: pd.DataFrame, name: str, seed: int) -> pd.DataFrame:
    """Materialize exactly the fields locked in R3_METADATA_FIELD_ENCODING.csv."""
    x=pd.DataFrame(index=df.index);rng=np.random.default_rng(int(seed)%(2**32))
    utility=np.clip(df.reported_utility.astype(float).to_numpy(),-1.0,1.0)
    if name=="Raw":
        x["client_id"]=df.client_id.astype(int);x["round"]=df["round"].astype(int);x["utility"]=df.reported_utility.astype(float);x["wire_bytes"]=df.wire_bytes.astype(float);x["n_fit"]=df.n_fit.astype(float);x["eligible"]=df.static_eligible.astype(int)
    elif name=="Noisy-Quantized":
        x["utility"]=utility+rng.normal(0.0,0.1,len(df));x["wire_bytes"]=np.maximum(1024,np.round(df.wire_bytes.astype(float)/1024)*1024);x["n_fit"]=np.maximum(0,np.round(df.n_fit.astype(float)/1000)*1000)
    elif name=="No-Size":
        x["utility"]=utility+rng.normal(0.0,0.2,len(df));x["wire_bytes"]=np.maximum(4096,np.round(df.wire_bytes.astype(float)/4096)*4096)
    elif name=="Minimal-Bucketed":
        x["utility_bucket"]=pd.qcut(df.reported_utility.rank(method="first"),3,labels=False,duplicates="raise").astype(int);x["cost_bucket"]=pd.qcut(df.wire_bytes.rank(method="first"),3,labels=False,duplicates="raise").astype(int)
    else: raise ValueError(name)
    if not np.isfinite(x.to_numpy(dtype=float)).all(): raise RuntimeError(f"Nonfinite values in metadata view {name}")
    return x

def _target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target=="client_size_above_seed_median": field="actual_client_size"
    elif target=="attack_ratio_above_seed_median": field="actual_client_attack_ratio"
    else: raise ValueError(target)
    med=df.groupby("training_seed")[field].transform("median")
    return (df[field]>med).astype(int).to_numpy()

def _model(name: str, cfg: dict, seed: int, columns: list[str]):
    hp=cfg["privacy"]["attacker_hyperparameters"][name]
    categorical=[c for c in columns if c=="client_id"];numeric=[c for c in columns if c not in categorical]
    if name in {"LogisticRegression","MLP"}:
        num_pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    else:
        num_pipe=Pipeline([("impute",SimpleImputer(strategy="median"))])
    transformers=[]
    if numeric: transformers.append(("numeric",num_pipe,numeric))
    if categorical: transformers.append(("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical))
    preprocess=ColumnTransformer(transformers=transformers,remainder="drop")
    if name=="LogisticRegression": estimator=LogisticRegression(random_state=seed,**hp)
    elif name=="RandomForest": estimator=RandomForestClassifier(random_state=seed,**hp)
    elif name=="MLP":
        hp=dict(hp);hp["hidden_layer_sizes"]=tuple(hp["hidden_layer_sizes"]);estimator=MLPClassifier(random_state=seed,**hp)
    else: raise ValueError(name)
    return Pipeline([("preprocess",preprocess),("attacker",estimator)])

def run_cell(cfg: dict, row: dict) -> dict:
    dataset=row["dataset_protocol"]; view=row["metadata_view"]; attacker=row["attacker"]; target=row["target"]; seed=int(row["privacy_split_seed"]); folds=int(row["group_folds"])
    df=_load_metadata(cfg,dataset); X=_view(df,view,seed); y=_target(df,target); groups=df.client_id.astype(int).to_numpy()
    if len(np.unique(groups))<folds: raise RuntimeError("Insufficient client groups for group-disjoint folds")
    aucs=[]; fold_rows=[]; oof=np.full(len(y),np.nan,dtype=float)
    unique_groups=np.unique(groups); rng=np.random.default_rng(seed); shuffled=unique_groups.copy(); rng.shuffle(shuffled)
    fold_groups=np.array_split(shuffled,folds)
    for fold,test_groups in enumerate(fold_groups):
        te=np.where(np.isin(groups,test_groups))[0]; tr=np.where(~np.isin(groups,test_groups))[0]
        if set(groups[tr]) & set(groups[te]): raise RuntimeError("Privacy group overlap")
        model=_model(attacker,cfg,seed+fold*1009,list(X.columns)); model.fit(X.iloc[tr],y[tr]); prob=model.predict_proba(X.iloc[te])[:,1]; oof[te]=prob
        auc=float(roc_auc_score(y[te],prob)) if len(np.unique(y[te]))==2 else float("nan")
        aucs.append(auc); fold_rows.append({"fold":fold,"auc":auc,"train_groups":sorted(map(int,np.unique(groups[tr]))),"test_groups":sorted(map(int,np.unique(groups[te]))),"train_rows":len(tr),"test_rows":len(te),"test_positive_rate":float(y[te].mean())})
    if np.isnan(oof).any(): raise RuntimeError("Incomplete out-of-fold privacy predictions")
    overall_auc=float(roc_auc_score(y,oof))
    boot_rng=np.random.default_rng(stable_seed("privacy-bootstrap",dataset,view,attacker,target,seed)); boot=[]
    for _ in range(int(cfg["privacy"]["group_bootstrap_resamples"])):
        sampled=boot_rng.choice(unique_groups,size=len(unique_groups),replace=True)
        idx=np.concatenate([np.where(groups==g)[0] for g in sampled])
        if len(np.unique(y[idx]))==2: boot.append(float(roc_auc_score(y[idx],oof[idx])))
    if not boot: raise RuntimeError("No finite group-bootstrap privacy AUC")
    boots=np.asarray(boot,dtype=float)
    return {"cell_id":row["cell_id"],"dataset_protocol":dataset,"metadata_view":view,"attacker":attacker,"target":target,"privacy_split_seed":seed,"observations":len(df),"groups":int(df.client_id.nunique()),"class_balance":float(y.mean()),"mean_auc":overall_auc,"fold_mean_auc":float(np.nanmean(aucs)),"auc_ci_low":float(np.quantile(boots,.025)),"auc_ci_high":float(np.quantile(boots,.975)),"uncertainty":"2000-resample client-group bootstrap over out-of-fold predictions","folds":fold_rows,"group_overlap_max":0,"hyperparameters":cfg["privacy"]["attacker_hyperparameters"][attacker],"view_fields":list(X.columns),"attacker_preprocessing":"median imputation; StandardScaler for LogisticRegression/MLP numeric fields; client_id one-hot only in Raw; RandomForest numeric fields unscaled","claim_boundary":"empirical pre-upload metadata leakage only; no formal privacy guarantee"}

def _prevalidate_design(cfg: dict, manifest: pd.DataFrame) -> None:
    errors=[]
    for dataset in sorted(manifest.dataset_protocol.unique()):
        df=_load_metadata(cfg,dataset)
        groups=df.client_id.astype(int).to_numpy(); unique_groups=np.unique(groups)
        for target in sorted(manifest[manifest.dataset_protocol==dataset].target.unique()):
            y=_target(df,target)
            for seed in sorted(map(int,manifest[(manifest.dataset_protocol==dataset)&(manifest.target==target)].privacy_split_seed.unique())):
                rng=np.random.default_rng(seed);shuffled=unique_groups.copy();rng.shuffle(shuffled);fold_groups=np.array_split(shuffled,5)
                for fold,test_groups in enumerate(fold_groups):
                    te=np.where(np.isin(groups,test_groups))[0];tr=np.where(~np.isin(groups,test_groups))[0]
                    if set(groups[tr])&set(groups[te]): errors.append(f"{dataset}/{target}/{seed}/fold{fold}: group overlap")
                    if len(np.unique(y[tr]))!=2: errors.append(f"{dataset}/{target}/{seed}/fold{fold}: train target single-class")
                    if len(np.unique(y[te]))!=2: errors.append(f"{dataset}/{target}/{seed}/fold{fold}: test target single-class")
        for view in sorted(manifest[manifest.dataset_protocol==dataset].metadata_view.unique()):
            sample=_view(df,view,42)
            if not np.isfinite(sample.to_numpy(dtype=float)).all(): errors.append(f"{dataset}/{view}: nonfinite transformed metadata")
    if errors: raise RuntimeError("Privacy design prevalidation failed: "+"; ".join(errors[:50]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--cell-id"); ap.add_argument("--all",action="store_true")
    args=ap.parse_args(); cfg_path=Path(args.config);cfg=load_yaml(cfg_path); root=Path(__file__).resolve().parents[1]; manifest_path=root/"manifests/R3_PRIVACY_CELL_MANIFEST.csv";manifest=pd.read_csv(manifest_path)
    _prevalidate_design(cfg,manifest)
    if args.cell_id: manifest=manifest[manifest.cell_id==args.cell_id]
    elif not args.all: raise RuntimeError("Use --cell-id or --all")
    outroot=Path(cfg["paths"]["output_root"])/"privacy_cells"; outroot.mkdir(parents=True,exist_ok=True); summaries=[]
    config_hash=sha256_file(cfg_path);manifest_hash=sha256_file(manifest_path);code_hash=sha256_file(Path(__file__))
    total=len(manifest)
    for i,row in enumerate(manifest.to_dict("records"),1):
        print(f"[privacy] {i}/{total} {row['cell_id']}",flush=True)
        out=outroot/row["cell_id"]; complete=out/"CELL_COMPLETE.json";lock_path=out/"configuration_lock.json"
        expected_lock={"execution_config_sha256":config_hash,"privacy_manifest_sha256":manifest_hash,"privacy_implementation_sha256":code_hash,"execution_amendment_id":"R4-EXEC-AMEND-02-RECOVERY"}
        if complete.exists() and json.loads(complete.read_text()).get("status")=="PASS":
            if not lock_path.exists(): raise RuntimeError(f"Privacy cell missing lock: {row['cell_id']}")
            lock=json.loads(lock_path.read_text())
            if any(lock.get(k)!=v for k,v in expected_lock.items()): raise RuntimeError(f"Privacy cell lock mismatch: {row['cell_id']}")
            continue
        out.mkdir(parents=True,exist_ok=True);atomic_json(lock_path,{**expected_lock,"cell_id":row["cell_id"],"row":row});result=run_cell(cfg,row);atomic_json(out/"result.json",result);atomic_json(complete,{"status":"PASS","cell_id":row["cell_id"],"execution_amendment_id":"R4-EXEC-AMEND-02-RECOVERY"});summaries.append(result)
    print(json.dumps({"status":"PASS","completed_now":len(summaries),"requested":total},indent=2),flush=True)
if __name__=="__main__": main()
