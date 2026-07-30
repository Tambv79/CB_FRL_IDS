from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from r4_common import atomic_json, load_yaml, sha256_file


def _copy_gzip_lines(parts: list[Path], out_path: Path, expected_rows: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        for p in parts:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    count += 1
    if count != expected_rows:
        raise RuntimeError(f"Line count {count} != expected rows {expected_rows}: {out_path}")


def build_split(processed_split: Path, cache_dir: Path, split: str, feature_keep: list[int] | None = None) -> dict:
    part_manifest = processed_split / "part_manifest.csv"
    frame = pd.read_csv(part_manifest).sort_values("part")
    rows = int(frame["rows"].sum())
    source_features = int(frame["features"].iloc[0])
    features = source_features if feature_keep is None else len(feature_keep)
    X_file = cache_dir / f"{split}_X_float32.dat"
    y_file = cache_dir / f"{split}_y_int8.npy"
    row_hash_file = cache_dir / f"{split}_row_hash.txt.gz"
    source_group_file = cache_dir / f"{split}_source_group.txt.gz"
    done = cache_dir / f"{split}_CACHE_DONE.json"
    signature = {
        "part_manifest_sha256": sha256_file(part_manifest),
        "feature_keep": feature_keep,
        "parts": [{"part":int(r.part),"X_sha256":str(r.X_sha256),"y_sha256":str(r.y_sha256),"rows":int(r.rows),"features":int(r.features)} for r in frame.itertuples()],
    }
    if done.exists() and X_file.exists() and y_file.exists() and row_hash_file.exists() and source_group_file.exists():
        old=json.loads(done.read_text(encoding="utf-8"))
        if old.get("source_signature")==signature and X_file.stat().st_size==rows*features*4:
            print(f"[CACHE-SKIP] {cache_dir.name}/{split}")
            return old
    cache_dir.mkdir(parents=True,exist_ok=True)
    X_mm=np.memmap(X_file,dtype=np.float32,mode="w+",shape=(rows,features))
    y_mm=np.lib.format.open_memmap(y_file,dtype=np.int8,mode="w+",shape=(rows,))
    offset=0; row_hash_parts=[]; source_group_parts=[]
    for rec in frame.itertuples():
        part=int(rec.part)
        X=sparse.load_npz(processed_split/f"X_part_{part:05d}.npz").tocsr()
        y=np.load(processed_split/f"y_part_{part:05d}.npy")
        if X.shape[0]!=len(y) or X.shape[1]!=source_features:
            raise RuntimeError(f"Part shape mismatch {processed_split} part={part}")
        end=offset+len(y)
        for start in range(0,len(y),25000):
            stop=min(len(y),start+25000)
            dense=X[start:stop].toarray().astype(np.float32,copy=False)
            if feature_keep is not None: dense=dense[:,feature_keep]
            X_mm[offset+start:offset+stop]=dense
        y_mm[offset:end]=y.astype(np.int8,copy=False)
        rh=processed_split/f"row_hash_part_{part:05d}.txt.gz"
        sg=processed_split/f"source_group_part_{part:05d}.txt.gz"
        if not rh.exists() or not sg.exists():
            raise FileNotFoundError(f"Missing processed row metadata for natural/leakage audits: {rh} or {sg}")
        row_hash_parts.append(rh); source_group_parts.append(sg)
        offset=end; X_mm.flush(); y_mm.flush()
        print(f"[CACHE] {cache_dir.name}/{split}: {offset:,}/{rows:,}")
    del X_mm,y_mm
    _copy_gzip_lines(row_hash_parts,row_hash_file,rows)
    _copy_gzip_lines(source_group_parts,source_group_file,rows)
    result={"split":split,"rows":rows,"features":features,"source_features":source_features,
            "X_file":X_file.name,"y_file":y_file.name,"row_hash_file":row_hash_file.name,"source_group_file":source_group_file.name,
            "source_signature":signature}
    atomic_json(done,result); return result


def _feature_removal(step6_dataset: Path, suspicious_path: Path | None = None) -> tuple[list[int] | None,list[str],list[str]]:
    names_path=step6_dataset/"processed"/"feature_names.json"
    suspicious=Path(suspicious_path) if suspicious_path is not None else step6_dataset/"suspicious_feature_decisions.csv"
    if not names_path.exists() or not suspicious.exists():
        raise FileNotFoundError("Feature-sensitivity cache requires feature_names.json and suspicious_feature_decisions.csv")
    names=json.loads(names_path.read_text(encoding="utf-8"))
    s=pd.read_csv(suspicious)
    if "feature" not in s: raise RuntimeError("suspicious_feature_decisions.csv lacks feature column")
    flagged=s["feature"].astype(str).tolist()
    removed=[n for n in names if n in set(flagged)]
    if len(removed)!=10: raise RuntimeError(f"Expected exactly 10 flagged features, found {len(removed)}: {removed}")
    keep=[i for i,n in enumerate(names) if n not in set(removed)]
    return keep,names,removed


def build_dataset_cache(step6_root: Path, cache_root: Path, dataset: str, remove_flagged10: bool=False, suspicious_path: Path | None=None) -> None:
    source=step6_root/dataset
    suffix="_REMOVE_FLAGGED10" if remove_flagged10 else ""
    out=cache_root/f"{dataset}{suffix}"
    keep=None; names=[]; removed=[]
    if remove_flagged10:
        keep,names,removed=_feature_removal(source,suspicious_path)
    elif (source/"processed"/"feature_names.json").exists():
        names=json.loads((source/"processed"/"feature_names.json").read_text(encoding="utf-8"))
    splits={}
    for split in ("train","validation","test"):
        splits[split]=build_split(source/"processed"/split,out,split,keep)
    manifest={"dataset":dataset+suffix,"base_dataset":dataset,"n_features":int(splits["train"]["features"]),
              "feature_names":[names[i] for i in keep] if keep is not None else names,"removed_features":removed,
              "source_step6_root":str(step6_root),"splits":splits}
    atomic_json(out/"cache_manifest.json",manifest)
    print(json.dumps({"status":"PASS","cache":str(out),"rows":{k:v['rows'] for k,v in splits.items()},"features":manifest['n_features'],"removed":removed},indent=2))


def dirichlet_assignment(y: np.ndarray, K: int, alpha: float, seed: int, min_rows: int=200) -> np.ndarray:
    y=np.asarray(y,dtype=np.int8); rng=np.random.default_rng(int(seed)%(2**32)); assignment=np.full(len(y),-1,dtype=np.int16)
    for label in sorted(np.unique(y)):
        idx=np.where(y==label)[0]; rng.shuffle(idx)
        proportions=rng.dirichlet(np.full(K,float(alpha)))
        raw=proportions*len(idx); counts=np.floor(raw).astype(int)
        for j in np.argsort(-(raw-counts))[:len(idx)-counts.sum()]: counts[j]+=1
        start=0
        for client,n in enumerate(counts):
            assignment[idx[start:start+n]]=client; start+=n
    # deterministic balancing repair only when a client has fewer than min_rows.
    counts=np.bincount(assignment,minlength=K)
    while counts.min()<min_rows:
        receiver=int(np.argmin(counts)); donor=int(np.argmax(counts))
        need=min_rows-int(counts[receiver]); movable=np.where(assignment==donor)[0]
        if len(movable)-need<min_rows: raise RuntimeError("Cannot satisfy minimum Dirichlet client rows")
        chosen=np.sort(movable)[:need]; assignment[chosen]=receiver; counts=np.bincount(assignment,minlength=K)
    if (assignment<0).any(): raise RuntimeError("Unassigned rows")
    return assignment


def natural_source_group_assignment(source_groups: list[str], K: int) -> np.ndarray:
    groups=pd.Series(source_groups,dtype="string")
    counts=groups.value_counts().sort_index()
    # deterministic largest-first balanced bin packing; source groups are never split.
    client_load=[0]*K; group_client={}
    for group,n in sorted(counts.items(),key=lambda x:(-int(x[1]),str(x[0]))):
        client=min(range(K),key=lambda i:(client_load[i],i)); group_client[str(group)]=client; client_load[client]+=int(n)
    return groups.map(group_client).to_numpy(dtype=np.int16)


def prepare_assignments(cfg: dict, cache_root: Path, out_root: Path) -> None:
    r3=load_yaml(Path(__file__).resolve().parents[1]/"locks/R3_PROTOCOL_LOCK/config/R3_PROTOCOL_LOCK.yaml")
    K=int(r3["primary_design"]["client_count_K"]); seeds=list(map(int,r3["primary_design"]["paired_seeds"])); alphas=list(map(float,r3["heterogeneity"]["dirichlet_alpha_sensitivity"]))
    out_root.mkdir(parents=True,exist_ok=True)
    for dataset in ("CICIDS2017","CICIoT2023"):
        cache=cache_root/dataset; manifest=json.loads((cache/"cache_manifest.json").read_text()); rec=manifest["splits"]["train"]
        y=np.load(cache/rec["y_file"],mmap_mode="r")
        for alpha in alphas:
            for seed in seeds:
                d=out_root/dataset; d.mkdir(parents=True,exist_ok=True)
                p=d/f"client_assignment_alpha_{alpha:g}_seed_{seed}_clients_{K}.npy"
                lock=p.with_suffix(".json")
                if p.exists() and lock.exists(): continue
                # The primary alpha=0.5 partition is copied byte-for-byte from the
                # R1-authoritative Step-6 assignment. Sensitivity alphas are generated
                # only for the new R3 protocol and are separately hash-locked.
                historical=Path(cfg["paths"]["step6_root"])/dataset/"client_partitions"/p.name
                if abs(alpha-0.5)<1e-12:
                    if not historical.exists():
                        raise FileNotFoundError(f"R1-authoritative primary assignment missing: {historical}")
                    a=np.load(historical,mmap_mode="r")
                    if len(a)!=len(y): raise RuntimeError(f"Historical assignment length mismatch: {historical}")
                    shutil.copy2(historical,p)
                    source="R1_authoritative_Step6_assignment"
                    source_sha256=sha256_file(historical)
                else:
                    a=dirichlet_assignment(y,K,alpha,seed)
                    np.save(p,a)
                    source="R3_new_deterministic_Dirichlet_assignment"
                    source_sha256=None
                atomic_json(lock,{"dataset":dataset,"partition":"dirichlet_label_skew","alpha":alpha,"seed":seed,"K":K,"rows":len(a),"counts":np.bincount(np.asarray(a),minlength=K).tolist(),"source":source,"source_sha256":source_sha256,"sha256":sha256_file(p)})
        if dataset=="CICIoT2023":
            source_groups=[]
            with gzip.open(cache/rec["source_group_file"],"rt",encoding="utf-8") as f:
                source_groups=[x.rstrip("\n") for x in f]
            if len(source_groups)!=len(y): raise RuntimeError("Source group mapping length mismatch")
            a=natural_source_group_assignment(source_groups,K)
            p=out_root/dataset/f"client_assignment_natural_source_group_clients_{K}.npy"; np.save(p,a)
            mapping=pd.DataFrame({"source_group":source_groups,"client_id":a}).drop_duplicates().sort_values(["client_id","source_group"])
            mapping.to_csv(out_root/dataset/"natural_source_group_mapping.csv",index=False)
            atomic_json(out_root/dataset/"natural_source_group_lock.json",{"dataset":dataset,"partition":"source_group_preserving_balanced_bin_packing","K":K,"rows":len(a),"counts":np.bincount(a,minlength=K).tolist(),"groups":int(mapping.source_group.nunique()),"sha256":sha256_file(p)})


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--action",choices=["cache","assignments"],required=True); ap.add_argument("--dataset")
    args=ap.parse_args(); cfg=load_yaml(Path(args.config)); paths=cfg["paths"]
    step6=Path(paths["step6_root"]); cache=Path(paths["primary_cache_root"]); assignments=Path(paths["assignment_root"])
    if args.action=="cache":
        datasets=[args.dataset] if args.dataset else ["CICIDS2017","CICIoT2023"]
        for d in datasets: build_dataset_cache(step6,cache,d,False)
        build_dataset_cache(step6,cache,"CICIDS2017",True,Path(paths["cicids_suspicious_feature_file"]))
    else: prepare_assignments(cfg,cache,assignments)

if __name__=="__main__": main()
