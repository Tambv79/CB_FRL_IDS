from __future__ import annotations

import argparse, hashlib, json, math, time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from r4_common import *
from r4_locked_method import Candidate, priority_values, select_clients, static_eligible

TRAIN_MANIFESTS = {
    "primary":"R3_PRIMARY_CELL_MANIFEST.csv",
    "heterogeneity":"R3_HETEROGENEITY_CELL_MANIFEST.csv",
    "natural_partition":"R3_NATURAL_PARTITION_CELL_MANIFEST.csv",
    "ood_stress":"R3_OOD_STRESS_CELL_MANIFEST.csv",
    "feature_sensitivity":"R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv",
    "robustness":"R3_ROBUSTNESS_CELL_MANIFEST.csv",
}

def base_dataset(protocol: str) -> str:
    return "CICIoT2023" if protocol.startswith("CICIoT2023") else "CICIDS2017"

def cache_path(cfg: Mapping, protocol: str) -> Path:
    if protocol == "CICIDS2017_FILE_DISJOINT_OOD":
        return Path(cfg["paths"]["legacy_ood_cache"])
    suffix="_REMOVE_FLAGGED10" if "REMOVE_FLAGGED10" in protocol else ""
    return Path(cfg["paths"]["primary_cache_root"])/(base_dataset(protocol)+suffix)

def assignment_path(cfg: Mapping, row: Mapping) -> Path:
    protocol=str(row["dataset_protocol"]); dataset=base_dataset(protocol); K=10; seed=int(row["seed"])
    if protocol=="CICIoT2023_R1_NATURAL_SOURCE_GROUP_K10":
        return Path(cfg["paths"]["assignment_root"])/dataset/f"client_assignment_natural_source_group_clients_{K}.npy"
    alpha=float(row.get("alpha_dir",.5)) if not pd.isna(row.get("alpha_dir",.5)) else .5
    root=Path(cfg["paths"]["legacy_ood_assignment_root"]) if protocol=="CICIDS2017_FILE_DISJOINT_OOD" else Path(cfg["paths"]["assignment_root"])/dataset
    return root/f"client_assignment_alpha_{alpha:g}_seed_{seed}_clients_{K}.npy"

def cell_output_dir(cfg: Mapping, phase: str, cell_id: str) -> Path:
    return Path(cfg["paths"]["output_root"])/"cells"/phase/cell_id


def deterministic_malicious_ids(phase: str, attack: str, reported: str, seed: int, cfg: Mapping) -> set[int]:
    if phase != "robustness" or (attack == "clean" and reported == "honest"):
        return set()
    rng=np.random.default_rng(int(seed)+99173)
    count=max(1,round(10*float(cfg["robustness"]["malicious_fraction"])))
    return set(map(int,rng.choice(np.arange(10),size=count,replace=False)))

def client_splits(assignment: np.ndarray, y: np.ndarray, seed: int, cfg: Mapping) -> dict:
    out={}; lc=cfg["learner"]
    for cid in range(10):
        idx=np.where(assignment==cid)[0]
        fit,util=deterministic_client_split(idx,seed+cid*1009,float(lc["local_utility_fraction"]),int(lc["local_utility_cap_rows"]))
        out[cid]={"fit":fit,"utility":util,"utility_benign":int((np.asarray(y[util])==0).sum()),"utility_attack":int((np.asarray(y[util])==1).sum())}
    return out

def initialization(cache: DatasetCache, dataset_protocol: str, seed: int, cfg: Mapping, out_root: Path) -> np.ndarray:
    d=out_root/"initializations"/dataset_protocol; d.mkdir(parents=True,exist_ok=True); p=d/f"seed_{seed}.npz"
    if p.exists(): return np.load(p)["vector"].astype(np.float32)
    set_determinism(seed,True); model=ContextualMLP(cache.features,cfg["learner"]["hidden_units"],cfg["learner"]["dropout"])
    v=model_to_vector(model); np.savez_compressed(p,vector=v); return v

def aggregate(global_vector: np.ndarray, candidate_dicts: list[dict], selected_ids: list[int], mode: str, cfg: Mapping) -> tuple[np.ndarray,dict]:
    if not selected_ids: return global_vector.copy(),{"empty_round":True,"aggregate_delta_l2":0.0,"clip_threshold":None}
    selected=[c for c in candidate_dicts if c["client_id"] in set(selected_ids)]
    deltas=np.stack([c["delta"] for c in selected]); norms=np.linalg.norm(deltas.astype(np.float64),axis=1); threshold=None
    if mode=="Norm-Clipped-Averaging":
        med=float(np.median(norms)); mad=float(np.median(np.abs(norms-med))); threshold=med+2.5*mad
        if threshold<=1e-12: threshold=max(med,1e-12)
        scales=np.minimum(1.0,threshold/np.maximum(norms,1e-12)); deltas=deltas*scales[:,None]
    weights=np.asarray([c["n_fit"] for c in selected],dtype=np.float64); weights/=weights.sum()
    delta=np.sum(deltas.astype(np.float64)*weights[:,None],axis=0)
    return (global_vector.astype(np.float64)+delta).astype(np.float32),{"empty_round":False,"aggregate_delta_l2":float(np.linalg.norm(delta)),"clip_threshold":threshold}

def train_cell(cfg_path: Path, phase: str, row: Mapping, device_name: str="cpu") -> None:
    package_root=Path(__file__).resolve().parents[1]
    r2_path=package_root/"locks/R2_METHOD_LOCK/config/R2_METHOD_LOCK.yaml"
    r3_path=package_root/"locks/R3_PROTOCOL_LOCK/config/R3_PROTOCOL_LOCK.yaml"
    cfg=load_yaml(cfg_path); r2_cfg=load_yaml(r2_path)
    execution_hash=sha256_file(cfg_path); r2_hash=sha256_file(r2_path); r3_hash=sha256_file(r3_path); method_hash=sha256_file(package_root/"src/r4_locked_method.py")
    cell_id=str(row["cell_id"]); out=cell_output_dir(cfg,phase,cell_id); out.mkdir(parents=True,exist_ok=True)
    complete=out/"CELL_COMPLETE.json"
    train_cell_hash=sha256_file(package_root/"src/r4_train_cell.py")
    if complete.exists() and json.loads(complete.read_text()).get("status")=="PASS":
        lock=json.loads((out/"configuration_lock.json").read_text())
        expected=(execution_hash,r2_hash,r3_hash,method_hash)
        observed=(lock.get("execution_config_sha256"),lock.get("r2_lock_sha256"),lock.get("r3_lock_sha256"),lock.get("method_hash"))
        if observed!=expected: raise RuntimeError(f"Completed cell lock mismatch; do not mix protocols: {cell_id}")
        if phase=="robustness":
            if lock.get("execution_amendment_id")!="R4-EXEC-AMEND-02-RECOVERY" or lock.get("train_cell_implementation_sha256")!=train_cell_hash:
                raise RuntimeError(f"Robustness cell was not created by the approved recovery implementation: {cell_id}")
        print(f"[CELL-SKIP] {cell_id}",flush=True); return
    protocol=str(row["dataset_protocol"]); dataset=base_dataset(protocol); cache=DatasetCache(cache_path(cfg,protocol)); seed=int(row["seed"]); rounds=int(row["rounds"])
    method=str(row.get("method",row.get("selection"))); beta=None if pd.isna(row.get("beta",np.nan)) else float(row["beta"])
    aggregation=str(row.get("aggregation","FedAvg")); attack=str(row.get("model_attack","clean")); reported=str(row.get("reported_utility","honest"))
    device=torch.device(device_name); torch.set_num_threads(int(cfg["machine_lock"]["torch_threads_per_cell"]))
    assignment=np.load(assignment_path(cfg,row),mmap_mode="r"); train=cache.splits["train"]; val=cache.splits["validation"]; test=cache.splits["test"]
    if len(assignment)!=train.rows: raise RuntimeError(f"Assignment length mismatch {cell_id}: {len(assignment)} != {train.rows}")
    Xtr,ytr=train.open_X(),train.load_y(); Xv,yv=val.open_X(),val.load_y(); splits=client_splits(np.asarray(assignment),np.asarray(ytr),seed,cfg)
    monitor_cap=min(val.rows,int(cfg["learner"]["validation_monitor_cap_rows"])); monitor_rng=np.random.default_rng(seed+700001)
    monitor_idx=np.arange(val.rows,dtype=np.int64) if val.rows<=monitor_cap else np.sort(monitor_rng.choice(val.rows,size=monitor_cap,replace=False).astype(np.int64))
    weights=class_weights_from_y(np.asarray(ytr)); global_vector=initialization(cache,protocol,seed,cfg,Path(cfg["paths"]["output_root"]))
    checkpoint=out/"checkpoint.pt"; start_round=0; cumulative={k:0 for k in ["selected_uplink","metadata_wire","downlink_broadcast","downlink_unicast","E2E_broadcast","E2E_unicast","selected_airtime_us","full_candidate_airtime_us"]}; cumulative.update({"sum_client_CPU_seconds":0.0,"max_client_seconds":0.0,"server_selection_seconds":0.0,"aggregation_seconds":0.0,"wall_clock_seconds":0.0,"empty_rounds":0})
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if (ck.get("execution_config_sha256"),ck.get("r2_lock_sha256"),ck.get("r3_lock_sha256"),ck.get("method_hash")) != (execution_hash,r2_hash,r3_hash,method_hash):
            raise RuntimeError("Checkpoint lock mismatch; refusing unsafe resume")
        global_vector=np.asarray(ck["global_vector"],dtype=np.float32); start_round=int(ck["next_round"]); cumulative=ck["cumulative"]
    alpha_value=None if pd.isna(row.get("alpha_dir",np.nan)) else float(row.get("alpha_dir"))
    if protocol=="CICIoT2023_R1_NATURAL_SOURCE_GROUP_K10": partition_name="source_group_preserving_balanced_bin_packing"
    elif protocol=="CICIDS2017_FILE_DISJOINT_OOD": partition_name="file_disjoint_cross_capture_dirichlet_alpha_0.5"
    elif "REMOVE_FLAGGED10" in protocol: partition_name="R1_primary_dirichlet_alpha_0.5_feature_removed10"
    else: partition_name=f"dirichlet_alpha_{alpha_value if alpha_value is not None else 0.5:g}"
    config_lock={"protocol_id":cfg["protocol_id"],"r2_protocol_id":cfg["r2_protocol_id"],"r3_protocol_id":cfg["r3_protocol_id"],"dataset":dataset,"dataset_protocol":protocol,"partition":partition_name,"method":method,"beta":beta,"alpha_dir":alpha_value if alpha_value is not None else 0.5,"seed":seed,"aggregation":aggregation,"model_attack":attack,"reported_utility":reported,"execution_config_sha256":execution_hash,"r2_lock_sha256":r2_hash,"r3_lock_sha256":r3_hash,"method_hash":method_hash,"data_hashes":{"cache_manifest":sha256_file(cache.cache_dir/"cache_manifest.json"),"assignment":sha256_file(assignment_path(cfg,row))},"threshold_rules":cfg["threshold"],"test_locked":True}
    if phase=="robustness":
        config_lock.update({"execution_amendment_id":"R4-EXEC-AMEND-02-RECOVERY","train_cell_implementation_sha256":train_cell_hash,"robustness_manifest_row_sha256":hashlib.sha256(json.dumps(dict(row),sort_keys=True,default=str,separators=(",",":")).encode("utf-8")).hexdigest()})
    atomic_json(out/"configuration_lock.json",config_lock)
    class malicious:
        pass
    # A forged-utility condition is itself adversarial even when model_attack is clean.
    malicious_ids=deterministic_malicious_ids(phase,attack,reported,seed,cfg)
    round_fields=["round","validation_F1","validation_PR_AUC","validation_ROC_AUC","validation_FPR","FP_per_10000","selected_count","static_eligible_count","budget_bytes","used_bytes","budget_utilization","empty_round","selected_uplink","metadata_wire","downlink_broadcast","downlink_unicast","E2E_broadcast","E2E_unicast","selected_airtime_us","full_candidate_airtime_us","airtime_reduction","sum_client_CPU_seconds","max_client_seconds_simulated_parallel_critical_path","server_selection_seconds","aggregation_seconds","wall_clock_seconds","malicious_selection_rate","malicious_budget_fraction","pre_validation_F1","pre_validation_PR_AUC","pre_validation_FPR","actual_aggregate_validation_gain","selected_proxy_utility_sum","selected_proxy_utility_mean"]
    sel_fields=["round","client_id","static_eligible","ranking_eligible","selected","utility","reported_utility","wire_bytes","priority","n_fit","n_utility","utility_benign","utility_attack","reported_metadata_view","malicious_flag","delta_l2","cosine_to_mean_benign","actual_client_size","actual_client_attack_ratio","uplink_rate_mbps","predicted_airtime_us"]
    cell_start=time.perf_counter()
    for rid in range(start_round,rounds):
        round_start=time.perf_counter(); global_model=ContextualMLP(cache.features,cfg["learner"]["hidden_units"],cfg["learner"]["dropout"]).to(device); vector_to_model(global_model,global_vector)
        pre_probs=predict_probabilities(global_model,Xv,monitor_idx,device,int(cfg["learner"]["evaluation_batch_size"])); pre_met=binary_metrics(np.asarray(yv[monitor_idx]),pre_probs,.5)
        cands=[]; local_times=[]
        for cid in range(10):
            sp=splits[cid]; local_seed=seed*1000003+rid*10007+cid*101; before=global_vector.copy()
            flip=float(cfg["robustness"]["label_flip_probability"]) if cid in malicious_ids and attack=="label-flip" else 0.0
            lv,diag=train_local_model(global_vector,cache.features,Xtr,ytr,sp["fit"],local_seed,cfg,device,weights,flip)
            if not np.array_equal(before,global_vector): raise RuntimeError("Global vector mutated during local training")
            lm=ContextualMLP(cache.features,cfg["learner"]["hidden_units"],cfg["learner"]["dropout"]).to(device); vector_to_model(lm,lv)
            utility,mg,ml=local_utility(global_model,lm,Xtr,ytr,sp["utility"],device,int(cfg["learner"]["evaluation_batch_size"]),float(cfg["method"]["utility_fpr_penalty"]))
            delta=(lv-global_vector).astype(np.float32)
            if cid in malicious_ids and attack=="sign-flip": delta*=float(cfg["robustness"]["sign_flip_multiplier"])
            raw,payload=serialize_vector_zlib(delta,6); wire=payload+int(cfg["communication"]["update_message_overhead_bytes"])
            reported_u=float(utility)+(float(cfg["robustness"]["forged_utility_addition"]) if cid in malicious_ids and reported=="forged_inflated" else 0.0)
            cands.append({"client_id":cid,"utility":reported_u,"actual_utility":float(utility),"delta":delta,"payload_bytes":payload,"wire_bytes":wire,"n_fit":len(sp["fit"]),"n_utility":len(sp["utility"]),"utility_benign":sp["utility_benign"],"utility_attack":sp["utility_attack"],"local_training_seconds":diag["local_training_seconds"],"delta_l2":float(np.linalg.norm(delta.astype(np.float64))),"actual_client_size":int(len(sp["fit"])+len(sp["utility"])),"actual_client_attack_ratio":float(np.asarray(ytr[np.concatenate([sp["fit"],sp["utility"]])]).mean())})
            local_times.append(diag["local_training_seconds"])
        rates=list(map(float,cfg["airtime"]["rates_mbps"]))
        for c in cands:
            rate=rates[(int(c["client_id"])+int(seed)) % len(rates)]
            c["uplink_rate_mbps"]=rate
            c["predicted_airtime_us"]=float(math.ceil(8*int(c["wire_bytes"])/rate))
        full_wire=sum(c["wire_bytes"] for c in cands); budget=full_wire if method=="Full-Contextual" else int(math.floor(float(beta)*full_wire))
        candidates=[Candidate(c["client_id"],c["utility"],c["wire_bytes"],c["n_fit"],c["n_utility"],c["utility_benign"],c["utility_attack"],predicted_airtime_us=c["predicted_airtime_us"],local_training_seconds=c["local_training_seconds"]) for c in cands]
        st=time.perf_counter(); selected,diag=select_clients(method,candidates,budget,seed+rid*104729,r2_cfg); selection_seconds=time.perf_counter()-st
        st=time.perf_counter(); global_vector,aggdiag=aggregate(global_vector,cands,selected,aggregation,cfg); aggregation_seconds=time.perf_counter()-st
        monitor=ContextualMLP(cache.features,cfg["learner"]["hidden_units"],cfg["learner"]["dropout"]).to(device); vector_to_model(monitor,global_vector)
        probs=predict_probabilities(monitor,Xv,monitor_idx,device,int(cfg["learner"]["evaluation_batch_size"])); met=binary_metrics(np.asarray(yv[monitor_idx]),probs,.5)
        selected_set=set(selected); selected_wire=sum(c["wire_bytes"] for c in cands if c["client_id"] in selected_set); metadata=0 if method=="Full-Contextual" else 10*(metadata_payload_bytes(method)+int(cfg["communication"]["metadata_message_overhead_bytes"])); _,downpayload=serialize_vector_zlib(global_vector,6); down=downpayload+int(cfg["communication"]["downlink_message_overhead_bytes"]); unicast=10*down
        e2eb=selected_wire+metadata+down; e2eu=selected_wire+metadata+unicast
        selected_airtime=float(sum(c["predicted_airtime_us"] for c in cands if c["client_id"] in selected_set)); full_airtime=float(sum(c["predicted_airtime_us"] for c in cands))
        malicious_selected=[c for c in cands if c["client_id"] in selected_set and c["client_id"] in malicious_ids]
        selected_actual_utilities=[c["actual_utility"] for c in cands if c["client_id"] in selected_set]
        proxy_sum=float(np.sum(selected_actual_utilities)) if selected_actual_utilities else 0.0
        proxy_mean=float(np.mean(selected_actual_utilities)) if selected_actual_utilities else 0.0
        fpr_penalty=float(cfg["method"]["utility_fpr_penalty"])
        actual_gain=float((met["f1"]-fpr_penalty*met["fpr"])-(pre_met["f1"]-fpr_penalty*pre_met["fpr"]))
        rowout={"round":rid,"validation_F1":met["f1"],"validation_PR_AUC":met["pr_auc"],"validation_ROC_AUC":met["roc_auc"],"validation_FPR":met["fpr"],"FP_per_10000":met["false_positives_per_10000_benign"],"selected_count":len(selected),"static_eligible_count":diag["static_eligible_count"],"budget_bytes":budget,"used_bytes":diag["used_bytes"],"budget_utilization":diag.get("budget_utilization",1.0),"empty_round":int(diag["empty_round"]),"selected_uplink":selected_wire,"metadata_wire":metadata,"downlink_broadcast":down,"downlink_unicast":unicast,"E2E_broadcast":e2eb,"E2E_unicast":e2eu,"selected_airtime_us":selected_airtime,"full_candidate_airtime_us":full_airtime,"airtime_reduction":1.0-selected_airtime/max(full_airtime,1.0),"sum_client_CPU_seconds":sum(local_times),"max_client_seconds_simulated_parallel_critical_path":max(local_times),"server_selection_seconds":selection_seconds,"aggregation_seconds":aggregation_seconds,"wall_clock_seconds":time.perf_counter()-round_start,"malicious_selection_rate":len(malicious_selected)/max(1,len(selected)),"malicious_budget_fraction":sum(c["wire_bytes"] for c in malicious_selected)/max(1,selected_wire),"pre_validation_F1":pre_met["f1"],"pre_validation_PR_AUC":pre_met["pr_auc"],"pre_validation_FPR":pre_met["fpr"],"actual_aggregate_validation_gain":actual_gain,"selected_proxy_utility_sum":proxy_sum,"selected_proxy_utility_mean":proxy_mean}
        append_csv(out/"round_metrics.csv",rowout,round_fields)
        priorities=priority_values(method,candidates,r2_cfg); benign=[c["delta"] for c in cands if c["client_id"] not in malicious_ids]; mean_benign=np.mean(np.stack(benign),axis=0) if benign else np.zeros_like(global_vector)
        for pos,c in enumerate(cands):
            cos=float(np.dot(c["delta"],mean_benign)/(max(np.linalg.norm(c["delta"])*np.linalg.norm(mean_benign),1e-12)))
            static=bool(static_eligible(candidates[pos],budget,r2_cfg))
            append_csv(out/"selection_records.csv",{"round":rid,"client_id":c["client_id"],"static_eligible":int(static),"ranking_eligible":int(priorities[pos]>0),"selected":int(c["client_id"] in selected_set),"utility":c["actual_utility"],"reported_utility":c["utility"],"wire_bytes":c["wire_bytes"],"priority":priorities[pos],"n_fit":c["n_fit"],"n_utility":c["n_utility"],"utility_benign":c["utility_benign"],"utility_attack":c["utility_attack"],"reported_metadata_view":"Raw","malicious_flag":int(c["client_id"] in malicious_ids),"delta_l2":c["delta_l2"],"cosine_to_mean_benign":cos,"actual_client_size":c["actual_client_size"],"actual_client_attack_ratio":c["actual_client_attack_ratio"],"uplink_rate_mbps":c["uplink_rate_mbps"],"predicted_airtime_us":c["predicted_airtime_us"]},sel_fields)
        for key,valx in [("selected_uplink",selected_wire),("metadata_wire",metadata),("downlink_broadcast",down),("downlink_unicast",unicast),("E2E_broadcast",e2eb),("E2E_unicast",e2eu),("selected_airtime_us",selected_airtime),("full_candidate_airtime_us",full_airtime)]: cumulative[key]+=valx
        cumulative["sum_client_CPU_seconds"]+=sum(local_times); cumulative["max_client_seconds"]+=max(local_times); cumulative["server_selection_seconds"]+=selection_seconds; cumulative["aggregation_seconds"]+=aggregation_seconds; cumulative["wall_clock_seconds"]+=rowout["wall_clock_seconds"]; cumulative["empty_rounds"]+=int(diag["empty_round"])
        torch.save({"global_vector":global_vector,"next_round":rid+1,"cumulative":cumulative,"execution_config_sha256":execution_hash,"r2_lock_sha256":r2_hash,"r3_lock_sha256":r3_hash,"method_hash":method_hash},checkpoint.with_suffix(".tmp")); checkpoint.with_suffix(".tmp").replace(checkpoint)
    final=ContextualMLP(cache.features,cfg["learner"]["hidden_units"],cfg["learner"]["dropout"]).to(device); vector_to_model(final,global_vector)
    val_probs=predict_probabilities(final,Xv,None,device,int(cfg["learner"]["evaluation_batch_size"])); threshold,threshold_lock,grid=choose_threshold(np.asarray(yv),val_probs,cfg); grid.rename(columns={"f1":"F1","pr_auc":"PR_AUC","roc_auc":"ROC_AUC","fpr":"FPR","false_positives_per_10000_benign":"FP_per_10000"}).to_csv(out/"validation_threshold_grid.csv",index=False)
    # Test is first opened only after configuration and method-specific threshold are locked.
    atomic_json(out/"THRESHOLD_LOCK.json",threshold_lock); Xt,yt=test.open_X(),test.load_y(); test_probs=predict_probabilities(final,Xt,None,device,int(cfg["learner"]["evaluation_batch_size"])); indices=np.arange(test.rows,dtype=np.int64)
    np.savez_compressed(out/"test_predictions.npz",y_true=np.asarray(yt,dtype=np.int8),probabilities=test_probs,row_hash_or_index=indices)
    metrics_method=binary_metrics(np.asarray(yt),test_probs,threshold); metrics_fixed=binary_metrics(np.asarray(yt),test_probs,.5)
    atomic_json(out/"test_metrics.json",{"method_specific":metrics_method,"shared_full_threshold":None,"fixed_0.5":metrics_fixed})
    summary={"cell_id":cell_id,"final_metrics":metrics_method,"cumulative_communication":cumulative,"communication_to_target":"computed_in_R4_analysis","runtime":{"cell_wall_clock_seconds":time.perf_counter()-cell_start,**{k:v for k,v in cumulative.items() if "seconds" in k}},"hashes":{"test_predictions":sha256_file(out/"test_predictions.npz"),"round_metrics":sha256_file(out/"round_metrics.csv"),"selection_records":sha256_file(out/"selection_records.csv")}}
    atomic_json(out/"summary.json",summary); atomic_json(complete,{"status":"PASS","cell_id":cell_id,"phase":phase,"test_accessed_after_threshold_lock":True})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--phase",choices=TRAIN_MANIFESTS,required=True); ap.add_argument("--cell-id",required=True); ap.add_argument("--device",default="cpu")
    args=ap.parse_args(); root=Path(__file__).resolve().parents[1]; df=load_csv_manifest(root/"manifests"/TRAIN_MANIFESTS[args.phase]); hit=df[df.cell_id==args.cell_id]
    if len(hit)!=1: raise RuntimeError(f"Cell not found or duplicated: {args.cell_id}")
    train_cell(Path(args.config),args.phase,hit.iloc[0].to_dict(),args.device)
if __name__=="__main__": main()
