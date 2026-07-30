from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np


LOCKED_METHODS = (
    "Full-Contextual",
    "Random-Budget",
    "Cost-Only",
    "Utility-Only",
    "Utility-Cost-Ratio",
    "CB-Score",
    "Historical-DoubleCost",
    "Oort-Style-Adapted",
    "Oracle-Score-Exact",
)


@dataclass(frozen=True)
class Candidate:
    client_id: int
    utility: float
    wire_bytes: int
    n_fit: int
    n_utility: int
    utility_benign: int
    utility_attack: int
    predicted_airtime_us: float | None = None
    local_training_seconds: float | None = None


def marginal_utility(
    global_f1: float,
    global_fpr: float,
    local_f1: float,
    local_fpr: float,
    fpr_penalty: float = 0.25,
) -> float:
    """Locked utility: J(local)-J(global), J=F1-lambda*FPR, same subset/threshold."""
    vals=(global_f1,global_fpr,local_f1,local_fpr,fpr_penalty)
    if not all(isfinite(float(x)) for x in vals):
        return float("nan")
    return float((local_f1-fpr_penalty*local_fpr)-(global_f1-fpr_penalty*global_fpr))


def winsorized_minmax(values: Sequence[float], lower: float=0.05, upper: float=0.95) -> np.ndarray:
    x=np.asarray(values,dtype=np.float64)
    if x.size==0: return x
    lo=float(np.quantile(x,lower)); hi=float(np.quantile(x,upper))
    y=np.clip(x,lo,hi)
    span=float(y.max()-y.min())
    if span<=1e-12: return np.zeros_like(y)
    return (y-y.min())/span


def static_eligible(c: Candidate, round_budget_bytes: int, cfg: Mapping) -> bool:
    """Finite metadata, support, both classes, and update can fit the whole round budget."""
    e=cfg["eligibility"]
    return bool(
        isfinite(float(c.utility))
        and isinstance(c.wire_bytes,(int,np.integer))
        and 0 < int(c.wire_bytes) <= int(round_budget_bytes)
        and int(c.n_utility) >= int(e["minimum_utility_rows"])
        and int(c.utility_benign) >= int(e["minimum_benign_utility_rows"])
        and int(c.utility_attack) >= int(e["minimum_attack_utility_rows"])
        and int(c.n_fit) >= int(e["minimum_fit_rows"])
    )


def _oort_value(c: Candidate, preferred_duration_us: float, gamma: float) -> float:
    """Adapted Oort utility x system penalty; no claim of exact Oort reproduction."""
    stat=max(float(c.utility),0.0)
    airtime=float(c.predicted_airtime_us if c.predicted_airtime_us is not None else c.wire_bytes)
    train_us=float(c.local_training_seconds or 0.0)*1_000_000.0
    duration=max(airtime+train_us,1e-12)
    penalty=(preferred_duration_us/duration)**gamma if duration>preferred_duration_us else 1.0
    return stat*penalty


def priority_values(method: str, candidates: Sequence[Candidate], cfg: Mapping) -> np.ndarray:
    if method not in LOCKED_METHODS:
        raise ValueError(f"Unknown locked method: {method}")
    u=np.asarray([float(c.utility) for c in candidates],dtype=np.float64)
    cost=np.asarray([float(c.wire_bytes) for c in candidates],dtype=np.float64)
    positive=np.maximum(u,0.0)
    if method=="Cost-Only":
        return 1.0/(cost+1e-12)
    if method=="Utility-Only":
        return positive
    if method=="Utility-Cost-Ratio":
        return positive/(cost+1e-12)
    if method in {"CB-Score","Historical-DoubleCost","Oracle-Score-Exact"}:
        s=np.maximum(
            float(cfg["ranking"]["alpha_u"])*winsorized_minmax(u,float(cfg["ranking"]["winsor_lower_quantile"]),float(cfg["ranking"]["winsor_upper_quantile"]))
            -float(cfg["ranking"]["alpha_c"])*winsorized_minmax(cost,float(cfg["ranking"]["winsor_lower_quantile"]),float(cfg["ranking"]["winsor_upper_quantile"])),
            0.0,
        )
        return s/(cost+1e-12) if method=="Historical-DoubleCost" else s
    if method=="Oort-Style-Adapted":
        durations=[]
        for c in candidates:
            airtime=float(c.predicted_airtime_us if c.predicted_airtime_us is not None else c.wire_bytes)
            durations.append(airtime+float(c.local_training_seconds or 0.0)*1_000_000.0)
        preferred=float(np.median(durations)) if durations else 1.0
        return np.asarray([_oort_value(c,preferred,float(cfg["oort_adapted"]["system_penalty_gamma"])) for c in candidates])
    if method in {"Full-Contextual","Random-Budget"}:
        return np.ones(len(candidates),dtype=np.float64)
    raise AssertionError(method)


def _exact_oracle(candidates: Sequence[Candidate], eligible_positions: Sequence[int], budget: int, values: np.ndarray):
    best=(float('-inf'),float('-inf'),float('inf'),tuple())
    for r in range(len(eligible_positions)+1):
        for subset in combinations(eligible_positions,r):
            used=sum(int(candidates[i].wire_bytes) for i in subset)
            if used>budget: continue
            value=sum(float(values[i]) for i in subset)
            utility=sum(float(candidates[i].utility) for i in subset)
            ids=tuple(sorted(int(candidates[i].client_id) for i in subset))
            key=(value,utility,-used,tuple(-x for x in ids))
            if key>best:
                best=key; best_subset=subset
    selected=[int(candidates[i].client_id) for i in best_subset]
    used=sum(int(candidates[i].wire_bytes) for i in best_subset)
    return selected,used


def select_clients(method: str, candidates: Sequence[Candidate], round_budget_bytes: int, seed: int, cfg: Mapping):
    if cfg.get("novelty_enabled",False):
        raise ValueError("Novelty is not part of the R2 locked core method.")
    budget=int(round_budget_bytes)
    if budget<0: raise ValueError("Budget must be non-negative")
    if method=="Full-Contextual":
        ids=[int(c.client_id) for c in candidates]
        used=sum(int(c.wire_bytes) for c in candidates)
        return ids,{"used_bytes":used,"budget_bytes":used,"static_eligible_count":len(candidates),"empty_round":False,"fallback_used":False}

    eligible=[i for i,c in enumerate(candidates) if static_eligible(c,budget,cfg)]
    priorities=priority_values(method,candidates,cfg)
    if method=="Oracle-Score-Exact":
        selected,used=_exact_oracle(candidates,[i for i in eligible if priorities[i]>0],budget,priorities)
    else:
        if method=="Random-Budget":
            rng=np.random.default_rng(int(seed)%(2**32))
            order=list(eligible); rng.shuffle(order)
        else:
            order=sorted(eligible,key=lambda i:(-float(priorities[i]),int(candidates[i].client_id)))
        selected=[]; used=0
        for i in order:
            if method!="Random-Budget" and float(priorities[i])<=0: continue
            wire=int(candidates[i].wire_bytes)
            if used+wire<=budget:
                selected.append(int(candidates[i].client_id)); used+=wire

    # Locked empty-round behavior: no forced fallback; keep global model unchanged.
    return selected,{
        "used_bytes":int(used),
        "budget_bytes":budget,
        "budget_utilization":float(used/budget) if budget>0 else 0.0,
        "static_eligible_count":len(eligible),
        "ranking_positive_count":int(sum(float(priorities[i])>0 for i in eligible)),
        "empty_round":not bool(selected),
        "fallback_used":False,
    }


def aggregate_or_keep(global_vector: np.ndarray, candidates: Sequence[Mapping], selected_ids: Sequence[int]):
    """Sample-weighted FedAvg; an empty selected set leaves the global vector unchanged."""
    if not selected_ids:
        return np.asarray(global_vector,dtype=np.float32).copy(),{"empty_round":True,"aggregate_delta_l2":0.0}
    selected=[c for c in candidates if int(c["client_id"]) in set(map(int,selected_ids))]
    weights=np.asarray([float(c["n_fit"]) for c in selected],dtype=np.float64)
    weights/=weights.sum()
    delta=np.zeros_like(np.asarray(global_vector),dtype=np.float64)
    for w,c in zip(weights,selected): delta+=w*np.asarray(c["delta"],dtype=np.float64)
    updated=(np.asarray(global_vector,dtype=np.float64)+delta).astype(np.float32)
    return updated,{"empty_round":False,"aggregate_delta_l2":float(np.linalg.norm(delta))}
