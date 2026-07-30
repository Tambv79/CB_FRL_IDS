from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj



def stable_seed(*parts: object) -> int:
    """Return a process-independent uint32 seed derived from canonical JSON."""
    payload=json.dumps(parts,sort_keys=True,separators=(",",":"),default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4],"big",signed=False)

def atomic_json(path: Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path, exclude_names: set[str] | None = None) -> str:
    root = Path(root)
    exclude_names = exclude_names or set()
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name not in exclude_names):
        rel = p.relative_to(root).as_posix().encode()
        h.update(rel); h.update(b"\0"); h.update(sha256_file(p).encode()); h.update(b"\n")
    return h.hexdigest()


def normalize_seed32(seed: int) -> int:
    return int(seed) % (2**32)


def set_determinism(seed: int, deterministic: bool = True) -> None:
    seed = normalize_seed32(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ContextualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: Sequence[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden_units:
            layers.extend([nn.Linear(prev, int(width)), nn.ReLU(), nn.Dropout(float(dropout))])
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def model_to_vector(model: nn.Module) -> np.ndarray:
    return torch.nn.utils.parameters_to_vector([p.detach().cpu() for p in model.parameters()]).numpy().astype(np.float32, copy=True)


def vector_to_model(model: nn.Module, vector: np.ndarray) -> None:
    arr = np.asarray(vector, dtype=np.float32)
    flat = torch.as_tensor(arr, dtype=torch.float32, device=next(model.parameters()).device)
    expected = sum(int(p.numel()) for p in model.parameters())
    if int(flat.numel()) != expected:
        raise ValueError(f"Parameter vector length {flat.numel()} != {expected}")
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            n = int(param.numel())
            param.copy_(flat[offset:offset+n].view_as(param))
            offset += n


def serialize_vector_zlib(vector: np.ndarray, level: int = 6) -> tuple[int, int]:
    arr = np.asarray(vector, dtype="<f4", order="C")
    payload = struct.pack("<IQ", 1, int(arr.size)) + arr.tobytes(order="C")
    return len(payload), len(zlib.compress(payload, int(level)))


def metadata_payload_bytes(method: str) -> int:
    if method == "Full-Contextual":
        return 0
    if method == "Random-Budget":
        return struct.calcsize("<HHQB")
    # client, round, utility, update bytes, n_fit, static eligibility
    return struct.calcsize("<HHfQIB")


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y_true) != len(probabilities):
        raise ValueError("Metric arrays have different lengths")
    pred = (probabilities >= float(threshold)).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fpr),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives_per_10000_benign": float(10000.0 * fp / max(1, fp + tn)),
        "threshold": float(threshold),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
    return result


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, cfg: Mapping) -> tuple[float, dict, pd.DataFrame]:
    tc = cfg["threshold"]
    thresholds = np.round(np.arange(float(tc["grid_start"]), float(tc["grid_stop"]) + float(tc["grid_step"])/2, float(tc["grid_step"])), 10)
    rows = [binary_metrics(y_true, probabilities, float(t)) for t in thresholds]
    frame = pd.DataFrame(rows)
    feasible = frame[frame.fpr <= float(tc["fpr_target"]) + 1e-12]
    if len(feasible):
        best = feasible.sort_values(["f1", "pr_auc", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
        rule = "maximize_F1_subject_to_FPR_target"
    else:
        best = frame.sort_values(["fpr", "f1", "pr_auc", "threshold"], ascending=[True, False, False, True]).iloc[0]
        rule = "minimum_FPR_then_maximum_F1"
    result = best.to_dict(); result["selection_rule"] = rule; result["target_fpr"] = float(tc["fpr_target"])
    return float(best.threshold), result, frame


@dataclass
class DenseMemmapSplit:
    X_path: Path
    y_path: Path
    rows: int
    features: int
    row_hash_path: Path | None = None
    source_group_path: Path | None = None

    def open_X(self) -> np.memmap:
        return np.memmap(self.X_path, dtype=np.float32, mode="r", shape=(self.rows, self.features))

    def load_y(self) -> np.ndarray:
        return np.load(self.y_path, mmap_mode="r")


class DatasetCache:
    def __init__(self, cache_dir: Path):
        cache_dir = Path(cache_dir)
        manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
        self.cache_dir = cache_dir
        self.dataset = str(manifest["dataset"])
        self.features = int(manifest["n_features"])
        self.feature_names = list(manifest.get("feature_names", []))
        self.splits: Dict[str, DenseMemmapSplit] = {}
        for split, rec in manifest["splits"].items():
            self.splits[split] = DenseMemmapSplit(
                cache_dir / rec["X_file"], cache_dir / rec["y_file"], int(rec["rows"]), self.features,
                cache_dir / rec["row_hash_file"] if rec.get("row_hash_file") else None,
                cache_dir / rec["source_group_file"] if rec.get("source_group_file") else None,
            )
        self.manifest = manifest


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def predict_probabilities(model: nn.Module, X: np.ndarray, indices: np.ndarray | None, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    if indices is None:
        indices = np.arange(X.shape[0], dtype=np.int64)
    probs = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            idx = indices[start:start+int(batch_size)]
            xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
            probs[start:start+len(idx)] = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32)
    return probs


def class_weights_from_y(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.int8)
    n = max(1, len(y)); neg = max(1, int((y == 0).sum())); pos = max(1, int((y == 1).sum()))
    return float(n/(2*neg)), float(n/(2*pos))


def deterministic_client_split(indices: np.ndarray, seed: int, fraction: float, cap: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        return indices.copy(), np.array([], dtype=np.int64)
    rng = np.random.default_rng(normalize_seed32(seed)); x = indices.copy(); rng.shuffle(x)
    utility_n = min(int(math.ceil(len(x)*float(fraction))), int(cap))
    utility_n = max(1, min(utility_n, len(x)-1))
    return np.sort(x[utility_n:]), np.sort(x[:utility_n])


def train_local_model(global_vector: np.ndarray, input_dim: int, X: np.ndarray, y: np.ndarray, fit_indices: np.ndarray,
                      seed: int, cfg: Mapping, device: torch.device, class_weights: tuple[float,float],
                      label_flip_probability: float = 0.0) -> tuple[np.ndarray, dict]:
    if len(fit_indices) == 0:
        raise RuntimeError("Empty fit subset")
    set_determinism(seed, bool(cfg["learner"]["deterministic_algorithms"]))
    model = ContextualMLP(input_dim, cfg["learner"]["hidden_units"], cfg["learner"]["dropout"]).to(device)
    vector_to_model(model, global_vector)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learner"]["learning_rate"]), weight_decay=float(cfg["learner"]["weight_decay"]))
    rng = np.random.default_rng(normalize_seed32(seed))
    w_neg, w_pos = class_weights; losses=[]; started=time.perf_counter()
    for _ in range(int(cfg["learner"]["local_steps_per_round"])):
        chosen = rng.choice(fit_indices, size=int(cfg["learner"]["batch_size"]), replace=len(fit_indices)<int(cfg["learner"]["batch_size"]))
        xb = torch.from_numpy(np.asarray(X[chosen], dtype=np.float32)).to(device)
        y_np = np.asarray(y[chosen], dtype=np.float32).copy()
        if label_flip_probability > 0:
            flips = rng.random(len(y_np)) < float(label_flip_probability)
            y_np[flips] = 1.0 - y_np[flips]
        yb = torch.from_numpy(y_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
        weights = torch.where(yb > .5, torch.tensor(w_pos, device=device), torch.tensor(w_neg, device=device))
        loss=(raw*weights).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["learner"]["gradient_clip_norm"])); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model_to_vector(model), {"local_training_seconds": time.perf_counter()-started, "mean_local_loss": float(np.mean(losses)), "effective_seed32": normalize_seed32(seed)}


def local_utility(global_model: nn.Module, local_model: nn.Module, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                  device: torch.device, batch_size: int, fpr_penalty: float) -> tuple[float,dict,dict]:
    if len(indices) == 0:
        return float("nan"), {}, {}
    yy=np.asarray(y[indices],dtype=np.int8)
    pg=predict_probabilities(global_model,X,indices,device,batch_size)
    pl=predict_probabilities(local_model,X,indices,device,batch_size)
    mg=binary_metrics(yy,pg,.5); ml=binary_metrics(yy,pl,.5)
    return float((ml["f1"]-fpr_penalty*ml["fpr"])-(mg["f1"]-fpr_penalty*mg["fpr"])),mg,ml


def append_csv(path: Path, row: Mapping, fieldnames: Sequence[str]) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
        if not exists: w.writeheader()
        w.writerow(row)


def load_csv_manifest(path: Path) -> pd.DataFrame:
    df=pd.read_csv(path)
    if "cell_id" not in df or df.cell_id.duplicated().any():
        raise RuntimeError(f"Invalid/duplicate cell manifest: {path}")
    return df


def environment_manifest() -> dict:
    import scipy, sklearn
    return {
        "python": os.sys.version,
        "platform": os.name,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
