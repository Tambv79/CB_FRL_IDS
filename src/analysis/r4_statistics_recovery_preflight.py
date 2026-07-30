from __future__ import annotations
import argparse,json,os,shutil,sys,time
from pathlib import Path
import pandas as pd,psutil
from r4_common import atomic_json,load_yaml

ROOT=Path(__file__).resolve().parents[1]
PHASES={
 "primary":("R3_PRIMARY_CELL_MANIFEST.csv",980),"heterogeneity":("R3_HETEROGENEITY_CELL_MANIFEST.csv",240),
 "natural_partition":("R3_NATURAL_PARTITION_CELL_MANIFEST.csv",60),"ood_stress":("R3_OOD_STRESS_CELL_MANIFEST.csv",60),
 "feature_sensitivity":("R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv",20),"robustness":("R3_ROBUSTNESS_CELL_MANIFEST.csv",360)}
CONFLICT_MARKERS=(
 'r4_train_cell.py','r4_privacy.py','r4_runner.py','r4_analysis.py',
 'r4_statistics_recovery_orchestrator.py'
)
ORCHESTRATOR_PATH=str((ROOT/'src/r4_statistics_recovery_orchestrator.py').resolve()).lower()


def is_conflicting_process(pid:int,cmd:str,current_pid:int,ancestor_pids:set[int])->bool:
    """Return True only for a separate R4 worker/orchestrator.

    The venv launcher and the base Python interpreter may both expose the same
    orchestrator command line on Windows.  They are legitimate ancestors of
    this preflight process and must not be mistaken for concurrent runs.
    """
    low=(cmd or '').lower()
    if pid==current_pid:
        return False
    if pid in ancestor_pids and ORCHESTRATOR_PATH in low:
        return False
    return any(marker in low for marker in CONFLICT_MARKERS)


def find_conflicting_processes()->list[dict]:
    current=psutil.Process()
    ancestor_pids={p.pid for p in current.parents()}
    active=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            pid=int(proc.info['pid'])
            cmd=' '.join(proc.info.get('cmdline') or [])
            if is_conflicting_process(pid,cmd,current.pid,ancestor_pids):
                active.append({'pid':pid,'cmd':cmd})
        except (psutil.NoSuchProcess,psutil.AccessDenied,psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return active


def reusable_stage12(out:Path):
    a=out/'analysis';errors=[]
    try:
        s=pd.read_csv(a/'shared_threshold_results.csv')
        if len(s)!=980 or s.cell_id.nunique()!=980: errors.append('shared threshold table invalid')
        t=pd.read_csv(a/'R4_THRESHOLD_DISTRIBUTION.csv')
        if len(t)!=980 or t.cell_id.nunique()!=980: errors.append('threshold distribution invalid')
        c=pd.read_csv(a/'R4_PR_ROC_CURVES.csv')
        cov=c.groupby(['cell_id','curve']).size().reset_index()
        if c.cell_id.nunique()!=980 or len(cov)!=1960 or set(cov.curve)!={'PR','ROC'}: errors.append('curve coverage invalid')
    except Exception as exc: errors.append(str(exc))
    return not errors,errors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=load_yaml(Path(args.config));out=Path(cfg['paths']['output_root']);errors=[];checks=[]

    # Refuse a genuinely separate R4 execution, but explicitly exclude this
    # preflight process and every ancestor that launches this same orchestrator.
    active=find_conflicting_processes()
    if active: errors.append(f'active external R4 processes: {active}')

    # All completed experimental evidence must exist. No training is invoked.
    for phase,(fn,expected) in PHASES.items():
        m=pd.read_csv(ROOT/'manifests'/fn);valid=0
        if len(m)!=expected: errors.append(f'{phase} manifest {len(m)}/{expected}')
        for rec in m.to_dict('records'):
            d=out/'cells'/phase/rec['cell_id']
            try:
                if json.loads((d/'CELL_COMPLETE.json').read_text()).get('status')!='PASS': raise ValueError('completion not PASS')
                rm=pd.read_csv(d/'round_metrics.csv',usecols=['round'])
                if len(rm)!=int(rec['rounds']): raise ValueError(f'rounds {len(rm)}/{rec["rounds"]}')
                valid+=1
            except Exception as exc:
                errors.append(f'{phase}/{rec["cell_id"]}: {exc}')
                if len(errors)>50: break
        checks.append({'check':phase,'expected':expected,'valid':valid,'pass':valid==expected})

    pm=pd.read_csv(ROOT/'manifests/R3_PRIVACY_CELL_MANIFEST.csv');pv=0
    for cid in pm.cell_id:
        try:
            d=out/'privacy_cells'/cid
            if json.loads((d/'CELL_COMPLETE.json').read_text()).get('status')!='PASS': raise ValueError('completion not PASS')
            r=json.loads((d/'result.json').read_text())
            if r.get('group_overlap_max')!=0: raise ValueError('group overlap')
            pv+=1
        except Exception as exc:
            errors.append(f'privacy/{cid}: {exc}')
            if len(errors)>50: break
    checks.append({'check':'privacy','expected':480,'valid':pv,'pass':pv==480})

    # Manifest-based primary lookup covers exact IDs, including beta=1.0.
    prim=pd.read_csv(ROOT/'manifests/R3_PRIMARY_CELL_MANIFEST.csv')
    if len(prim)!=980 or prim.cell_id.nunique()!=980: errors.append('primary manifest cardinality/uniqueness failure')
    beta1=prim[(prim.beta==1.0)&(prim.method!='Full-Contextual')]
    if len(beta1)!=160 or not beta1.cell_id.str.contains('__b1.0__',regex=False).all(): errors.append('beta=1.0 authoritative ID invariant failed')

    free=shutil.disk_usage(out.anchor or 'D:\\').free/1024**3
    if free<1.5: errors.append(f'insufficient disk {free:.2f} GB')
    reuse,reuse_errors=reusable_stage12(out)
    recommended='communication' if reuse else 'shared_thresholds'

    validation=out/'validation';validation.mkdir(parents=True,exist_ok=True)
    amendment03={
      'amendment_id':'R4-EXEC-AMEND-03-STATISTICS-RECOVERY','status':'LOCKED_BEFORE_STATISTICS_RERUN',
      'reason':'V5 analysis reconstructed beta=1.0 directory names as b1 instead of using locked manifest IDs b1.0.',
      'scope':'analysis, final validation, finalization only; no training or privacy reruns',
      'preserved_evidence':{'training_cells':1720,'privacy_evaluations':480},
      'scientific_effect':'none on trained models, predictions, selection, communication, robustness, or privacy outputs',
      'fixes':['authoritative manifest-based cell resolution','stage-level output validation','statistics-only fail-fast preflight','full traceback capture'],
      'recommended_from_stage':recommended,'stage_1_2_reuse_errors':reuse_errors,'created_epoch':time.time()}
    amendment04={
      'amendment_id':'R4-EXEC-AMEND-04-SELF-PROCESS-GUARD','status':'LOCKED_BEFORE_V7_EXECUTION',
      'reason':'V6 preflight incorrectly classified its own Windows venv/base-interpreter orchestrator ancestors as concurrent R4 processes.',
      'scope':'process-safety guard only; analysis, validation, and finalization remain unchanged',
      'preserved_evidence':{'training_cells':1720,'privacy_evaluations':480},
      'scientific_effect':'none; no experimental artifact, method, manifest, threshold, hypothesis, or statistic is altered',
      'guard_rule':'exclude current PID and same-orchestrator ancestor PIDs; reject every separate R4 worker or orchestrator',
      'created_epoch':time.time()}
    atomic_json(validation/'R4_EXECUTION_AMENDMENT_03_STATISTICS_RECOVERY.json',amendment03)
    atomic_json(validation/'R4_EXECUTION_AMENDMENT_04_SELF_PROCESS_GUARD.json',amendment04)
    result={
      'status':'PASS' if not errors else 'FAIL','errors':errors,'checks':checks,
      'active_external_processes':active,'free_disk_gb':free,
      'recommended_from_stage':recommended,'reused_stage_1_2':reuse,
      'self_process_guard':'PASS' if not active else 'FAIL'}
    atomic_json(validation/'R4_STATISTICS_RECOVERY_PREFLIGHT.json',result)
    print(json.dumps(result,indent=2));sys.exit(0 if not errors else 1)

if __name__=='__main__':main()
