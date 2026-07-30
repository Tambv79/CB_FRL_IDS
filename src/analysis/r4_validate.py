from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from r4_common import atomic_json,load_yaml,sha256_file
TRAIN_MANIFESTS={
 "primary":"R3_PRIMARY_CELL_MANIFEST.csv",
 "heterogeneity":"R3_HETEROGENEITY_CELL_MANIFEST.csv",
 "natural_partition":"R3_NATURAL_PARTITION_CELL_MANIFEST.csv",
 "ood_stress":"R3_OOD_STRESS_CELL_MANIFEST.csv",
 "feature_sensitivity":"R3_FEATURE_SENSITIVITY_CELL_MANIFEST.csv",
 "robustness":"R3_ROBUSTNESS_CELL_MANIFEST.csv",
}

REQUIRED={
 'configuration_lock.json':['protocol_id','method_hash','data_hashes','dataset','partition','method','beta','alpha_dir','seed','threshold_rules','test_locked'],
 'round_metrics.csv':['round','validation_F1','validation_PR_AUC','validation_ROC_AUC','validation_FPR','FP_per_10000','selected_count','static_eligible_count','budget_bytes','used_bytes','budget_utilization','empty_round','selected_uplink','metadata_wire','downlink_broadcast','downlink_unicast','E2E_broadcast','E2E_unicast','selected_airtime_us','full_candidate_airtime_us','airtime_reduction','actual_aggregate_validation_gain','selected_proxy_utility_sum','selected_proxy_utility_mean'],
 'selection_records.csv':['round','client_id','static_eligible','ranking_eligible','selected','utility','wire_bytes','priority','n_fit','n_utility','utility_benign','utility_attack','reported_metadata_view','malicious_flag'],
 'validation_threshold_grid.csv':['threshold','F1','PR_AUC','ROC_AUC','FPR','FP_per_10000'],
 'test_metrics.json':['method_specific','shared_full_threshold','fixed_0.5'],
 'summary.json':['final_metrics','cumulative_communication','communication_to_target','runtime','hashes'],
}
def validate_cell(d:Path,rounds:int,final:bool,require_shared:bool=False):
 e=[]
 for fn,fields in REQUIRED.items():
  p=d/fn
  if not p.exists(): e.append(f'missing {fn}'); continue
  if p.suffix=='.json':
   o=json.loads(p.read_text()); e.extend([f'{fn} missing {x}' for x in fields if x not in o]);
   if fn=='test_metrics.json' and final and require_shared and o.get('shared_full_threshold') is None: e.append('shared_full_threshold not materialized for primary analysis')
  else:
   df=pd.read_csv(p); e.extend([f'{fn} missing {x}' for x in fields if x not in df]);
   if fn=='round_metrics.csv':
    if len(df)!=rounds or set(df['round'])!=set(range(rounds)): e.append(f'round rows invalid {len(df)}/{rounds}')
    if (df.used_bytes>df.budget_bytes).any(): e.append('hard budget violation')
   if fn=='selection_records.csv' and len(df)!=rounds*10: e.append(f'selection rows {len(df)} != {rounds*10}')
 p=d/'test_predictions.npz'
 if not p.exists(): e.append('missing test_predictions.npz')
 else:
  z=np.load(p); e.extend([f'test_predictions missing {x}' for x in ['y_true','probabilities','row_hash_or_index'] if x not in z]);
  if len(z['y_true'])!=len(z['probabilities']): e.append('test prediction length mismatch')
 if final and not (d/'CELL_COMPLETE.json').exists(): e.append('missing CELL_COMPLETE.json')
 return e

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--mode',choices=['partial','final'],default='partial');args=ap.parse_args();cfg=load_yaml(Path(args.config));root=Path(__file__).resolve().parents[1];out=Path(cfg['paths']['output_root']);errors=[];counts={}
 for phase,fn in TRAIN_MANIFESTS.items():
  m=pd.read_csv(root/'manifests'/fn); done=0
  for row in m.to_dict('records'):
   d=out/'cells'/phase/row['cell_id']
   if not d.exists():
    if args.mode=='final': errors.append(f'{phase}/{row["cell_id"]}: missing cell directory')
    continue
   ce=validate_cell(d,int(row['rounds']),args.mode=='final',require_shared=(phase=='primary'))
   if phase=='robustness' and not ce and (d/'CELL_COMPLETE.json').exists():
    lock=json.loads((d/'configuration_lock.json').read_text());sr=pd.read_csv(d/'selection_records.csv')
    if lock.get('execution_amendment_id')!='R4-EXEC-AMEND-02-RECOVERY': ce.append('missing recovery amendment lock')
    adversarial=(row['model_attack']!='clean' or row['reported_utility']!='honest');expected_mal=3 if adversarial else 0;observed=sr.groupby('round').malicious_flag.sum()
    if len(observed)!=int(row['rounds']) or not (observed==expected_mal).all(): ce.append('malicious client cardinality mismatch')
    if row['reported_utility']=='forged_inflated':
     mal=sr[sr.malicious_flag==1]
     if mal.empty or not np.allclose(mal.reported_utility-mal.utility,1.0,rtol=0,atol=1e-9): ce.append('forged utility addition mismatch')
    elif not np.allclose(sr.reported_utility-sr.utility,0.0,rtol=0,atol=1e-9): ce.append('honest reported utility altered')
   if ce: errors.extend([f'{phase}/{row["cell_id"]}: {x}' for x in ce])
   else: done+=1
  counts[phase]={'expected':len(m),'valid':done}
 pm=pd.read_csv(root/'manifests/R3_PRIVACY_CELL_MANIFEST.csv');pdone=0
 for cid in pm.cell_id:
  d=out/'privacy_cells'/cid;complete=d/'CELL_COMPLETE.json'
  if not complete.exists(): continue
  local=[]
  try:
   if json.loads(complete.read_text()).get('status')!='PASS': local.append('completion status not PASS')
  except Exception as exc: local.append(f'invalid completion JSON: {exc}')
  for fn in ['configuration_lock.json','result.json']:
   if not (d/fn).exists(): local.append(f'missing {fn}')
  if (d/'result.json').exists():
   r=json.loads((d/'result.json').read_text())
   if r.get('group_overlap_max')!=0: local.append('group overlap')
   for c in ['mean_auc','auc_ci_low','auc_ci_high']:
    if not np.isfinite(float(r.get(c,np.nan))): local.append(f'nonfinite {c}')
  if local: errors.extend([f'privacy/{cid}: {x}' for x in local])
  else: pdone+=1
 counts['privacy']={'expected':len(pm),'valid':pdone}
 if args.mode=='final' and pdone!=len(pm): errors.append(f'privacy cells {pdone}/{len(pm)}')
 if args.mode=='final':
  required_analysis=[
   'R4_ALL_CELL_SUMMARY.csv','R4_LOCKED_HYPOTHESIS_RESULTS.csv','R4_THRESHOLD_SENSITIVITY_HYPOTHESIS_RESULTS.csv',
   'shared_threshold_results.csv','R4_THRESHOLD_DISTRIBUTION.csv','R4_PR_ROC_CURVES.csv',
   'R4_SAME_CUMULATIVE_COMMUNICATION.csv','R4_COMMUNICATION_TO_TARGET.csv',
   'R4_PRIVACY_PER_ATTACKER_RESULTS.csv','R4_ROBUSTNESS_DIAGNOSTICS.csv','R4_METADATA_FALSIFICATION_DIAGNOSTICS.csv','R4_PROXY_ACTUAL_GAIN_CORRELATION.csv','ANALYSIS_COMPLETE.json']
  for name in required_analysis:
   p=out/'analysis'/name
   if not p.exists(): errors.append(f'missing analysis {name}')
  hp=out/'analysis/R4_LOCKED_HYPOTHESIS_RESULTS.csv'
  if hp.exists():
   h=pd.read_csv(hp)
   if len(h)!=108: errors.append(f'hypothesis rows {len(h)}/108')
   for c in ['raw_p','holm_adjusted_p','one_sided_lower_95','one_sided_upper_95','decision']:
    if c not in h: errors.append(f'hypothesis output missing {c}')
  pv=out/'analysis/R4_PRIVACY_PER_ATTACKER_RESULTS.csv'
  if pv.exists() and len(pd.read_csv(pv))!=480: errors.append(f'privacy aggregate rows {len(pd.read_csv(pv))}/480')
  rd=out/'analysis/R4_ROBUSTNESS_DIAGNOSTICS.csv'
  if rd.exists():
   rdf=pd.read_csv(rd)
   if len(rdf)!=360: errors.append(f'robustness diagnostics rows {len(rdf)}/360')
   for c in ['forged_minus_honest_F1','forged_minus_honest_PR_AUC','forged_minus_honest_FPR','mean_malicious_selection_rate','mean_malicious_budget_fraction']:
    if c not in rdf: errors.append(f'robustness diagnostics missing {c}')
  md=out/'analysis/R4_METADATA_FALSIFICATION_DIAGNOSTICS.csv'
  if md.exists() and len(pd.read_csv(md))!=240: errors.append(f'metadata falsification rows {len(pd.read_csv(md))}/240')
  for provenance in ['R4_EXECUTED_ENVIRONMENT_ACTUAL.json','R4_EXECUTION_AMENDMENT_02_RECOVERY.json','R4_RECOVERY_PREFLIGHT.json','R4_EXECUTION_AMENDMENT_03_STATISTICS_RECOVERY.json','R4_EXECUTION_AMENDMENT_04_SELF_PROCESS_GUARD.json']:
   pp=out/'validation'/provenance
   if not pp.exists(): errors.append(f'missing provenance {provenance}')
  pf=out/'validation/R4_RECOVERY_PREFLIGHT.json'
  if pf.exists() and json.loads(pf.read_text()).get('status')!='PASS': errors.append('recovery preflight not PASS')
  # Robustness semantics: exactly three malicious clients in every adversarial round,
  # and forged utility must differ from honest utility by the locked +1.0 addition.
  robm=pd.read_csv(root/'manifests/R3_ROBUSTNESS_CELL_MANIFEST.csv')
  for rec in robm.to_dict('records'):
   d=out/'cells/robustness'/rec['cell_id'];lockp=d/'configuration_lock.json';selp=d/'selection_records.csv'
   if lockp.exists():
    lock=json.loads(lockp.read_text())
    if lock.get('execution_amendment_id')!='R4-EXEC-AMEND-02-RECOVERY': errors.append(f'robustness/{rec["cell_id"]}: missing recovery amendment lock')
   if selp.exists():
    sr=pd.read_csv(selp);adversarial=(rec['model_attack']!='clean' or rec['reported_utility']!='honest')
    expected_mal=3 if adversarial else 0
    observed=sr.groupby('round').malicious_flag.sum()
    if len(observed)!=30 or not (observed==expected_mal).all(): errors.append(f'robustness/{rec["cell_id"]}: malicious client cardinality mismatch')
    if rec['reported_utility']=='forged_inflated':
     mal=sr[sr.malicious_flag==1]
     if mal.empty or not np.allclose(mal.reported_utility-mal.utility,1.0,rtol=0,atol=1e-9): errors.append(f'robustness/{rec["cell_id"]}: forged utility addition mismatch')
    elif not np.allclose(sr.reported_utility-sr.utility,0.0,rtol=0,atol=1e-9): errors.append(f'robustness/{rec["cell_id"]}: honest reported utility altered')
  allsum=out/'analysis/R4_ALL_CELL_SUMMARY.csv'
  if allsum.exists() and len(pd.read_csv(allsum))!=1720: errors.append(f'all-cell summary rows {len(pd.read_csv(allsum))}/1720')
  ts=out/'analysis/R4_THRESHOLD_SENSITIVITY_HYPOTHESIS_RESULTS.csv'
  if ts.exists() and len(pd.read_csv(ts))!=324: errors.append(f'threshold sensitivity hypothesis rows {len(pd.read_csv(ts))}/324')
  sh=out/'analysis/shared_threshold_results.csv'
  if sh.exists() and len(pd.read_csv(sh))!=980: errors.append(f'shared threshold rows {len(pd.read_csv(sh))}/980')
  ac=out/'analysis/ANALYSIS_COMPLETE.json'
  if ac.exists() and json.loads(ac.read_text()).get('status')!='PASS': errors.append('analysis complete gate not PASS')
  # Every privacy cell must have a reproducibility lock and zero group overlap.
  for cid in pm.cell_id:
   d=out/'privacy_cells'/cid
   for fn in ['configuration_lock.json','result.json','CELL_COMPLETE.json']:
    if not (d/fn).exists(): errors.append(f'privacy/{cid}: missing {fn}')
   if (d/'CELL_COMPLETE.json').exists() and json.loads((d/'CELL_COMPLETE.json').read_text()).get('status')!='PASS': errors.append(f'privacy/{cid}: completion status not PASS')
   if (d/'result.json').exists():
    r=json.loads((d/'result.json').read_text())
    if r.get('group_overlap_max')!=0: errors.append(f'privacy/{cid}: group overlap')
    for c in ['mean_auc','auc_ci_low','auc_ci_high']:
     if not np.isfinite(float(r.get(c,np.nan))): errors.append(f'privacy/{cid}: nonfinite {c}')
 result={'status':'PASS' if not errors else 'FAIL','mode':args.mode,'counts':counts,'errors':errors}; (out/'validation').mkdir(parents=True,exist_ok=True);atomic_json(out/'validation'/f'R4_VALIDATION_{args.mode.upper()}.json',result);print(json.dumps(result,indent=2));sys.exit(1 if errors else 0)
if __name__=='__main__':main()
