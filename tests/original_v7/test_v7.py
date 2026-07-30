import sys,unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import r4_analysis as a
import r4_statistics_recovery_preflight as pf

class V7Tests(unittest.TestCase):
 def test_manifest_exact_roundtrip(self):
  for rec in a.PRIMARY_MANIFEST.to_dict('records'):
   beta=None if rec['method']=='Full-Contextual' else float(rec['beta'])
   self.assertEqual(a.cell_id_for(rec['dataset_protocol'],rec['method'],int(rec['seed']),beta),rec['cell_id'])
 def test_beta_one_uses_decimal_manifest_id(self):
  cid=a.cell_id_for('CICIDS2017_COVERAGE_R1','Random-Budget',42,1.0)
  self.assertIn('__b1.0__',cid);self.assertNotIn('__b1__',cid)
 def test_complete_primary_design_matrix(self):
  m=a.PRIMARY_MANIFEST
  self.assertEqual(len(m),980);self.assertEqual(m.cell_id.nunique(),980)
  for ds in m.dataset_protocol.unique():
   self.assertEqual(len(m[(m.dataset_protocol==ds)&(m.method=='Full-Contextual')]),10)
   for method in [x for x in m.method.unique() if x!='Full-Contextual']:
    for beta in a.BETAS:
     self.assertEqual(len(m[(m.dataset_protocol==ds)&(m.method==method)&(m.beta==beta)]),10)
 def test_hypothesis_references_resolve(self):
  h=pd.read_csv(ROOT/'manifests/R3_HYPOTHESIS_MANIFEST.csv')
  self.assertEqual(len(h),108)
  for r in h.to_dict('records'):
   rhs=r['comparison'].split('minus',1)[1].strip();beta=float(r['beta'])
   for seed in a.SEEDS:
    a.cell_id_for(r['dataset'],'CB-Score',seed,beta)
    a.cell_id_for(r['dataset'],rhs,seed,None if rhs=='Full-Contextual' else beta)
 def test_no_training_launcher(self):
  txt=(ROOT/'src/r4_statistics_recovery_orchestrator.py').read_text()+(ROOT/'RUN_STATISTICS_FINALIZE_RECOVERY.cmd').read_text()
  self.assertNotIn('r4_train_cell.py',txt);self.assertNotIn('r4_privacy.py',txt)
 def test_statistics_seed_is_stable(self):
  self.assertEqual(a.stable_seed('hypothesis','method_specific','X'),a.stable_seed('hypothesis','method_specific','X'))
 def test_holm_monotone(self):
  x=a.holm([.01,.02,.5]);self.assertTrue(all(0<=v<=1 for v in x))
 def test_self_orchestrator_ancestors_are_not_conflicts(self):
  cmd=f'python -u {pf.ORCHESTRATOR_PATH}'
  self.assertFalse(pf.is_conflicting_process(101,cmd,999,{101,102}))
  self.assertFalse(pf.is_conflicting_process(102,cmd,999,{101,102}))
 def test_separate_orchestrator_is_still_blocked(self):
  cmd=f'python -u {pf.ORCHESTRATOR_PATH}'
  self.assertTrue(pf.is_conflicting_process(201,cmd,999,{101,102}))
 def test_training_and_analysis_processes_are_blocked(self):
  self.assertTrue(pf.is_conflicting_process(301,'python r4_train_cell.py',999,set()))
  self.assertTrue(pf.is_conflicting_process(302,'python r4_analysis.py',999,set()))
 def test_unrelated_and_current_processes_are_not_blocked(self):
  self.assertFalse(pf.is_conflicting_process(999,'python r4_analysis.py',999,set()))
  self.assertFalse(pf.is_conflicting_process(400,'notepad.exe',999,set()))

if __name__=='__main__':unittest.main(verbosity=2)
