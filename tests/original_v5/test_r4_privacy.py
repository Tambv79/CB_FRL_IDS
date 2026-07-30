import sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r4_privacy import run_cell
class TestPrivacy(unittest.TestCase):
 def test_group_disjoint_privacy_cell(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);out=td/'out';base=yaml.safe_load((ROOT/'config/R4_EXECUTION_CONFIG.yaml').read_text());base['paths']['output_root']=str(out);base['privacy']['group_bootstrap_resamples']=20;base['privacy']['attacker_hyperparameters']['LogisticRegression']['max_iter']=200
   dataset='CICIDS2017_COVERAGE_R1'
   rng=np.random.default_rng(4)
   for seed in [42,123,777,2026,3407,11,314,2718,9001,2027]:
    cid=f'PRIMARY__{dataset}__CB-Score__b0.4__seed{seed}';d=out/'cells/primary'/cid;d.mkdir(parents=True)
    rows=[]
    for rnd in range(30):
     for client in range(10):
      size=200+client*10+(seed%7);ratio=.05+.08*client
      rows.append({'round':rnd,'client_id':client,'reported_utility':float(rng.normal(client/10,.1)),'wire_bytes':1000+client*100,'n_fit':int(size*.9),'selected':int(client%2==0),'static_eligible':1,'actual_client_size':size,'actual_client_attack_ratio':ratio})
    pd.DataFrame(rows).to_csv(d/'selection_records.csv',index=False)
   row={'cell_id':'PRIV_TEST','dataset_protocol':dataset,'metadata_view':'Raw','attacker':'LogisticRegression','target':'client_size_above_seed_median','privacy_split_seed':42,'group_folds':5}
   r=run_cell(base,row);self.assertEqual(r['group_overlap_max'],0);self.assertEqual(r['groups'],10);self.assertTrue(0<=r['mean_auc']<=1);self.assertLessEqual(r['auc_ci_low'],r['auc_ci_high'])
if __name__=='__main__':unittest.main(verbosity=2)
