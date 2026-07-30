import json, sys, tempfile, unittest
from pathlib import Path
import numpy as np, yaml, pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r4_train_cell import train_cell
class TestR4Integration(unittest.TestCase):
 def test_two_round_cell_end_to_end(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);cache=td/'cache'/'CICIDS2017';cache.mkdir(parents=True);assign=td/'assign'/'CICIDS2017';assign.mkdir(parents=True);out=td/'out'
   rng=np.random.default_rng(7);features=6
   splits={}
   for name,n in [('train',2400),('validation',300),('test',300)]:
    X=rng.normal(size=(n,features)).astype(np.float32);y=(X[:,0]+.4*X[:,1]>.0).astype(np.int8)
    xp=cache/f'{name}_X.dat';mm=np.memmap(xp,dtype=np.float32,mode='w+',shape=X.shape);mm[:]=X;mm.flush();del mm
    yp=cache/f'{name}_y.npy';np.save(yp,y);splits[name]={'X_file':xp.name,'y_file':yp.name,'rows':n}
   (cache/'cache_manifest.json').write_text(json.dumps({'dataset':'CICIDS2017','n_features':features,'splits':splits}))
   a=np.repeat(np.arange(10,dtype=np.int16),240);ap=assign/'client_assignment_alpha_0.5_seed_42_clients_10.npy';np.save(ap,a)
   base=yaml.safe_load((ROOT/'config/R4_EXECUTION_CONFIG.yaml').read_text())
   base['paths'].update({'primary_cache_root':str(td/'cache'),'assignment_root':str(td/'assign'),'output_root':str(out),'legacy_ood_cache':str(cache),'legacy_ood_assignment_root':str(assign)})
   base['learner'].update({'hidden_units':[8],'dropout':0.0,'batch_size':32,'local_steps_per_round':1,'local_utility_fraction':.1,'local_utility_cap_rows':64,'validation_monitor_cap_rows':300,'evaluation_batch_size':256})
   base['machine_lock']['torch_threads_per_cell']=1
   cp=td/'cfg.yaml';cp.write_text(yaml.safe_dump(base,sort_keys=False))
   row={'cell_id':'SYNTHETIC_CELL','dataset_protocol':'CICIDS2017_COVERAGE_R1','method':'Utility-Only','beta':.4,'alpha_dir':.5,'seed':42,'rounds':2}
   train_cell(cp,'primary',row,'cpu')
   d=out/'cells/primary/SYNTHETIC_CELL'
   self.assertTrue((d/'CELL_COMPLETE.json').exists());rm=pd.read_csv(d/'round_metrics.csv');self.assertEqual(len(rm),2);self.assertTrue((rm.used_bytes<=rm.budget_bytes).all())
   tm=json.loads((d/'test_metrics.json').read_text());self.assertIn('method_specific',tm);self.assertIn('pr_auc',tm['method_specific'])
if __name__=='__main__':unittest.main(verbosity=2)
