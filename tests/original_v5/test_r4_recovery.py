import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from r4_common import stable_seed
from r4_manifest_schema import validate_manifests
from r4_train_cell import deterministic_malicious_ids, train_cell
from r4_validate import validate_cell

class TestR4Recovery(unittest.TestCase):
    def test_manifest_schema_and_locked_relation(self):
        self.assertEqual(validate_manifests(ROOT),[])

    def test_forged_clean_assigns_malicious_clients(self):
        cfg=yaml.safe_load((ROOT/'config/R4_EXECUTION_CONFIG.yaml').read_text())
        self.assertEqual(deterministic_malicious_ids('robustness','clean','honest',42,cfg),set())
        forged=deterministic_malicious_ids('robustness','clean','forged_inflated',42,cfg)
        self.assertEqual(len(forged),3)
        self.assertEqual(forged,deterministic_malicious_ids('robustness','clean','forged_inflated',42,cfg))
        self.assertEqual(len(deterministic_malicious_ids('robustness','sign-flip','honest',42,cfg)),3)


    def test_forged_clean_robustness_cell_end_to_end(self):
        with tempfile.TemporaryDirectory() as td_raw:
            td=Path(td_raw);cache=td/'cache'/'CICIDS2017';cache.mkdir(parents=True);assign=td/'assign'/'CICIDS2017';assign.mkdir(parents=True);out=td/'out'
            rng=np.random.default_rng(71);features=5;splits={}
            for name,n in [('train',2400),('validation',300),('test',300)]:
                X=rng.normal(size=(n,features)).astype(np.float32);y=(X[:,0]+.3*X[:,1]>0).astype(np.int8)
                xp=cache/f'{name}_X.dat';mm=np.memmap(xp,dtype=np.float32,mode='w+',shape=X.shape);mm[:]=X;mm.flush();del mm
                np.save(cache/f'{name}_y.npy',y);splits[name]={'X_file':xp.name,'y_file':f'{name}_y.npy','rows':n}
            (cache/'cache_manifest.json').write_text(json.dumps({'dataset':'CICIDS2017','n_features':features,'splits':splits}))
            np.save(assign/'client_assignment_alpha_0.5_seed_42_clients_10.npy',np.repeat(np.arange(10,dtype=np.int16),240))
            cfg=yaml.safe_load((ROOT/'config/R4_EXECUTION_CONFIG.yaml').read_text());cfg['paths'].update({'primary_cache_root':str(td/'cache'),'assignment_root':str(td/'assign'),'output_root':str(out),'legacy_ood_cache':str(cache),'legacy_ood_assignment_root':str(assign)});cfg['learner'].update({'hidden_units':[8],'dropout':0.0,'batch_size':32,'local_steps_per_round':1,'local_utility_fraction':.1,'local_utility_cap_rows':64,'validation_monitor_cap_rows':300,'evaluation_batch_size':256});cfg['machine_lock']['torch_threads_per_cell']=1
            cp=td/'cfg.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
            row={'cell_id':'ROB_SYNTH','dataset_protocol':'CICIDS2017_COVERAGE_R1','selection':'CB-Score','aggregation':'Norm-Clipped-Averaging','model_attack':'clean','reported_utility':'forged_inflated','beta':.4,'seed':42,'rounds':2}
            train_cell(cp,'robustness',row,'cpu')
            d=out/'cells/robustness/ROB_SYNTH';sr=pd.read_csv(d/'selection_records.csv')
            self.assertEqual(sr.groupby('round').malicious_flag.sum().tolist(),[3,3])
            mal=sr[sr.malicious_flag==1]
            self.assertTrue(np.allclose(mal.reported_utility-mal.utility,1.0))
            lock=json.loads((d/'configuration_lock.json').read_text());self.assertEqual(lock['execution_amendment_id'],'R4-EXEC-AMEND-02-RECOVERY')
            self.assertEqual(validate_cell(d,2,False),[])

    def test_stable_seed_ignores_python_hash_randomization(self):
        code=f"import sys;sys.path.insert(0,{str(ROOT/'src')!r});from r4_common import stable_seed;print(stable_seed('a',1,2.0))"
        vals=[]
        for hs in ['1','987654']:
            env=dict(os.environ);env['PYTHONHASHSEED']=hs
            vals.append(subprocess.check_output([sys.executable,'-c',code],env=env,text=True).strip())
        self.assertEqual(vals[0],vals[1])
        self.assertEqual(int(vals[0]),stable_seed('a',1,2.0))

if __name__=='__main__':unittest.main(verbosity=2)
