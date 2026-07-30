import sys, tempfile, unittest
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from r4_common import ContextualMLP, class_weights_from_y, local_utility, model_to_vector, set_determinism, train_local_model, vector_to_model
from r4_data import dirichlet_assignment, natural_source_group_assignment
from r4_locked_method import Candidate, select_clients
from r4_common import load_yaml
CFG=load_yaml(ROOT/'locks/R2_METHOD_LOCK/config/R2_METHOD_LOCK.yaml')
class TestR4Core(unittest.TestCase):
 def test_manifest_method_lock(self):
  self.assertFalse(CFG['novelty_enabled']); self.assertEqual(CFG['eligibility']['minimum_utility_rows'],20)
 def test_hard_budget_and_empty_round(self):
  cs=[Candidate(0,.2,100,180,20,10,10),Candidate(1,.1,100,180,20,10,10)]
  ids,d=select_clients('Utility-Only',cs,100,42,CFG); self.assertLessEqual(d['used_bytes'],100); self.assertEqual(len(ids),1)
  cs=[Candidate(0,-.2,100,180,20,10,10)];ids,d=select_clients('Utility-Only',cs,100,42,CFG);self.assertEqual(ids,[]);self.assertTrue(d['empty_round'])
 def test_dirichlet_deterministic_complete(self):
  y=np.array([0]*1000+[1]*3000,dtype=np.int8); a=dirichlet_assignment(y,10,.5,42,200); b=dirichlet_assignment(y,10,.5,42,200)
  np.testing.assert_array_equal(a,b);self.assertTrue((a>=0).all());self.assertEqual(len(a),len(y));self.assertGreaterEqual(np.bincount(a,minlength=10).min(),200)
 def test_natural_groups_not_split(self):
  groups=['a']*10+['b']*8+['c']*5; a=natural_source_group_assignment(groups,3)
  for g in set(groups): self.assertEqual(len(set(a[np.array(groups)==g])),1)
 def test_synthetic_local_training_no_global_mutation(self):
  rng=np.random.default_rng(1);X=rng.normal(size=(256,6)).astype(np.float32);y=(X[:,0]+X[:,1]>.1).astype(np.int8)
  cfg={'learner':{'hidden_units':[8],'dropout':0.0,'deterministic_algorithms':True,'learning_rate':.001,'weight_decay':0.0,'batch_size':32,'local_steps_per_round':2,'gradient_clip_norm':5.0}}
  set_determinism(42);m=ContextualMLP(6,[8],0.0);g=model_to_vector(m);before=g.copy();lv,diag=train_local_model(g,6,X,y,np.arange(200),42,cfg,torch.device('cpu'),class_weights_from_y(y))
  np.testing.assert_array_equal(g,before);self.assertGreater(np.linalg.norm(lv-g),0)
if __name__=='__main__':unittest.main(verbosity=2)
