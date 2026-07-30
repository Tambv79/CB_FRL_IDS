import unittest
import numpy as np
from pathlib import Path
import sys, yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from r2_locked_method import Candidate, marginal_utility, static_eligible, priority_values, select_clients, aggregate_or_keep

CFG=yaml.safe_load((Path(__file__).resolve().parents[1]/"config/R2_METHOD_LOCK.yaml").read_text())

def c(i,u,cost=100,n_u=20,b=10,a=10,n_fit=180,air=None,train=0.0):
    return Candidate(i,u,cost,n_fit,n_u,b,a,air,train)

class TestR2MethodLock(unittest.TestCase):
    def test_marginal_utility(self):
        self.assertAlmostEqual(marginal_utility(.8,.1,.9,.2,.25),.075)
    def test_static_eligibility(self):
        self.assertTrue(static_eligible(c(0,.1),200,CFG))
        self.assertFalse(static_eligible(c(0,.1,n_u=19),200,CFG))
        self.assertFalse(static_eligible(c(0,.1,b=4,a=16),200,CFG))
        self.assertFalse(static_eligible(c(0,.1,cost=201),200,CFG))
    def test_full_denominator(self):
        cs=[c(0,float('nan')),c(1,-1)]
        ids,d=select_clients('Full-Contextual',cs,1,1,CFG)
        self.assertEqual(ids,[0,1])
    def test_nonpositive_not_forced(self):
        ids,d=select_clients('Utility-Only',[c(0,-.1),c(1,0)],200,1,CFG)
        self.assertEqual(ids,[]); self.assertTrue(d['empty_round']); self.assertFalse(d['fallback_used'])
    def test_hard_budget(self):
        cs=[c(0,.4,80),c(1,.3,80),c(2,.2,80)]
        ids,d=select_clients('Utility-Only',cs,160,1,CFG)
        self.assertEqual(ids,[0,1]); self.assertLessEqual(d['used_bytes'],160)
    def test_score_no_second_cost_penalty(self):
        cs=[c(0,.4,100),c(1,.2,50),c(2,.1,200)]
        score=priority_values('CB-Score',cs,CFG)
        hist=priority_values('Historical-DoubleCost',cs,CFG)
        self.assertFalse(np.allclose(score,hist))
    def test_oracle_exact(self):
        cs=[c(0,.9,70),c(1,.8,60),c(2,.7,40)]
        ids,d=select_clients('Oracle-Score-Exact',cs,100,1,CFG)
        self.assertLessEqual(d['used_bytes'],100)
        # Exact oracle should choose the best feasible score subset, deterministically.
        self.assertEqual(ids,sorted(ids))
    def test_random_deterministic(self):
        cs=[c(i,.1,40) for i in range(5)]
        self.assertEqual(select_clients('Random-Budget',cs,120,42,CFG)[0],select_clients('Random-Budget',cs,120,42,CFG)[0])
    def test_empty_round_keeps_global(self):
        g=np.array([1,2],dtype=np.float32)
        new,d=aggregate_or_keep(g,[],[])
        np.testing.assert_array_equal(g,new); self.assertTrue(d['empty_round'])
    def test_novelty_rejected(self):
        bad=dict(CFG); bad['novelty_enabled']=True
        with self.assertRaises(ValueError): select_clients('Utility-Only',[c(0,.1)],200,1,bad)

if __name__=='__main__': unittest.main(verbosity=2)
