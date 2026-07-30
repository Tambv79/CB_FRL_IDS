import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r4_analysis import exact_sign_flip,holm,bootstrap_ci
class TestStatistics(unittest.TestCase):
 def test_exact(self): self.assertLess(exact_sign_flip(np.ones(10)),.01)
 def test_holm_monotone(self):
  p=np.array([.001,.02,.04]);a=holm(p);self.assertTrue(np.all(a>=p));self.assertTrue(np.all(a<=1))
 def test_bootstrap_repro(self): self.assertEqual(bootstrap_ci(np.arange(10),100,4),bootstrap_ci(np.arange(10),100,4))
if __name__=='__main__':unittest.main(verbosity=2)
