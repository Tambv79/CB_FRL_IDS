import sys,unittest
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from r4_train_cell import aggregate
class TestRobustness(unittest.TestCase):
 def test_norm_clipping_bounds_outlier(self):
  g=np.zeros(3,dtype=np.float32);c=[{'client_id':0,'delta':np.array([1.,0,0],np.float32),'n_fit':100},{'client_id':1,'delta':np.array([1.,0,0],np.float32),'n_fit':100},{'client_id':2,'delta':np.array([100.,0,0],np.float32),'n_fit':100}]
  cfg=yaml.safe_load((ROOT/'config/R4_EXECUTION_CONFIG.yaml').read_text());fed,_=aggregate(g,c,[0,1,2],'FedAvg',cfg);clip,d=aggregate(g,c,[0,1,2],'Norm-Clipped-Averaging',cfg);self.assertLess(clip[0],fed[0]);self.assertIsNotNone(d['clip_threshold'])
if __name__=='__main__':unittest.main(verbosity=2)
