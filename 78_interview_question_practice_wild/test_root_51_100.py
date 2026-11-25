import unittest
import copy
import numpy as np
from exact_numeric_51_100 import *
from stopping_075 import stopping_distribution,solve
from sampling_074 import Sampler,reward_interval
from focal_077 import focal_loss

class ExactTests(unittest.TestCase):
 def test_patches(self):
  np.testing.assert_array_equal(patch_embed(np.array([[[[1,2],[3,4]]]]),np.array([[2]]),np.array([1]),1),[[[3],[5],[7],[9]]])
  image=np.arange(2*4*4).reshape(1,2,4,4)
  actual=patch_embed(image,np.eye(8),np.zeros(8),2)
  expected=[image[0,:,r:r+2,c:c+2].reshape(-1) for r in (0,2) for c in (0,2)]
  np.testing.assert_array_equal(actual[0],expected)
 def test_norm(self):
  np.testing.assert_allclose(layer_norm(np.array([1.,2.,3.]),1,0,0),[-np.sqrt(1.5),0,np.sqrt(1.5)])
  np.testing.assert_allclose(softmax(np.array([[1000.,1000.]])),[[.5,.5]])
 def test_attention(self):
  rng=np.random.default_rng(1); x=rng.normal(size=(2,3,4)); w=rng.normal(size=(4,12)); b=rng.normal(size=12)
  out=rng.normal(size=(4,4)); ob=rng.normal(size=4)
  q,k,v=np.split(x@w+b,3,axis=-1)
  heads=[]
  for h in range(2):
   qh=q[:,:,h*2:(h+1)*2]; kh=k[:,:,h*2:(h+1)*2]; vh=v[:,:,h*2:(h+1)*2]
   heads.append(softmax(qh@kh.swapaxes(-1,-2)/np.sqrt(2))@vh)
  expected=np.concatenate(heads,axis=-1)@out+ob
  np.testing.assert_allclose(multi_head_attention(x,w,b,out,ob,2),expected)
 def test_vit(self):
  # Zero branches preserve residual; CLS + position determine exact result.
  D=4; z=lambda *s:np.zeros(s)
  block=dict(ln1_g=np.ones(D),ln1_b=z(D),qkv_w=z(D,3*D),qkv_b=z(3*D),proj_w=z(D,D),proj_b=z(D),ln2_g=np.ones(D),ln2_b=z(D),fc1_w=z(D,7),fc1_b=z(7),fc2_w=z(7,D),fc2_b=z(D))
  p=dict(patch_w=z(4,D),patch_b=z(D),cls=np.arange(D,dtype=float),pos=z(5,D),blocks=[block],ln_g=np.ones(D),ln_b=z(D),head_w=np.eye(D),head_b=z(D))
  images=np.arange(32,dtype=float).reshape(2,1,4,4); before=images.copy(); saved=copy.deepcopy(p)
  result=vit_forward(images,p,2,2)
  np.testing.assert_allclose(result,np.broadcast_to(layer_norm(p['cls'],1,0),(2,D)))
  np.testing.assert_array_equal(images,before)
  np.testing.assert_array_equal(p['cls'],saved['cls'])
 def test_ace(self):
  self.assertAlmostEqual(ace_calc(np.log([.9,.95,1,.5,.55]),[1,1,1,0,1],5),.0375)
  self.assertAlmostEqual(ace_calc(np.log([.99,.3]),[1,0]),.155)
  self.assertEqual(ace_calc(np.array([]),np.array([])),0)
  self.assertEqual(ace_calc(np.array([0.,0.]),np.array([1,1])),0)
 def test_mlp_examples(self):
  m=MLP(2,2,2,1); m.params={k:np.array(v,dtype=float) for k,v in dict(W1=[[1,-1],[2,1]],b1=[0,.5],W2=[[1,0],[-1,1]],b2=[0,-1],W3=[[2],[1]],b3=[.5]).items()}
  x=np.array([[1,2],[1,-1.]]); y=np.array([[7],[1.5]])
  before=copy.deepcopy(m.params)
  np.testing.assert_array_equal(m.forward(x),[[8],[.5]]); self.assertEqual(m.loss(x,y),1)
  g=m.backward(x,y); expected=dict(W3=[[3.5],[.5]],b3=[0],W2=[[10,5],[3,1.5]],b2=[2,1],W1=[[2,-1],[4,-2]],b1=[2,-1])
  for name in g:
   np.testing.assert_array_equal(g[name],expected[name]); np.testing.assert_array_equal(m.params[name],before[name])
  m.step(g,.1); np.testing.assert_allclose(m.params['W1'],[[.8,-.9],[1.6,1.2]])
  m.params=before; m.params['b2']=np.array([1.,0.]); g=m.backward(np.array([[1.,-.5]]),np.array([[1.5]]))
  np.testing.assert_array_equal(g['b2'],[4,0]); np.testing.assert_array_equal(g['W1'],np.zeros((2,2)))
 def test_mlp_finite_differences(self):
  m=MLP(2,3,2,2); rng=np.random.default_rng(5)
  for name in m.params: m.params[name]=rng.normal(size=m.params[name].shape)*.3
  x=rng.normal(size=(4,2)); y=rng.normal(size=(4,2)); grads=m.backward(x,y)
  for name,p in m.params.items():
   for index in np.ndindex(p.shape):
    value=p[index]; p[index]=value+1e-6; plus=m.loss(x,y); p[index]=value-1e-6; minus=m.loss(x,y); p[index]=value
    self.assertAlmostEqual(grads[name][index],(plus-minus)/2e-6,places=6)
 def test_stopping(self):
  for p,expected,mean in [([0],[1],1),([1,0,0],[1,0,0],1),([0,0,0],[0,0,1],3),([.5,.25,.9],[.5,.125,.375],1.875)]:
   pmf,e=stopping_distribution(p); np.testing.assert_allclose(pmf,expected); self.assertEqual(e,mean)
  data=dict(N=2,p=[.5,0],decisions=['A','B'],adversaries=[dict(p=[1,0],rewards=[[2,2],[1,1]]),dict(p=[0,0],rewards=[[0,0],[1,1]])])
  self.assertEqual(solve(data)['decision'],'B'); self.assertEqual(solve(data)['worst_expected_reward'],1)
  with self.assertRaises(ValueError): stopping_distribution([1.1])
 def test_sampler(self):
  sampler=Sampler(lambda s:[0],lambda ids:''.join(map(str,ids)),lambda ids:[-1000,0],eos_id=1)
  self.assertEqual(sampler.sample('x'),'')
  mean,interval=reward_interval(['a','b'],lambda s:7,replicates=100)
  self.assertEqual((mean,interval),(7.,(7.,7.)))
 def test_focal(self):
  losses=focal_loss([1,0,1],[.9,.1,.2],reduction='none')
  self.assertAlmostEqual(sum(losses)/3,.08618789038201145,places=8)
  self.assertAlmostEqual(focal_loss([1,0],[.8,.3],gamma=0),(-.25*np.log(.8)-.75*np.log(.7))/2)
  self.assertTrue(np.isfinite(focal_loss([1,0],[0,1])))

if __name__=='__main__':unittest.main()
