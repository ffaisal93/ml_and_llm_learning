"""Forward/manual-gradient/reference/autograd checks for six optional Torch studies."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from review_torch_six_supplements import (
    attention_torch,multi_head_attention_torch,grouped_query_attention_torch,
    layernorm_forward_torch,layernorm_backward_torch,masked_cross_entropy_torch,
    sigmoid_torch,sigmoid_derivative_torch,dense_layer_forward_torch,dense_layer_backward_torch,
)

ROOT=Path(__file__).parent

def manual(i):
    with redirect_stdout(io.StringIO()):return runpy.run_path(str(ROOT/f'review_w{i:03}_implementation.py'))

def leaf(value):return torch.tensor(value,dtype=torch.float64,requires_grad=True)

class TorchSupplementTests(unittest.TestCase):
    def setUp(self):self.rng=np.random.default_rng(602);torch.manual_seed(602)
    def compare(self,a,b,atol=1e-10):
        expected=b.detach().numpy() if torch.is_tensor(b) else np.asarray(b)
        np.testing.assert_allclose(a.detach().numpy(),expected,atol=atol,rtol=atol)

    def test_w002_forward_masks_and_projection_gradients(self):
        ref=manual(2);x=self.rng.normal(size=(3,4));weights=[self.rng.normal(size=(4,d)) for d in (2,2,3)]
        mask=np.tril(np.ones((3,3),bool));args=[leaf(x)]+[leaf(w) for w in weights]
        actual=attention_torch(*args,mask)
        self.compare(actual,ref['attention'](x,*weights,mask))
        self.assertTrue(torch.autograd.gradcheck(lambda *a:attention_torch(*a,mask),tuple(args),atol=1e-5))
        self.compare(attention_torch(*args,np.eye(3,dtype=bool)),x@weights[2])
        with self.assertRaises(ValueError):attention_torch(*args,np.zeros((3,3),bool))
        q=torch.eye(2,dtype=torch.float64)*1000;w=torch.eye(2,dtype=torch.float64)
        self.assertTrue(bool(torch.isfinite(attention_torch(q,w,w,w)).all()))

    def test_w003_no_projection_head_order_mask_and_gradients(self):
        ref=manual(3);arrays=[self.rng.normal(size=(2,3,8)) for _ in range(3)]
        args=[leaf(a) for a in arrays];mask=np.tril(np.ones((3,3),bool))
        self.compare(multi_head_attention_torch(*args,4,mask),ref['multi_head_attention'](*arrays,4,mask))
        small=tuple(leaf(self.rng.normal(size=(1,2,4))) for _ in range(3))
        self.assertTrue(torch.autograd.gradcheck(lambda *a:multi_head_attention_torch(*a,2,[[1,0],[1,1]]),small,atol=1e-5))
        base=multi_head_attention_torch(*args,4,mask)
        k2=args[1].detach().clone();v2=args[2].detach().clone();k2[:,2]+=100;v2[:,2]+=100
        self.compare(multi_head_attention_torch(args[0],k2,v2,4,mask)[:,:2],base[:,:2].detach())
        with self.assertRaises(ValueError):multi_head_attention_torch(*args,3)
        with self.assertRaises(ValueError):multi_head_attention_torch(*args,4,torch.zeros((3,3),dtype=torch.bool))

    def test_w004_mapping_shared_kv_forward_and_gradient_sum(self):
        ref=manual(4);arrays=[self.rng.normal(size=(1,4,2,2)),self.rng.normal(size=(1,2,3,2)),self.rng.normal(size=(1,2,3,2))]
        args=tuple(leaf(a) for a in arrays)
        actual=grouped_query_attention_torch(*args)
        self.compare(actual,ref['grouped_query_attention'](*arrays))
        kr=args[1].repeat_interleave(2,dim=1);vr=args[2].repeat_interleave(2,dim=1)
        expected=(args[0]@kr.transpose(-1,-2)/(2**.5)).softmax(dim=-1)@vr
        upstream=torch.randn_like(actual)
        ga=torch.autograd.grad((actual*upstream).sum(),args,retain_graph=True)
        ge=torch.autograd.grad((expected*upstream).sum(),args)
        for a,b in zip(ga,ge):self.compare(a,b)
        self.assertTrue(torch.autograd.gradcheck(grouped_query_attention_torch,args,atol=1e-5))
        q=torch.zeros((1,4,2,1),dtype=torch.float64,requires_grad=True)
        k=torch.zeros((1,2,3,1),dtype=torch.float64,requires_grad=True)
        v=leaf(np.array([[[[2.],[2.],[2.]],[[9.],[9.],[9.]]]]))
        result=grouped_query_attention_torch(q,k,v)
        self.compare(result[0,:2],np.full((2,2,1),2.));self.compare(result[0,2:],np.full((2,2,1),9.))
        result.sum().backward();self.compare(v.grad,np.full((1,2,3,1),4/3))

    def test_w007_forward_manual_backward_and_autograd(self):
        ref=manual(7);x=self.rng.normal(size=(3,4));gamma=self.rng.normal(size=4);beta=self.rng.normal(size=4);dout=self.rng.normal(size=(3,4))
        args=tuple(leaf(a) for a in (x,gamma,beta));up=torch.tensor(dout,dtype=torch.float64)
        out,cache=layernorm_forward_torch(*args)
        y,manual_cache=ref['layernorm_forward'](x,gamma,beta);self.compare(out,y)
        explicit=layernorm_backward_torch(up,cache);expected=ref['layernorm_backward'](dout,manual_cache)
        automatic=torch.autograd.grad((out*up).sum(),args)
        for a,b,c in zip(explicit,expected,automatic):self.compare(a,b);self.compare(a,c)
        self.assertTrue(torch.autograd.gradcheck(lambda *a:layernorm_forward_torch(*a)[0],args,atol=1e-5))

    def test_w007_offset_constant_and_single_feature(self):
        ref=manual(7)
        for x in (self.rng.normal(size=(2,4))+1e8,np.full((2,4),2.),np.array([[5.],[8.]])):
            D=x.shape[1];g=np.ones(D);b=np.arange(D,dtype=float)
            args=tuple(leaf(a) for a in (x,g,b));y,cache=layernorm_forward_torch(*args)
            self.compare(y,ref['layernorm_forward'](x,g,b)[0],atol=1e-7)
            up=torch.arange(x.size,dtype=torch.float64).reshape(x.shape)
            explicit=layernorm_backward_torch(up,cache);auto=torch.autograd.grad((y*up).sum(),args)
            for a,c in zip(explicit,auto):self.compare(a,c,atol=1e-6)
            if D==1:self.compare(explicit[0],np.zeros_like(x))

    def test_w008_forward_smoothing_denominator_and_gradient(self):
        ref=manual(8);z=self.rng.normal(size=(4,5));labels=np.array([1,0,3,2]);mask=np.array([1,0,1,0]);epsilon=.2
        logits=leaf(z);loss=masked_cross_entropy_torch(logits,labels,mask,epsilon)
        self.compare(loss,ref['masked_cross_entropy'](z,labels,mask,epsilon))
        independent=F.cross_entropy(logits,torch.tensor(labels),reduction='none',label_smoothing=epsilon)[torch.tensor(mask).bool()].mean()
        self.compare(loss,independent)
        loss.backward();target=torch.full_like(logits,epsilon/5)
        target[torch.arange(4),torch.tensor(labels)]+=1-epsilon
        expected=(logits.detach().softmax(-1)-target)*torch.tensor(mask)[:,None]/mask.sum()
        self.compare(logits.grad,expected)
        self.assertTrue(torch.autograd.gradcheck(lambda a:masked_cross_entropy_torch(a,labels,mask,epsilon),(leaf(z),),atol=1e-5))

    def test_w008_all_masked_extremes_and_validation(self):
        z=leaf([[1000.,-1000.]])
        self.assertEqual(masked_cross_entropy_torch(z,[1],[1],0).item(),2000)
        zero=masked_cross_entropy_torch(z,[1],[0],.2)
        self.assertTrue(zero.requires_grad);self.assertEqual(zero.item(),0.)
        zero.backward();self.compare(z.grad,np.zeros((1,2)))
        loss=masked_cross_entropy_torch(torch.zeros((3,4),dtype=torch.float64),[0,1,2],[1,1,1],.8)
        self.assertAlmostEqual(loss.item(),np.log(4))
        with self.assertRaises(ValueError):masked_cross_entropy_torch(z,[2],[0])
        with self.assertRaises(ValueError):masked_cross_entropy_torch(z,[1],[.5])
        with self.assertRaises(ValueError):masked_cross_entropy_torch(z,[1],[1],1.)

    def test_w147_forward_manual_backward_and_autograd(self):
        ref=manual(147);arrays=[self.rng.normal(size=(3,2)),self.rng.normal(size=(2,4)),self.rng.normal(size=4)]
        args=tuple(leaf(a) for a in arrays);up=leaf(self.rng.normal(size=(3,4)))
        y=dense_layer_forward_torch(*args);self.compare(y,ref['dense_layer_forward'](*arrays))
        explicit=dense_layer_backward_torch(up,*args)
        expected=ref['dense_layer_backward'](up.detach().numpy(),*arrays)
        auto=torch.autograd.grad((y*up).sum(),args)
        for a,b,c in zip(explicit,expected,auto):self.compare(a,b);self.compare(a,c)
        self.assertTrue(torch.autograd.gradcheck(dense_layer_forward_torch,args,atol=1e-5))
        zeros=dense_layer_backward_torch(torch.zeros_like(up),*args)
        self.assertTrue(all(bool((a==0).all()) for a in zeros))

    def test_w147_sigmoid_input_convention_and_example(self):
        z=leaf([-1000.,0.,1000.]);s=sigmoid_torch(z)
        self.compare(s,[0.,.5,1.]);self.compare(sigmoid_derivative_torch(z),[0.,.25,0.])
        self.compare(torch.autograd.grad(s.sum(),z)[0],sigmoid_derivative_torch(z).detach())
        X=leaf([[.5,.2]]);W=leaf([[.3,.8],[.5,.2]]);b=leaf([.1,.4])
        self.compare(dense_layer_forward_torch(X,W,b),[[.5866175789,.698465216]],atol=1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
