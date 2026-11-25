"""Explicit-fixture behavior checks, actual process election, exact probability calculations."""
import ast
from collections import Counter,deque
from contextlib import redirect_stdout
from fractions import Fraction
from functools import lru_cache,partial
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

tree=load('review_w253_implementation');gaussian=load('review_w287_implementation')
images=load('review_w288_implementation');pods=load('review_w290_implementation');pubsub=load('review_w294_implementation')

def fail_on_7(img):
    if int(img.flat[0])==7:raise ValueError('corrupt image')
    return img

def plus_one(img):return np.clip(img.astype(int)+1,0,255).astype(np.uint8)
def double(img):return np.clip(img.astype(int)*2,0,255).astype(np.uint8)
def require_child(img,parent):
    if os.getpid()==parent:raise RuntimeError('pipeline ran in caller')
    return img

def wait_then_return(img):
    time.sleep(float(img.flat[0])*.005);return img

class Pod:
    def __init__(self,name):self.name=name

class FakeAPI:
    def __init__(self,n):self.pods=[Pod(str(i)) for i in range(n)];self.serial=n;self.fail_list=False;self.ambiguous=False;self.creates=0
    def list_pods(self):
        if self.fail_list:raise ConnectionError('API down')
        return list(self.pods)
    def create_pod(self):
        self.serial+=1;self.creates+=1;pod=Pod(str(self.serial));self.pods.append(pod)
        if self.ambiguous:self.ambiguous=False;raise TimeoutError('applied, response lost')
        return pod
    def delete_pod(self,name):self.pods=[p for p in self.pods if p.name!=name]

class ReviewTests(unittest.TestCase):
    def test_tree_lost_duplicate_reordered(self):
        pending=deque();dropped=set();nodes={}
        class TransportNode(tree.Node):
            def send_async(self,to,message):
                key=(self.node_id,to,message.kind,message.request_id)
                if key not in dropped:dropped.add(key);return
                pending.append((self.node_id,to,message));pending.append((self.node_id,to,message))
        for ident,parent,children in [('r',None,['a','b']),('a','r',[]),('b','r',['c','d']),('c','b',[]),('d','b',[])]:nodes[ident]=TransportNode(ident,parent,children)
        buf=io.StringIO()
        with redirect_stdout(buf):
            nodes['r'].call(None,tree.Message('request_count','req'))
            for _ in range(20):
                for node in nodes.values():node.retry_pending()
                while pending:
                    src,to,msg=pending.pop();nodes[to].call(src,msg)
            nodes['r'].call(None,tree.Message('request_count','req'))
        self.assertEqual(buf.getvalue(),'5\n')
        self.assertEqual(nodes['r'].requests['req']['total'],5)

    def test_tree_independent_requests_and_leaf(self):
        root=tree.Node('r',None,[]);buf=io.StringIO()
        with redirect_stdout(buf):
            root.call(None,tree.Message('request_count','a'))
            root.call(None,tree.Message('request_count','b'))
            root.call(None,tree.Message('request_count','a'))
        self.assertEqual(buf.getvalue(),'1\n1\n')

    def test_real_three_process_join_and_failover(self):
        handles=[];ports=[]
        for _ in range(3):
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));ports.append(sock.getsockname()[1])
        def query(port):
            with urlopen(f'http://127.0.0.1:{port}/leader',timeout=.3) as response:return json.load(response)['leader']
        def wait_for(ids,target):
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                try:
                    if all(query(ports[i])==target for i in ids):return
                except Exception:pass
                time.sleep(.05)
            self.fail(f'nodes {ids} did not converge to {target}')
        try:
            for i,port in enumerate(ports):
                cmd=[sys.executable,str(ROOT/'review_w254_implementation.py'),'--id',str(i+1),'--port',str(port),'--timeout','1']
                if i:cmd+=['--seed',f'127.0.0.1:{ports[0]}']
                handles.append(subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
            wait_for([0,1,2],3)
            handles[2].kill();handles[2].wait(timeout=3)
            wait_for([0,1],2)
        finally:
            for handle in handles:
                if handle.poll() is None:handle.terminate()
            for handle in handles:
                try:handle.wait(timeout=3)
                except subprocess.TimeoutExpired:handle.kill();handle.wait()

    def test_gaussian_direct_reference_edges_and_constants(self):
        source=[[0.,20.,90.],[200.,80.,5.]]
        def direct(a,k,sigma):
            h,w=len(a),len(a[0]);r=k//2
            kernel=[(dy,dx,math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))) for dy in range(-r,r+1) for dx in range(-r,r+1)]
            z=sum(v for _,_,v in kernel)
            return [[sum(v*a[min(h-1,max(0,y+dy))][min(w-1,max(0,x+dx))] for dy,dx,v in kernel)/z for x in range(w)] for y in range(h)]
        original=[row[:] for row in source]
        for k in (1,3,5):np.testing.assert_allclose(gaussian.gaussian_filter(source,k,.7),direct(source,k,.7),atol=1e-11)
        self.assertEqual(source,original)
        np.testing.assert_allclose(gaussian.gaussian_filter([[7.]*4]*3,15,10),7)
        self.assertEqual(gaussian.gaussian_filter([[9.]],3,1e-310),[[9.]])
        np.testing.assert_allclose(gaussian.gaussian_filter([[0,100]],3,1),[[27.4068619061,72.5931380939]],atol=1e-8)

    def test_gaussian_cli_and_invalid(self):
        result=subprocess.run([sys.executable,str(ROOT/'review_w287_implementation.py')],input='2 2 1 1\n1 2\n3 4\n',text=True,capture_output=True,check=True)
        self.assertEqual(result.stdout,'1.000000 2.000000\n3.000000 4.000000\n')
        for k,s in [(2,1),(1,0),(1,float('nan'))]:
            with self.assertRaises(ValueError):gaussian.gaussian_filter([[1]],k,s)

    def test_transformation_exact_and_nonmutation(self):
        rgb=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],np.uint8);old=rgb.copy()
        np.testing.assert_array_equal(images.to_grayscale(rgb),[[76,150],[29,255]])
        np.testing.assert_array_equal(rgb,old)
        source=np.array([[10,20],[30,40]],np.uint8)
        np.testing.assert_array_equal(images.resize(source,3,4),[[10,10,20,20],[10,10,20,20],[30,30,40,40]])
        center=np.zeros((3,3),np.uint8);center[1,1]=90
        np.testing.assert_array_equal(images.box_blur(center),np.full((3,3),10,np.uint8))
        np.testing.assert_array_equal(images.box_blur(np.full((1,1,3),255,np.uint8)),np.full((1,1,3),255,np.uint8))
        for transform in (partial(images.resize,height=2,width=2),images.to_grayscale,images.box_blur):
            result=transform(rgb);self.assertEqual(result.dtype,np.uint8);self.assertFalse(np.shares_memory(result,rgb));np.testing.assert_array_equal(rgb,old)
        with self.assertRaises(ValueError):images.resize(rgb,0,2)

    def test_actual_process_pipeline_order_errors(self):
        pipeline=[partial(require_child,parent=os.getpid()),fail_on_7,plus_one,double]
        processor=images.BatchImageProcessor(pipeline,num_workers=2)
        inputs=[np.full((2,2),v,np.uint8) for v in (3,7,10)]
        result,errors=processor.process(inputs)
        np.testing.assert_array_equal(result[0],8);self.assertIsNone(result[1]);np.testing.assert_array_equal(result[2],22)
        self.assertEqual(errors,{1:'ValueError: corrupt image'})
        np.testing.assert_array_equal(inputs[0],3)
        self.assertEqual(processor.process([]),([] ,{}))

    def test_injected_executor_out_of_order_empty_pipeline(self):
        from concurrent.futures import ThreadPoolExecutor
        factories=[]
        def factory(n):factories.append(n);return ThreadPoolExecutor(max_workers=n)
        inputs=[np.full((1,1),i,np.uint8) for i in (9,1,5)]
        result,errors=images.BatchImageProcessor([wait_then_return],3,factory).process(inputs)
        self.assertEqual(factories,[3]);self.assertFalse(errors)
        self.assertEqual([int(a.flat[0]) for a in result],[9,1,5])
        result,_=images.BatchImageProcessor([],2,factory).process(inputs)
        for a,b in zip(result,inputs):np.testing.assert_array_equal(a,b);self.assertFalse(np.shares_memory(a,b))
        with self.assertRaises(ValueError):images.BatchImageProcessor([],0)

    def test_controller_examples_and_ambiguous_timeout(self):
        for initial,want in ((0,2),(6,4),(3,3)):
            api=FakeAPI(initial);c=pods.PodController(api,2,4)
            self.assertEqual(c.reconcile(),want);self.assertEqual(c.reconcile(),want)
        api=FakeAPI(0);api.ambiguous=True;c=pods.PodController(api,2,4)
        with self.assertRaises(TimeoutError):c.reconcile()
        self.assertEqual(c.reconcile(),2);self.assertEqual(api.creates,2)
        api.fail_list=True
        with self.assertRaises(ConnectionError):c.reconcile()
        self.assertEqual(api.creates,2)

    def test_controller_loop_error_backoff_then_stop(self):
        api=FakeAPI(0);api.fail_list=True;c=pods.PodController(api,1,2);stop=threading.Event();errors=[]
        def record(exc):errors.append(str(exc));api.fail_list=False
        t=threading.Thread(target=c.run,args=(stop,),kwargs={'interval':.01,'max_backoff':.03,'on_error':record});t.start()
        deadline=time.monotonic()+2
        while not api.pods and time.monotonic()<deadline:time.sleep(.01)
        stop.set();t.join(1)
        self.assertFalse(t.is_alive());self.assertTrue(errors);self.assertEqual(len(api.pods),1)

    def test_pubsub_priority_self_unsubscribe_and_exceptions(self):
        b=pubsub.Broker();seen=[]
        def low(x):seen.append(('low',x))
        def high(x):seen.append(('high',x));b.unsubscribe(high,'a')
        def bad(x):raise ValueError('boom')
        b.subscribe(low,'a',1);b.subscribe(high,'a',4);b.subscribe(bad,'a',2);b.subscribe(low,'a',1)
        errors=b.publish('a',3)
        self.assertEqual(seen,[('high',3),('low',3)]);self.assertEqual(errors[0][1],'ValueError: boom')
        b.publish('a',4);self.assertEqual(seen[-1],('low',4))
        b.subscribe(low,'b');b.publish('b',5);self.assertEqual(seen[-1],('low',5))

    def test_bag_exact_two_methods(self):
        @lru_cache(None)
        def E(a,b,c):
            if min(a,b,c)==0:return Fraction(a+b+c)
            return (a*E(a-1,b,c)+b*E(a,b-1,c)+c*E(a,b,c-1))/(a+b+c)
        self.assertEqual(E(10,10,10),Fraction(310,77))
        def choose(n,k):return math.comb(n,k) if 0<=k<=n else 0
        tail=sum((Fraction(3*choose(20,k)-3*choose(10,k),choose(30,k)) for k in range(1,21)),Fraction())
        self.assertEqual(tail,Fraction(310,77))

    def test_random_walk_and_renewal_values(self):
        u=[Fraction(0)]*101;u[0]=Fraction(1)
        for s in range(1,101):u[s]=sum((u[s-d] for d in range(1,7) if s>=d),Fraction())/6
        self.assertAlmostEqual(float((u[99]+u[100])/6),.09523809523809523,places=12)
        A=np.eye(81);rhs=np.ones(81)
        for x in range(1,10):
            for y in range(1,10):
                i=(x-1)*9+y-1
                for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0<xx<10 and 0<yy<10:A[i,(xx-1)*9+yy-1]-=.25
        h=np.linalg.solve(A,rhs)
        self.assertLess(np.max(np.abs(A@h-rhs)),1e-12);self.assertAlmostEqual(h[40],29.239374213664465)
        self.assertEqual(Fraction(1,1)/(1+Fraction(2,3)**10),Fraction(59049,60073))

    def test_coffee_exact_discrete_and_continuous(self):
        counts=Counter()
        for a,b,c,d in itertools.product(range(1,6),repeat=4):
            counts['sweep']+=min(a,b)>max(c,d);counts['best']+=max(a,b)>max(c,d)
            counts['best_tie']+=max(a,b)==max(c,d);counts['total']+=a+b>c+d;counts['total_tie']+=a+b==c+d
        wants={'sweep':Fraction(14,125),'best':Fraction(46,125),'best_tie':Fraction(33,125),'total':Fraction(54,125),'total_tie':Fraction(17,125)}
        self.assertEqual({k:Fraction(v,625) for k,v in counts.items()},wants)
        wins=Counter()
        for A in itertools.combinations(range(4),2):wins[sum(a>b for a in A for b in range(4) if b not in A)]+=1
        self.assertEqual(wins,{0:1,1:1,2:2,3:1,4:1})
        self.assertEqual([math.comb(2*n,n) for n in (3,5,10)],[20,252,184756])

    def test_noise_geometric_and_eight_numeric(self):
        self.assertLess(Fraction(95,96),Fraction(99,100));self.assertGreater(Fraction(114,115),Fraction(99,100))
        p=1/6;q=5/6;M=500
        total=sum(p*q**(n-1)/n for n in range(1,M+1))
        self.assertAlmostEqual(total,math.log(6)/5,places=14)
        self.assertLess(q**M/(M+1),1e-40)
        favorable=sum(sum(roll)%5==0 for roll in itertools.product(range(1,7),repeat=5))
        self.assertEqual(favorable,1556)
        runs=sum(1+sum(a!=b for a,b in zip(bits,bits[1:])) for bits in itertools.product((0,1),repeat=9))
        self.assertEqual(Fraction(runs,512),5)
        self.assertEqual(sum(max(roll)==2 for roll in itertools.product(range(1,7),repeat=3)),7)
        self.assertEqual(Fraction(math.comb(48,1),4*math.comb(48,2)+math.comb(48,1)),Fraction(1,95))
        self.assertEqual(sum(2*x-1 for x in range(1,13)),144)
        self.assertAlmostEqual(math.exp(-2)-math.exp(-4),.11701964434787852)

    def test_all_code_and_diagrams(self):
        for i in range(251,301):
            path=ROOT/'questions'/f'W{i:03}.md';text=path.read_text();self.assertIn(f'# W{i:03}.',text)
            answer=text.split('## Worked solution',1)[1]
            for code in re.findall(r'```python\n(.*?)```',answer,re.S):ast.parse(code)
            for ref in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',answer):
                asset=(path.parent/ref).resolve();self.assertTrue(asset.exists(),asset)
                if asset.suffix=='.svg':ET.parse(asset)

if __name__=='__main__':unittest.main(verbosity=2)
