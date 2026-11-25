"""Behavior checks for exact answers W151-W200, using only explicit fixtures."""
import ast
from collections import Counter
from fractions import Fraction
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

vm=load('review_w157_implementation')
gauss=load('review_w160_implementation')
spam=load('review_w194_implementation')

class FakeClock:
    def __init__(self): self.now=0.0
    def clock(self): return self.now
    def sleep(self,t): self.now+=t

class ReviewChecks(unittest.TestCase):
    def test_vm_all_provider_completion_and_stop(self):
        for name in ('crusoe','lambda','nebius'):
            with self.subTest(provider=name):
                timer=FakeClock()
                adapter=vm.DemoAdapter(name,asynchronous=name!='lambda')
                f=vm.Facade({name:adapter},timer.clock,timer.sleep)
                item=f.create(name,'DEMO-GPU',2,'unit','region')
                self.assertEqual((item.provider,item.count,item.state),(name,2,'RUNNING'))
                self.assertEqual(f.get(name,item.id),item)
                self.assertEqual(f.stop(name,item.id).state,'STOPPED')
                self.assertEqual(f.stop(name,item.id).state,'STOPPED')

    def test_vm_partial_failure_timeout_errors(self):
        good=vm.DemoAdapter('lambda'); good.create('DEMO',1,'','')
        f=vm.Facade({'lambda':good,'nebius':vm.DemoAdapter('nebius',unavailable=True)})
        items,errors=f.list()
        self.assertEqual(len(items),1); self.assertIn('nebius',errors)
        with self.assertRaisesRegex(ValueError,'positive'): f.create('lambda','D',0)
        with self.assertRaisesRegex(vm.ProviderError,'not found'): f.get('lambda','missing')
        with self.assertRaisesRegex(vm.ProviderError,'quota'): f.create('lambda','D',9)
        clock=FakeClock(); stuck=vm.DemoAdapter('crusoe',asynchronous=True,stuck=True)
        f=vm.Facade({'crusoe':stuck},clock.clock,clock.sleep)
        with self.assertRaisesRegex(TimeoutError,'crusoe-1'): f.create('crusoe','D',1,timeout=.5)
        self.assertEqual(len(stuck.instances),1)
        bad=vm.DemoAdapter('nebius',asynchronous=True,fail_operation=True)
        with self.assertRaisesRegex(vm.ProviderError,'operation failed'):
            vm.Facade({'nebius':bad}).create('nebius','D',1)

    def test_vm_cli_commands_and_invalid_count(self):
        script=str(ROOT/'review_w157_implementation.py')
        cases=[(['list'],0),(['create','--provider','lambda','--gpu','D','--count','2'],0),
               (['get','--provider','crusoe','crusoe-demo'],0),
               (['stop','--provider','nebius','nebius-demo'],0),
               (['create','--provider','lambda','--gpu','D','--count','0'],1),
               (['list','--provider','missing'],2)]
        for args,code in cases:
            result=subprocess.run([sys.executable,script]+args,capture_output=True,text=True)
            self.assertEqual(result.returncode,code,(args,result.stderr))

    def test_box_muller_exact_contract(self):
        cases=[((.5,0),1.1774100225154747),((.5,.5),-1.1774100225154747),
               ((0,.3),0.),((.75,0),1.6651092223153954)]
        for draws,want in cases:
            calls=[]; values=iter(draws)
            def uniform():
                calls.append(1); return next(values)
            gauss.uniform=uniform
            value=gauss.sample_gaussian()
            self.assertIs(type(value),float)
            self.assertAlmostEqual(value,want,delta=1e-9)
            self.assertEqual(len(calls),2)
        values=iter([math.nextafter(1.,0.),.2]); gauss.uniform=lambda:next(values)
        self.assertTrue(math.isfinite(gauss.sample_gaussian()))

    def test_box_muller_seeded_distribution(self):
        rng=random.Random(731); gauss.uniform=rng.random
        values=[gauss.sample_gaussian() for _ in range(200_000)]
        self.assertLess(abs(statistics.mean(values)),.012)
        self.assertLess(abs(statistics.pvariance(values)-1),.025)
        self.assertLess(abs(sum(-1<x<1 for x in values)/len(values)-.682689),.008)

    def test_order_statistics_exact_integral(self):
        # Expand (v-u)^(n-2), integrate each polynomial term exactly.
        for n in range(2,9):
            mixed=sum((Fraction((-1)**k*math.comb(n-2,k),(k+2)*(n+2))
                       for k in range(n-1)),Fraction())*n*(n-1)
            self.assertEqual(mixed,Fraction(1,n+2))
            self.assertEqual(mixed-Fraction(n,(n+1)**2),Fraction(1,(n+1)**2*(n+2)))

    def test_mutual_maximum_exact_enumeration(self):
        for n in (2,3,4):
            edges=list(itertools.combinations(range(n),2)); total=0
            orders=0
            for weights in itertools.permutations(range(len(edges))):
                best=[None]*n
                for edge,w in zip(edges,weights):
                    for a in edge:
                        if best[a] is None or w>best[a][0]: best[a]=(w,edge)
                total+=sum(best[a][1]==(a,b) and best[b][1]==(a,b) for a,b in edges)
                orders+=1
            self.assertEqual(Fraction(total,orders),Fraction(n*(n-1),2*(2*n-3)))

    def test_deepseek_computed_figures(self):
        mla=(512+64)*2; mha=2*128*128*2; length=131072
        self.assertEqual((mla,mha),(1152,65536))
        self.assertEqual((mla*length,mha*length),(150994944,8589934592))
        self.assertEqual((mla*length*61,mha*length*61),(9210691584,523986010112))
        self.assertEqual(Fraction(mha,mla),Fraction(512,9))
        self.assertAlmostEqual(100*37/671,5.514157973174367)
        self.assertEqual([Fraction(15,m+15) for m in (16,64,256)],
                         [Fraction(15,31),Fraction(15,79),Fraction(15,271)])
        self.assertAlmostEqual(.9**8+.1**8,.43046722)
        self.assertAlmostEqual(.99**8+.01**8,.9227446944279201)

    def test_spam_class_and_metrics(self):
        detector=spam.SpamDetector(lambda f:.99,threshold=.99)
        mail=spam.Email('one','hello','body',.5,False)
        self.assertEqual(detector.classify(mail)['action'],'QUARANTINE')
        self.assertEqual(spam.SpamDetector(lambda f:.98).classify(mail)['action'],'DELIVER')
        self.assertIn('\n',detector.features(mail)['text'])
        for p in (float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError): spam.SpamDetector(lambda f:p).classify(mail)
        with self.assertRaises(ValueError): spam.SpamDetector(lambda f:.5,2)
        tp,fp,fn,tn=900,90,100,8910
        self.assertAlmostEqual(tp/(tp+fp),.90909090909)
        self.assertAlmostEqual(tp/(tp+fn),.9)
        self.assertAlmostEqual(2*tp/(2*tp+fp+fn),.904522613065)
        self.assertAlmostEqual(fp/(fp+tn),.01)

    def test_ticket_atomic_claim_race(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'claims.db')
            con=sqlite3.connect(path)
            con.execute('CREATE TABLE ticket(id TEXT PRIMARY KEY,owner TEXT,state TEXT,version INTEGER)')
            con.execute("INSERT INTO ticket VALUES ('t',NULL,'OPEN',0)"); con.commit(); con.close()
            barrier=threading.Barrier(2); results=[]
            def claim(agent):
                conn=sqlite3.connect(path,timeout=5)
                barrier.wait()
                cursor=conn.execute("UPDATE ticket SET owner=?,state='CLAIMED',version=version+1 WHERE id='t' AND owner IS NULL AND state='OPEN' AND version=0",(agent,))
                conn.commit(); results.append(cursor.rowcount); conn.close()
            threads=[threading.Thread(target=claim,args=(a,)) for a in ('a','b')]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(sorted(results),[0,1])
            con=sqlite3.connect(path)
            self.assertEqual(con.execute('SELECT version,state FROM ticket').fetchone(),(1,'CLAIMED'))
            con.close()

    def test_ticket_bucket_corrections_and_boundary(self):
        ledger={}
        def apply(key,version,timestamp,duration):
            if key not in ledger or ledger[key][0]<version:
                ledger[key]=(version,timestamp,duration)
        apply('a',1,10,10); apply('a',1,10,10)
        apply('b',1,20,10)
        for i in range(8): apply(f'c{i}',1,70,5)
        self.assertEqual(sum(x[2] for x in ledger.values())/len(ledger),6)
        apply('a',2,10,20); apply('a',1,10,10)
        self.assertEqual(sum(x[2] for x in ledger.values())/len(ledger),7)
        chosen=[x for x in ledger.values() if 20<=x[1]<70]
        self.assertEqual(len(chosen),1)

    def test_lease_and_fenced_completion_model(self):
        # Explicit small state model, not a production scheduler implementation.
        job={'state':'ACTIVE'}
        stage={'state':'READY','attempt':0,'token':None,'expiry':None}
        def claim(now):
            if job['state']!='ACTIVE' or stage['state']!='READY': return None
            stage.update(state='RUNNING',attempt=stage['attempt']+1,
                         token=stage['attempt']+1,expiry=now+120)
            return stage['token']
        def renew(token,now):
            if job['state']!='ACTIVE' or stage['state']!='RUNNING' or stage['token']!=token or now>=stage['expiry']: return False
            stage['expiry']=now+120; return True
        def complete(token,now):
            if job['state']!='ACTIVE' or stage['state']!='RUNNING' or token!=stage['token'] or now>=stage['expiry']: return False
            stage['state']='SUCCEEDED'; return True
        first=claim(0)
        self.assertIsNone(claim(1)); self.assertFalse(renew(first,120))
        stage.update(state='READY',token=None)
        second=claim(121)
        self.assertFalse(complete(first,122)); self.assertTrue(renew(second,151))
        self.assertTrue(complete(second,152)); self.assertFalse(complete(second,153))
        for state in ('FAILING','CANCELING','FAILED','CANCELED'):
            job['state']=state; stage.update(state='RUNNING',token=9,expiry=500)
            self.assertFalse(complete(9,200)); self.assertFalse(renew(9,200))

    def test_answer_python_and_assets(self):
        import re
        for i in range(151,201):
            path=ROOT/'questions'/f'W{i:03}.md'; text=path.read_text()
            self.assertIn(f'# W{i:03}.',text)
            for code in re.findall(r'```python\n(.*?)```',text,re.S): ast.parse(code)
            for ref in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
                asset=(path.parent/ref).resolve()
                self.assertTrue(asset.exists(),asset)
                if asset.suffix=='.svg': ET.parse(asset)

if __name__=='__main__': unittest.main(verbosity=2)
