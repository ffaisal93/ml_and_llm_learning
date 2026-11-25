import concurrent.futures
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import solutions_201_250 as s
import numerical_201_250 as num
import data_201_250 as data
import tweet_engagement_201_250 as engagement
import numpy as np
import torch
import pandas as pd

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).parent/'tweet_pipeline_201_250/app.py')
pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)

class Algorithms(unittest.TestCase):
    def test_piecewise_bruteforce(self):
        rng = random.Random(0)
        for n in range(1,25):
            pts = [(rng.randrange(5), rng.randrange(-5,6)) for _ in range(n)]
            t,a,b = s.fit_two_segments(pts)
            actual = sum((y-(a if x<=t else b))**2 for x,y in pts)
            expected = float('inf')
            for cut in sorted(set(x for x,y in pts)):
                left=[y for x,y in pts if x<=cut]; right=[y for x,y in pts if x>cut]
                score = sum((v-sum(left)/len(left))**2 for v in left)
                if right: score += sum((v-sum(right)/len(right))**2 for v in right)
                expected=min(expected,score)
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(s.fit_two_segments([(2,3)]),(2,3.,3.))
    def test_piecewise_sample(self):
        self.assertEqual(s.fit_two_segments([(1,1),(2,2),(3,2),(4,4),(5,5)])[0],3)
    def test_ucb_reference(self):
        self.assertEqual(s.ucb1(3,10,42,[0,0,0]),0)
        self.assertEqual(s.ucb1(3,2,42,[1,1,1]),2)
        rng=random.Random(42); c=[0,0,0]; sums=[0,0,0]; p=[.1,.5,.9]
        for t in range(1,101):
            a=next((a for a in range(3) if not c[a]),None)
            if a is None: a=max(range(3),key=lambda a:sums[a]/c[a]+math.sqrt(2*math.log(t)/c[a]))
            r=int(rng.random()<p[a]);c[a]+=1;sums[a]+=r
        self.assertEqual(s.ucb1(3,100,42,p),sum(sums))
    def test_compiler_examples(self):
        cases=[('res1=var1+var2\nres2=var3-var4\nres3=res2+var5\nres=res1+res3',(16,2)),
               ('res1=10*var2\nres2=var3*100-var4\nres3=res1/2\nres=res2+res3',(27,2)),
               ('a=x+y\nb=x-y\nc=x*y\nr=a+b',(20,3)),
               ('a=x+x\nb=a+a',(8,1)),('a=1\n\nb=2',(2,2)),('',(0,0))]
        for program,expected in cases:self.assertEqual(s.extract_time_and_mem_cost(program),expected)
    def test_crossword_examples(self):
        board=[list('#..#'),list('....'),list('#..#')]
        words=['THIS','THAT','IS','AT','HAT','CAT','IHA','SAT']
        before=[r[:] for r in board]
        self.assertEqual(s.solve_crossword(board,words),[list('#IS#'),list('THAT'),list('#AT#')])
        self.assertEqual(board,before)
        self.assertIsNone(s.solve_crossword(board,[w for w in words if w!='IHA']))
        self.assertEqual(s.solve_crossword([list('Z#..')],['AB']),[list('Z#AB')])
        self.assertIsNone(s.solve_crossword([list('.#..')],['AB']))
        self.assertIsNone(s.solve_crossword([list('XY')],['AB']))
        self.assertEqual(s.solve_crossword([list('..#..')],['AB','AB']),[list('AB#AB')])
    def test_crossword_max_fixture(self):
        alphabet='ABCDEFGH'
        truth=[[alphabet[(r+c)%8] for c in range(8)] for r in range(8)]
        board=[[truth[r][c] if r in (0,7) or c in (0,7) else '.' for c in range(8)] for r in range(8)]
        rng=random.Random(123)
        words=set(''.join(row) for row in truth)
        while len(words)<2000:words.add(''.join(rng.choice(alphabet) for _ in range(8)))
        start=time.monotonic(); result=s.solve_crossword(board,list(words)); elapsed=time.monotonic()-start
        self.assertIsNotNone(result)
        for r in result:self.assertIn(''.join(r),words)
        for c in zip(*result):self.assertIn(''.join(c),words)
        self.assertLess(elapsed,5)
    def test_snapshots(self):
        sn=s.VersionedSocialNetwork();sn.follow(1,2);self.assertEqual(sn.createSnapshot(),0)
        sn.unfollow(1,2);self.assertTrue(sn.isFollowing(1,2,0));self.assertFalse(sn.isFollowing(1,2,1))
        sn.follow(1,2);sn.unfollow(1,2);sn.createSnapshot();sn.follow(1,2)
        self.assertFalse(sn.isFollowing(1,2,1));self.assertTrue(sn.isFollowing(1,2,2))
        self.assertFalse(sn.isFollowing(2,1,0))
        with self.assertRaises(ValueError):sn.isFollowing(1,2,99)
    def test_bots_legacy(self):
        def old(u,t):
            if t.startswith('!help'):return '[help] Available commands: !help, !schedule, !weather'
            if t.startswith('!schedule'):return f'[schedule] Okay {u}, your meeting is scheduled.'
            if t.startswith('!weather'):return '[weather] It is sunny today.'
            if t.startswith('!echo'):return t[5:]
            return '[error] Unknown command.'
        for t in ('','x','!help me','!helpful','!schedule','!weathering','!echo hello','!echoes'):
            self.assertEqual(s.llm_response('u',t),old('u',t))
        server=s.ChatServer();server.register_channel('a',[s.HelpBot()]);server.register_channel('b',[s.EchoBot()])
        server.send_message('a','1','!help');self.assertEqual(server.send_message('b','2','!help'),'[error] Unknown command.')
        self.assertEqual(len(server.channels['a'].messages),1)
        server.register_channel('a',[]);self.assertEqual(server.channels['a'].messages,[])
        with self.assertRaises(ValueError):server.send_message('missing','x','!help')
        server.register_channel('x',[s.HelpBot(),s.HelpBot()]);self.assertEqual(len(server.channels['x'].bots),2)
    def test_meeting_demo(self):
        h=data.SchedulingHandler(data.SchedulingService(data.SchedulingRepository()))
        self.assertEqual(h.post({'slots_a':[[10,50]],'slots_b':[[20,40]],'duration':15}),(200,{'meeting':[20,35]}))
        self.assertEqual(data.earliest_meeting([[0,1]],[[1,3]],1),None)
        self.assertEqual(h.post({'slots_a':[[2,1]],'slots_b':[],'duration':1})[0],422)

class LimitsAndConcurrency(unittest.TestCase):
    def test_sliding_window(self):
        l=s.SlidingWindowLimiter(2,10)
        self.assertEqual([l.allow('a',t) for t in [0,1,9,10]],[True,True,False,True])
        self.assertTrue(l.allow('b',9))
        with self.assertRaises(ValueError):l.allow('a',8)
        l.expire_idle(30);self.assertEqual(l.events,{})
    def test_clock_wrapper_logger(self):
        now=[0]; l=s.RateLimiter(5,10,lambda:now[0])
        self.assertEqual([l.allow('a') for _ in range(6)],[True]*5+[False]);now[0]=11
        self.assertTrue(l.allow('a'))
        log=s.Logger();self.assertTrue(log.shouldPrintMessage(1,'x'));self.assertFalse(log.shouldPrintMessage(5,'x'));self.assertTrue(log.shouldPrintMessage(11,'x'))
    def test_concurrent_final_slot(self):
        l=s.SlidingWindowLimiter(1,10)
        with concurrent.futures.ThreadPoolExecutor(20) as pool:
            self.assertEqual(sum(pool.map(lambda _:l.allow('u',0),range(100))),1)
        b=s.TokenBucket(1,1,clock=lambda:0)
        with concurrent.futures.ThreadPoolExecutor(20) as pool:
            self.assertEqual(sum(pool.map(lambda _:b.allow()[0],range(100))),1)
    def test_token_refill(self):
        now=[0];b=s.TokenBucket(2,1,lambda:now[0])
        self.assertTrue(b.allow()[0]);self.assertTrue(b.allow()[0]);self.assertEqual(b.allow(),(False,1.))
        now[0]=.5;self.assertEqual(b.allow(),(False,.5));now[0]=1;self.assertTrue(b.allow()[0])
    def test_cache_same_key_and_none(self):
        entered,release=threading.Event(),threading.Event();count=[0]
        def compute(k):count[0]+=1;entered.set();release.wait(2);return None
        cache=s.SingleFlightCache(compute)
        with concurrent.futures.ThreadPoolExecutor(10) as pool:
            fs=[pool.submit(cache.get,2) for _ in range(10)];self.assertTrue(entered.wait(1));release.set()
            self.assertEqual([f.result(2) for f in fs],[None]*10)
        self.assertEqual(count[0],1);self.assertIsNone(cache.get(2))
    def test_cache_distinct_and_failure(self):
        barrier=threading.Barrier(2)
        cache=s.SingleFlightCache(lambda k:(barrier.wait(2),k)[1])
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fs=[pool.submit(cache.get,k) for k in (3,4)];self.assertEqual([f.result(3) for f in fs],[3,4])
        calls=[0]
        def flaky(k):
            calls[0]+=1
            if calls[0]==1:raise ValueError('demo')
            return k
        cache=s.SingleFlightCache(flaky)
        with self.assertRaises(ValueError):cache.get(1)
        self.assertEqual(cache.get(1),1)
    def test_scheduler_delay_cancel_and_exception(self):
        scheduler=s.JobScheduler(2);times=[];ran=threading.Event();start=time.monotonic()
        try:
            job=scheduler.schedule(lambda:(times.append(time.monotonic()),ran.set()),.08)
            self.assertFalse(ran.wait(.025));self.assertTrue(ran.wait(1));self.assertGreaterEqual(times[0]-start,.065)
            self.assertTrue(scheduler.deschedule(job));self.assertFalse(scheduler.deschedule(job))
            cancelled=threading.Event();j=scheduler.schedule(cancelled.set,.05);scheduler.deschedule(j)
            self.assertFalse(cancelled.wait(.1))
            ok=threading.Event()
            def bad():raise ValueError('intentional job failure')
            with self.assertLogs(level='ERROR'):
                bj=scheduler.schedule(bad,.02);scheduler.schedule(ok.set,.03);self.assertTrue(ok.wait(1));scheduler.deschedule(bj)
        finally:scheduler.close()
    def test_scheduler_slow_and_queued_cancel(self):
        scheduler=s.JobScheduler(2);release=threading.Event();entered=threading.Event();fast=threading.Event()
        running=[0,0];guard=threading.Lock()
        def slow():
            with guard:running[0]+=1;running[1]=max(running)
            entered.set();release.wait(2)
            with guard:running[0]-=1
        try:
            slow_id=scheduler.schedule(slow,.01);self.assertTrue(entered.wait(1))
            scheduler.schedule(fast.set,.02);self.assertTrue(fast.wait(1));scheduler.deschedule(slow_id)
            self.assertEqual(running[1],1)
            # Saturate workers, queue a victim, then cancel before it starts.
            blocker_enter=threading.Event();scheduler.schedule(lambda:(blocker_enter.set(),release.wait(2)),.01)
            self.assertTrue(blocker_enter.wait(1))
            victim=threading.Event();v=scheduler.schedule(victim.set,.01);time.sleep(.04);scheduler.deschedule(v)
            release.set();self.assertFalse(victim.wait(.08))
        finally:release.set();scheduler.close()
    def test_scheduler_concurrent_registration(self):
        scheduler=s.JobScheduler(2)
        try:
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                ids=list(pool.map(lambda _:scheduler.schedule(lambda:None,10),range(100)))
                self.assertEqual(len(set(ids)),100)
                self.assertTrue(all(pool.map(scheduler.deschedule,ids)))
        finally:scheduler.close()

class Numerical(unittest.TestCase):
    def test_qr_addition_vs_refit(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(40,3));z=rng.normal(size=(40,5));z[:,0]=x[:,0];y=rng.normal(size=40)
        j,b,drop=num.add_best_regressor(x,z,y,qr=np.linalg.qr(x,mode='reduced'))
        old=np.linalg.lstsq(x,y,rcond=None)[0];old_rss=np.sum((y-x@old)**2)
        drops=[]
        for col in range(5):
            design=np.column_stack((x,z[:,col]));fit=np.linalg.lstsq(design,y,rcond=None)[0]
            drops.append(old_rss-np.sum((y-design@fit)**2))
        self.assertEqual(j,int(np.argmax(drops)))
        np.testing.assert_allclose(b,np.linalg.lstsq(np.column_stack((x,z[:,j])),y,rcond=None)[0],atol=1e-10)
        self.assertAlmostEqual(drop,max(drops));self.assertIsNone(num.add_best_regressor(x,x[:,:1],y)[0])
    def test_two_task_training(self):
        torch.set_num_threads(1);torch.manual_seed(2);x=torch.randn(120,3)
        y=torch.stack((-200000-10000*x[:,0],100+5*x[:,1]),dim=1)
        model,stats=num.fit_two_targets(x[:90],y[:90],epochs=200)
        pred=num.predict_two_targets(model,stats,x[90:]);self.assertEqual(tuple(pred.shape),(30,2))
        self.assertTrue(bool((pred[:,0]<0).all()));self.assertTrue(bool((pred[:,1]>0).all()))
        mean=y[:90].mean(0);self.assertLess(float((((pred-y[90:])/y[:90].std(0))**2).mean()),.25)
        _,st=num.fit_two_targets(x[:10],torch.tensor([[-3.,2.]]).repeat(10,1),epochs=2)
        self.assertTrue(bool(torch.isfinite(st[-1]).all()))
    def test_logistic_gradient_finite_difference(self):
        rng=np.random.default_rng(0);X=rng.normal(size=(7,3));y=np.array([0,1,0,1,1,0,1]);w=rng.normal(size=3);b=.2;eta=.1
        wn,bn=num.logistic_step(X,y,w,b,eta)
        def loss(w,b):z=X@w+b;return np.mean(np.logaddexp(0,z)-y*z)
        eps=1e-6
        grad=np.array([(loss(w+np.eye(3)[i]*eps,b)-loss(w-np.eye(3)[i]*eps,b))/(2*eps) for i in range(3)])
        np.testing.assert_allclose((w-wn)/eta,grad,atol=1e-8)
        self.assertAlmostEqual((b-bn)/eta,(loss(w,b+eps)-loss(w,b-eps))/(2*eps),places=7)
    def test_ctr_demo(self):
        rng=np.random.default_rng(2);n=500;features=rng.exponential(size=(n,3));cat=rng.choice(list('ABC'),n)
        y=rng.random(n)<(1/(1+np.exp(1.3-features[np.arange(n),np.array([ord(c)-65 for c in cat])])))
        df=pd.DataFrame(features,columns=num.FEATURES);df['current_post_category']=cat;df['click']=y.astype(int);df.loc[::10,num.FEATURES[0]]=np.nan
        with patch('sys.stdout',new=io.StringIO()): result=num.compare_ctr(df)
        self.assertEqual(len(result['validation']),3)
        self.assertTrue(0<=result['test']['ap']<=1);self.assertTrue(0<=result['test']['f1']<=1)
    def test_mdp_fixture(self):
        # Explicit fixture implementing the question's P contract; not pasted class.
        class Env:
            nS,nA=16,4
            def __init__(self,slip):
                desc=['SFFF','FHFH','FFFH','HFFG'];self.P={s:{} for s in range(16)}
                for state in range(16):
                    r,c=divmod(state,4)
                    for a in range(4):
                        if desc[r][c] in 'HG':self.P[state][a]=[(1.,state,0.,True)];continue
                        moves=[(a,1.)] if not slip else [(a,.8),((a-1)%4,.1),((a+1)%4,.1)]
                        out=[]
                        for direction,prob in moves:
                            dr,dc=[(0,-1),(1,0),(0,1),(-1,0)][direction];rr=min(3,max(0,r+dr));cc=min(3,max(0,c+dc));ch=desc[rr][cc]
                            out.append((prob,rr*4+cc,float(ch=='G'),ch in 'HG'))
                        self.P[state][a]=out
        for slippery in [False,True]:
            env=Env(slippery);v,p=num.value_iteration(env);v2,p2=num.policy_iteration(env)
            np.testing.assert_allclose(v,v2,atol=1e-4)
            q=num._q_values(env,v,.95);self.assertLess(np.max(q.max(1)-q[np.arange(16),p]),1e-6)
            self.assertEqual(list(v[[5,7,11,12,15]]),[0]*5)
            self.assertAlmostEqual(v[0],.5312 if slippery else .95**5,places=4)
            if not slippery:self.assertEqual(p[0],1)
        env=Env(False)
        for a in range(4):env.P[0][a]=[(1.,0,0.,False)]
        self.assertEqual(num.value_iteration(env)[0][0],0)

class PipelinesAndSQL(unittest.TestCase):
    def test_cleaning(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tx.csv';p.write_text('date,amount,customer_id,payment_method,notes\n"Jan 5, 2024","USD 1,234.56",00123,Credit Crad,ok\n01/06/2024,12.34,7,cash,n\n2024-01-07,EUR9,9,cash,bad\n')
            rows,rejects=data.clean_transactions(p);self.assertEqual(len(rows),2);self.assertEqual(len(rejects),1)
            self.assertEqual(rows[0]['customer_id'],'00123');self.assertEqual(rows[0]['amount'],'1234.56');self.assertEqual(rows[0]['payment_method'],'card')
            padded,_=data.clean_transactions(p,id_width=5);self.assertEqual(padded[1]['customer_id'],'00007')
    def test_tweet_pipeline_validation_idempotency_fence(self):
        with tempfile.TemporaryDirectory() as d:
            old=pipeline.DB;pipeline.DB=str(Path(d)/'db.sqlite');pipeline.init()
            try:
                with patch.object(pipeline,'acquire',lambda *args:None):
                    fixture=Path(__file__).parent/'tweet_pipeline_201_250/demo.json'
                    self.assertEqual(pipeline.ingest(str(fixture)),{'inserted':2,'rejected_this_scan':1})
                    self.assertEqual(pipeline.ingest(str(fixture))['inserted'],0)
                    self.assertTrue(pipeline.process_one(lambda r:{'sentiment':'made_up','topic':'x','confidence':.9}))
                    with pipeline.connect() as db:
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM insights').fetchone()[0],0)
                        db.execute('UPDATE tweets SET next_try=0')
                    self.assertTrue(pipeline.process_one(lambda r:{'sentiment':'neutral','topic':'x','confidence':.2}))
                    with pipeline.connect() as db:
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM insights').fetchone()[0],0)
                        db.execute('UPDATE tweets SET next_try=0')
                    while pipeline.process_one(lambda r:{'sentiment':'neutral','topic':'x','confidence':.9}):pass
                    with pipeline.connect() as db:
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM insights').fetchone()[0],2)
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM rejects').fetchone()[0],1)
                        self.assertEqual(db.execute("SELECT COUNT(*) FROM tweets WHERE status='done'").fetchone()[0],2)
                    # New record; simulate a newer lease holder before old commit.
                    with pipeline.connect() as db:db.execute("INSERT INTO tweets(id,payload) VALUES ('z','{}')")
                    def steal(row):
                        with pipeline.connect() as db:db.execute('UPDATE tweets SET fence=fence+1 WHERE id=?',(row['id'],))
                        return {'sentiment':'neutral','topic':'x','confidence':.9}
                    pipeline.process_one(steal)
                    with pipeline.connect() as db:self.assertIsNone(db.execute("SELECT 1 FROM insights WHERE tweet_id='z'").fetchone())
            finally:pipeline.DB=old
    def test_tweet_pipeline_two_workers_claim(self):
        with tempfile.TemporaryDirectory() as d:
            old=pipeline.DB;pipeline.DB=str(Path(d)/'db.sqlite');pipeline.init()
            try:
                with pipeline.connect() as db:db.execute("INSERT INTO tweets(id,payload) VALUES ('a','{}')")
                with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda _:pipeline.claim(),range(2)))
                self.assertEqual(sum(r is not None for r in rows),1)
            finally:pipeline.DB=old
    def test_tweet_rate_independent(self):
        with tempfile.TemporaryDirectory() as d:
            old=pipeline.DB;pipeline.DB=str(Path(d)/'db.sqlite');pipeline.init()
            try:
                pipeline.acquire('inbound',100)
                t=time.monotonic();pipeline.acquire('outbound',10);pipeline.acquire('outbound',10)
                self.assertGreaterEqual(time.monotonic()-t,.085)
                with pipeline.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM limits').fetchone()[0],2)
            finally:pipeline.DB=old
    def test_engagement_demo_cursor(self):
        class NoWait:
            def wait(self):pass
        pages={None:{'items':[dict(id='1',text='a',likes=2,retweets=1,comments=0)],'next_cursor':'b'},
               'b':{'items':[dict(id='1',text='a',likes=3,retweets=1,comments=0),dict(id='2',text='b',likes=1,retweets=0,comments=2)],'next_cursor':None}}
        with tempfile.TemporaryDirectory() as d:
            p=str(Path(d)/'db.sqlite');result=engagement.analyze('x',lambda k,c:pages[c],p,NoWait(),NoWait())
            self.assertEqual(result,dict(tweets=2,likes=4,retweets=1,comments=2,engagement_per_tweet=3.5))
            result2=engagement.analyze('x',lambda k,c:(_ for _ in ()).throw(AssertionError('must not fetch again')),p,NoWait(),NoWait())
            self.assertEqual(result,result2)
    def test_sql_window_semantics(self):
        db=sqlite3.connect(':memory:')
        db.executescript("CREATE TABLE sales(store_id,date,amount);INSERT INTO sales VALUES(1,'2024-01-01',10),(1,'2024-01-01',5),(1,'2024-01-02',7),(1,'2024-02-01',2);")
        rows=db.execute("WITH daily AS(SELECT store_id,date,SUM(amount) amount FROM sales GROUP BY store_id,date) SELECT SUM(amount) OVER(PARTITION BY store_id,substr(date,1,7) ORDER BY date ROWS UNBOUNDED PRECEDING) FROM daily ORDER BY date").fetchall()
        self.assertEqual(rows,[(15,),(22,),(2,)])
        db.executescript("CREATE TABLE events(user_id,event_id,event_time,kind,value);INSERT INTO events VALUES(1,1,1,'view',10),(1,2,2,'purchase',20),(1,3,3,'view',25),(1,4,4,'purchase',40);")
        row=db.execute("WITH all_events AS(SELECT *,LAG(value) OVER(PARTITION BY user_id ORDER BY event_time,event_id) prev FROM events), q AS(SELECT *,ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY event_time DESC,event_id DESC) rn FROM all_events WHERE kind='purchase') SELECT value-prev FROM q WHERE rn=1").fetchone()
        self.assertEqual(row,(15,))
        db.executescript("CREATE TABLE orders(status,amount,date);INSERT INTO orders VALUES('completed',5,'x'),('pending',7,'x'),('cancelled',9,'x');")
        self.assertEqual(db.execute("SELECT SUM(CASE WHEN status='completed' THEN amount ELSE 0 END),SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END),SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) FROM orders").fetchone(),(5,1,1))
        db.executescript("CREATE TABLE organization(employee_id,manager_id,name);INSERT INTO organization VALUES(1,3,'root'),(2,1,'a'),(3,2,'b');")
        rows=db.execute("WITH RECURSIVE r AS(SELECT employee_id,1 level,','||1||','||employee_id||',' path FROM organization WHERE manager_id=1 AND employee_id<>1 UNION ALL SELECT e.employee_id,r.level+1,r.path||e.employee_id||',' FROM organization e JOIN r ON e.manager_id=r.employee_id WHERE instr(r.path,','||e.employee_id||',')=0) SELECT employee_id,level FROM r").fetchall()
        self.assertEqual(rows,[(2,1),(3,2)])
        db.close()

if __name__=='__main__':unittest.main(verbosity=2)
