import concurrent.futures,itertools,json,math,sqlite3,tempfile,unittest
from pathlib import Path
from decimal import Decimal
import numpy as np
from scipy.stats import norm
import solutions_451_500 as s

class ProbabilityFinance(unittest.TestCase):
    def test_handshake_edges_enumeration(self):
        def edges(m,n,diagonal=False):
            moves=[(0,1),(1,0)]+([(1,1),(1,-1)] if diagonal else [])
            return [((r,c),(r+dr,c+dc)) for r in range(m) for c in range(n) for dr,dc in moves if 0<=r+dr<m and 0<=c+dc<n]
        self.assertEqual(len(edges(10,10))/6,30);self.assertEqual(len(edges(10,10,True))/6,57)
        e=edges(2,2);total=0
        for rolls in itertools.product(range(6),repeat=4):
            total+=sum(rolls[r*2+c]==rolls[rr*2+cc] for (r,c),(rr,cc) in e)
        self.assertAlmostEqual(total/6**4,4/6)
    def test_likelihood_demo(self):
        a=norm.cdf(10/math.sqrt(50));c=norm.sf(2);gap=norm.cdf(-15/math.sqrt(50))+norm.sf(25/math.sqrt(50))
        self.assertGreater(a,c);self.assertGreater(c,gap);self.assertAlmostEqual(gap,.01715,places=4)
    def test_stones_modular_proof_examples(self):
        for a in range(1,25):
            for b in range(1,25):
                total=a+b;x=a;seen=set()
                while x and x not in seen:seen.add(x);x=(2*x)%total
                q=total//math.gcd(a,b)
                self.assertEqual(x==0,q&(q-1)==0)
        for b in [23,24,25]:self.assertNotEqual((1+b)//math.gcd(1,b)&(((1+b)//math.gcd(1,b))-1),0)
    def test_var_quantile_examples_and_paths(self):
        self.assertAlmostEqual(-1e6*(-.105+.5*.01),100000)
        self.assertAlmostEqual(1e6*(-.105+.5*.99),390000)
        self.assertAlmostEqual(-1e6*(-.102+.2*.01),100000)
        self.assertAlmostEqual(1e6*(-.102+.2*.99),96000)
        sigma=.1/norm.ppf(.99);density=norm.pdf(norm.ppf(.99))/(1e6*sigma)
        n=norm.ppf(.975)**2*.99*.01/(1000**2*density**2)
        self.assertTrue(98000<n<100000)
    def test_mc_reproducible_precision_and_budget(self):
        sigma=.1/norm.ppf(.99)
        draw=lambda n,rng:-1e6*rng.normal(0,sigma,n)
        result=s.quantile_mc(draw,seed=0)
        again=s.quantile_mc(draw,seed=0)
        self.assertEqual(result,again);self.assertTrue(result['converged'])
        lo,hi=result['interval'];self.assertLessEqual(max(result['estimate']-lo,hi-result['estimate']),1000)
        self.assertLessEqual(lo,100000);self.assertGreaterEqual(hi,100000)
        self.assertLessEqual(sum(r['look_alpha'] for r in result['history']),.05)
        unconverged=s.quantile_mc(draw,half_width=.0001,initial=100,max_paths=100)
        self.assertFalse(unconverged['converged'])
    def test_business_arithmetic(self):
        visits=5000*200*.7;self.assertEqual(visits,700000)
        self.assertEqual(visits*(50+20-25)-20e6,11.5e6)
        self.assertAlmostEqual(20e6/visits+25-20,33.57142857)
        earnings=10e9/12+20e9/20;self.assertAlmostEqual(earnings/1e9,1.8333333333)
        self.assertAlmostEqual(30e9/earnings,16.36363636)

class CacheTests(unittest.TestCase):
    def test_ttl_overwrite_expiry_none_lru(self):
        now=[0.];cache=s.LocalTTLCache(2,lambda:now[0])
        cache.set('a',None,ttl=1);self.assertIsNone(cache.get('a'))
        now[0]=.5;cache.set('a','new',ttl=10);now[0]=1
        self.assertEqual(cache.get('a'),'new')
        cache.set('b',2);cache.get('a');cache.set('c',3)
        with self.assertRaises(KeyError):cache.get('b')
        now[0]=11
        with self.assertRaises(KeyError):cache.get('a')
        self.assertTrue(cache.delete('c'));self.assertFalse(cache.delete('c'))
    def test_bounded_expiry_heap_and_concurrent(self):
        cache=s.LocalTTLCache(10,lambda:0.)
        for i in range(1000):cache.set('a',i,ttl=1000)
        self.assertLess(len(cache.expiry),125)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i:cache.set(str(i%5),i,ttl=10),range(1000)))
        self.assertLessEqual(len(cache.data),10)
        for ttl in [0,-1,float('nan'),True]:
            with self.assertRaises(ValueError):cache.set('x',1,ttl)

class URLTests(unittest.TestCase):
    def test_generated_collision_alias_and_expiry(self):
        with tempfile.TemporaryDirectory() as d:
            now=[100.];codes=iter(['aaaa','aaaa','bbbb'])
            store=s.ShortCodeStore(str(Path(d)/'links.db'),lambda:now[0],lambda:next(codes))
            self.assertEqual(store.create('https://a.example'), 'aaaa')
            self.assertEqual(store.create('https://b.example'), 'bbbb')
            self.assertEqual(store.resolve('aaaa'),'https://a.example')
            code=store.create('https://c.example',110,'sale');self.assertEqual(code,'sale')
            with self.assertRaises(s.AliasTaken):store.create('https://d.example',optional_custom_alias='sale')
            now[0]=110
            with self.assertRaises(s.LinkExpired):store.resolve('sale')
            with self.assertRaises(s.LinkNotFound):store.resolve('missing')
            again=s.ShortCodeStore(str(Path(d)/'links.db'),lambda:now[0]);self.assertEqual(again.resolve('bbbb'),'https://b.example')
    def test_idempotent_retry_including_expired(self):
        with tempfile.TemporaryDirectory() as d:
            now=[0.];store=s.ShortCodeStore(str(Path(d)/'db'),lambda:now[0])
            code=store.create('https://example.com',10,idempotency_key='request1')
            self.assertEqual(store.create('https://example.com',10,idempotency_key='request1'),code)
            with self.assertRaises(ValueError):store.create('https://other.example',10,idempotency_key='request1')
            now[0]=20
            self.assertEqual(store.create('https://example.com',10,idempotency_key='request1'),code)
            with self.assertRaises(ValueError):store.create('https://fresh.example',10,idempotency_key='new')
    def test_concurrent_alias_one_winner(self):
        with tempfile.TemporaryDirectory() as d:
            store=s.ShortCodeStore(str(Path(d)/'db'))
            def create(i):
                try:return store.create('https://example.com/'+str(i),optional_custom_alias='sale')
                except s.AliasTaken:return None
            with concurrent.futures.ThreadPoolExecutor(8) as pool:results=list(pool.map(create,range(20)))
            self.assertEqual(results.count('sale'),1)
    def test_concurrent_idempotency_one_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'db');store=s.ShortCodeStore(path)
            with concurrent.futures.ThreadPoolExecutor(8) as pool:results=list(pool.map(lambda _:store.create('https://example.com',idempotency_key='same'),range(20)))
            self.assertEqual(len(set(results)),1)
            db=sqlite3.connect(path);self.assertEqual(db.execute('SELECT COUNT(*) FROM links').fetchone()[0],1);db.close()
    def test_validation_disable_and_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'db');store=s.ShortCodeStore(path)
            for url in ['javascript:alert(1)','https://','https://u:p@example.com','https://a.example\r\nX: bad','http://a.example:bad']:
                with self.assertRaises(ValueError):store.create(url)
            for alias in ['admin','API','x/y','']:
                with self.assertRaises(ValueError):store.create('https://example.com',optional_custom_alias=alias)
            code=store.create('https://example.com')
            db=sqlite3.connect(path);db.execute('UPDATE links SET disabled=1');db.commit();db.close()
            with self.assertRaises(s.LinkExpired):store.resolve(code)
            self.assertGreater(62**10,182.5e9);self.assertAlmostEqual(182.5e9*500/1e12,91.25)

class FeeTests(unittest.TestCase):
    def test_all_schedules_and_validation(self):
        self.assertEqual([f(True,4) for f in [s.fee_young,s.fee_middle,s.fee_old]],[Decimal('.80'),Decimal('1.20'),Decimal('1.50')])
        self.assertEqual(s.fee_old(False,4),Decimal('1.60'));self.assertEqual(s.fee_young(True,0),Decimal('0.00'))
        for x in [-1,float('nan'),float('inf'),'bad']:
            with self.assertRaises(ValueError):s.fee_old(True,x)
        with self.assertRaises(ValueError):s.fee_old(1,4)

if __name__=='__main__':unittest.main(verbosity=2)
