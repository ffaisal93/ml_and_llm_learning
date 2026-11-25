import concurrent.futures,threading,importlib.util,io,itertools,json,math,os,sqlite3,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import solutions_301_350 as s
from sql_301_350 import SQL
from flask_301_350 import create_app

class ProbabilityAlgorithms(unittest.TestCase):
    def test_umbrella_enumeration_closed_form(self):
        for n in range(6):
            for p in [0,.3,.5,1]:
                expected=0.
                for rain in itertools.product([False,True],repeat=2*n):
                    person=umbrella=0;prob=1.;dry=True
                    for wet in rain:
                        prob*=p if wet else 1-p
                        if wet:
                            if umbrella!=person:dry=False;break
                            umbrella=1-person
                        person=1-person
                    if dry:expected+=prob
                self.assertAlmostEqual(s.staying_dry(n,p),expected)
                q=1-p;delta=math.sqrt(p*p+4*q*q);a,b=(p+delta)/2,(p-delta)/2
                closed=((1-b)*a**(2*n)+(a-1)*b**(2*n))/delta
                self.assertAlmostEqual(closed,expected)
        self.assertAlmostEqual(s.staying_dry(5,.3),.2588142316)
    def test_cauchy_sampling_distribution(self):
        a=s.sample_hit_positions(100000,2,np.random.default_rng(2));b=s.sample_hit_positions(100000,2,np.random.default_rng(2))
        np.testing.assert_array_equal(a,b)
        for y in [-10,-2,0,2,10]:self.assertLess(abs(np.mean(a<=y)-(.5+math.atan(y/2)/math.pi)),.007)
        self.assertEqual(a.shape,(100000,))
    def test_cauchy_pdf_stability_arguments(self):
        self.assertAlmostEqual(float(s.cauchy_pdf(0,2)),1/(2*math.pi))
        value=float(s.cauchy_pdf(1e150,1e-6));self.assertGreater(value,0);self.assertAlmostEqual(value/(1e-306/math.pi),1)
        self.assertGreater(float(s.cauchy_pdf(1e308,1e308)),0)
        for bad in [0,-1,float('nan'),float('inf'),True,'x',None]:
            with self.assertRaises(ValueError):s.cauchy_pdf(1,bad)
        for bad in [0,-1,1.,True]:
            with self.assertRaises(ValueError):s.sample_hit_positions(bad)
        with self.assertRaises(ValueError):s.density_histogram([])
        with self.assertRaises(ValueError):s.density_histogram([1],n_bins=True)
    def test_histogram_total_mass(self):
        e,h=s.density_histogram(np.array([-100,-.5,.5,100]),limit=1,n_bins=2)
        np.testing.assert_allclose(e,[-1,0,1]);np.testing.assert_allclose(h,[.25,.25])
        hits=s.sample_hit_positions(200000,rng=np.random.default_rng(42));edges,height=s.density_histogram(hits)
        self.assertLess(abs(np.sum(height*np.diff(edges))-2*math.atan(20)/math.pi),.005)
    def test_plot_recording_stub(self):
        calls=[];plt=types.ModuleType('matplotlib.pyplot');parent=types.ModuleType('matplotlib');parent.__path__=[];parent.pyplot=plt
        for name in ['bar','plot','show','xlabel','ylabel','legend']:
            setattr(plt,name,lambda *a,_name=name,**kw:calls.append((_name,a,kw)))
        with patch.dict(sys.modules,{'matplotlib':parent,'matplotlib.pyplot':plt}):s.plot_simulation(1000,2,10,4)
        self.assertEqual([x[0] for x in calls].count('bar'),1);self.assertEqual([x[0] for x in calls].count('plot'),1);self.assertEqual(calls[-1][0],'show')
        edges,h=s.density_histogram(s.sample_hit_positions(1000,2,np.random.default_rng(4)),10)
        bar=next(x for x in calls if x[0]=='bar');np.testing.assert_allclose(bar[1][1],h);self.assertEqual(bar[2]['align'],'edge')
        curve=next(x for x in calls if x[0]=='plot');np.testing.assert_allclose(curve[1][1],s.cauchy_pdf(curve[1][0],2))
    def test_gaussian_modes(self):
        for d in [1.,2.,4.]:
            x=np.linspace(-4,4,100001);f=np.exp(-(x-d/2)**2/2)+np.exp(-(x+d/2)**2/2)
            mode=x[np.argmax(f)]
            if d<2:self.assertAlmostEqual(mode,0,places=4)
            if d==4:self.assertLess(abs(abs(mode)-1.99865),1e-4)
    def test_completion(self):
        events=[('u1','m1',20),('u1','m1',85),('u1','m1',40)]
        self.assertEqual(s.classify_completion(events),[('u1','m1',80)])
        ev=[('u2','m1',50),('u1','m1',29.9),('u3','m1',30),('u1','m0',79.99),('u2','m0',100)]
        self.assertEqual(s.classify_completion(ev),[('u1','m0',50),('u2','m0',80),('u2','m1',50),('u3','m1',30)])
        self.assertEqual(s.classify_completion([]),[]);self.assertEqual(s.classify_completion(ev[::-1]),s.classify_completion(ev))
    def test_reciprocal_power_dynamics(self):
        def iterate(S,v,n=80):
            out=[v.copy()]
            for _ in range(n):a=S@v;v=a/(a@a);out.append(v.copy())
            return out
        seq=iterate(np.array([[4.]]),np.array([1.]),4)
        np.testing.assert_allclose(np.array(seq).ravel(),[1,.25,1,.25,1])
        S=np.diag([1.,2.,2.]);v=np.array([1.,2.,3.]);out=iterate(S,v);scaled=iterate(S,3*v)
        target=np.array([0.,2.,3.])/math.sqrt(13)
        np.testing.assert_allclose(out[-1]/np.linalg.norm(out[-1]),target,atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(out[-1])*np.linalg.norm(out[-2]),.5)
        np.testing.assert_allclose(scaled[-1],3*out[-1]);np.testing.assert_allclose(scaled[-2],out[-2]/3)
        fixed=np.array([0,1/math.sqrt(2),0]);np.testing.assert_allclose(iterate(S,fixed,1)[1],fixed)
    def test_interval_union_and_etl(self):
        rows=[{'run_id':'a','start_time':a,'end_time':b} for a,b in [(58,70),(57,69),(55,72),(56,71),(80,100)]]
        self.assertEqual(s.cumulative_deduped_time(rows),{'a':37});self.assertEqual(s.cumulative_deduped_time([]),{})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'in.csv').write_text('id,name,amount\n001, Alice  Smith ,12.30\n001,Again,4\n002,Bob,bad\n003,,2\n004,Joe,NaN\n005,Eva,1.234\n006,Sam,0\n')
            self.assertEqual(s.run_etl(p/'in.csv',p/'out',p/'reject'),{'accepted':2,'rejected':5})
            clean=[json.loads(x) for x in (p/'out').read_text().splitlines()];self.assertEqual(clean[0],{'id':'001','name':'Alice Smith','amount':'12.30'})
            before=(p/'out').read_bytes();s.run_etl(p/'in.csv',p/'out',p/'reject');self.assertEqual((p/'out').read_bytes(),before)
            (p/'bad.csv').write_text('wrong,header\na,b\n')
            with self.assertRaises(ValueError):s.run_etl(p/'bad.csv',p/'out',p/'reject')

class FlaskTests(unittest.TestCase):
    def test_pagination_cache_auth_limits(self):
        class FakeRedis:
            def __init__(self):self.data={};self.count={};self.lock=threading.Lock()
            def eval(self,script,n,key):
                with self.lock:
                    self.count[key]=self.count.get(key,0)+1;return [self.count[key],60000]
            def get(self,key):return self.data.get(key)
            def setex(self,key,ttl,value):self.data[key]=value
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'db');db=sqlite3.connect(path);db.executescript('CREATE TABLE entries(id INTEGER PRIMARY KEY,tenant TEXT,payload TEXT);CREATE INDEX tenant_id ON entries(tenant,id);')
            db.executemany('INSERT INTO entries VALUES(?,?,?)',[(i,'a',str(i)) for i in range(1,206)]+[(999,'b','secret')]);db.commit();db.close()
            redis=FakeRedis();app=create_app(path,redis,{'aa':'a','bb':'b'},'testsecret');app.testing=True;c=app.test_client();h={'Authorization':'Bearer aa'}
            self.assertEqual(c.get('/entries').status_code,401)
            first=c.get('/entries',headers=h).get_json();self.assertEqual(len(first['items']),100)
            with patch('flask_301_350.sqlite3.connect',side_effect=AssertionError('cached read must not hit DB')):
                self.assertEqual(c.get('/entries',headers=h).get_json(),first)
            second=c.get('/entries',query_string={'cursor':first['next_cursor']},headers=h).get_json();self.assertEqual(second['items'][0]['id'],101)
            self.assertEqual(c.get('/entries?limit=101',headers=h).status_code,400)
            self.assertEqual(c.get('/entries',query_string={'cursor':first['next_cursor']},headers={'Authorization':'Bearer bb'}).status_code,400)
            self.assertEqual(c.get('/entries?cursor=malformed',headers=h).status_code,400)
            def concurrent_read(_):
                with app.test_client() as client:return client.get('/entries',headers={'Authorization':'Bearer bb'}).get_json()
            with concurrent.futures.ThreadPoolExecutor(10) as pool:
                pages=list(pool.map(concurrent_read,range(20)))
            self.assertTrue(all(page['items']==[{'id':999,'payload':'secret'}] for page in pages))
            for _ in range(60):last=c.get('/entries',headers=h)
            self.assertEqual(last.status_code,429);self.assertIn('Retry-After',last.headers)

class SQLTests(unittest.TestCase):
    def setUp(self):self.db=sqlite3.connect(':memory:')
    def tearDown(self):self.db.close()
    def runsql(self,i,query=None):return self.db.execute(query or SQL[i]).fetchall()
    def test_lot_winners(self):
        self.db.executescript('CREATE TABLE lots(lot_id,title);CREATE TABLE bids(bid_id,lot_id,bidder_id,amount);INSERT INTO lots VALUES(1,"a"),(2,"b");INSERT INTO bids VALUES(1,1,7,10),(2,1,8,15),(3,1,9,15),(4,1,8,15);')
        self.assertEqual(self.runsql(319),[(1,8,15),(1,9,15),(2,None,None)])
    def fixture_four(self):
        self.db.executescript('''CREATE TABLE blocks(workspace_id,block_id,parent_id,block_type);
        INSERT INTO blocks VALUES(1,1,NULL,'page'),(1,2,1,'todo'),(1,3,1,'todo'),(2,1,NULL,'todo'),(2,4,1,'page');
        CREATE TABLE users(id PRIMARY KEY,subscription_id,country,created_at);
        INSERT INTO users VALUES(1,10,'US','x'),(2,20,'US','x'),(3,NULL,'CA','x');
        CREATE TABLE subscription(id PRIMARY KEY,user_id,plan_type);
        INSERT INTO subscription VALUES(10,1,'paid'),(11,1,'old'),(20,2,'free');
        CREATE TABLE events(event_id,block_id,user_id,event_type,platform);
        INSERT INTO events VALUES(1,1,1,'view','web'),(2,1,1,'view','web'),(3,2,2,'edit_text','web'),(4,2,3,'view','mobile');''')
    def test_parent_type_country_event_plan(self):
        self.fixture_four();self.assertEqual(self.runsql(321),[('page',2)])
        self.assertEqual(self.runsql(322),[('US',3)]);self.assertEqual(self.runsql(330),[('view',3)])
        self.assertEqual(self.runsql(334),[('view','paid',2)])
        self.assertAlmostEqual(self.runsql(335)[0][0],80.)
        self.db.execute("INSERT INTO events VALUES(5,2,3,'edit_text','web')")
        self.assertAlmostEqual(self.runsql(335)[0][0],80.)
    def test_never_edited_null_empty(self):
        self.db.executescript("CREATE TABLE blocks(block_id);CREATE TABLE events(block_id,event_type);INSERT INTO blocks VALUES(1),(2),(3),(4);INSERT INTO events VALUES(1,'edit_text'),(1,'edit_text'),(NULL,'edit_text'),(2,'view');")
        self.assertEqual(self.runsql(335),[(75.,)]);self.db.execute('DELETE FROM blocks');self.assertEqual(self.runsql(335),[(None,)])
    def test_event_ties_nulls_empty(self):
        self.db.executescript('CREATE TABLE events(event_type);INSERT INTO events VALUES("a"),("b"),(NULL);')
        self.assertEqual(len(self.runsql(330)),3);self.db.execute('DELETE FROM events');self.assertEqual(self.runsql(330),[])
    def test_interval_sql_python_nested(self):
        self.db.execute('CREATE TABLE intervals(run_id,start_time,end_time)')
        rows=[('a',1,10),('a',1,10),('a',2,3),('a',4,5),('a',10,12),('a',20,21),('b',2,2),('b',5,9)]
        self.db.executemany('INSERT INTO intervals VALUES(?,?,?)',rows)
        expected=s.cumulative_deduped_time([dict(zip(['run_id','start_time','end_time'],r)) for r in rows])
        self.assertEqual(dict(self.runsql(323)),expected)
    def test_workout_fixture_metrics(self):
        self.db.executescript('CREATE TABLE workouts(id,user_id,workout_type_id,started_at);')
        rows=[(1,1,1,'2026-03-02 08:00'),(2,1,2,'2026-03-02 18:30'),(3,2,1,'2026-03-02 09:15'),(4,1,1,'2026-03-03 07:45'),(5,3,3,'2026-03-04 12:00'),(6,2,1,'2026-03-09 10:00'),(7,3,1,'2026-03-10 11:00')]
        self.db.executemany('INSERT INTO workouts VALUES(?,?,?,?)',rows)
        dau=self.db.execute("SELECT date(started_at),COUNT(DISTINCT user_id) FROM workouts WHERE started_at>='2026-03-02' AND started_at<'2026-03-05' GROUP BY date(started_at)").fetchall()
        self.assertEqual(dau,[('2026-03-02',2),('2026-03-03',1),('2026-03-04',1)])
        mau=self.db.execute("WITH days(day) AS(VALUES('2026-03-02'),('2026-03-03'),('2026-03-04')) SELECT day,COUNT(DISTINCT user_id) FROM days LEFT JOIN workouts ON date(started_at) BETWEEN date(day,'-29 days') AND day GROUP BY day").fetchall();self.assertEqual([r[1] for r in mau],[2,2,3])
        favorite=SQL[324].split('-- Exactly three')[1].split('\n',1)[1];self.assertEqual(self.db.execute(favorite).fetchall(),[(1,1,2,1),(1,2,1,2),(2,1,2,1),(3,1,1,1),(3,3,1,2)])
        retained=self.db.execute("WITH firsts AS(SELECT user_id,MIN(date(started_at,'weekday 0','-6 days')) cohort FROM workouts GROUP BY user_id), weekly AS(SELECT DISTINCT user_id,date(started_at,'weekday 0','-6 days') week FROM workouts) SELECT cohort,CAST((julianday(week)-julianday(cohort))/7 AS INTEGER),COUNT(*) FROM firsts JOIN weekly USING(user_id) GROUP BY cohort,week").fetchall();self.assertEqual(retained,[('2026-03-02',0,3),('2026-03-02',1,2)])
    def test_listing_weighted_average(self):
        self.db.executescript("CREATE TABLE listings(id,host_id);CREATE TABLE listing_stats_daily(listing_id,date,nights_booked,gross_revenue,price_count);INSERT INTO listings VALUES(1,7),(2,7),(3,8);INSERT INTO listing_stats_daily VALUES(1,'2024-01-01',1,100.0,1),(1,'2024-01-02',2,300.0,2);")
        rows=self.db.execute(SQL[331],{'host_id':7,'start_date':'2024-01-01','end_date':'2024-01-02'}).fetchall()
        self.assertEqual(rows[0][:2],(1,3));self.assertAlmostEqual(rows[0][2],400/3);self.assertEqual(rows[1],(2,0,None))
    def test_basic_selects(self):
        self.db.executescript(SQL[333].split('SELECT')[0]);self.assertEqual(self.db.execute('SELECT name FROM students ORDER BY id LIMIT 2').fetchall(),[('Alice',),('Bob',)])
        self.assertEqual(self.db.execute('SELECT name FROM students WHERE age>24 ORDER BY id').fetchall(),[('Bob',),('Charlie',)])
        self.db.execute('INSERT INTO students VALUES(?,?,?)',(4,"O'Neil",None));self.assertEqual(self.db.execute('SELECT * FROM students WHERE age>100').fetchall(),[]);self.assertEqual(self.db.execute('SELECT * FROM students LIMIT 0').fetchall(),[])
    def test_partial_index_plan(self):
        self.db.executescript("CREATE TABLE job(id INTEGER PRIMARY KEY,status TEXT,created_on TEXT);CREATE INDEX job_new_latest ON job(created_on DESC,id DESC) WHERE status='new';")
        self.db.executemany('INSERT INTO job VALUES(?,?,?)',[(i,'new' if i%5==0 else 'done',f'{i:09}') for i in range(1000)])
        plan=self.db.execute("EXPLAIN QUERY PLAN SELECT * FROM job WHERE status='new' ORDER BY created_on DESC,id DESC LIMIT 1").fetchall()
        self.assertIn('job_new_latest',str(plan));self.assertEqual(self.db.execute("SELECT id FROM job WHERE status='new' ORDER BY created_on DESC,id DESC LIMIT 1").fetchone(),(995,))
    def test_resource_dashboard(self):
        self.db.executescript('CREATE TABLE cpu_metrics(application_id,dt,usage_percentage);CREATE TABLE memory_metrics(application_id,dt,usage_percentage);')
        self.db.executemany('INSERT INTO cpu_metrics VALUES(?,?,?)',[(1,'2024-02-01 00:00:00',50),(1,'2024-02-29 23:59:59',50.008),(1,'2024-03-01 00:00:00',100),(2,'2024-02-10 00:00:00',50)])
        self.db.execute("INSERT INTO memory_metrics VALUES(1,'2024-02-10 00:00:00',60)")
        q=SQL[337].replace("TO_CHAR(ROUND(AVG(usage_percentage),2),'FM999990.00')","printf('%.2f',AVG(usage_percentage))")
        self.assertEqual(self.db.execute(q).fetchall(),[('CPU',1,'50.00'),('Memory',1,'60.00')])
    def test_employee_duration_shares(self):
        self.db.executescript('''CREATE TABLE employees(employee_id,name);CREATE TABLE projects(project_id,start_date,end_date,budget,status);CREATE TABLE assignments(employee_id,project_id);
        INSERT INTO employees VALUES(1,'a'),(2,'b'),(3,'c');INSERT INTO projects VALUES(1,'2024-01-01','2024-04-02',120,'completed'),(2,'2024-01-01','2024-03-31',100,'completed');
        INSERT INTO assignments VALUES(1,1),(2,1),(3,1),(1,2),(2,2),(1,2);''')
        queries=SQL[339].split(';');q=queries[0].replace('p.end_date::date-p.start_date::date','julianday(p.end_date)-julianday(p.start_date)')
        self.assertEqual(self.db.execute(q).fetchall(),[(1,'a'),(2,'b'),(3,'c')]);q=queries[1].replace('p.budget::numeric','1.0*p.budget')
        self.assertEqual(self.db.execute(q).fetchall(),[(1,45.),(2,45.),(3,40.)])
    def test_login_islands(self):
        self.db.execute('CREATE TABLE user_login_log(user_id,login_date)')
        self.db.executemany('INSERT INTO user_login_log VALUES(?,?)',[('a',f'2024-01-{d:02}') for d in [1,2,4,5,6,6]]+[('b',f'2024-01-{d:02}') for d in range(1,21)]+[('c','2024-02-01')])
        q=SQL[340].split(';')[0].replace('login_date::date','date(login_date)').replace('day-(ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY day))::int','julianday(day)-ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY day)').replace('LEAST(l.max_streak,15)','MIN(l.max_streak,15)')
        self.assertEqual(self.db.execute(q).fetchall(),[('a',3,3,5),('b',20,15,20),('c',1,1,1)])
    def test_vendor_variants(self):
        self.db.executescript('''CREATE TABLE transactions(transaction_id,vendor_id,amount_charged,transaction_type,refund_transaction_id,timestamp,transaction_time,transaction_dollars);CREATE TABLE vendors(vendor_id,state,city);INSERT INTO vendors VALUES(1,'CA','SJ'),(2,'CA','SJ'),(3,'TX','Austin'),(4,'CA','SJ');''')
        rows=[(1,1,100,'PURCHASE',None,'2024-01-01'),(2,1,100,'REFUND',1,'2024-02-01'),(3,1,150,'PURCHASE',None,'2024-03-01'),(4,2,300,'PURCHASE',None,'2024-03-01'),(5,3,500,'PURCHASE',None,'2024-03-01'),(6,4,300,'PURCHASE',None,'2024-03-01'),(7,2,1000,'PURCHASE',None,'2022-01-01'),(8,2,40,'REFUND',7,'2024-03-02')]
        self.db.executemany('INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?)',[r+(r[-1],r[2]) for r in rows])
        def dated(q):return q.replace("CURRENT_DATE-INTERVAL '2 years'","'2023-04-01'").replace("CURRENT_DATE+INTERVAL '1 day'","'2025-04-02'")
        self.assertEqual(self.db.execute(dated(SQL[342])).fetchall(),[(3,500),(4,300),(2,260)])
        self.assertEqual(self.db.execute(dated(SQL[343])).fetchall(),[(3,500),(2,300),(4,300)])
        city=self.db.execute(dated(SQL[341])).fetchall();self.assertEqual(city[0],('CA','SJ',4,300,1));self.assertIn(('TX','Austin',3,500,1),city)

if __name__=='__main__':unittest.main(verbosity=2)
