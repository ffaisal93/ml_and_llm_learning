import itertools
import json
import math
from pathlib import Path
import sqlite3
import unittest
import numpy as np
import additional_solutions as a


class AdditionalPracticeTests(unittest.TestCase):
    def test_compiler_examples_and_variable_occurrences(self):
        self.assertEqual(a.compiler_cost('res1=var1+var2\nres2=var3-var4\nres3=res2+var5\nres=res1+res3'),(16,2))
        self.assertEqual(a.compiler_cost('res1=10*var2\nres2=var3*100-var4\nres3=res1/2\nres=res2+res3'),(27,2))
        self.assertEqual(a.compiler_cost('a=x+y\nb=x-y\nc=x*y\nr=a+b'),(20,3))
        self.assertEqual(a.compiler_cost('a=x+x'),(4,1))

    def test_idle_union_boundary_and_invalid_mutation(self):
        t=a.GPUIdleIntervalTracker()
        for start,end in [(10,20),(30,40),(18,35)]:t.add_task(start,end,'A')
        self.assertEqual(t.get_idle_intervals('A',0,50),[(0,10),(40,50)])
        self.assertEqual(t.get_idle_intervals('B',0,50),[(0,50)])
        self.assertEqual(t.get_idle_intervals('A',10,10),[])
        with self.assertRaises(ValueError):t.add_task(4,4,'A')
        self.assertEqual(t.busy['A'],[(10,40)])

    def test_interval_total(self):
        rows=[{'run_id':'a','start_time':x,'end_time':y} for x,y in [(58,70),(57,69),(55,72),(56,71),(80,100)]]
        self.assertEqual(a.cumulative_deduped_time(rows),{'a':37})

    def test_stopping_all_five_edges(self):
        for p,expected in [([.5,.5,.7],[.5,.25,.25]),([0,0,0],[0,0,1]),([1,0,0],[1,0,0]),([0,1,1],[0,1,0]),([0],[1])]:
            masses,mean=a.stopping_distribution(p)
            np.testing.assert_allclose(masses,expected)
            self.assertAlmostEqual(sum(masses),1)
            self.assertAlmostEqual(mean,sum((i+1)*x for i,x in enumerate(expected)))
        with self.assertRaises(ValueError):a.stopping_distribution([])

    def test_minimax_not_mean(self):
        def reward(decision,t,adversary):return [[3,3],[10,-1]][decision][adversary]
        self.assertEqual(a.minimax_decision([0,1],[[1],[1]],reward),(0,3.0))

    def test_kmedian_lower_median_and_nonincreasing(self):
        centers,labels,total,history=a.manhattan_pickups([(0,0),(2,0),(10,0)],1)
        np.testing.assert_array_equal(centers,[[2,0]])
        self.assertEqual(total,10)
        self.assertTrue(all(y<=x+1e-10 for x,y in zip(history,history[1:])))
        centers,_,_,_=a.manhattan_pickups([(1,1),(5,5)],1)
        np.testing.assert_array_equal(centers,[[1,1]])

    def test_bag_and_grid_exact_special_cases(self):
        self.assertAlmostEqual(a.expected_remaining_balls(10,10,10),310/77)
        self.assertEqual(a.expected_remaining_balls(0,2,3),5)
        self.assertAlmostEqual(a.grid_exit_time(2),1)
        self.assertAlmostEqual(a.grid_exit_time(10),29.239374213664465)

    def test_five_dice_residue_by_enumeration(self):
        good=sum(sum(rolls)%5==0 for rolls in itertools.product(range(1,7),repeat=5))
        self.assertEqual(good,1556)
        self.assertAlmostEqual(good/6**5,389/1944)
        self.assertAlmostEqual((math.comb(6,4)+math.comb(6,5)+math.comb(6,6))/64,11/32)

    def test_spd_two_cycle_and_scaling(self):
        def step(x):return 4*x/(4*x)**2
        self.assertEqual(step(1),.25)
        self.assertEqual(step(step(1)),1)
        self.assertEqual(step(.5),.5)
        self.assertAlmostEqual(step(3),step(1)/3)

    def test_darts_nonzero_root_and_boundary(self):
        lo,hi=1e-6,2.
        for _ in range(100):
            mid=(lo+hi)/2
            if 2*math.tanh(2*mid)-mid>0:lo=mid
            else:hi=mid
        self.assertAlmostEqual((lo+hi)/2,1.998651346,places=7)
        self.assertLess(math.tanh(.1),.1)

    def test_harvest_and_storage(self):
        fields={'A':(4,240000),'B':(2,210000),'C':(3,90000)}
        losses={}
        for order in itertools.permutations(fields):
            elapsed=0;loss=0
            for key in order:
                days,value=fields[key];elapsed+=days;loss+=.01*value*elapsed
            losses[order]=loss
        self.assertEqual(min(losses,key=losses.get),('B','A','C'))
        self.assertEqual(min(losses.values()),26700)
        values=[(100000+2000*t)*(1-.5*p) for t,p in enumerate([0,.01,.03,.07,.14,.28])]
        self.assertEqual(int(np.argmax(values)),2)
        self.assertAlmostEqual(max(values),102440)
        self.assertAlmostEqual((6.25-.18-.087-5.8)*50000,9150)
        self.assertAlmostEqual((6.25-5.8-.087)/3,.121)

    def test_noise_correlation_and_sequence_rules(self):
        self.assertLess(1/(1+1/(19*5)),.99)
        self.assertGreaterEqual(1/(1+1/(19*6)),.99)
        self.assertEqual(2*55.5+23,134)
        self.assertEqual(6*12*144,10368)
        self.assertEqual(10*9+10*9,180)

    def test_interval_sql_handles_nested_ranges(self):
        import re
        text=Path(__file__).with_name('ADDITIONAL_LAB.md').read_text()
        query=re.findall(r'```sql\n(.*?)```',text,re.S)[0]
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE activity(run_id,start_time,end_time)')
        db.executemany('INSERT INTO activity VALUES (?,?,?)',[("a",58,70),("a",57,69),("a",55,72),("a",56,71),("a",80,100),("b",43,62)])
        self.assertEqual(dict(db.execute(query).fetchall()),{'a':37,'b':19})
        db.close()

    def test_all_new_queries_execute_with_ties_and_boundaries(self):
        rows=json.loads(Path(__file__).with_name('practice_questions.json').read_text())
        db=sqlite3.connect(':memory:')
        db.executescript('''
        CREATE TABLE blocks(workspace_id,block_id,parent_id,block_type);
        CREATE TABLE events(event_id,block_id,user_id,event_type,platform);
        CREATE TABLE users(id,subscription_id,country,created_at);
        CREATE TABLE students(id,name,age);
        CREATE TABLE cpu_metrics(application_id,dt,usage_percentage);
        CREATE TABLE memory_metrics(application_id,dt,usage_percentage);
        CREATE TABLE lots(lot_id); CREATE TABLE bids(lot_id,bidder_id,amount);
        INSERT INTO blocks VALUES (1,1,NULL,'page'),(1,2,1,'todo'),(1,3,1,'todo'),(1,4,NULL,'page');
        INSERT INTO users VALUES (1,NULL,'US',NULL),(2,NULL,'CA',NULL);
        INSERT INTO events VALUES (1,2,1,'edit_text','web'),(2,2,1,'edit_text','web'),(3,3,2,'visit','web');
        INSERT INTO students VALUES (1,'Alice',22),(2,'Bob',25),(3,'Charlie',30);
        INSERT INTO cpu_metrics VALUES (1,'2024-02-29',60),(1,'2024-03-01',100);
        INSERT INTO memory_metrics VALUES (1,'2024-02-01',50),(2,'2024-02-01',75);
        INSERT INTO lots VALUES (1),(2); INSERT INTO bids VALUES (1,'A',10),(1,'B',10),(1,'C',9);
        ''')
        by_title={x['title']:x for x in rows}
        for item in rows:
            code=item['solution'].get('code')
            if code:
                queries=[q.strip() for q in code.split(';') if q.strip()]
                outputs=[db.execute(q).fetchall() for q in queries]
                title=item['title']
                if 'Never Edited' in title:self.assertEqual(outputs[0],[(75.,)])
                elif title=='Country with the Most Events':self.assertEqual(outputs[0],[('US',2)])
                elif title=='Block Type with the Most Children':self.assertEqual(outputs[0],[('page',2)])
                elif title=='Event Type with the Most Events':self.assertEqual(outputs[0],[('edit_text',2)])
                elif title=='SQL Cloud Resource Utilization Dashboard':self.assertEqual(outputs[0],[('CPU',1,'60.00'),('Memory',2,'75.00')])
                elif 'Auction Web Service' in title:self.assertEqual(outputs[0],[(1,'A',10),(1,'B',10),(2,None,None)])
                elif title=='Implement Basic SQL Query Operations':self.assertEqual(outputs[-1],[(2,'Bob',25),(3,'Charlie',30)])
        db.close()


if __name__=='__main__':unittest.main()
