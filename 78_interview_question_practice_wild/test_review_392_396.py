"""Explicit small behavioral models and exact arithmetic for design answers392–396."""
from bisect import bisect_left,bisect_right
from datetime import datetime,date,time,timedelta,timezone
from heapq import merge
import itertools
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT=Path(__file__).parent

class ReviewChecks(unittest.TestCase):
    def test_notification_intent_device_dedup_and_tombstone(self):
        parents={};receipts={};inbox={}
        def expand(post,user,channel,devices):
            key=(post,user,channel);parents.setdefault(key,{'state':'QUEUED'})
            inbox.setdefault(key,{'deleted':False})
            for device in devices:receipts.setdefault((key,device),{'state':'PENDING'})
            return key
        key=expand('p','u','push',['phone','web'])
        expand('p','u','push',['phone','web'])
        self.assertEqual((len(parents),len(receipts)),(1,2))
        inbox[key]['deleted']=True;expand('p','u','push',['phone','web'])
        self.assertTrue(inbox[key]['deleted'])
        receipts[(key,'phone')]['state']='ACCEPTED';receipts[(key,'web')]['state']='OPENED'
        self.assertTrue(any(v['state']=='OPENED' for (k,_),v in receipts.items() if k==key))

    def test_unknown_send_identity_and_late_preference(self):
        delivery={'id':'d','provider_key':'stable','state':'QUEUED'};provider={};calls=0
        def send():
            nonlocal calls
            calls+=1;provider.setdefault(delivery['provider_key'],'accepted')
        send();delivery['state']='UNKNOWN';send()
        self.assertEqual((calls,len(provider)),(2,1))
        delivery['state']='ACCEPTED'
        promotional={'queued':True,'consent_version':1};current={'version':2,'opt_in':False}
        if not current['opt_in']:promotional['state']='SUPPRESSED'
        self.assertEqual(promotional['state'],'SUPPRESSED')

    def test_quiet_hours_timezone_and_local_day_dst(self):
        tz=ZoneInfo('America/New_York')
        # Regular overnight quiet window22:00–08:00: defer rather than discard.
        queued=datetime(2026,11,25,23,0,tzinfo=tz)
        next_send=datetime.combine(queued.date()+timedelta(days=1),time(8),tz)
        self.assertEqual(next_send.hour,8);self.assertGreater(next_send,queued)
        for day,hours in [(date(2026,3,8),23),(date(2026,11,1),25)]:
            start=datetime.combine(day,time(),tz).astimezone(timezone.utc)
            end=datetime.combine(day+timedelta(days=1),time(),tz).astimezone(timezone.utc)
            self.assertEqual((end-start).total_seconds(),hours*3600)

    def test_threshold_exact_bounds_and_jump(self):
        thresholds=[98,99,100,101,102]
        def up(old,new):return thresholds[bisect_right(thresholds,old):bisect_right(thresholds,new)] if new>old else []
        def down(old,new):return thresholds[bisect_left(thresholds,new):bisect_left(thresholds,old)] if new<old else []
        self.assertEqual(up(99,100),[100]);self.assertEqual(up(100,100),[])
        self.assertEqual(up(98,102),[99,100,101,102]);self.assertEqual(down(100,99),[99])
        self.assertEqual(down(102,98),[98,99,100,101]);self.assertEqual(down(99,99),[])

    def test_quote_dedup_one_shot_and_recurring_rearm(self):
        last_seq=0;last_price=99;armed=True;triggers=set()
        def quote(seq,price):
            nonlocal last_seq,last_price,armed
            if seq<=last_seq:return
            if armed and last_price<100<=price:triggers.add(('alert-v1',seq));armed=False
            elif not armed and price<99:armed=True
            last_seq,last_price=seq,price
        quote(1,100);quote(1,100);quote(2,101);quote(0,90)
        self.assertEqual(len(triggers),1)
        quote(3,98);quote(4,100);self.assertEqual(len(triggers),2)

    def test_single_puzzle_submission_two_devices(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'puzzle.db');db=sqlite3.connect(path)
            db.execute('CREATE TABLE submission(user TEXT,puzzle TEXT,answer TEXT,score INTEGER,UNIQUE(user,puzzle))');db.commit();db.close()
            barrier=threading.Barrier(2);results=[]
            def submit(answer):
                con=sqlite3.connect(path,timeout=3);barrier.wait()
                try:
                    con.execute('INSERT INTO submission VALUES(?,?,?,?)',('u','p',answer,100 if answer=='A' else 0));con.commit();results.append('accepted')
                except sqlite3.IntegrityError:results.append('conflict')
                finally:con.close()
            threads=[threading.Thread(target=submit,args=(a,)) for a in ('A','B')]
            for t in threads:t.start()
            for t in threads:t.join()
            self.assertEqual(sorted(results),['accepted','conflict'])
            con=sqlite3.connect(path);self.assertEqual(con.execute('SELECT count(*) FROM submission').fetchone()[0],1);con.close()

    def test_all_time_contribution_replay_and_correction(self):
        contribution={};total=0
        def apply(puzzle,version,score):
            nonlocal total
            old=contribution.get(puzzle)
            if old and old[0]>=version:return
            total+=score-(old[1] if old else 0);contribution[puzzle]=(version,score)
        apply('p1',1,100);apply('p1',1,100);apply('p2',1,100)
        self.assertEqual(total,200)
        apply('p1',2,0);apply('p1',1,100);self.assertEqual(total,100)

    def test_shard_top_k_exact_and_global_rank_not_local(self):
        rng=random.Random(396)
        rows=[(-rng.randrange(20),rng.randrange(1000),f's{i:04}',i) for i in range(500)]
        shards=[sorted(r for r in rows if r[3]%7==s) for s in range(7)]
        expected=sorted(rows)
        for k in (1,10,100):self.assertEqual(list(merge(*(s[:k] for s in shards)))[:k],expected[:k])
        player=shards[0][4];local=5;global_rank=1+sum(r<player for r in rows)
        self.assertNotEqual(local,global_rank)
        self.assertEqual(global_rank,expected.index(player)+1)

    def test_best_filter_ties_and_window_counterexample(self):
        best={};accepted=[]
        def submit(user,score,when,ident):
            key=(-score,when,ident)
            if user in best and key>=best[user]:return False
            best[user]=key;accepted.append((user,key));return True
        self.assertTrue(submit('u',100,0,'a'))
        self.assertFalse(submit('u',90,1,'b'))
        self.assertFalse(submit('u',100,2,'c'))
        self.assertTrue(submit('u',101,3,'d'));self.assertEqual(len(accepted),2)
        # At expiry of100@t0,90@t1 should win the sliding window, yet never persisted.
        kept=[(100,0)];actual=[(100,0),(90,1)];cutoff=.5
        self.assertEqual(max(s for s,t in actual if t>=cutoff),90)
        self.assertEqual([x for x in kept if x[1]>=cutoff],[])

    def test_capacity_arithmetic(self):
        self.assertEqual(10_000*200,2_000_000)
        self.assertEqual(15_000_000*200,3_000_000_000)
        self.assertEqual(3_000_000_000*250,750_000_000_000)
        self.assertEqual(20_000_000//50_000,400)
        self.assertEqual(100_000*100*86400,864_000_000_000)
        self.assertEqual(1_000_000*100*32,3_200_000_000)

    def test_pages_and_matching_diagrams(self):
        for i in range(392,397):
            p=ROOT/'questions'/f'W{i:03}.md';text=p.read_text()
            self.assertIn(f'# W{i:03}.',text);self.assertIn('## Question',text);self.assertIn('## Worked solution',text)
            assets=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)
            self.assertEqual(len(assets),1)
            for ref in assets:
                asset=(p.parent/ref).resolve();self.assertTrue(asset.exists());ET.parse(asset)

if __name__=='__main__':unittest.main(verbosity=2)
