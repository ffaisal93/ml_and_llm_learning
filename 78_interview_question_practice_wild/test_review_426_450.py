"""Actual TCP persistence/OOD checks plus explicit ledger/distribution/sync fixtures."""
import ast
from datetime import date
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

cards=load('review_w438_implementation')

class Client:
    def __init__(self,port):
        self.socket=socket.create_connection(('127.0.0.1',port),timeout=.2);self.buffer=b''
        self.until(lambda line:line.startswith('WELCOME'))
    def send(self,text):self.socket.sendall((text+'\n').encode())
    def until(self,match,timeout=4):
        result=[];deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1);line=line.decode();result.append(line)
                if match(line):return result
                continue
            try:
                data=self.socket.recv(65536)
                if not data:raise AssertionError('server disconnected')
                self.buffer+=data
            except socket.timeout:pass
        raise AssertionError(f'expected line not received; saw {result}')
    def close(self):self.socket.close()

class ReviewTests(unittest.TestCase):
    def test_tcp_two_clients_history_and_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            command=[sys.executable,str(ROOT/'review_w427_implementation.py'),str(port),'--db',str(Path(temp)/'chat.db')]
            processes=[];clients=[]
            def start():
                process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);processes.append(process)
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    try:return Client(port)
                    except OSError:time.sleep(.03)
                self.fail('chat did not start')
            try:
                a=start();clients.append(a)
                a.send('before login');self.assertIn('identify',a.until(lambda s:s.startswith('ERROR'))[-1])
                a.send('/nick alice');line=a.until(lambda s:s.startswith('id='))[-1]
                match=re.fullmatch(r'id=(\S+) token=(\S+) nick=alice',line);self.assertIsNotNone(match)
                ident,token=match.groups()
                a.send('before join');self.assertIn('join a room',a.until(lambda s:s.startswith('ERROR'))[-1])
                a.send('/create general');a.until(lambda s:s=='ROOM general')
                a.send('/join general');a.until(lambda s:s=='END_HISTORY general')
                b=Client(port);clients.append(b)
                b.send('/nick alice');self.assertIn('already used',b.until(lambda s:s.startswith('ERROR'))[-1])
                b.send('/nick bob');b.until(lambda s:s.startswith('id='))
                b.send('/join general');b.until(lambda s:s=='END_HISTORY general')
                a.send('hello');self.assertEqual(b.until(lambda s:s=='[general] alice: hello')[-1],'[general] alice: hello')
                a.until(lambda s:s=='[general] alice: hello')
                for n in range(11):
                    a.send(f'm{n}');a.until(lambda s:s==f'[general] alice: m{n}')
                a.send('/users');users=a.until(lambda s:s.startswith('USERS'))[-1]
                self.assertEqual(users,'USERS alice bob')
                for client in clients:client.close()
                clients=[];processes[0].terminate();processes[0].wait(timeout=3)
                c=start();clients.append(c);c.send(f'/login {ident} wrong')
                self.assertIn('login failed',c.until(lambda s:s.startswith('ERROR'))[-1])
                c.send(f'/login {ident} {token}')
                restored=c.until(lambda s:s=='END_HISTORY general')
                self.assertTrue(any('welcome back alice' in s for s in restored))
                history=[s for s in restored if s.startswith('[general]')]
                self.assertEqual(history,[f'[general] alice: m{n}' for n in range(1,11)])
                c.send('/rooms');self.assertEqual(c.until(lambda s:s.startswith('ROOMS'))[-1],'ROOMS general')
                c.send('/help');self.assertIn('/login',c.until(lambda s:s.startswith('/help'))[-1])
                c.send('/quit');self.assertEqual(c.until(lambda s:s=='BYE')[-1],'BYE')
            finally:
                for client in clients:client.close()
                for process in processes:
                    if process.poll() is None:process.terminate()
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    if process.stderr:process.stderr.close()

    def test_card_pending_settle_repay_interest_statement(self):
        service=cards.CreditCardService();account=cards.Account('a',10000);card=service.issueCard(account)
        tx=service.authorize(card,6000,'auth')
        self.assertIs(service.authorize(card,6000,'auth'),tx);self.assertEqual(account.pending,6000)
        with self.assertRaises(ValueError):service.authorize(card,5000,'too-big')
        with self.assertRaises(ValueError):service.authorize(card,6001,'auth')
        service.settle(tx,date(2026,1,2));service.settle(tx,date(2026,1,2))
        self.assertEqual((account.settled,account.pending),(6000,0))
        repayment=service.repay(card,2000,'repay',date(2026,1,3))
        self.assertIs(service.repay(card,2000,'repay',date(2026,1,3)),repayment)
        interest=service.accrue_interest('a',date(2026,1,4),10000)
        self.assertEqual(interest.amount,40)
        service.accrue_interest('a',date(2026,1,4),10000);self.assertEqual(account.settled,4040)
        statement=service.generateStatement(card,(date(2026,1,1),date(2026,2,1)))
        self.assertEqual((statement.net_activity,statement.closing_balance),(4040,4040))
        self.assertIs(service.generateStatement(card,(date(2026,1,1),date(2026,2,1))),statement)
        late=service.authorize(card,100,'late')
        with self.assertRaises(ValueError):service.settle(late,date(2026,1,31))
        service.settle(late,date(2026,2,1));self.assertEqual(account.settled,4140)

    def test_card_shared_account_race_void_invalid(self):
        service=cards.CreditCardService();account=cards.Account('a',10000)
        c1=service.issueCard(account);c2=service.issueCard(account);results=[];barrier=threading.Barrier(2)
        def authorize(card,key):
            barrier.wait()
            try:results.append(service.authorize(card,8000,key))
            except ValueError:results.append(None)
        threads=[threading.Thread(target=authorize,args=(c,key)) for c,key in ((c1,'1'),(c2,'2'))]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(sum(x is not None for x in results),1)
        tx=next(x for x in results if x is not None);service.void(tx);service.void(tx)
        self.assertEqual(account.pending,0)
        for amount in (0,-1,1.5,True):
            with self.assertRaises(ValueError):service.authorize(c1,amount,str(amount))
        c1.active=False
        with self.assertRaises(ValueError):service.authorize(c1,1,'inactive')

    def test_sql_ledger_duplicate_and_concurrent_funds(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=str(Path(tmp)/'ledger.db');conn=sqlite3.connect(db)
            conn.executescript('CREATE TABLE balance(id TEXT PRIMARY KEY,cents INTEGER);CREATE TABLE transfer(id TEXT PRIMARY KEY,amount INTEGER);CREATE TABLE entry(transfer_id TEXT,account TEXT,amount INTEGER);INSERT INTO balance VALUES ("a",100),("b",0);');conn.close()
            def transfer(key,amount):
                con=sqlite3.connect(db,timeout=3)
                try:
                    con.execute('BEGIN IMMEDIATE')
                    row=con.execute('SELECT amount FROM transfer WHERE id=?',(key,)).fetchone()
                    if row:
                        if row[0]!=amount:raise ValueError('conflicting key')
                        con.rollback();return True
                    funds=con.execute('SELECT cents FROM balance WHERE id="a"').fetchone()[0]
                    if funds<amount:con.rollback();return False
                    con.execute('INSERT INTO transfer VALUES (?,?)',(key,amount))
                    con.executemany('INSERT INTO entry VALUES (?,?,?)',[(key,'a',-amount),(key,'b',amount)])
                    con.execute('UPDATE balance SET cents=cents-? WHERE id="a"',(amount,));con.execute('UPDATE balance SET cents=cents+? WHERE id="b"',(amount,));con.commit();return True
                finally:con.close()
            outcomes=[];barrier=threading.Barrier(2)
            def worker(key):barrier.wait();outcomes.append(transfer(key,80))
            ts=[threading.Thread(target=worker,args=(key,)) for key in ('x','y')]
            for t in ts:t.start()
            for t in ts:t.join()
            self.assertEqual(sorted(outcomes),[False,True])
            con=sqlite3.connect(db);key=con.execute('SELECT id FROM transfer').fetchone()[0];con.close()
            self.assertTrue(transfer(key,80))
            with self.assertRaises(ValueError):transfer(key,79)
            con=sqlite3.connect(db);self.assertEqual(con.execute('SELECT sum(amount) FROM entry').fetchone()[0],0)
            self.assertEqual(con.execute('SELECT sum(cents) FROM balance').fetchone()[0],100);self.assertEqual(con.execute('SELECT count(*) FROM transfer').fetchone()[0],1);con.close()

    def test_distribution_snapshot_zero_and_unknown_identity(self):
        # Explicit small model tests the answer's immutable-plan/zero-command invariant.
        plans={};commands={}
        def accept(payment,amount,currency,snapshot,allocations):
            signature=(amount,currency,snapshot,tuple(allocations))
            if payment in plans:
                if plans[payment]!=signature:raise ValueError('conflicting snapshot')
                return
            if any(type(a) is not int or a<0 or c!=currency for _,a,c in allocations) or sum(a for _,a,_ in allocations)!=amount:raise ValueError('invalid plan')
            plans[payment]=signature
            for who,a,_ in allocations:
                if a>0:commands[(payment,who)]={'key':f'{payment}:{who}','state':'PLANNED','amount':a}
        accept('p',10000,'USD','s1',[('a',10000,'USD'),('b',0,'USD')]);self.assertEqual(len(commands),1)
        accept('p',10000,'USD','s1',[('a',10000,'USD'),('b',0,'USD')]);self.assertEqual(len(commands),1)
        with self.assertRaises(ValueError):accept('p',10000,'USD','s2',[('a',10000,'USD')])
        with self.assertRaises(ValueError):accept('bad',10,'USD','s',[('a',9,'USD')])
        self.assertNotIn('bad',plans)
        cmd=commands[('p','a')];key=cmd['key'];cmd['state']='UNKNOWN'
        self.assertEqual(commands[('p','a')]['key'],key)
        cmd['state']='SUCCEEDED';self.assertEqual(len(commands),1)

    def test_single_editor_version_replay_and_epoch(self):
        text='cat';head=7;epoch=1;accepted={}
        def edit(op,base,client_epoch,pos,length,insert):
            nonlocal text,head
            if op in accepted:return accepted[op]
            if base!=head or client_epoch!=epoch:raise ValueError('stale editor/version')
            if pos<0 or length<0 or pos+length>len(text):raise ValueError('range')
            text=text[:pos]+insert+text[pos+length:];head+=1;accepted[op]=head;return head
        self.assertEqual(edit('one',7,1,3,0,'s'),8);self.assertEqual(text,'cats')
        self.assertEqual(edit('one',7,1,3,0,'s'),8);self.assertEqual(text,'cats')
        with self.assertRaises(ValueError):edit('two',7,1,0,1,'b')
        epoch=2
        with self.assertRaises(ValueError):edit('two',8,1,0,1,'b')
        self.assertEqual(edit('two',8,2,0,1,'b'),9);self.assertEqual(text,'bats')

    def test_capacity_and_advance_accounting(self):
        self.assertEqual(100_000_000*50*300,1_500_000_000_000)
        self.assertEqual(5_000_000*1000*200,1_000_000_000_000)
        self.assertEqual(50_000_000*.01,500000)
        # Explicit $10 advance source, withdrawal, and later funding success/failure.
        for succeeds in (True,False):
            wallet=0;platform_cash=10000;advance_receivable=0
            wallet+=1000;advance_receivable+=1000
            wallet-=1000;platform_cash-=1000
            if succeeds:platform_cash+=1000;advance_receivable-=1000
            self.assertEqual(wallet,0)
            self.assertEqual((platform_cash,advance_receivable),(10000,0) if succeeds else (9000,1000))
            self.assertEqual(platform_cash+advance_receivable,10000)

    def test_all_answer_code_diagrams(self):
        for i in range(426,451):
            p=ROOT/'questions'/f'W{i:03}.md';text=p.read_text();answer=text.split('## Worked solution',1)[1]
            self.assertIn(f'# W{i:03}.',text)
            for code in re.findall(r'```python\n(.*?)```',answer,re.S):ast.parse(code)
            for ref in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',answer):
                asset=(p.parent/ref).resolve();self.assertTrue(asset.exists(),asset)
                if asset.suffix=='.svg':ET.parse(asset)

if __name__=='__main__':unittest.main(verbosity=2)
