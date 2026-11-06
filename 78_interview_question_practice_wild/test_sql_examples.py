"""Run the SQL printed in SQL_LAB.md, rather than a separate copy of the queries."""
from pathlib import Path
import re
import sqlite3
import unittest

BLOCKS = re.findall(r'```sql\n(.*?)```', Path(__file__).with_name('SQL_LAB.md').read_text(), re.S)


class SqlPracticeTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript(BLOCKS[0])

    def tearDown(self):
        self.db.close()

    def test_net_totals_and_half_open_dates(self):
        self.db.executemany('INSERT INTO transactions VALUES (?,?,?,?)', [
            (1,'A','2024-11-06',100),(2,'A','2025-01-01',-30),
            (3,'B','2025-01-01',80),(4,'C','2025-01-01',10),
            (5,'D','2026-11-06',999),(6,'D','2024-11-05',999)])
        rows = self.db.execute(BLOCKS[1], {'start_day':'2024-11-06','end_day':'2026-11-06'}).fetchall()
        self.assertEqual(rows,[('B',80.),('A',70.),('C',10.)])

    def test_duplicate_login_days(self):
        self.db.executemany('INSERT INTO logins VALUES (?,?)', [('a','2026-01-01'),('a','2026-01-02'),('a','2026-01-02'),('a','2026-01-04')])
        rows=self.db.execute(BLOCKS[2]).fetchall()
        self.assertEqual(rows,[('a','2026-01-01','2026-01-02',2),('a','2026-01-04','2026-01-04',1)])

    def test_plan_popularity_is_conditional(self):
        rows=[('open','free')]*3+[('open','paid')]+[('close','paid')]*3
        self.db.executemany('INSERT INTO events VALUES (?,?)',rows)
        self.assertEqual(self.db.execute(BLOCKS[3]).fetchall(),[('free',3)])

    def test_no_future_feature_and_missing_entity(self):
        self.db.executemany('INSERT INTO predictions VALUES (?,?,?)',[(1,'a',5),(2,'a',10),(3,'b',10)])
        self.db.executemany('INSERT INTO features VALUES (?,?,?)',[('a',3,30),('a',8,80),('a',11,110)])
        self.assertEqual(self.db.execute(BLOCKS[4]).fetchall(),[(1,'a',5,30.),(2,'a',10,80.),(3,'b',10,None)])

    def test_duplicate_clicks_do_not_inflate_ctr(self):
        self.db.executemany('INSERT INTO impressions VALUES (?,?)',[(1,'a'),(2,'a'),(3,'b')])
        self.db.executemany('INSERT INTO clicks VALUES (?)',[(1,),(1,),(3,)])
        self.assertEqual(self.db.execute(BLOCKS[5]).fetchall(),[('a',2,1,.5),('b',1,1,1.)])


if __name__ == '__main__':
    unittest.main()
