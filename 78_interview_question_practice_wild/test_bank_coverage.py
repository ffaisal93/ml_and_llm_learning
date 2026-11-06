"""Protect the complete bank and keep foundation exercises out of its counts."""
from pathlib import Path
from collections import Counter
import json
import re
import unittest

ROOT=Path(__file__).parent
BANDS=['FIRST_100.md','NEXT_100.md','THIRD_100.md','FOURTH_100.md','FIFTH_100.md','FINAL_47.md']


class BankCoverageTests(unittest.TestCase):
    def setUp(self):self.rows=json.loads((ROOT/'practice_questions.json').read_text())

    def test_all_547_unique_ids_and_positions(self):
        self.assertEqual(len(self.rows),547)
        self.assertEqual([r['id'] for r in self.rows],[f'W{i:03d}' for i in range(1,548)])
        self.assertEqual([r['position'] for r in self.rows],list(range(1,548)))
        self.assertEqual(len({r['path'] for r in self.rows}),547)

    def test_six_groups_with_exact_one_membership(self):
        self.assertEqual(Counter(r['group'] for r in self.rows),{1:100,2:100,3:100,4:100,5:100,6:47})
        found=[]
        for group,name in enumerate(BANDS,1):
            ids=re.findall(r'^- \[(W\d{3})\.',(ROOT/name).read_text(),re.M)
            self.assertEqual(set(ids),{r['id'] for r in self.rows if r['group']==group})
            self.assertEqual(len(ids),100 if group<6 else 47)
            found.extend(ids)
        self.assertEqual(len(found),len(set(found)))

    def test_each_question_and_answer_exists(self):
        for row in self.rows:
            text=(ROOT/row['path']).read_text()
            self.assertTrue(row['question'].strip(),row['id'])
            self.assertGreaterEqual(len(row['solution']['steps']),3,row['id'])
            self.assertTrue(row['solution']['example'],row['id'])
            self.assertTrue(row['solution']['trap'],row['id'])
            self.assertIn('## Worked solution',text)
            self.assertNotIn('Loading problem…',text)
            self.assertNotIn('The full problem is locked',text)

    def test_supporting_explainers_are_separate_and_linked(self):
        foundation=json.loads((ROOT/'foundation_questions.json').read_text())
        self.assertEqual([r['id'] for r in foundation],[f'F{i:03d}' for i in range(1,201)])
        for row in self.rows:
            text=(ROOT/row['path']).read_text()
            for path,anchor in re.findall(r'\]\(\.\./([^#)]+)#([^)]*)\)',text):
                self.assertIn(f'id="{anchor}"',(ROOT/path).read_text(),(row['id'],path,anchor))


if __name__=='__main__':unittest.main()
