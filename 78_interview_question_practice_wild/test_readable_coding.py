from pathlib import Path
import ast
import copy
import json
import re
import unittest
import readable_coding as r

ROOT=Path(__file__).parent


class ReadabilityTests(unittest.TestCase):
    def test_bpe_roundtrip_unicode_and_determinism(self):
        samples=['low lower','lowest low','café 🙂 café']
        first=r.train_bpe(samples,8)
        self.assertEqual(first,r.train_bpe(samples,8))
        for text in samples+['unseen🙂']:
            self.assertEqual(''.join(r.encode_bpe(text,first)),text)

    def test_bpe_nonoverlap_and_empty(self):
        self.assertEqual(r.train_bpe(['aaaa'],1),[('a','a')])
        self.assertEqual(r.encode_bpe('aaaa',[('a','a')]),['aa','aa'])
        self.assertEqual(r.train_bpe([],3),[])
        self.assertEqual(r.encode_bpe('',[]),[])

    def test_crossword_reuse_and_input_immutability(self):
        board=[['.','.'],['.','.']]
        before=copy.deepcopy(board)
        self.assertEqual(r.solve_crossword(board,['AA']),[['A','A'],['A','A']])
        self.assertEqual(board,before)

    def test_crossword_isolated_and_fixed_mismatch(self):
        self.assertIsNone(r.solve_crossword([['.']],['A']))
        self.assertEqual(r.solve_crossword([['A']],[]),[['A']])
        self.assertIsNone(r.solve_crossword([['A','A']],['BB']))

    def test_every_bank_page_has_structured_statement_and_depth(self):
        rows=json.loads((ROOT/'practice_questions.json').read_text())
        self.assertEqual(len(rows),547)
        for row in rows:
            text=(ROOT/row['path']).read_text()
            self.assertNotIn('<pre class="question-text"',text,row['id'])
            self.assertIn('## Worked solution',text,row['id'])
            self.assertGreaterEqual(len(text.split('## Worked solution',1)[1].split()),300,row['id'])
            self.assertTrue(row['solution']['teaching_notes_markdown'],row['id'])

    def test_python_blocks_and_layernorm_topic(self):
        for path in (ROOT/'questions').glob('*.md'):
            for block in re.findall(r'```python\n(.*?)```',path.read_text(),re.S):
                ast.parse(block,filename=str(path))
        layernorm=(ROOT/'questions/W007.md').read_text().lower()
        self.assertNotIn('tokenizer',layernorm)
        self.assertIn('central differences',layernorm)
        self.assertIn('$$',layernorm)


if __name__=='__main__':unittest.main()
