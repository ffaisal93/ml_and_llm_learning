from pathlib import Path
import asyncio
import contextlib
import importlib.util
import io
import re
import unittest
from rag_agent_examples import (Chunk, retrieve, extractive_answer, word_chunks,
    reciprocal_rank_fusion, retrieval_recall, pack_context, validate_search_call,
    run_agent, success_summary, rank_answers)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks=[Chunk('a','public',1,'Annual leave allowance is 20 days.'),
                     Chunk('b','private',1,'Private salary records.'),
                     Chunk('c','public',1,'Carry over allowance is 5 days.')]
        self.permissions={'public':{'alice'},'private':{'director'}}

    def test_permissions_and_revocation_precede_evidence_output(self):
        self.assertEqual(extractive_answer('salary',self.chunks,self.permissions,'alice')['status'],'no_evidence')
        self.assertEqual(extractive_answer('annual leave',self.chunks,self.permissions,'alice')['citations'],['public@v1#a'])
        self.permissions['public'].remove('alice')
        self.assertEqual(retrieve('annual leave',self.chunks,self.permissions,'alice'),[])

    def test_empty_unknown_queries_and_stable_versions(self):
        self.assertEqual(retrieve('',self.chunks,self.permissions,'alice'),[])
        self.assertEqual(retrieve('unseen-word',self.chunks,self.permissions,'alice'),[])
        self.assertNotEqual(Chunk('a','public',1,'x').citation,Chunk('a','public',2,'x').citation)

    def test_word_chunk_boundaries_and_invalid_stride(self):
        self.assertEqual(word_chunks('a b c d e f g h',4,1),['a b c d','d e f g','g h'])
        self.assertEqual(word_chunks('',4,0),[])
        self.assertEqual(word_chunks('a b c d',4,1),['a b c d'])
        with self.assertRaises(ValueError):word_chunks('text',2,2)

    def test_rank_fusion_duplicate_ids_do_not_add_votes(self):
        a=reciprocal_rank_fusion([['a','a','b'],['b','a']])
        b=reciprocal_rank_fusion([['a','b'],['b','a']])
        self.assertEqual(a,b)
        self.assertEqual([item for item,score in a],['a','b'])

    def test_multi_evidence_recall_and_context_budget(self):
        self.assertEqual(retrieval_recall(['a','a','x'],['a','b']),.5)
        with self.assertRaises(ValueError):retrieval_recall(['a'],[])
        a=Chunk('a','doc',1,'one two three')
        b=Chunk('b','doc',1,'four five')
        self.assertEqual(pack_context([(1,a),(.9,a),(.8,b)],5),[a,b])
        self.assertEqual(pack_context([(1,a),(.8,b)],2),[b])

    def test_validator_rejects_invalid_scope_shape_and_boolean_limit(self):
        base={'type':'tool','name':'search','arguments':{'query':'leave','collection':'staff','limit':3}}
        self.assertEqual(validate_search_call(base,{'staff'})['limit'],3)
        for limit in [True,-1,11]:
            bad={**base,'arguments':{**base['arguments'],'limit':limit}}
            with self.assertRaises(ValueError):validate_search_call(bad,{'staff'})
        for collection in ['private',[]]:
            bad={**base,'arguments':{**base['arguments'],'collection':collection}}
            with self.assertRaises(PermissionError):validate_search_call(bad,{'staff'})

    def test_task_weighting_differs_from_run_weighting(self):
        result=success_summary([{'task':'a','success':True},{'task':'a','success':False},{'task':'b','success':True}])
        self.assertEqual(result['macro_success'],.75)

    def test_all_documented_rag_examples_execute(self):
        namespace={'__name__':'__main__'}
        page=Path(__file__).parent/'08_rag_and_agents.md'
        for number,code in enumerate(re.findall(r'```python\n(.*?)```',page.read_text(),re.S)):
            if code.startswith('# Optional PyTorch') and importlib.util.find_spec('torch') is None:continue
            with self.subTest(example=number),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code,f'RAG article example {number}','exec'),namespace)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_repetition_and_unverified_finish(self):
        action={'type':'tool','name':'search','arguments':{'query':'leave','collection':'staff','limit':1}}
        calls=[]
        async def search(**arguments):calls.append(arguments);return '20 days'
        async def planner(history):return action if not history else {'type':'finish','answer':'20 days'}
        result=await run_agent(planner,search,lambda a,h:a=='20 days' and len(h)==1,{'staff'})
        self.assertEqual(result['status'],'completed')
        async def repeated(history):return action
        result=await run_agent(repeated,search,lambda a,h:False,{'staff'})
        self.assertEqual(result['status'],'repeated_call')
        self.assertEqual(len(calls),2)
        async def unsupported(history):return {'type':'finish','answer':'unsupported'}
        result=await run_agent(unsupported,search,lambda a,h:False,{'staff'})
        self.assertEqual(result['status'],'unverified_answer')

    async def test_forbidden_action_never_executes(self):
        called=[]
        async def planner(history):return {'type':'tool','name':'upload','arguments':{}}
        async def tool(**args):called.append(args)
        result=await run_agent(planner,tool,lambda a,h:False,{'staff'})
        self.assertEqual(result['status'],'invalid_action')
        self.assertEqual(called,[])

    async def test_timeout_and_bounded_tool_failure(self):
        action={'type':'tool','name':'search','arguments':{'query':'q','collection':'staff','limit':1}}
        async def planner(history):return action
        async def slow(**args):await asyncio.sleep(.1)
        result=await run_agent(planner,slow,lambda a,h:False,{'staff'},per_call_seconds=.005)
        self.assertEqual(result['status'],'timeout')
        attempts=[]
        async def failed(**args):attempts.append(1);raise OSError('temporary failure')
        result=await run_agent(planner,failed,lambda a,h:False,{'staff'})
        self.assertEqual(result['status'],'tool_failure')
        self.assertEqual(len(attempts),2)

    async def test_pairwise_cycle_is_a_tie_not_a_total_order(self):
        async def judge(a,b):return 'left' if (a,b) in {('A','B'),('B','C'),('C','A')} else 'right'
        result=await rank_answers({'A':'A','B':'B','C':'C'},judge)
        self.assertEqual(result['scores'],dict.fromkeys(['A','B','C'],.5))
        self.assertEqual(len(result['comparisons']),3)
        with self.assertRaises(ValueError):await rank_answers({'A':'A','B':'B','C':'C'},judge,max_pairs=2)

    async def test_judge_concurrency_and_failure_reporting(self):
        active=0;maximum=0
        async def judge(a,b):
            nonlocal active,maximum
            active+=1;maximum=max(maximum,active)
            await asyncio.sleep(.001)
            active-=1
            return 'left'
        await rank_answers({str(i):str(i) for i in range(5)},judge,concurrency=2)
        self.assertLessEqual(maximum,2)
        attempts=[]
        async def invalid(a,b):attempts.append(1);return {'unexpected':'shape'}
        result=await rank_answers({'a':'a','b':'b'},invalid,retries=1)
        self.assertEqual(result['failures'],[('a','b')])
        self.assertEqual(result['scores'],{'a':None,'b':None})
        self.assertEqual(len(attempts),2)


if __name__=='__main__':unittest.main()
