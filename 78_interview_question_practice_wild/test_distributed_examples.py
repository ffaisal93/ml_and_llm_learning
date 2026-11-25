from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import contextlib
import io
import re
import unittest
from distributed_examples import TokenBucket,ready_jobs


class DistributedExamplesTests(unittest.TestCase):
    def test_burst_refill_and_capacity(self):
        bucket=TokenBucket(5,1,0)
        self.assertTrue(bucket.allow(3,0))
        self.assertFalse(bucket.allow(3,0))
        self.assertTrue(bucket.allow(3,1))
        self.assertFalse(bucket.allow(6,100))
        self.assertEqual(bucket.tokens,5)

    def test_parallel_consumers_cannot_spend_one_token_twice(self):
        bucket=TokenBucket(1,1,0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            allowed=list(pool.map(lambda _:bucket.allow(1,0),range(30)))
        self.assertEqual(sum(allowed),1)

    def test_invalid_costs_and_backward_clock_are_rejected(self):
        bucket=TokenBucket(1,1,0)
        with self.assertRaises(ValueError):bucket.allow(0,0)
        with self.assertRaises(ValueError):bucket.allow(float('nan'),0)
        self.assertTrue(bucket.allow(1,2))
        with self.assertRaises(ValueError):bucket.allow(1,1)

    def test_dependencies_require_success_not_dispatch(self):
        graph={'a':[],'b':['a'],'c':['b']}
        self.assertEqual(ready_jobs(graph,{'a':'RUNNING','b':'PENDING','c':'PENDING'}),[])
        self.assertEqual(ready_jobs(graph,{'a':'SUCCEEDED','b':'PENDING','c':'PENDING'}),['b'])
        with self.assertRaises(ValueError):ready_jobs({'b':['missing']},{'b':'PENDING'})

    def test_documented_distributed_examples_execute(self):
        namespace={'__name__':'__main__'}
        text=(Path(__file__).parent/'17_distributed_systems_depth.md').read_text()
        for number,code in enumerate(re.findall(r'```python\n(.*?)```',text,re.S)):
            with self.subTest(example=number),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code,f'Distributed article example {number}','exec'),namespace)


if __name__=='__main__':unittest.main()
