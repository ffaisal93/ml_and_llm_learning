from pathlib import Path
import asyncio
import contextlib
import io
import re
import unittest
from ml_system_examples import capacity_estimate,known_event_count,canary_assignment,classify_batch


class SystemExamplesTests(unittest.TestCase):
    def test_capacity_units(self):
        self.assertEqual(capacity_estimate(8_640_000,5,.02),{'peak_rps':500.,'worker_slots':15})
        with self.assertRaises(ValueError):capacity_estimate(100,1,.02,0)

    def test_future_available_events_do_not_leak_into_history(self):
        events=[{'entity':'a','event_time':90,'available_time':95},
                {'entity':'a','event_time':91,'available_time':105},
                {'entity':'a','event_time':80,'available_time':80},
                {'entity':'b','event_time':90,'available_time':95}]
        self.assertEqual(known_event_count(events,'a',100,20),1)
        self.assertEqual(known_event_count(events,'a',110,30),2)

    def test_canary_assignment_is_stable_and_extremes_are_exact(self):
        self.assertEqual(canary_assignment('user','release',5),canary_assignment('user','release',5))
        self.assertEqual(canary_assignment('user','release',0),'control')
        self.assertEqual(canary_assignment('user','release',100),'canary')

    def test_all_documented_system_examples_execute(self):
        namespace={'__name__':'__main__'}
        page=Path(__file__).parent/'09_ml_system_design.md'
        for number,code in enumerate(re.findall(r'```python\n(.*?)```',page.read_text(),re.S)):
            with self.subTest(example=number),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code,f'System article example {number}','exec'),namespace)


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_results_are_not_silently_counted_as_success(self):
        async def provider(text):return {'label':{'invalid':'type'}} if text=='bad' else {'label':'yes'}
        result=await classify_batch([{'id':'a','text':'bad'},{'id':'b','text':'ok'}],provider,{'yes'})
        self.assertEqual(result['a']['status'],'invalid_output')
        self.assertEqual(result['b']['status'],'completed')

    async def test_transient_errors_retry_with_bounded_concurrency(self):
        active=0;maximum=0;attempts={}
        async def provider(text):
            nonlocal active,maximum
            active+=1;maximum=max(maximum,active)
            try:
                await asyncio.sleep(.001)
                attempts[text]=attempts.get(text,0)+1
                if text=='a' and attempts[text]==1:raise OSError('temporary')
                return {'label':'yes'}
            finally:active-=1
        result=await classify_batch([{'id':str(i),'text':str(i)} for i in range(4)]+[{'id':'a','text':'a'}],provider,{'yes'},concurrency=2)
        self.assertTrue(all(v['status']=='completed' for v in result.values()))
        self.assertEqual(attempts['a'],2)
        self.assertLessEqual(maximum,2)


if __name__=='__main__':unittest.main()
