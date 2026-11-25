import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from airline_406 import Airline, HoldExpired, PaymentDeclined, PaymentStub
from event_logger_425 import EventLogger


class FakeClock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


class BookingTests(unittest.TestCase):
    def make_airline(self, directory, payments=None, clock=None):
        clock = clock or FakeClock()
        payments = payments or PaymentStub()
        airline = Airline(Path(directory) / 'flights.db', payments, clock=clock)
        airline.add_flight('F', 'BOS', 'LAX', 1000, rows=1)
        return airline, payments, clock

    def seat_owner(self, airline):
        with sqlite3.connect(airline.path) as db:
            return db.execute("SELECT state,hold_id FROM seats WHERE flight_id='F' AND seat='1A'").fetchone()

    def test_expiry_purchase_cancel_and_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory)
            self.assertEqual(len(a.search(departure='BOS', end=1001)), 1)
            self.assertEqual(a.search(arrival='NYC'), [])
            first = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(ValueError):
                a.hold('F', '1A', 'bob')
            clock.now = 10
            with self.assertRaises(HoldExpired):
                a.purchase(first, 'alice', 100)
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))
            second = a.hold('F', '1A', 'bob')
            for _ in range(2):
                self.assertEqual(a.purchase(second, 'bob', 100), 'charge:' + second)
            self.assertEqual(len(payments.operations), 1)
            a.cancel(second, 'bob')
            a.cancel(second, 'bob')
            self.assertEqual(len(payments.operations), 2)
            third = a.hold('F', '1A', 'carol')
            a.cancel(second, 'bob')
            self.assertEqual(self.seat_owner(a), ('HELD', third))

    def test_two_threads_one_seat(self):
        with tempfile.TemporaryDirectory() as directory:
            a, _, _ = self.make_airline(directory)
            barrier = threading.Barrier(2)

            def hold(user):
                barrier.wait(timeout=2)
                try:
                    return a.hold('F', '1A', user)
                except ValueError:
                    return None

            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                results = list(pool.map(hold, ['a', 'b']))
            self.assertEqual(sum(result is not None for result in results), 1)

    def test_stale_start_snapshot_cannot_extend_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory)
            hid = a.hold('F', '1A', 'a', ttl=10)
            clock.now = 10
            with self.assertRaises(HoldExpired):
                a.purchase(hid, 'a', 100, now=0)
            self.assertEqual(payments.operations, {})
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))

    def test_clock_advanced_inside_provider_triggers_refund(self):
        clock = FakeClock()

        class Slow(PaymentStub):
            def charge(self, key, amount):
                reference = super().charge(key, amount)
                clock.now = 10  # Exactly the deadline, after request admission.
                return reference

        with tempfile.TemporaryDirectory() as directory:
            a, payments, _ = self.make_airline(directory, Slow(), clock)
            hid = a.hold('F', '1A', 'a', ttl=10)
            with self.assertRaises(HoldExpired):
                a.purchase(hid, 'a', 100, now=0)
            self.assertEqual(a.booking_status(hid, 'a')['state'], 'COMPENSATED')
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))
            self.assertEqual(payments.operations['refund:' + hid], ('refund', 'charge:' + hid, 100))

    def test_late_success_cannot_steal_new_held_or_sold_seat(self):
        for sell_replacement in (False, True):
            with self.subTest(sell_replacement=sell_replacement), tempfile.TemporaryDirectory() as directory:
                entered, release = threading.Event(), threading.Event()

                class DelayedReply(PaymentStub):
                    blocked_key = None

                    def charge(self, key, amount):
                        reference = super().charge(key, amount)
                        if key == self.blocked_key:
                            entered.set()
                            if not release.wait(3):
                                raise TimeoutError('Test release did not arrive')
                        return reference

                a, payments, clock = self.make_airline(directory, DelayedReply())
                old = a.hold('F', '1A', 'alice', ttl=10)
                payments.blocked_key = 'charge:' + old
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(a.purchase, old, 'alice', 100)
                    try:
                        self.assertTrue(entered.wait(2))
                        clock.now = 10
                        new = a.hold('F', '1A', 'bob')  # Lazy expiry also expires PAYMENT_PENDING.
                        if sell_replacement:
                            a.purchase(new, 'bob', 200)
                        release.set()
                        with self.assertRaises(HoldExpired):
                            future.result(timeout=3)
                    finally:
                        release.set()
                expected = 'SOLD' if sell_replacement else 'HELD'
                self.assertEqual(self.seat_owner(a), (expected, new))
                status = a.booking_status(old, 'alice')
                self.assertEqual((status['state'], status['refund_state']), ('COMPENSATED', 'SUCCEEDED'))
                self.assertEqual(payments.operations['refund:' + old][2], 100)
                self.assertEqual(a.reconcile(), [])
                self.assertEqual(a.reconcile(), [])
                self.assertEqual(self.seat_owner(a), (expected, new))
                self.assertEqual(sum(op[0] == 'refund' for op in payments.operations.values()), 1)

    def test_lost_charge_and_refund_responses_recover_after_restart(self):
        class LostReplies(PaymentStub):
            lost_charge = False
            lost_refund = False
            refund_calls = 0

            def charge(self, key, amount):
                reference = super().charge(key, amount)
                if not self.lost_charge:
                    self.lost_charge = True
                    raise TimeoutError('Lost response after successful charge')
                return reference

            def refund(self, key, charge_id, amount):
                self.refund_calls += 1
                reference = super().refund(key, charge_id, amount)
                if not self.lost_refund:
                    self.lost_refund = True
                    raise TimeoutError('Lost response after successful refund')
                return reference

        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory, LostReplies())
            old = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(TimeoutError):
                a.purchase(old, 'alice', 100)
            clock.now = 10
            self.assertEqual(dict(a.seat_map('F'))['1A'], 'AVAILABLE')
            new = a.hold('F', '1A', 'bob')
            errors = a.reconcile()
            self.assertEqual(len(errors), 1)
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'COMPENSATION_PENDING')
            self.assertEqual(self.seat_owner(a), ('HELD', new))
            recovered = Airline(a.path, payments, clock=clock)
            self.assertEqual(recovered.reconcile(), [])
            self.assertEqual(recovered.booking_status(old, 'alice')['state'], 'COMPENSATED')
            self.assertEqual(recovered.reconcile(), [])
            self.assertEqual(payments.refund_calls, 2)  # Two attempts, one provider refund effect.
            self.assertEqual(sum(op[0] == 'refund' for op in payments.operations.values()), 1)
            self.assertEqual(self.seat_owner(recovered), ('HELD', new))
            with sqlite3.connect(a.path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM payment_operation_errors").fetchone()[0], 2)

    def test_unseen_expired_charge_is_voided_not_replayed(self):
        class LostBeforeAcceptance(PaymentStub):
            def charge(self, key, amount):
                raise TimeoutError('Request never reached provider')

        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory, LostBeforeAcceptance())
            old = a.hold('F', '1A', 'alice', ttl=1)
            with self.assertRaises(TimeoutError):
                a.purchase(old, 'alice', 100)
            clock.now = 1
            self.assertEqual(a.reconcile(), [])
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'EXPIRED')
            self.assertEqual(payments.operations, {'charge:' + old: ('void',)})
            with self.assertRaises(PaymentDeclined):
                PaymentStub.charge(payments, 'charge:' + old, 100)
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))

    def test_unexpired_lost_response_finalizes_without_second_charge(self):
        class LostOnce(PaymentStub):
            def charge(self, key, amount):
                super().charge(key, amount)
                raise TimeoutError('Lost successful response')

        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory, LostOnce())
            hid = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(TimeoutError):
                a.purchase(hid, 'alice', 100)
            clock.now = 9
            self.assertEqual(a.reconcile(), [])
            self.assertEqual(a.booking_status(hid, 'alice')['state'], 'PAID')
            self.assertEqual(self.seat_owner(a), ('SOLD', hid))
            self.assertEqual(len(payments.operations), 1)

    def test_unknown_provider_status_does_not_extend_inventory_ttl(self):
        class StillUnknown(PaymentStub):
            def charge(self, key, amount):
                raise TimeoutError('Pending provider outcome')

            def void_charge(self, key):
                return {'state': 'PENDING'}

        with tempfile.TemporaryDirectory() as directory:
            a, _, clock = self.make_airline(directory, StillUnknown())
            old = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(TimeoutError):
                a.purchase(old, 'alice', 100)
            clock.now = 10
            new = a.hold('F', '1A', 'bob')
            self.assertEqual(a.reconcile(), [])
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'EXPIRED_PAYMENT_PENDING')
            self.assertEqual(self.seat_owner(a), ('HELD', new))



    def test_concurrent_compensation_has_one_refund_effect(self):
        refund_barrier = threading.Barrier(2)

        class ConcurrentRefund(PaymentStub):
            def charge(self, key, amount):
                super().charge(key, amount)
                raise TimeoutError('Charge accepted but reply lost')

            def refund(self, key, charge_id, amount):
                reference = super().refund(key, charge_id, amount)
                refund_barrier.wait(timeout=3)
                return reference

        with tempfile.TemporaryDirectory() as directory:
            a, payments, clock = self.make_airline(directory, ConcurrentRefund())
            old = a.hold('F', '1A', 'alice', ttl=1)
            with self.assertRaises(TimeoutError):
                a.purchase(old, 'alice', 100)
            clock.now = 1
            new = a.hold('F', '1A', 'bob')
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(a.reconcile) for _ in range(2)]
                self.assertEqual([future.result(timeout=5) for future in futures], [[], []])
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'COMPENSATED')
            self.assertEqual(sum(op[0] == 'refund' for op in payments.operations.values()), 1)
            self.assertEqual(self.seat_owner(a), ('HELD', new))



    def test_deadline_rechecked_between_claim_and_charge_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            a, payments, _ = self.make_airline(directory)
            old = a.hold('F', '1A', 'alice', ttl=10)
            readings = iter([0, 10, 10])
            a.clock = lambda: next(readings, 10)
            with self.assertRaises(HoldExpired):
                a.purchase(old, 'alice', 100, now=0)
            self.assertEqual(payments.operations, {})
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))
            self.assertEqual(a.reconcile(), [])
            self.assertEqual(payments.operations, {'charge:' + old: ('void',)})

    def test_definitive_decline_releases_before_deadline(self):
        class Declined(PaymentStub):
            def charge(self, key, amount):
                raise PaymentDeclined('Definitive decline, no debit')

        with tempfile.TemporaryDirectory() as directory:
            a, _, _ = self.make_airline(directory, Declined())
            old = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(PaymentDeclined):
                a.purchase(old, 'alice', 100)
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'PAYMENT_FAILED')
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))



    def test_unknown_response_after_deadline_releases_immediately(self):
        clock = FakeClock()

        class SlowLostReply(PaymentStub):
            def charge(self, key, amount):
                super().charge(key, amount)
                clock.now = 10
                raise TimeoutError('Unknown response at hold deadline')

        with tempfile.TemporaryDirectory() as directory:
            a, _, _ = self.make_airline(directory, SlowLostReply(), clock)
            old = a.hold('F', '1A', 'alice', ttl=10)
            with self.assertRaises(TimeoutError):
                a.purchase(old, 'alice', 100)
            # Direct DB inspection: no seat_map/hold/manual expiry call needed here.
            self.assertEqual(self.seat_owner(a), ('AVAILABLE', None))
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'EXPIRED_PAYMENT_PENDING')
            self.assertEqual(a.reconcile(), [])
            self.assertEqual(a.booking_status(old, 'alice')['state'], 'COMPENSATED')


class LoggerTests(unittest.TestCase):
 def test_group_commit_and_recovery(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'events.log';syncs=[]
   def sync(fd):syncs.append(1);os.fsync(fd)
   logger=EventLogger(path,max_batch=10,flush_seconds=.2,sync=sync)
   futures=[logger.append({'value':i}) for i in range(10)]
   self.assertEqual([f.result(timeout=2) for f in futures],list(range(1,11)))
   self.assertEqual(len(syncs),1);logger.close()
   with path.open('ab') as file:file.write(b'{torn')
   recovered=EventLogger(path);self.assertEqual(recovered.append('after').result(timeout=2),11);recovered.close()
   records=[json.loads(line) for line in path.read_text().splitlines()]
   self.assertEqual(len(records),11);self.assertEqual(records[-1]['event'],'after')
 def test_failure_fails_future_and_future_appends(self):
  with tempfile.TemporaryDirectory() as directory:
   def bad_sync(fd):raise OSError('Disk failure')
   logger=EventLogger(Path(directory)/'events.log',sync=bad_sync)
   future=logger.append('x')
   with self.assertRaises(OSError):future.result(timeout=2)
   with self.assertRaises(RuntimeError):logger.append('later')
   with self.assertRaises(RuntimeError):logger.close()
 def test_close_drains_and_payload_snapshot(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'events.log';logger=EventLogger(path)
   event={'value':1};future=logger.append(event);event['value']=99
   logger.close();self.assertEqual(future.result(),1)
   self.assertEqual(json.loads(path.read_text())['event']['value'],1)
   self.assertFalse(future.cancel())
   with self.assertRaises(RuntimeError):logger.append('later')

if __name__=='__main__':unittest.main()
