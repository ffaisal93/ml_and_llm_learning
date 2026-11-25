from bisect import bisect_left, bisect_right


class GPUIdleIntervalTracker:
    def __init__(self):
        self.busy = {}

    def add_task(self, start, end, node_id):
        if start >= end:
            raise ValueError("task must be nonempty")
        intervals = self.busy.setdefault(node_id, [])
        left = bisect_left(intervals, (start,))
        if left and intervals[left - 1][1] >= start:
            left -= 1
        right = left
        while right < len(intervals) and intervals[right][0] <= end:
            start = min(start, intervals[right][0])
            end = max(end, intervals[right][1])
            right += 1
        intervals[left:right] = [(start, end)]

    def get_idle_intervals(self, node_id, window_start, window_end):
        if window_start > window_end:
            raise ValueError("reversed window")
        if window_start == window_end:
            return []
        intervals = self.busy.get(node_id, [])
        result = []
        cursor = window_start
        position = bisect_right(intervals, (window_start, float("inf")))
        if position and intervals[position - 1][1] > window_start:
            position -= 1
        while position < len(intervals):
            start, end = intervals[position]
            if start >= window_end:
                break
            if start > cursor:
                result.append((cursor, min(start, window_end)))
            cursor = max(cursor, min(end, window_end))
            position += 1
        if cursor < window_end:
            result.append((cursor, window_end))
        return result


t = GPUIdleIntervalTracker()
for a, b in [(10, 20), (30, 40), (18, 35)]:
    t.add_task(a, b, "A")
assert t.get_idle_intervals("A", 0, 50) == [(0, 10), (40, 50)]
for a, b in [(5, 10), (15, 20), (10, 15)]:
    t.add_task(a, b, "B")
assert t.get_idle_intervals("B", 0, 25) == [(0, 5), (20, 25)]
assert t.get_idle_intervals("B", 7, 17) == [] and t.get_idle_intervals("C", 3, 8) == [
    (3, 8)
]
assert t.get_idle_intervals("B", 3, 3) == []
# Independent integer-cell coverage oracle, random insertion order.
import random

rng = random.Random(0)
t = GPUIdleIntervalTracker()
occupied = set()
for _ in range(100):
    a = rng.randrange(-30, 30)
    b = a + rng.randrange(1, 8)
    t.add_task(a, b, "x")
    occupied.update(range(a, b))
    gaps = t.get_idle_intervals("x", -35, 40)
    actual = {p for a, b in gaps for p in range(a, b)}
    assert actual == set(range(-35, 40)) - occupied
before = {k: v[:] for k, v in t.busy.items()}
try:
    t.add_task(1, 1, "new")
except ValueError:
    pass
assert t.busy == before
print(
    "W028: examples, bisected windows, no-mutation rejection and independent coverage oracle passed"
)
