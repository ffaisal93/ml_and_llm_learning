"""Runnable reference implementations for W201–W250; Python 3.9+."""
# BEGIN W221
import sys

def fit_two_segments(points):
    if not points:
        raise ValueError('at least one point required')
    points = sorted(points)
    n = len(points)
    total = sum(y for _, y in points)
    # The all-left partition is legal, including n=1/all x equal.
    best_score, best = total * total / n, (points[-1][0], total / n, total / n)
    left = 0.0
    for i, (x, y) in enumerate(points[:-1], 1):
        left += y
        if x == points[i][0]:
            continue
        right = total - left
        score = left * left / i + right * right / (n - i)
        if score > best_score:
            best_score = score
            # x itself avoids midpoint overflow and gives the same partition.
            best = (x, left / i, right / (n - i))
    return best   # t, b1, b2

def piecewise_main():
    data = sys.stdin.buffer
    n = int(data.readline())
    print(*fit_two_segments([tuple(map(float, data.readline().split())) for _ in range(n)]))
# END W221

# BEGIN W226
import math
import random

def ucb1(K, T, seed, probabilities):
    if len(probabilities) != K or K < 1 or T < 0:
        raise ValueError('invalid dimensions')
    rng = random.Random(seed)
    counts, sums = [0] * K, [0] * K
    total = 0
    for t in range(1, T + 1):
        if t <= K:
            arm = t - 1
        else:
            log_term = 2.0 * math.log(t)
            arm = max(range(K), key=lambda a: sums[a] / counts[a] + math.sqrt(log_term / counts[a]))
        reward = int(rng.random() < probabilities[arm])
        counts[arm] += 1
        sums[arm] += reward
        total += reward
    return total

def bandit_main():
    K, T, seed = map(int, sys.stdin.readline().split())
    print(ucb1(K, T, seed, list(map(float, sys.stdin.readline().split()))))
# END W226

# BEGIN W234
from collections import deque
import threading
import time

class SlidingWindowLimiter:
    """Accepted requests in (timestamp - window, timestamp]; monotonic per key."""
    def __init__(self, limit, window):
        if limit <= 0 or window <= 0:
            raise ValueError('positive limit/window required')
        self.limit, self.window = limit, window
        self.events, self.last = {}, {}
        self.lock = threading.Lock()

    def allow(self, user_id, timestamp):
        with self.lock:
            if timestamp < self.last.get(user_id, -float('inf')):
                raise ValueError('timestamps must not go backward per user')
            self.last[user_id] = timestamp
            q = self.events.setdefault(user_id, deque())
            while q and q[0] <= timestamp - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(timestamp)
            return True

    def expire_idle(self, now):
        """Call periodically with the same monotonic clock as allow()."""
        with self.lock:
            for key in list(self.last):
                if self.last[key] <= now - self.window:
                    self.last.pop(key)
                    self.events.pop(key, None)
# END W234

# BEGIN W235
class RateLimiter:
    def __init__(self, limit=5, window=10, clock=time.monotonic):
        self.clock = clock
        self.impl = SlidingWindowLimiter(limit, window)

    def allow(self, client_id):
        # Read clock under the same serialization as request ordering.
        with self.impl.lock:
            now = self.clock()
            q = self.impl.events.setdefault(client_id, deque())
            while q and q[0] <= now - self.impl.window:
                q.popleft()
            self.impl.last[client_id] = now
            if len(q) == self.impl.limit:
                return False
            q.append(now)
            return True

class Logger:
    def __init__(self, window=10):
        self.limiter = SlidingWindowLimiter(1, window)

    def shouldPrintMessage(self, timestamp, message):
        return self.limiter.allow(message, timestamp)
# END W235

# BEGIN W239
class TokenBucket:
    def __init__(self, capacity, refill_rate, clock=time.monotonic):
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError('positive policy required')
        self.capacity, self.rate, self.clock = capacity, refill_rate, clock
        self.tokens, self.updated = float(capacity), clock()
        self.lock = threading.Lock()

    def allow(self, cost=1):
        if not 0 < cost <= self.capacity:
            raise ValueError('cost must fit bucket')
        with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + max(0, now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < cost:
                return False, (cost - self.tokens) / self.rate
            self.tokens -= cost
            return True, 0.0
# END W239

# BEGIN W241
import re

def extract_time_and_mem_cost(program):
    tokens = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[+*/-]')
    rows, last_read, total = [], {}, 0
    for line in program.splitlines():
        if not line.strip():
            continue
        lhs, rhs = map(str.strip, line.split('=', 1))
        ts = tokens.findall(rhs)
        reads = [t for t in ts if t[0].isalpha() or t[0] == '_']
        total += 1 + len(reads) + sum(5 if t in '*/' else 1 for t in ts if t in ('+', '-', '*', '/'))
        i = len(rows)
        for name in reads:
            last_read[name] = i
        rows.append((lhs, reads))
    live, peak = set(), 0
    for i, (lhs, reads) in enumerate(rows):
        for name in reads:
            if last_read[name] == i:
                live.discard(name)
        live.add(lhs)
        peak = max(peak, len(live))
    return total, peak
# END W241

# BEGIN W242
from collections import defaultdict

def solve_crossword(board, dictionary):
    grid = [row[:] for row in board]
    m, n = len(grid), len(grid[0])
    slots = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '#':
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < m and 0 <= pc < n and grid[pr][pc] != '#':
                    continue
                cells, rr, cc = [], r, c
                while 0 <= rr < m and 0 <= cc < n and grid[rr][cc] != '#':
                    cells.append((rr, cc)); rr += dr; cc += dc
                if len(cells) >= 2:
                    slots.append(cells)
    owners = defaultdict(list)
    for s, cells in enumerate(slots):
        for pos, cell in enumerate(cells):
            owners[cell].append((s, pos))
    if any(grid[r][c] == '.' and (r, c) not in owners for r in range(m) for c in range(n)):
        return None
    words = defaultdict(list)
    for word in sorted(set(dictionary)):
        words[len(word)].append(word)
    masks = {}
    for length, ws in words.items():
        for j, w in enumerate(ws):
            for pos, char in enumerate(w):
                key = length, pos, char
                masks[key] = masks.get(key, 0) | (1 << j)
    domains = []
    for cells in slots:
        length = len(cells)
        mask = (1 << len(words[length])) - 1
        for pos, (r, c) in enumerate(cells):
            if grid[r][c] != '.':
                mask &= masks.get((length, pos, grid[r][c]), 0)
        domains.append(mask)
    def search(ds, remaining):
        if not remaining:
            return [row[:] for row in grid]
        s = min(remaining, key=lambda j: bin(ds[j]).count('1'))
        mask = ds[s]
        if not mask:
            return None
        cells, rest = slots[s], remaining - {s}
        while mask:
            bit = mask & -mask; mask -= bit
            word = words[len(cells)][bit.bit_length() - 1]
            new_ds, changed, valid = ds[:], [], True
            for (r, c), char in zip(cells, word):
                if grid[r][c] == '.':
                    changed.append((r, c)); grid[r][c] = char
                for other, pos in owners[(r, c)]:
                    if other in rest:
                        new_ds[other] &= masks.get((len(slots[other]), pos, char), 0)
                        if not new_ds[other]:
                            valid = False
            if valid:
                result = search(new_ds, rest)
                if result is not None:
                    return result
            for r, c in changed:
                grid[r][c] = '.'
        return None
    return search(domains, set(range(len(slots))))
# END W242

# BEGIN W243
from bisect import bisect_right

class VersionedSocialNetwork:
    def __init__(self):
        self.version = 0
        self.history = {}  # edge -> sorted (version, value), coalesced within version

    def _write(self, follower, followee, value):
        h = self.history.setdefault((follower, followee), [])
        if h and h[-1][0] == self.version:
            h[-1] = (self.version, value)
        elif (h[-1][1] if h else False) != value:
            h.append((self.version, value))

    def follow(self, followerId, followeeId):
        self._write(followerId, followeeId, True)

    def unfollow(self, followerId, followeeId):
        self._write(followerId, followeeId, False)

    def createSnapshot(self):
        result = self.version
        self.version += 1
        return result

    def isFollowing(self, followerId, followeeId, snapId):
        # The example queries the mutable current version, so allow it explicitly.
        if not 0 <= snapId <= self.version:
            raise ValueError('unknown version')
        h = self.history.get((followerId, followeeId), [])
        i = bisect_right(h, (snapId, True)) - 1
        return h[i][1] if i >= 0 else False
# END W243

# BEGIN W244
from abc import ABC, abstractmethod

class Bot(ABC):
    @abstractmethod
    def can_handle(self, text: str) -> bool:
        pass

    @abstractmethod
    def handle(self, user_id: str, text: str) -> str:
        pass

class HelpBot(Bot):
    def can_handle(self, text): return text.startswith('!help')
    def handle(self, user_id, text): return '[help] Available commands: !help, !schedule, !weather'

class ScheduleBot(Bot):
    def can_handle(self, text): return text.startswith('!schedule')
    def handle(self, user_id, text): return f'[schedule] Okay {user_id}, your meeting is scheduled.'

class WeatherBot(Bot):
    def can_handle(self, text): return text.startswith('!weather')
    def handle(self, user_id, text): return '[weather] It is sunny today.'

class EchoBot(Bot):
    def can_handle(self, text): return text.startswith('!echo')
    def handle(self, user_id, text): return text[5:]

class ChatChannel:
    def __init__(self, name: str, bots):
        self.name, self.bots, self.messages = name, list(bots), []

    def record_message(self, user_id, text):
        self.messages.append({'user': user_id, 'text': text})

    def dispatch(self, user_id, text):
        for bot in self.bots:
            if bot.can_handle(text):
                return bot.handle(user_id, text)
        return '[error] Unknown command.'

class ChatServer:
    def __init__(self): self.channels = {}

    def register_channel(self, name: str, bots):
        self.channels[name] = ChatChannel(name, bots)

    def send_message(self, channel: str, user_id: str, text: str):
        if channel not in self.channels:
            raise ValueError('unknown channel')
        target = self.channels[channel]
        target.record_message(user_id, text)
        return target.dispatch(user_id, text)

_server = ChatServer()
_server.register_channel('default', [HelpBot(), ScheduleBot(), WeatherBot(), EchoBot()])

def llm_response(user_id: str, text: str) -> str:
    return _server.send_message('default', user_id, text)
# END W244

# BEGIN W247
class SingleFlightCache:
    def __init__(self, compute):
        self.compute, self.values, self.inflight = compute, {}, set()
        self.cv = threading.Condition(threading.Lock())

    def get(self, key):
        with self.cv:
            while key in self.inflight:
                self.cv.wait()  # always recheck after spurious wakeups
            if key in self.values:
                return self.values[key]  # None is a valid value
            self.inflight.add(key)
        try:
            value = self.compute(key)  # no global lock during expensive work
        except BaseException:
            with self.cv:
                self.inflight.remove(key)
                self.cv.notify_all()
            raise
        with self.cv:
            self.values[key] = value
            self.inflight.remove(key)
            self.cv.notify_all()
            return value
# END W247

# BEGIN W248
import heapq
import logging
from concurrent.futures import ThreadPoolExecutor

class JobScheduler:
    """Fixed-rate, skip missed ticks, no overlap per job; bounded dispatch queue."""
    def __init__(self, workers=8):
        self.cv = threading.Condition()
        self.jobs, self.heap, self.next_id = {}, [], 1
        self.closed = False
        self.pending, self.max_pending = 0, workers * 2
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def schedule(self, fn, frequency_seconds):
        if not callable(fn) or not math.isfinite(frequency_seconds) or frequency_seconds <= 0:
            raise ValueError('callable and finite positive interval required')
        with self.cv:
            if self.closed:
                raise RuntimeError('scheduler closed')
            job_id = self.next_id; self.next_id += 1
            self.jobs[job_id] = {'fn': fn, 'period': frequency_seconds, 'busy': False}
            heapq.heappush(self.heap, (time.monotonic() + frequency_seconds, job_id))
            self.cv.notify_all()
            return job_id

    def deschedule(self, job_id):
        with self.cv:
            found = self.jobs.pop(job_id, None) is not None
            self.cv.notify_all()
            return found

    def _loop(self):
        with self.cv:
            while not self.closed:
                # Lazy deletion is bounded by compaction below.
                while self.heap and self.heap[0][1] not in self.jobs:
                    heapq.heappop(self.heap)
                if len(self.heap) > 2 * len(self.jobs) + 100:
                    self.heap = [(t, j) for t, j in self.heap if j in self.jobs]
                    heapq.heapify(self.heap)
                if not self.heap:
                    self.cv.wait(); continue
                due, job_id = self.heap[0]
                delay = due - time.monotonic()
                if delay > 0:
                    self.cv.wait(delay); continue
                heapq.heappop(self.heap)
                job = self.jobs[job_id]
                now = time.monotonic()
                ticks = max(1, math.floor((now - due) / job['period']) + 1)
                heapq.heappush(self.heap, (due + ticks * job['period'], job_id))
                if not job['busy'] and self.pending < self.max_pending:
                    job['busy'] = True
                    self.pending += 1
                    self.pool.submit(self._execute, job_id)

    def _execute(self, job_id):
        with self.cv:
            job = self.jobs.get(job_id)
            if job is None or self.closed:
                self.pending -= 1
                return
            fn = job['fn']  # invocation starts/linearizes under this same lock
        try:
            fn()
        except BaseException:
            logging.exception('Scheduled job failed: %s', job_id)
        finally:
            with self.cv:
                job['busy'] = False
                self.pending -= 1
                self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.jobs.clear()
            self.cv.notify_all()
        self.thread.join()
        self.pool.shutdown(wait=True)
# END W248
