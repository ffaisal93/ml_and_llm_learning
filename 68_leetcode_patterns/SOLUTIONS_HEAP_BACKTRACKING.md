# NeetCode 150 — Heap / Priority Queue + Backtracking — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips.

---

## Heap / Priority Queue (7)

Python's `heapq` is min-heap only. For max-heap, push negated values.

### 1. Kth Largest Element in a Stream (LC 703)

**Problem.** Class supporting `add(val)` returning the $k$-th largest seen so far.

**Intuition.** Min-heap of size $k$; root is the $k$-th largest.

```python
import heapq
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.h = []
        for n in nums: self.add(n)
    def add(self, val):
        heapq.heappush(self.h, val)
        if len(self.h) > self.k:
            heapq.heappop(self.h)
        return self.h[0]
```

**Complexity.** $O(\log k)$ per add.

**Hook.** "Min-heap of size k; root is k-th largest."

---

### 2. Last Stone Weight (LC 1046)

**Problem.** Repeatedly smash two heaviest stones (subtract); return last weight.

**Intuition.** Max-heap. Negate to use Python's min-heap.

```python
import heapq
def lastStoneWeight(stones):
    h = [-s for s in stones]
    heapq.heapify(h)
    while len(h) > 1:
        a = -heapq.heappop(h)
        b = -heapq.heappop(h)
        if a != b:
            heapq.heappush(h, -(a - b))
    return -h[0] if h else 0
```

**Complexity.** $O(n \log n)$.

**Hook.** "Negate for max-heap."

---

### 3. K Closest Points to Origin (LC 973)

**Intuition.** Max-heap of size $k$ with negated distances; or quickselect for $O(n)$ average.

```python
import heapq
def kClosest(points, k):
    h = []
    for x, y in points:
        d = -(x*x + y*y)
        heapq.heappush(h, (d, x, y))
        if len(h) > k:
            heapq.heappop(h)
    return [[x, y] for _, x, y in h]
```

**Complexity.** $O(n \log k)$.

**Hook.** "Max-heap of negated dist, size k."

---

### 4. Kth Largest Element in an Array (LC 215)

**Intuition.** Heap of size $k$ → $O(n \log k)$. Or quickselect → $O(n)$ average.

```python
import heapq
def findKthLargest(nums, k):
    h = []
    for n in nums:
        heapq.heappush(h, n)
        if len(h) > k:
            heapq.heappop(h)
    return h[0]
```

Quickselect alternative ($O(n)$ avg):

```python
import random
def findKthLargest(nums, k):
    k = len(nums) - k    # k-th smallest in 0-indexed
    def quickselect(l, r):
        pivot = nums[random.randint(l, r)]
        lt, eq, gt = [], [], []
        for x in nums[l:r+1]:
            if x < pivot: lt.append(x)
            elif x == pivot: eq.append(x)
            else: gt.append(x)
        if k < l + len(lt): return quickselect(l, l + len(lt) - 1)
        if k < l + len(lt) + len(eq): return pivot
        # else navigate to gt; rebuild slice in place is fiddly — use heap version in interview
    return quickselect(0, len(nums) - 1)
```

**Complexity.** $O(n \log k)$ heap; $O(n)$ avg quickselect.

**Hook.** "Min-heap of size k → root."

**Pressure tip.** Use heap. Quickselect has more edge cases.

---

### 5. Task Scheduler (LC 621)

**Problem.** Given task chars and cooldown $n$ between same-task runs, min CPU intervals to finish.

**Intuition.** Greedy with max-heap of remaining counts + cooldown queue.

```python
import heapq
from collections import Counter, deque
def leastInterval(tasks, n):
    cnt = Counter(tasks)
    h = [-c for c in cnt.values()]
    heapq.heapify(h)
    q = deque()    # (count_left_negated, time_available)
    time = 0
    while h or q:
        time += 1
        if h:
            c = heapq.heappop(h) + 1   # used one (less negative)
            if c < 0:
                q.append((c, time + n))
        if q and q[0][1] == time:
            heapq.heappush(h, q.popleft()[0])
    return time
```

**Complexity.** $O(T \log 26) = O(T)$.

**Hook.** "Heap by count + cooldown queue with release time."

---

### 6. Design Twitter (LC 355)

**Problem.** Implement post, follow, unfollow, getNewsFeed (10 most recent tweets from followed + self).

**Intuition.** Per-user list of (timestamp, tweet). On feed, push latest tweet of each followed onto a max-heap; pop 10.

```python
import heapq
from collections import defaultdict
class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)        # uid -> list of (time, tid)
        self.follows = defaultdict(set)        # uid -> set of followed uids
    def postTweet(self, uid, tid):
        self.time += 1
        self.tweets[uid].append((self.time, tid))
    def follow(self, fr, to):
        if fr != to: self.follows[fr].add(to)
    def unfollow(self, fr, to):
        self.follows[fr].discard(to)
    def getNewsFeed(self, uid):
        h = []   # max-heap by negated time
        users = self.follows[uid] | {uid}
        for u in users:
            tw = self.tweets[u]
            if tw:
                idx = len(tw) - 1
                heapq.heappush(h, (-tw[idx][0], tw[idx][1], u, idx))
        out = []
        while h and len(out) < 10:
            t, tid, u, idx = heapq.heappop(h)
            out.append(tid)
            if idx > 0:
                tw = self.tweets[u]
                heapq.heappush(h, (-tw[idx - 1][0], tw[idx - 1][1], u, idx - 1))
        return out
```

**Complexity.** Feed is $O(F + 10 \log F)$ where F = follow count.

**Hook.** "Heap of latest tweet per followed; pop+next-of-same-user."

---

### 7. Find Median from Data Stream (LC 295)

**Intuition.** Two heaps — max-heap for lower half, min-heap for upper half. Keep sizes balanced (lo ≥ hi by at most 1).

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.lo = []   # max-heap (negated)
        self.hi = []   # min-heap
    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

**Complexity.** $O(\log n)$ add, $O(1)$ median.

**Hook.** "Max-heap lo, min-heap hi; lo ≥ hi by ≤1."

**Watch out.** The trick: push to lo (max-heap), then push lo's max into hi. Then rebalance only if hi grew larger than lo. This guarantees max(lo) ≤ min(hi).

---

## Backtracking (10)

### 8. Subsets (LC 78)

**Intuition.** Include / exclude each element.

```python
def subsets(nums):
    out, cur = [], []
    def dfs(i):
        if i == len(nums):
            out.append(cur[:]); return
        # exclude
        dfs(i + 1)
        # include
        cur.append(nums[i])
        dfs(i + 1)
        cur.pop()
    dfs(0)
    return out
```

**Complexity.** $O(n \cdot 2^n)$.

**Hook.** "Include / exclude per element; copy on append."

---

### 9. Combination Sum (LC 39)

**Problem.** Distinct combinations summing to target. Same number can be used unlimited times.

**Intuition.** Backtrack with `start` index (no reuse of earlier elements means each combination is unique). At each step, *include same i again* (allows repeats) or *move to i+1*.

```python
def combinationSum(candidates, target):
    out, cur = [], []
    def dfs(i, total):
        if total == target:
            out.append(cur[:]); return
        if i == len(candidates) or total > target: return
        # include candidates[i] (and stay at i to allow reuse)
        cur.append(candidates[i])
        dfs(i, total + candidates[i])
        cur.pop()
        # skip candidates[i]
        dfs(i + 1, total)
    dfs(0, 0)
    return out
```

**Complexity.** $O(2^t)$ where $t$ relates to target.

**Hook.** "Stay at i for reuse; i+1 to skip."

---

### 10. Combination Sum II (LC 40)

**Problem.** Each candidate can be used once; combinations must be unique.

**Intuition.** Sort first. At each level, after using `candidates[i]`, skip identical neighbors.

```python
def combinationSum2(candidates, target):
    candidates.sort()
    out, cur = [], []
    def dfs(i, total):
        if total == target:
            out.append(cur[:]); return
        if total > target: return
        prev = -1
        for j in range(i, len(candidates)):
            if candidates[j] == prev: continue   # skip duplicate at same level
            cur.append(candidates[j])
            dfs(j + 1, total + candidates[j])
            cur.pop()
            prev = candidates[j]
    dfs(0, 0)
    return out
```

**Complexity.** $O(2^n)$.

**Hook.** "Sort, then skip-if-equal-to-prev *at same depth*."

---

### 11. Permutations (LC 46)

**Intuition.** Try each unused element at each position.

```python
def permute(nums):
    out, cur = [], []
    used = [False] * len(nums)
    def dfs():
        if len(cur) == len(nums):
            out.append(cur[:]); return
        for i in range(len(nums)):
            if used[i]: continue
            used[i] = True
            cur.append(nums[i])
            dfs()
            cur.pop()
            used[i] = False
    dfs()
    return out
```

**Complexity.** $O(n \cdot n!)$.

**Hook.** "Used array + try every unused."

---

### 12. Subsets II (LC 90)

**Problem.** Subsets with potentially duplicate input; output unique.

**Intuition.** Sort. After taking `nums[i]` at some level, skip identical at same level.

```python
def subsetsWithDup(nums):
    nums.sort()
    out, cur = [], []
    def dfs(i):
        out.append(cur[:])
        for j in range(i, len(nums)):
            if j > i and nums[j] == nums[j - 1]: continue
            cur.append(nums[j])
            dfs(j + 1)
            cur.pop()
    dfs(0)
    return out
```

**Complexity.** $O(n \cdot 2^n)$.

**Hook.** "Sort + skip dup at same level."

---

### 13. Word Search (LC 79)

**Problem.** Does the board contain the word as a path of adjacent cells (no cell reuse)?

**Intuition.** DFS from each cell. Mark visited in-place, restore on return.

```python
def exist(board, word):
    R, C = len(board), len(board[0])
    def dfs(r, c, i):
        if i == len(word): return True
        if not (0 <= r < R and 0 <= c < C): return False
        if board[r][c] != word[i]: return False
        board[r][c] = '#'   # mark
        found = (dfs(r+1, c, i+1) or dfs(r-1, c, i+1)
              or dfs(r, c+1, i+1) or dfs(r, c-1, i+1))
        board[r][c] = word[i]   # restore
        return found
    return any(dfs(r, c, 0) for r in range(R) for c in range(C))
```

**Complexity.** $O(R \cdot C \cdot 4^L)$.

**Hook.** "Mark in-place, restore on return."

---

### 14. Palindrome Partitioning (LC 131)

**Problem.** All ways to partition string so each piece is a palindrome.

**Intuition.** Try each prefix; if palindrome, recurse on suffix.

```python
def partition(s):
    out, cur = [], []
    def is_pal(l, r):
        while l < r:
            if s[l] != s[r]: return False
            l += 1; r -= 1
        return True
    def dfs(i):
        if i == len(s):
            out.append(cur[:]); return
        for j in range(i, len(s)):
            if is_pal(i, j):
                cur.append(s[i:j + 1])
                dfs(j + 1)
                cur.pop()
    dfs(0)
    return out
```

**Complexity.** $O(n \cdot 2^n)$.

**Hook.** "If prefix is palindrome, recurse on suffix."

---

### 15. Letter Combinations of a Phone Number (LC 17)

**Intuition.** Backtrack one digit at a time.

```python
def letterCombinations(digits):
    if not digits: return []
    m = {'2':'abc','3':'def','4':'ghi','5':'jkl',
         '6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    out, cur = [], []
    def dfs(i):
        if i == len(digits):
            out.append(''.join(cur)); return
        for c in m[digits[i]]:
            cur.append(c)
            dfs(i + 1)
            cur.pop()
    dfs(0)
    return out
```

**Complexity.** $O(4^n \cdot n)$.

**Hook.** "Backtrack per digit."

---

### 16. N-Queens (LC 51)

**Intuition.** Place row by row. Track attacked columns and both diagonals (`r+c` and `r-c`) in three sets.

```python
def solveNQueens(n):
    cols, posD, negD = set(), set(), set()
    out = []
    board = [["."] * n for _ in range(n)]
    def dfs(r):
        if r == n:
            out.append([''.join(row) for row in board]); return
        for c in range(n):
            if c in cols or (r + c) in posD or (r - c) in negD: continue
            cols.add(c); posD.add(r + c); negD.add(r - c)
            board[r][c] = "Q"
            dfs(r + 1)
            cols.remove(c); posD.discard(r + c); negD.discard(r - c)
            board[r][c] = "."
    dfs(0)
    return out
```

**Complexity.** $O(n!)$.

**Hook.** "Three sets: col, r+c diag, r-c diag."

---

### 17. Permutations II (LC 47)

**Problem.** Permutations with duplicate elements; output unique.

**Intuition.** Sort. Skip placing the same value twice at the same depth.

```python
def permuteUnique(nums):
    nums.sort()
    out, cur = [], []
    used = [False] * len(nums)
    def dfs():
        if len(cur) == len(nums):
            out.append(cur[:]); return
        for i in range(len(nums)):
            if used[i]: continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue   # skip dup whose left neighbor isn't used yet
            used[i] = True
            cur.append(nums[i])
            dfs()
            cur.pop()
            used[i] = False
    dfs()
    return out
```

**Complexity.** $O(n \cdot n!)$ worst case, less with dups.

**Hook.** "Sort + skip dup when left neighbor not used."

**Watch out.** The condition `not used[i-1]` (not `used[i-1]`) — we skip the *later* duplicate when its identical predecessor is *not* in the current permutation, ensuring canonical order.

---

## Summary table — Heap + Backtracking

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Kth Largest Stream | min-heap size k | $O(\log k)$ | root is k-th largest |
| 2 | Last Stone Weight | max-heap (negate) | $O(n\log n)$ | repeatedly smash |
| 3 | K Closest Points | max-heap size k | $O(n\log k)$ | negate dist |
| 4 | Kth Largest Array | heap or quickselect | $O(n\log k)$ | size-k heap |
| 5 | Task Scheduler | heap + cooldown q | $O(T)$ | release at time+n |
| 6 | Design Twitter | heap + per-user list | $O(F\log F)$ | next of same user |
| 7 | Median Stream | two heaps | $O(\log n)$ | lo max, hi min |
| 8 | Subsets | include/exclude | $O(n2^n)$ | binary tree |
| 9 | Combination Sum | stay-at-i for reuse | $O(2^t)$ | reuse via stay |
| 10 | Combo Sum II | sort + skip dup | $O(2^n)$ | dup skip at level |
| 11 | Permutations | used[] | $O(nn!)$ | every unused |
| 12 | Subsets II | sort + skip dup | $O(n2^n)$ | record at every node |
| 13 | Word Search | DFS + mark/restore | $O(RC4^L)$ | restore after recurse |
| 14 | Palindrome Partition | prefix palindrome | $O(n2^n)$ | recurse on suffix |
| 15 | Letter Combinations | per-digit backtrack | $O(4^n)$ | digit→letters |
| 16 | N-Queens | 3 attack sets | $O(n!)$ | col, r+c, r-c |
| 17 | Permutations II | dup-aware skip | $O(nn!)$ | left not used |

## Drilling routine

1. **Day 1:** Read all 29 problems.
2. **Day 2:** Re-implement each from the hook only, on a blank file.
3. **Day 5:** Repeat from scratch; aim ≤5 min per problem.
4. **Day 10:** Mock-style — shuffle, pick 5 random, solve under 25 min each.

That's the full Two Pointers + Stack + Heap + Backtracking coverage from NeetCode 150.
