# NeetCode 150 — Greedy + Math & Geometry — Solutions

> Simplest-optimal Python with intuition, memory hooks, pressure tips.

---

## Greedy (8)

### 1. Maximum Subarray (LC 53)

**Problem.** Max contiguous subarray sum (Kadane's algorithm).

**Intuition.** At each position, decide: extend current subarray, or start fresh.

```python
def maxSubArray(nums):
    cur = best = nums[0]
    for n in nums[1:]:
        cur = max(n, cur + n)
        best = max(best, cur)
    return best
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Kadane: extend or restart at each element."

---

### 2. Jump Game (LC 55)

**Problem.** Each value is max jump length. Can you reach the last index?

**Intuition.** Track farthest reachable. If at any point current index > farthest, fail.

```python
def canJump(nums):
    farthest = 0
    for i, n in enumerate(nums):
        if i > farthest: return False
        farthest = max(farthest, i + n)
    return True
```

**Complexity.** $O(n)$.

**Hook.** "Track farthest reachable; bail if current > farthest."

---

### 3. Jump Game II (LC 45)

**Problem.** Min jumps to reach the last index.

**Intuition.** BFS-style: process the current "level" (positions reachable in the current jump count); next level extends as far as possible.

```python
def jump(nums):
    jumps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            jumps += 1
            end = farthest
    return jumps
```

**Complexity.** $O(n)$.

**Hook.** "Greedy BFS; consume current level, bump jumps when crossing."

---

### 4. Gas Station (LC 134)

**Problem.** Circular route; can you complete it starting from some station?

**Intuition.** If `sum(gas) >= sum(cost)`, an answer exists. Greedy: walk through; when tank goes negative, restart from the next station.

```python
def canCompleteCircuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

**Complexity.** $O(n)$.

**Hook.** "Reset start to i+1 when tank goes negative; total ≥ 0 ⇒ unique start."

---

### 5. Hand of Straights (LC 846)

**Problem.** Can the hand be split into groups of `groupSize` consecutive cards?

**Intuition.** Counter + min-heap (or sorted keys). Always start a group at the smallest available card; consume groupSize consecutive.

```python
from collections import Counter
import heapq
def isNStraightHand(hand, groupSize):
    if len(hand) % groupSize: return False
    cnt = Counter(hand)
    h = list(cnt.keys())
    heapq.heapify(h)
    while h:
        smallest = h[0]
        for x in range(smallest, smallest + groupSize):
            if cnt[x] == 0: return False
            cnt[x] -= 1
            if cnt[x] == 0:
                if x != h[0]: return False   # gap
                heapq.heappop(h)
    return True
```

**Complexity.** $O(n \log n)$.

**Hook.** "Min-heap of distinct cards; build groups starting from smallest."

---

### 6. Merge Triplets to Form Target (LC 1899)

**Problem.** Given triplets, can you pick some so that element-wise max equals target?

**Intuition.** Discard any triplet that exceeds target in any coordinate. From the survivors, the max in each position must hit target's.

```python
def mergeTriplets(triplets, target):
    seen = [False, False, False]
    for a, b, c in triplets:
        if a > target[0] or b > target[1] or c > target[2]: continue
        if a == target[0]: seen[0] = True
        if b == target[1]: seen[1] = True
        if c == target[2]: seen[2] = True
    return all(seen)
```

**Complexity.** $O(n)$.

**Hook.** "Filter exceeders; check each coord can be hit."

---

### 7. Partition Labels (LC 763)

**Problem.** Partition into substrings so each char appears in only one part. Return part lengths.

**Intuition.** For each char, record last index. Sweep; track current part end; when current i == end, close the part.

```python
def partitionLabels(s):
    last = {c: i for i, c in enumerate(s)}
    out = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            out.append(end - start + 1)
            start = i + 1
    return out
```

**Complexity.** $O(n)$.

**Hook.** "Last-index map; close part when i hits running end."

---

### 8. Valid Parenthesis String (LC 678)

**Problem.** String of `(`, `)`, `*`. `*` can be `(`, `)`, or empty. Valid?

**Intuition.** Track range `[lo, hi]` of possible open counts. `(` increments both; `)` decrements both; `*` widens. If hi < 0, impossible. At end, lo must reach 0.

```python
def checkValidString(s):
    lo = hi = 0
    for c in s:
        if c == '(':
            lo += 1; hi += 1
        elif c == ')':
            lo -= 1; hi -= 1
        else:   # '*'
            lo -= 1; hi += 1
        if hi < 0: return False
        lo = max(lo, 0)
    return lo == 0
```

**Complexity.** $O(n)$.

**Hook.** "Range [lo, hi] of possible opens; clamp lo at 0."

**Watch out.** `lo = max(lo, 0)` — closing-paren count never goes effectively negative; we floor at 0 since impossible-too-many-closes drives one branch out, but valid branches survive.

---

## Math & Geometry (8)

### 9. Rotate Image (LC 48)

**Problem.** Rotate $n \times n$ matrix 90° clockwise in place.

**Intuition.** Two-step: transpose, then reverse each row.

```python
def rotate(matrix):
    n = len(matrix)
    # transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # reverse each row
    for row in matrix:
        row.reverse()
```

**Complexity.** $O(n^2)$ time, $O(1)$ space.

**Hook.** "Transpose + reverse rows = 90° CW. (Reverse rows + transpose = 90° CCW.)"

---

### 10. Spiral Matrix (LC 54)

**Problem.** Return all elements in spiral order.

**Intuition.** Maintain four shrinking bounds (top, bottom, left, right). Walk the perimeter; shrink.

```python
def spiralOrder(matrix):
    out = []
    top, bot = 0, len(matrix) - 1
    lft, rgt = 0, len(matrix[0]) - 1
    while top <= bot and lft <= rgt:
        for j in range(lft, rgt + 1): out.append(matrix[top][j])
        top += 1
        for i in range(top, bot + 1): out.append(matrix[i][rgt])
        rgt -= 1
        if top <= bot:
            for j in range(rgt, lft - 1, -1): out.append(matrix[bot][j])
            bot -= 1
        if lft <= rgt:
            for i in range(bot, top - 1, -1): out.append(matrix[i][lft])
            lft += 1
    return out
```

**Complexity.** $O(mn)$.

**Hook.** "Four bounds shrinking; check before going back."

**Watch out.** The two `if` checks before the bottom and left passes prevent re-visiting in skinny matrices.

---

### 11. Set Matrix Zeroes (LC 73)

**Problem.** If a cell is 0, zero out its row and column. In place, $O(1)$ space.

**Intuition.** Use first row/column as markers. Track separately whether the first row/col themselves need zeroing.

```python
def setZeroes(matrix):
    R, C = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][c] == 0 for c in range(C))
    first_col_zero = any(matrix[r][0] == 0 for r in range(R))
    for r in range(1, R):
        for c in range(1, C):
            if matrix[r][c] == 0:
                matrix[r][0] = 0
                matrix[0][c] = 0
    for r in range(1, R):
        for c in range(1, C):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0
    if first_row_zero:
        for c in range(C): matrix[0][c] = 0
    if first_col_zero:
        for r in range(R): matrix[r][0] = 0
```

**Complexity.** $O(RC)$ time, $O(1)$ space.

**Hook.** "First row/col as markers; track them separately."

---

### 12. Happy Number (LC 202)

**Problem.** Replace number with sum of digit-squares; happy if it eventually reaches 1.

**Intuition.** Either reaches 1 or enters a cycle. Use Floyd's tortoise-and-hare on the digit-square function.

```python
def isHappy(n):
    def step(x):
        s = 0
        while x:
            x, r = divmod(x, 10)
            s += r * r
        return s
    slow = fast = n
    while True:
        slow = step(slow)
        fast = step(step(fast))
        if fast == 1: return True
        if slow == fast: return False
```

**Complexity.** $O(\log n)$ per step; bounded total iterations.

**Hook.** "Floyd cycle on digit-squares."

---

### 13. Plus One (LC 66)

**Problem.** Add 1 to an integer represented as a digit array.

**Intuition.** Walk from right; carry until non-9 digit.

```python
def plusOne(digits):
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    return [1] + digits   # all 9s
```

**Complexity.** $O(n)$.

**Hook.** "Walk right; bail on first <9; prepend 1 if all 9."

---

### 14. Pow(x, n) (LC 50)

**Problem.** $x^n$ in $O(\log n)$.

**Intuition.** Fast exponentiation: square the base, halve the exponent.

```python
def myPow(x, n):
    if n < 0:
        x = 1 / x
        n = -n
    result = 1.0
    while n > 0:
        if n & 1: result *= x
        x *= x
        n >>= 1
    return result
```

**Complexity.** $O(\log |n|)$ time, $O(1)$ space.

**Hook.** "Square base, halve exponent."

**Watch out.** Negative `n`: invert `x` first.

---

### 15. Multiply Strings (LC 43)

**Problem.** Multiply two non-negative integers represented as strings.

**Intuition.** Schoolbook: each pair of digits contributes to position $i+j$ and possibly carries to $i+j-1$.

```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0": return "0"
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + res[p2]
            res[p1] += total // 10
            res[p2] = total % 10
    s = ''.join(map(str, res)).lstrip('0')
    return s or "0"
```

**Complexity.** $O(mn)$.

**Hook.** "Each (i, j) contributes to (i+j) carry, (i+j+1) digit."

---

### 16. Detect Squares (LC 2013)

**Problem.** Stream of points; queries return number of axis-aligned squares formed with stored points.

**Intuition.** Hash counts of points. For a query (x, y), iterate stored points; if same x or same y is degenerate; else if `|x - px| == |y - py|` and px ≠ x, count contribution = `count[(px, y)] * count[(x, py)] * count[(px, py)]`.

```python
from collections import defaultdict, Counter
class DetectSquares:
    def __init__(self):
        self.count = Counter()
        self.points = []
    def add(self, point):
        p = tuple(point)
        self.count[p] += 1
        self.points.append(p)
    def count_squares(self, point):
        x, y = point
        ans = 0
        for px, py in self.points:
            if abs(px - x) != abs(py - y) or px == x or py == y: continue
            ans += self.count[(px, y)] * self.count[(x, py)] * self.count[(px, py)]
        return ans
```

**Complexity.** $O(n)$ per query, $O(1)$ add.

**Hook.** "Diagonal corner ⇒ multiply 3 partner counts."

**Watch out.** Diagonals only; same-row or same-column doesn't make a square.

---

## Summary table — Greedy + Math/Geometry

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Max Subarray | Kadane | $O(n)$ | extend or restart |
| 2 | Jump Game | farthest reach | $O(n)$ | bail if i > far |
| 3 | Jump Game II | greedy BFS | $O(n)$ | level boundaries |
| 4 | Gas Station | reset on negative | $O(n)$ | start = i+1 |
| 5 | Hand of Straights | min-heap groups | $O(n\log n)$ | start at smallest |
| 6 | Merge Triplets | filter + flag | $O(n)$ | discard exceeders |
| 7 | Partition Labels | last-index sweep | $O(n)$ | close at running end |
| 8 | Valid Paren String | range [lo, hi] | $O(n)$ | clamp lo at 0 |
| 9 | Rotate Image | transpose + reverse | $O(n^2)$ | T+reverse rows |
| 10 | Spiral Matrix | 4 bounds shrink | $O(mn)$ | check before go back |
| 11 | Set Matrix Zeroes | row/col markers | $O(RC)$ | first row/col flags |
| 12 | Happy Number | Floyd cycle | $O(\log n)$ | digit-square fn |
| 13 | Plus One | rightward carry | $O(n)$ | bail on <9 |
| 14 | Pow(x, n) | fast exp | $O(\log n)$ | square + halve |
| 15 | Multiply Strings | schoolbook | $O(mn)$ | (i+j) and (i+j+1) |
| 16 | Detect Squares | diag + 3 multipliers | $O(n)$ q | corner counts |

---

## Cumulative NeetCode 150 coverage

After all solution files in this folder:

- Arrays & Hashing 9/9 ✅
- Two Pointers 5/5 ✅
- Sliding Window 6/6 ✅
- Stack 7/7 ✅
- Binary Search 7/7 ✅
- Linked List 11/11 ✅
- Trees 15/15 ✅
- Heap / Priority Queue 7/7 ✅
- Backtracking 10/10 ✅
- Graphs 13/13 ✅
- Advanced Graphs 6/6 ✅
- 1-D DP 12/12 ✅
- 2-D DP 11/11 ✅
- Greedy 8/8 ✅
- Math & Geometry 8/8 ✅

**Total: 135/150 problems solved.**

Remaining (all in the deep-dive patterns chapter, but not as full worked solutions):
- Tries (3)
- Intervals (~6)
- Bit Manipulation (~7)

Say the word and I'll add those too.
