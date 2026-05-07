# NeetCode 150 — 1-D + 2-D Dynamic Programming — Solutions

> Simplest-optimal Python with intuition, memory hooks, pressure tips. DP shows up in nearly every senior-level interview.

**General DP recipe.** (1) Define state precisely. (2) Write recurrence. (3) Identify base cases. (4) Choose top-down memo or bottom-up table. (5) Optimize space if only last few values needed.

---

## 1-D DP (12)

### 1. Climbing Stairs (LC 70)

**Problem.** Each step you can climb 1 or 2 stairs. Number of ways to climb $n$.

**State.** `dp[i]` = ways to reach step $i$. **Recurrence.** `dp[i] = dp[i-1] + dp[i-2]`. (Fibonacci.)

```python
def climbStairs(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Fibonacci."

---

### 2. Min Cost Climbing Stairs (LC 746)

**Problem.** Each step has a cost. Min cost to reach top (past last step).

```python
def minCostClimbingStairs(cost):
    a, b = 0, 0
    for c in cost:
        a, b = b, min(a, b) + c
    return min(a, b)
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Fib variant; pay then advance."

---

### 3. House Robber (LC 198)

**Problem.** Max sum without robbing two adjacent houses.

**Recurrence.** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` (skip vs take).

```python
def rob(nums):
    a, b = 0, 0   # a = dp[i-2], b = dp[i-1]
    for n in nums:
        a, b = b, max(b, a + n)
    return b
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Skip or take + previous-previous."

---

### 4. House Robber II (LC 213)

**Problem.** Houses arranged in a circle (first and last are adjacent).

**Intuition.** Run House Robber twice: nums[:-1] and nums[1:]. Take max.

```python
def rob(nums):
    if len(nums) == 1: return nums[0]
    def rob_line(arr):
        a, b = 0, 0
        for n in arr:
            a, b = b, max(b, a + n)
        return b
    return max(rob_line(nums[:-1]), rob_line(nums[1:]))
```

**Complexity.** $O(n)$.

**Hook.** "Skip first or skip last; take max."

---

### 5. Longest Palindromic Substring (LC 5)

**Intuition.** For each center (n + n−1 centers), expand outward while characters match. Track the longest.

```python
def longestPalindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l + 1:r]
    best = ""
    for i in range(len(s)):
        for a, b in ((i, i), (i, i + 1)):
            cand = expand(a, b)
            if len(cand) > len(best): best = cand
    return best
```

**Complexity.** $O(n^2)$ time, $O(1)$ space.

**Hook.** "Expand around 2n−1 centers."

---

### 6. Palindromic Substrings (LC 647)

**Intuition.** Same expand-around-center; count instead of tracking longest.

```python
def countSubstrings(s):
    def count_at(l, r):
        c = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            c += 1; l -= 1; r += 1
        return c
    return sum(count_at(i, i) + count_at(i, i + 1) for i in range(len(s)))
```

**Complexity.** $O(n^2)$.

**Hook.** "Count, not track longest."

---

### 7. Decode Ways (LC 91)

**Problem.** Count decodings of digit string ('A'=1..'Z'=26).

**Recurrence.** `dp[i] = (dp[i-1] if s[i-1] != '0') + (dp[i-2] if 10 <= int(s[i-2:i]) <= 26)`.

```python
def numDecodings(s):
    if not s or s[0] == '0': return 0
    a, b = 1, 1   # a = dp[i-2], b = dp[i-1]
    for i in range(1, len(s)):
        cur = 0
        if s[i] != '0': cur += b
        two = int(s[i-1:i+1])
        if 10 <= two <= 26: cur += a
        a, b = b, cur
    return b
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Single-digit valid? Two-digit in 10..26?"

**Watch out.** '0' alone is invalid; '00' is invalid; '06' is invalid (single digit '0' bad).

---

### 8. Coin Change (LC 322)

**Problem.** Min coins to make amount (or -1).

**Recurrence.** `dp[a] = min(dp[a - c] + 1 for c in coins)`. Unbounded knapsack — coin can repeat.

```python
def coinChange(coins, amount):
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != INF else -1
```

**Complexity.** $O(\text{amount} \cdot |\text{coins}|)$.

**Hook.** "Min over coins; INF sentinel."

---

### 9. Maximum Product Subarray (LC 152)

**Problem.** Max product of a contiguous subarray.

**Intuition.** Track both max and min ending at $i$ (negative × negative = positive flip).

```python
def maxProduct(nums):
    cur_max = cur_min = best = nums[0]
    for n in nums[1:]:
        if n < 0: cur_max, cur_min = cur_min, cur_max
        cur_max = max(n, cur_max * n)
        cur_min = min(n, cur_min * n)
        best = max(best, cur_max)
    return best
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Track max AND min; swap on negative."

---

### 10. Word Break (LC 139)

**Problem.** Can `s` be segmented into words from dictionary?

**Recurrence.** `dp[i] = any(dp[j] and s[j:i] in dict for j < i)`.

```python
def wordBreak(s, wordDict):
    words = set(wordDict)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[len(s)]
```

**Complexity.** $O(n^2 \cdot L)$.

**Hook.** "Prefix breakable + suffix in dict."

---

### 11. Longest Increasing Subsequence (LC 300)

**Intuition (O(n log n)).** Maintain `tails` array — `tails[k]` = smallest possible tail of an LIS of length $k+1$. Binary-search to find where each new element fits.

```python
from bisect import bisect_left
def lengthOfLIS(nums):
    tails = []
    for n in nums:
        i = bisect_left(tails, n)
        if i == len(tails): tails.append(n)
        else:               tails[i] = n
    return len(tails)
```

**Complexity.** $O(n \log n)$.

**Hook.** "Patience sorting; bisect_left."

**Watch out.** `tails` does not contain the actual LIS — only its length. For strictly-increasing use `bisect_left`; for non-decreasing use `bisect_right`.

---

### 12. Partition Equal Subset Sum (LC 416)

**Problem.** Can the array be split into two subsets with equal sum?

**Intuition.** Reduces to: can we reach `total / 2`? 0/1 knapsack on bool DP.

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp |= {x + n for x in dp if x + n <= target}
        if target in dp: return True
    return target in dp
```

**Complexity.** $O(n \cdot \text{target})$ time, $O(\text{target})$ space.

**Hook.** "Subset sum to total/2; set of reachable sums."

---

## 2-D DP (11)

### 13. Unique Paths (LC 62)

**Problem.** Number of paths in $m \times n$ grid from top-left to bottom-right (only right/down).

**Recurrence.** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`. Or closed form $\binom{m+n-2}{m-1}$.

```python
def uniquePaths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```

**Complexity.** $O(mn)$ time, $O(n)$ space.

**Hook.** "Pascal triangle; one row."

---

### 14. Longest Common Subsequence (LC 1143)

**Recurrence.** Match: `dp[i][j] = dp[i-1][j-1] + 1`. Mismatch: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.

```python
def longestCommonSubsequence(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**Complexity.** $O(mn)$.

**Hook.** "Match → diag+1; else → max of left/up."

---

### 15. Best Time Buy/Sell with Cooldown (LC 309)

**Problem.** Multiple buy/sell, but after selling you must wait one day.

**State.** Three states: holding, sold (just sold), rest.

```python
def maxProfit(prices):
    hold, sold, rest = float('-inf'), 0, 0
    for p in prices:
        prev_sold = sold
        sold = hold + p
        hold = max(hold, rest - p)
        rest = max(rest, prev_sold)
    return max(sold, rest)
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "3 states: hold / sold / rest. sold ⇒ rest only."

---

### 16. Coin Change II (LC 518)

**Problem.** Number of ways to make amount (unbounded coins).

**Intuition.** Iterate coins outer, amount inner — avoids double-counting permutations as different ways.

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]
```

**Complexity.** $O(\text{amount} \cdot |\text{coins}|)$.

**Hook.** "Coins outer; combinations not perms."

---

### 17. Target Sum (LC 494)

**Problem.** Number of ways to assign +/− to reach target.

**Reduction.** If we let $P$ = sum of positives and $N$ = sum of negatives: $P + N = S$ and $P - N = T$, so $P = (S + T)/2$. Reduces to subset-sum count for $P$.

```python
def findTargetSumWays(nums, target):
    s = sum(nums)
    if abs(target) > s or (s + target) % 2: return 0
    P = (s + target) // 2
    dp = [0] * (P + 1)
    dp[0] = 1
    for n in nums:
        for a in range(P, n - 1, -1):
            dp[a] += dp[a - n]
    return dp[P]
```

**Complexity.** $O(n \cdot P)$.

**Hook.** "Reduce to subset sum count for (S+T)/2."

**Watch out.** Loop amount *backwards* to make 0/1 (each num once).

---

### 18. Interleaving String (LC 97)

**Problem.** Is `s3` an interleaving of `s1` and `s2`?

**State.** `dp[i][j]` = is `s3[:i+j]` an interleaving of `s1[:i]` and `s2[:j]`?

```python
def isInterleave(s1, s2, s3):
    if len(s1) + len(s2) != len(s3): return False
    m, n = len(s1), len(s2)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0 and dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]:
                dp[i][j] = True
            if j > 0 and dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]:
                dp[i][j] = True
    return dp[m][n]
```

**Complexity.** $O(mn)$.

**Hook.** "Match s3 char with s1 (left) or s2 (up)."

---

### 19. Longest Increasing Path in a Matrix (LC 329)

**Intuition.** DFS + memoization. Each cell's longest path is independent of how you got there.

```python
def longestIncreasingPath(matrix):
    R, C = len(matrix), len(matrix[0])
    memo = {}
    def dfs(r, c):
        if (r, c) in memo: return memo[r, c]
        best = 1
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        memo[r, c] = best
        return best
    return max(dfs(r, c) for r in range(R) for c in range(C))
```

**Complexity.** $O(RC)$ — each cell computed once.

**Hook.** "DFS + memo; visit only strict increases (DAG, no visited needed)."

---

### 20. Distinct Subsequences (LC 115)

**Problem.** Number of times `t` appears as subsequence in `s`.

**Recurrence.** `dp[i][j]` = ways `t[:j]` appears in `s[:i]`. If chars match: `dp[i-1][j-1] + dp[i-1][j]`; else `dp[i-1][j]`.

```python
def numDistinct(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]
```

**Complexity.** $O(mn)$.

**Hook.** "Match → take or skip; else → skip only."

---

### 21. Edit Distance (LC 72)

**Recurrence.** Match: `dp[i-1][j-1]`. Else: `1 + min(insert, delete, replace) = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])`.

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
```

**Complexity.** $O(mn)$.

**Hook.** "Insert/delete/replace = +1 each."

---

### 22. Burst Balloons (LC 312)

**Problem.** Max coins from popping balloons (coins = left × cur × right neighbor in the moment of pop).

**Intuition.** Range DP. `dp[l][r]` = max coins from bursting all in `(l, r)` exclusive. Try each `k` as *last* burst in the range: `dp[l][r] = max(nums[l] * nums[k] * nums[r] + dp[l][k] + dp[k][r])`. Pad with 1 on both ends.

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for l in range(n - length):
            r = l + length
            for k in range(l + 1, r):
                dp[l][r] = max(dp[l][r],
                               nums[l] * nums[k] * nums[r] + dp[l][k] + dp[k][r])
    return dp[0][n - 1]
```

**Complexity.** $O(n^3)$.

**Hook.** "Range DP; k = LAST balloon burst in range."

**Watch out.** Critical insight: think of `k` as the *last* one popped, not the first — that way `nums[l]` and `nums[r]` are still intact when you multiply.

---

### 23. Regular Expression Matching (LC 10)

**Problem.** Match string against pattern with `.` (any char) and `*` (zero or more of previous).

**Recurrence.**
- If `p[j] == '*'`: `dp[i][j] = dp[i][j-2] (zero copies) or (matches[i-1, j-1] and dp[i-1][j])`.
- Else: `dp[i][j] = matches[i-1, j-1] and dp[i-1][j-1]`.

```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    # patterns like a*, a*b*, etc. can match empty
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]   # zero copies
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]
```

**Complexity.** $O(mn)$.

**Hook.** "`*` = zero copies (j-2) OR match-and-stay (i-1, j)."

---

## Summary table — DP

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Climbing Stairs | Fib | $O(n)$ | dp[i] = dp[i-1] + dp[i-2] |
| 2 | Min Cost Climb | Fib variant | $O(n)$ | pay then advance |
| 3 | House Robber | skip or take | $O(n)$ | max(prev, prev2 + cur) |
| 4 | House Robber II | circle | $O(n)$ | skip first OR last |
| 5 | Longest Pal Substring | expand center | $O(n^2)$ | 2n−1 centers |
| 6 | Palindromic Substrings | count centers | $O(n^2)$ | count instead of track |
| 7 | Decode Ways | 1-digit + 2-digit | $O(n)$ | valid? 10..26? |
| 8 | Coin Change | unbounded knapsack | $O(\text{amt} \cdot c)$ | min over coins |
| 9 | Max Product Subarray | track min too | $O(n)$ | swap on negative |
| 10 | Word Break | prefix DP | $O(n^2 L)$ | dp[j] and s[j:i] in dict |
| 11 | LIS | patience | $O(n\log n)$ | bisect_left tails |
| 12 | Partition Equal | subset sum | $O(n \cdot S)$ | sum to total/2 |
| 13 | Unique Paths | grid DP | $O(mn)$ | left + up |
| 14 | LCS | match diag, else max | $O(mn)$ | dp[i-1][j-1] + 1 |
| 15 | Buy/Sell Cooldown | 3-state | $O(n)$ | hold/sold/rest |
| 16 | Coin Change II | combos | $O(\text{amt} \cdot c)$ | coins outer |
| 17 | Target Sum | reduce subset | $O(nS)$ | (S+T)/2 |
| 18 | Interleaving | grid DP | $O(mn)$ | from left or up |
| 19 | Longest Inc Path | DFS + memo | $O(RC)$ | DAG via strict increase |
| 20 | Distinct Subseq | take or skip | $O(mn)$ | match → both |
| 21 | Edit Distance | 3 ops | $O(mn)$ | ins/del/repl |
| 22 | Burst Balloons | range DP | $O(n^3)$ | k = last burst |
| 23 | RegEx Match | star handling | $O(mn)$ | * = j-2 OR match&stay |
