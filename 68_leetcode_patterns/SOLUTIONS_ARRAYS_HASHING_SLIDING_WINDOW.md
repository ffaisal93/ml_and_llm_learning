# NeetCode 150 — Arrays & Hashing + Sliding Window — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips. Read once; re-implement the next day from the hook alone.

---

## Arrays & Hashing (9)

### 1. Contains Duplicate (LC 217)

**Problem.** Return True if any value appears at least twice.

**Intuition.** Compare set size to list size; equal means all unique.

```python
def containsDuplicate(nums):
    return len(set(nums)) != len(nums)
```

**Complexity.** $O(n)$ time, $O(n)$ space.

**Hook.** "Set vs len."

---

### 2. Valid Anagram (LC 242)

**Problem.** Are `s` and `t` anagrams?

**Intuition.** Same character counts.

```python
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
```

**Complexity.** $O(n)$ time, $O(1)$ space (26 chars).

**Hook.** "Counter equal."

**Watch out.** Different lengths → False (Counter handles this).

---

### 3. Two Sum (LC 1)

**Problem.** Indices of two numbers summing to target.

**Intuition.** As you scan, ask "have I already seen `target - x`?" Hash map gives $O(1)$ lookup.

```python
def twoSum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```

**Complexity.** $O(n)$ time, $O(n)$ space.

**Hook.** "Have I seen the complement."

**Watch out.** Insert *after* checking, so `nums[i] + nums[i] == target` doesn't return `[i, i]`.

---

### 4. Group Anagrams (LC 49)

**Problem.** Group strings that are anagrams of each other.

**Intuition.** Anagrams share a canonical key — sorted chars or 26-char-count tuple.

```python
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))   # or [count of each char]
        groups[key].append(s)
    return list(groups.values())
```

**Complexity.** $O(n \cdot k \log k)$ with sorted key (k = max length); $O(n \cdot k)$ with count tuple.

**Hook.** "Hash by canonical form."

---

### 5. Top K Frequent Elements (LC 347)

**Problem.** Return the $k$ most frequent elements.

**Intuition.** Bucket sort: index = frequency, value = list of elements with that frequency. Walk from high to low. $O(n)$.

```python
from collections import Counter
def topKFrequent(nums, k):
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for x, c in freq.items():
        buckets[c].append(x)
    out = []
    for c in range(len(buckets) - 1, 0, -1):
        out.extend(buckets[c])
        if len(out) >= k:
            return out[:k]
```

**Complexity.** $O(n)$ time and space.

**Hook.** "Frequency-as-index bucket."

**Alt.** Heap of size $k$ → $O(n \log k)$. Bucket sort is cleaner here.

---

### 6. Encode and Decode Strings (LC 271)

**Problem.** Encode list of strings to one string; decode back.

**Intuition.** Length-prefix each string with delimiter so decoder knows where each ends.

```python
def encode(strs):
    return ''.join(f"{len(s)}#{s}" for s in strs)

def decode(s):
    out, i = [], 0
    while i < len(s):
        j = s.index('#', i)
        L = int(s[i:j])
        out.append(s[j+1:j+1+L])
        i = j + 1 + L
    return out
```

**Complexity.** $O(n)$ both.

**Hook.** "Length#string."

**Watch out.** Strings can contain `#`, but length-prefix makes that safe — you read exactly `L` chars after the `#`.

---

### 7. Product of Array Except Self (LC 238)

**Problem.** `out[i] = product of all elements except nums[i]`. No division.

**Intuition.** Two passes: prefix product from left, then suffix product from right multiplied in.

```python
def productExceptSelf(nums):
    n = len(nums)
    out = [1] * n
    pre = 1
    for i in range(n):
        out[i] = pre
        pre *= nums[i]
    suf = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suf
        suf *= nums[i]
    return out
```

**Complexity.** $O(n)$ time, $O(1)$ extra (output not counted).

**Hook.** "Left prefix, right suffix, multiply in place."

---

### 8. Valid Sudoku (LC 36)

**Problem.** Is the partially filled 9x9 board valid?

**Intuition.** For each cell, build three keys: (row, val), (col, val), (box, val). All keys must be unique.

```python
def isValidSudoku(board):
    seen = set()
    for i in range(9):
        for j in range(9):
            v = board[i][j]
            if v == '.': continue
            keys = (f"r{i}{v}", f"c{j}{v}", f"b{i//3}{j//3}{v}")
            if any(k in seen for k in keys): return False
            seen.update(keys)
    return True
```

**Complexity.** $O(81) = O(1)$.

**Hook.** "Three keys per cell."

---

### 9. Longest Consecutive Sequence (LC 128)

**Problem.** Longest run of consecutive integers (in any order in the array).

**Intuition.** Put nums in a set. For each `n`, only start counting if `n-1` not in set (so each run starts at its smallest). Each number visited at most twice → $O(n)$ amortized.

```python
def longestConsecutive(nums):
    s = set(nums)
    best = 0
    for n in s:
        if n - 1 in s: continue   # not a run start
        cur = n
        while cur + 1 in s: cur += 1
        best = max(best, cur - n + 1)
    return best
```

**Complexity.** $O(n)$ time and space.

**Hook.** "Only start at run-starts (n where n-1 not in set)."

**Watch out.** Without the `n-1` skip, you re-walk runs and become $O(n^2)$.

---

## Sliding Window (6)

### 10. Best Time to Buy and Sell Stock (LC 121)

**Problem.** Max profit from one buy + one sell.

**Intuition.** Track running min price; max profit at each day = price - running_min.

```python
def maxProfit(prices):
    lo = prices[0]
    best = 0
    for p in prices[1:]:
        lo = min(lo, p)
        best = max(best, p - lo)
    return best
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Running min, running profit."

---

### 11. Longest Substring Without Repeating Characters (LC 3)

**Problem.** Longest substring with all unique characters.

**Intuition.** Sliding window. When you see a duplicate, shrink from the left until it's gone.

```python
def lengthOfLongestSubstring(s):
    seen = {}
    l, best = 0, 0
    for r, c in enumerate(s):
        if c in seen and seen[c] >= l:
            l = seen[c] + 1
        seen[c] = r
        best = max(best, r - l + 1)
    return best
```

**Complexity.** $O(n)$ time, $O(\min(n, |\Sigma|))$ space.

**Hook.** "Jump l past last duplicate."

**Watch out.** `seen[c] >= l` check — old positions before `l` are stale.

---

### 12. Longest Repeating Character Replacement (LC 424)

**Problem.** Longest substring after replacing at most $k$ chars with the same char.

**Intuition.** Window is valid iff `(window length) - (max char count in window) ≤ k`. Expand always; shrink when invalid.

```python
from collections import Counter
def characterReplacement(s, k):
    cnt = Counter()
    l, best, max_c = 0, 0, 0
    for r, c in enumerate(s):
        cnt[c] += 1
        max_c = max(max_c, cnt[c])
        if (r - l + 1) - max_c > k:
            cnt[s[l]] -= 1
            l += 1
        best = max(best, r - l + 1)
    return best
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Window − max-count ≤ k."

**Watch out.** `max_c` doesn't need to be re-validated on shrink — best length only grows when a higher max_c is seen.

---

### 13. Permutation in String (LC 567)

**Problem.** Does `s2` contain a permutation of `s1` as substring?

**Intuition.** Fixed-size sliding window of length $|s1|$. Compare character counts.

```python
from collections import Counter
def checkInclusion(s1, s2):
    if len(s1) > len(s2): return False
    need = Counter(s1)
    have = Counter(s2[:len(s1)])
    if need == have: return True
    for r in range(len(s1), len(s2)):
        have[s2[r]] += 1
        have[s2[r - len(s1)]] -= 1
        if have[s2[r - len(s1)]] == 0:
            del have[s2[r - len(s1)]]
        if have == need: return True
    return False
```

**Complexity.** $O(n)$ time, $O(1)$ space (26 letters).

**Hook.** "Fixed window, counter equality."

---

### 14. Minimum Window Substring (LC 76)

**Problem.** Smallest substring of $s$ containing all chars of $t$.

**Intuition.** Expand right; once all needed chars satisfied, shrink left as far as possible while staying valid; track min.

```python
from collections import Counter
def minWindow(s, t):
    if not t or not s: return ""
    need = Counter(t)
    have = Counter()
    matches = 0       # count of distinct chars satisfied
    target = len(need)
    l, best = 0, (float('inf'), 0, 0)
    for r, c in enumerate(s):
        have[c] += 1
        if c in need and have[c] == need[c]:
            matches += 1
        while matches == target:
            if r - l + 1 < best[0]:
                best = (r - l + 1, l, r)
            have[s[l]] -= 1
            if s[l] in need and have[s[l]] < need[s[l]]:
                matches -= 1
            l += 1
    return "" if best[0] == float('inf') else s[best[1]:best[2] + 1]
```

**Complexity.** $O(n + m)$ time.

**Hook.** "Track distinct-chars-satisfied; expand-then-shrink."

**Watch out.** Decrement matches only when a need-char drops *below* its required count.

---

### 15. Sliding Window Maximum (LC 239)

**Problem.** Max in each window of size $k$.

**Intuition.** Monotonic deque of indices, decreasing values. Front is the max. Pop back if smaller, pop front if out of window.

```python
from collections import deque
def maxSlidingWindow(nums, k):
    q = deque()   # indices, nums[q] decreasing
    out = []
    for i, x in enumerate(nums):
        while q and nums[q[-1]] < x:
            q.pop()
        q.append(i)
        if q[0] <= i - k:
            q.popleft()
        if i >= k - 1:
            out.append(nums[q[0]])
    return out
```

**Complexity.** $O(n)$ amortized.

**Hook.** "Monotonic deque; front is max."

**Watch out.** Pop from back while back is smaller (it can't be the max anymore for any future window).

---

## How to drill these

1. Read each problem's **Hook** alone; try to write the solution. If you can't, re-read.
2. Re-implement on a blank file 24 hours later from the hook only.
3. Re-attempt 1 week later. Three reps over 30 days = locked in.

## Summary table

| # | Problem | Pattern | Time | Space | Hook |
|---|---------|---------|------|-------|------|
| 1 | Contains Duplicate | hash set | $O(n)$ | $O(n)$ | set vs len |
| 2 | Valid Anagram | counter | $O(n)$ | $O(1)$ | counter equal |
| 3 | Two Sum | hash map | $O(n)$ | $O(n)$ | seen complement |
| 4 | Group Anagrams | hash by canonical | $O(nk)$ | $O(nk)$ | canonical key |
| 5 | Top K Frequent | bucket sort | $O(n)$ | $O(n)$ | freq-as-index |
| 6 | Encode/Decode | length-prefix | $O(n)$ | $O(n)$ | len#str |
| 7 | Product Except Self | prefix+suffix | $O(n)$ | $O(1)$ | left-then-right |
| 8 | Valid Sudoku | 3 keys per cell | $O(1)$ | $O(1)$ | row/col/box keys |
| 9 | Longest Consecutive | set + skip non-starts | $O(n)$ | $O(n)$ | start where n-1 missing |
| 10 | Buy/Sell Stock | running min | $O(n)$ | $O(1)$ | min-then-diff |
| 11 | Longest Unique Substring | sliding window | $O(n)$ | $O(\Sigma)$ | jump l past dup |
| 12 | Char Replacement | window-max-cnt ≤ k | $O(n)$ | $O(1)$ | window−maxc≤k |
| 13 | Permutation in String | fixed window counter | $O(n)$ | $O(1)$ | counter eq |
| 14 | Min Window Substring | matches counter | $O(n)$ | $O(\Sigma)$ | matches==target |
| 15 | Sliding Window Max | monotonic deque | $O(n)$ | $O(k)$ | front is max |
