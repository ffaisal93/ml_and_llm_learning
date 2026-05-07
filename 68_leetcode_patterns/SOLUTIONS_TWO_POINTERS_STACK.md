# NeetCode 150 — Two Pointers + Stack — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips.

---

## Two Pointers (5)

### 1. Valid Palindrome (LC 125)

**Problem.** Is the string a palindrome considering only alphanumeric characters, ignoring case?

**Intuition.** Two pointers from outside; skip non-alphanumerics.

```python
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum(): l += 1
        while l < r and not s[r].isalnum(): r -= 1
        if s[l].lower() != s[r].lower(): return False
        l += 1; r -= 1
    return True
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Skip junk; compare lower."

---

### 2. Two Sum II — Input Array Is Sorted (LC 167)

**Problem.** Sorted array; return 1-indexed pair summing to target.

**Intuition.** Opposing pointers. Sum > target → move right left; sum < target → move left right.

```python
def twoSum(numbers, target):
    l, r = 0, len(numbers) - 1
    while l < r:
        s = numbers[l] + numbers[r]
        if s == target: return [l + 1, r + 1]
        if s < target: l += 1
        else: r -= 1
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Sorted ⇒ opposing pointers, narrow toward target."

---

### 3. 3Sum (LC 15)

**Problem.** All unique triplets summing to zero.

**Intuition.** Sort. Fix first element (skip duplicates). Two-pointer the rest.

```python
def threeSum(nums):
    nums.sort()
    out = []
    for i in range(len(nums) - 2):
        if nums[i] > 0: break
        if i > 0 and nums[i] == nums[i - 1]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                out.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
                while l < r and nums[l] == nums[l - 1]: l += 1
                while l < r and nums[r] == nums[r + 1]: r -= 1
            elif s < 0: l += 1
            else: r -= 1
    return out
```

**Complexity.** $O(n^2)$ time, $O(1)$ space (output excluded).

**Hook.** "Sort + fix one + two-pointer + skip dups (3 places)."

**Watch out.** Three deduplication points: outer `i`, inner `l` after match, inner `r` after match.

---

### 4. Container With Most Water (LC 11)

**Problem.** Max water area between two vertical lines.

**Intuition.** Two pointers from outside. Move the *shorter* side inward (the only way to potentially increase area, since width shrinks).

```python
def maxArea(height):
    l, r = 0, len(height) - 1
    best = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        best = max(best, area)
        if height[l] < height[r]: l += 1
        else: r -= 1
    return best
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Move the shorter side."

---

### 5. Trapping Rain Water (LC 42)

**Problem.** Total water trapped after rain.

**Intuition.** At each position, water = min(maxLeft, maxRight) − height. Two pointers tracking running max from each side; advance the side with smaller max.

```python
def trap(height):
    l, r = 0, len(height) - 1
    lmax = rmax = total = 0
    while l < r:
        if height[l] < height[r]:
            lmax = max(lmax, height[l])
            total += lmax - height[l]
            l += 1
        else:
            rmax = max(rmax, height[r])
            total += rmax - height[r]
            r -= 1
    return total
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Smaller side determines water; advance it."

**Watch out.** Update `lmax`/`rmax` *before* counting; height equals max means 0 contribution (correctly).

---

## Stack (7)

### 6. Valid Parentheses (LC 20)

**Intuition.** Push openers; on closer, top must match.

```python
def isValid(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in '([{':
            stack.append(c)
        else:
            if not stack or stack.pop() != pairs[c]: return False
    return not stack
```

**Complexity.** $O(n)$ time and space.

**Hook.** "Pop matches closer."

---

### 7. Min Stack (LC 155)

**Problem.** Stack with `getMin` in $O(1)$.

**Intuition.** Each frame stores `(val, current_min)`. Or two stacks (values + running min).

```python
class MinStack:
    def __init__(self):
        self.stack = []   # list of (val, min_so_far)
    def push(self, val):
        cur_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, cur_min))
    def pop(self):
        self.stack.pop()
    def top(self):
        return self.stack[-1][0]
    def getMin(self):
        return self.stack[-1][1]
```

**Complexity.** $O(1)$ all ops.

**Hook.** "Carry min in each frame."

---

### 8. Evaluate Reverse Polish Notation (LC 150)

**Intuition.** Stack of operands; on operator, pop two and push result.

```python
def evalRPN(tokens):
    stack = []
    for t in tokens:
        if t in "+-*/":
            b = stack.pop(); a = stack.pop()
            if t == '+': stack.append(a + b)
            elif t == '-': stack.append(a - b)
            elif t == '*': stack.append(a * b)
            else: stack.append(int(a / b))   # truncate toward zero
        else:
            stack.append(int(t))
    return stack[0]
```

**Complexity.** $O(n)$ time and space.

**Hook.** "Stack pops two on operator."

**Watch out.** Division truncates *toward zero* in this problem — use `int(a / b)`, not `a // b` (which floors).

---

### 9. Generate Parentheses (LC 22)

**Intuition.** Backtracking; track open/close counts. Add '(' if open < n; add ')' if close < open.

```python
def generateParenthesis(n):
    out = []
    def bt(s, op, cl):
        if len(s) == 2 * n:
            out.append(s); return
        if op < n: bt(s + '(', op + 1, cl)
        if cl < op: bt(s + ')', op, cl + 1)
    bt('', 0, 0)
    return out
```

**Complexity.** $O(4^n / \sqrt{n})$ — Catalan number.

**Hook.** "Open < n; close < open."

---

### 10. Daily Temperatures (LC 739)

**Problem.** For each day, days until a warmer temperature.

**Intuition.** Monotonic decreasing stack of indices. When current > top's temperature, pop and record the gap.

```python
def dailyTemperatures(temperatures):
    out = [0] * len(temperatures)
    stack = []   # indices, decreasing temps
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out
```

**Complexity.** $O(n)$ amortized.

**Hook.** "Monotonic stack; pop when current > top."

---

### 11. Car Fleet (LC 853)

**Problem.** Cars at positions with speeds heading to target. A faster car catches up and forms a fleet. Count fleets.

**Intuition.** Sort by position descending. Compute time-to-target for each. Iterate; if current car's time > stack top's time, it's slower (or equal) → forms its own fleet (push); else it catches up (skip).

```python
def carFleet(target, position, speed):
    cars = sorted(zip(position, speed), reverse=True)
    stack = []
    for p, s in cars:
        time = (target - p) / s
        if not stack or time > stack[-1]:
            stack.append(time)
    return len(stack)
```

**Complexity.** $O(n \log n)$.

**Hook.** "Sort by pos desc; stack of fleet times."

---

### 12. Largest Rectangle in Histogram (LC 84)

**Problem.** Largest rectangle area in a histogram of bar heights.

**Intuition.** For each bar, find the nearest smaller bar to the left and right. Width = right − left − 1. Monotonic increasing stack.

```python
def largestRectangleArea(heights):
    stack = []   # (start_index, height)
    best = 0
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, ht = stack.pop()
            best = max(best, ht * (i - idx))
            start = idx
        stack.append((start, h))
    for idx, ht in stack:
        best = max(best, ht * (len(heights) - idx))
    return best
```

**Complexity.** $O(n)$ amortized.

**Hook.** "Pop while top is taller; extend the popped bar's start back."

**Watch out.** After main loop, drain the stack — remaining bars extend to the right edge.

---

## Summary table — Two Pointers + Stack

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Valid Palindrome | two pointers + skip | $O(n)$ | skip junk, compare lower |
| 2 | Two Sum II | opposing pointers | $O(n)$ | narrow toward target |
| 3 | 3Sum | sort + fix + 2p | $O(n^2)$ | dedup 3 places |
| 4 | Container Most Water | move shorter | $O(n)$ | shorter side moves |
| 5 | Trapping Rain Water | running max both sides | $O(n)$ | advance smaller side |
| 6 | Valid Parens | stack match | $O(n)$ | pop matches closer |
| 7 | Min Stack | (val, min) frames | $O(1)$ | min in each frame |
| 8 | Eval RPN | stack ops | $O(n)$ | int(a/b) for trunc |
| 9 | Generate Parens | backtrack | $O(4^n/\sqrt n)$ | open<n, close<open |
| 10 | Daily Temps | monotonic stack | $O(n)$ | pop when bigger seen |
| 11 | Car Fleet | sort + stack | $O(n\log n)$ | sort pos desc, time stack |
| 12 | Largest Rect | monotonic + extend | $O(n)$ | extend popped index |
