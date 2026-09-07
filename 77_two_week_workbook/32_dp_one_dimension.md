# Dynamic programming I: one dimension

Dynamic programming is recursion where you stop recomputing the same subproblem. That is the whole
idea. A recursive solution explores a tree of calls, the same call appears in many branches, and the
tree is exponential only because the work is repeated. Store each answer the first time you compute it
and the exponential tree collapses to one entry per distinct subproblem, which is usually linear.

The hard part is never the memoisation. Adding a dictionary or a decorator takes one line. The hard
part is **defining the state**. If you can say "let `dp[i]` be the answer for the first `i` items,
ending here" in one sentence, the transition usually follows in one more sentence, because the
transition is just "what are the last choices that could produce this state". If you cannot say the
state in one sentence, you are not ready to write code, and no amount of staring at the array helps.
Say the sentence out loud, write it as a comment, and only then write the recurrence.

The method is five steps, in order. Define the state in one sentence. Write the transition. Find the
base case. Decide the iteration order so that every value you read is already computed. Then, only if
you need it, reduce the space. Write the memoised recursion first, because it is far easier to get
right — the order takes care of itself. Convert it to a table only when you need the space or the
interviewer asks.

## Recognising it from the phrasing

| The interviewer says | They mean | The transition |
|---|---|---|
| "how many ways" | a counting DP | **sum** the transitions |
| "minimum cost", "maximum profit" | an optimisation DP | **min** or **max** of the transitions |
| "can you reach", "is it possible" | a boolean DP | **OR** of the transitions |
| "you cannot use two adjacent" | the house-robber shape | `dp[i] = max(dp[i-1], dp[i-2] + a[i])` |
| "longest something ending here" | the state must END at `i` | take the **max over all `i`** at the end |
| "unlimited reuse of the items" | complete knapsack | items outer, capacity **ascending** |
| "each item used at most once" | 0/1 knapsack | capacity **descending** |
| "in how many distinct orders" | a permutation count | total outer, items inner |

Before you write anything, ask two questions. What does the smallest piece of the answer look like?
And what information do you need to extend that piece by one element? If that information is a fixed
number of previous answers — the answer one step back, or one and two steps back, or the best over all
earlier positions — then the state is a single index and this is one-dimensional DP. If you need a
second index to say where you are — a position in a second string, a remaining capacity, a count of
transactions left — then the state is a pair and you need the next chapter. Answer these two questions
in words before you touch the keyboard. The recurrence is then a transcription, not an invention.

## The templates

Templates 1, 2 and 3 solve the same problem, Climbing Stairs, in three forms. They are printed
together on purpose: the recurrence never changes, only the machinery that stores it. Template 4 is
the one shape that differs, and it differs in where the answer lives.

**Template 1 — memoised recursion.** Use this first, always. Write the recurrence exactly as you said
it in words and let the cache handle the order.

```python
from functools import lru_cache

def climb_memo(n):
    @lru_cache(maxsize=None)
    def best(i):                                  ## state: ways to reach step i
        if i <= 1:                                ## 3. base case
            return 1
        return best(i - 1) + best(i - 2)          ## 2. transition
    return best(n)

## tests

assert climb_memo(2) == 2
assert climb_memo(3) == 3
assert climb_memo(10) == 89
print(climb_memo(10))
```

```
89
```

**Template 2 — bottom-up table.** Use when you want no recursion depth limit, or when you will reduce
the space afterwards. The iteration order is now your responsibility.

```python
def climb_table(n):
    dp = [0] * (n + 1)                            ## 1. dp[i] = ways to reach step i
    dp[0] = 1                                     ## 3. base cases
    if n >= 1:
        dp[1] = 1
    for i in range(2, n + 1):                     ## 4. ascending: i-1 and i-2 are ready
        dp[i] = dp[i - 1] + dp[i - 2]             ## 2. transition
    return dp[n]

## tests

assert climb_table(2) == 2
assert climb_table(3) == 3
assert climb_table(10) == 89
print(climb_table(10))
```

```
89
```

**Template 3 — the same table with rolling variables.** Use when the transition reads only a fixed
number of recent cells. Compare it line by line with template 2: `dp[i-2]` became `two_back`,
`dp[i-1]` became `one_back`, and the assignment at the end of the body slides the pair forward.

```python
def climb_rolling(n):
    two_back, one_back = 1, 1                     ## dp[i-2], dp[i-1]
    for i in range(2, n + 1):
        current = one_back + two_back             ## the SAME transition
        two_back, one_back = one_back, current    ## slide the two variables forward
    return one_back

## tests

assert climb_rolling(2) == 2
assert climb_rolling(3) == 3
assert climb_rolling(10) == 89
print(climb_rolling(10))
```

```
89
```

**Template 4 — "longest something ending at `i`".** Use when the answer can finish at any index. The
state is forced to end at `i` so that the transition has something to attach to, and therefore
`dp[n-1]` is **not** the answer. The answer is `max(dp)`.

```python
def longest_increasing_run(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)                          ## dp[i] = longest subsequence ENDING at i
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)                                ## answer is the MAX over all states

## tests

assert longest_increasing_run([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert longest_increasing_run([7, 7, 7]) == 1
assert longest_increasing_run([]) == 0
print(longest_increasing_run([10, 9, 2, 5, 3, 7, 101, 18]))
```

```
4
```

The difference between templates 2 and 4 is worth stating plainly, because it costs people whole
rounds. In template 2 the state is "the answer for the first `i` items", so the last cell already
covers everything and `dp[n]` is the answer. In template 4 the state is "the answer for something that
ends exactly at `i`", so the last cell only covers the subsequences that happen to end at the last
element. If you define a state with the word "ending", you must scan the table at the end.

## Space reduction: from a table to two variables

Look at the house-robber transition, `dp[i] = max(dp[i-1], dp[i-2] + a[i])`. It reads exactly two
cells, both within two steps of `i`. Every other cell of the table is dead the moment the loop passes
it. So the array is doing no work, and two integers can replace it.

The rewrite is mechanical. Start from the table form, and for each cell the transition reads,
introduce a variable named after its offset. Here `dp[i-2]` becomes `two_back` and `dp[i-1]` becomes
`one_back`. Replace the reads. Replace the write with a temporary called `current`. Then, at the end
of the body, shift: the old `one_back` becomes the new `two_back`, and `current` becomes the new
`one_back`. Do the shift as a single tuple assignment, because doing it in two statements overwrites
`one_back` before `two_back` has read it. The result runs in $O(1)$ space and computes value for value
what the table computed, which the cross-check below confirms on every array of length up to seven
over the values 0 to 2.

```python
def rob_table(nums):
    n = len(nums)
    dp = [0] * (n + 1)                            ## dp[i] = best over the first i houses
    if n >= 1:
        dp[1] = nums[0]
    for i in range(2, n + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
    return dp[n]

def rob_rolling(nums):
    two_back, one_back = 0, 0                     ## dp[i-2] and dp[i-1], nothing else
    for x in nums:
        current = max(one_back, two_back + x)     ## the SAME transition
        two_back, one_back = one_back, current    ## slide the pair forward
    return one_back

def rob_brute(nums):                              ## check: every non-adjacent subset
    n, best = len(nums), 0
    for mask in range(1 << n):
        if mask & (mask >> 1):                    ## two adjacent bits set: illegal
            continue
        best = max(best, sum(nums[i] for i in range(n) if mask >> i & 1))
    return best

def small_arrays(max_len, top):                   ## every array of length <= max_len, values 0..top
    out, level = [[]], [[]]
    for _ in range(max_len):
        level = [a + [v] for a in level for v in range(top + 1)]
        out += level
    return out

## tests

assert rob_table([1, 2, 3, 1]) == rob_rolling([1, 2, 3, 1]) == 4
assert rob_table([2, 7, 9, 3, 1]) == rob_rolling([2, 7, 9, 3, 1]) == 12
checked = 0
for a in small_arrays(7, 2):                      ## cross-check both forms against brute force
    assert rob_table(a) == rob_rolling(a) == rob_brute(a)
    checked += 1
print(rob_table([2, 7, 9, 3, 1]), rob_rolling([2, 7, 9, 3, 1]), "brute force agrees on", checked)
```

```
12 12 brute force agrees on 3280
```

The caveat is real and you should say it before the interviewer does. Two variables give you the
value of the optimum and nothing else. Reconstructing which houses you robbed needs the table, because
reconstruction walks backwards from `dp[n]` and asks at each step which branch of the `max` won, and
that question needs the older cells you just threw away. So: if the question asks for a number, reduce
the space. If it asks for the sequence, the subset, or the actual path, keep the table and keep a
parent pointer next to it.

## The problems

### P1. Climbing Stairs — count the ways to climb `n` steps taking one or two at a time

**Which template.** Template 3, after writing template 1 to see the recurrence.
**The trick.** The last move is either one step or two steps, so the ways to reach step `n` split
cleanly into the ways to reach `n-1` and the ways to reach `n-2`. That is the Fibonacci recurrence, and
saying "this is Fibonacci" out loud is the fastest way to show you have the state right.

```python
def climb_stairs(n):
    if n <= 2:
        return max(n, 1)
    two_back, one_back = 1, 2                     ## ways to reach step 1 and step 2
    for _ in range(3, n + 1):
        two_back, one_back = one_back, one_back + two_back
    return one_back

## tests

assert climb_stairs(1) == 1
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(5) == 8
assert climb_stairs(45) == 1836311903
print(climb_stairs(5), climb_stairs(45))
```

```
8 1836311903
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P2. Min Cost Climbing Stairs — reach the top when stepping on stair `i` costs `cost[i]`

**Which template.** Template 3, with `min` in place of `+`.
**The trick.** Define the state as the cost to **stand on** step `i`, not the cost to have passed it.
Then arriving at `i` costs the cheaper of coming from `i-1` or from `i-2`, plus the cost of the step
you left. The top is index `n`, one past the last stair, which is why the loop runs to `n` inclusive
and why you may start free on either step 0 or step 1.

```python
def min_cost_climbing_stairs(cost):
    n = len(cost)
    two_back, one_back = 0, 0                     ## cost to STAND on step 0 and step 1
    for i in range(2, n + 1):
        current = min(one_back + cost[i - 1], two_back + cost[i - 2])
        two_back, one_back = one_back, current
    return one_back

## tests

assert min_cost_climbing_stairs([10, 15, 20]) == 15
assert min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) == 6
assert min_cost_climbing_stairs([5, 5]) == 5
print(min_cost_climbing_stairs([10, 15, 20]),
      min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]))
```

```
15 6
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P3. House Robber — maximum sum of a subset of an array with no two adjacent elements

**Which template.** Template 3. This is the shape the recognition table names.
**The trick.** At house `i` there are exactly two choices and they are exhaustive: skip it, which
leaves `dp[i-1]`, or take it, which forces you to skip `i-1` and leaves `dp[i-2] + a[i]`. Write those
two expressions inside a `max` and the problem is finished. The cross-check enumerates every legal
subset for the listed cases.

```python
def rob(nums):
    two_back, one_back = 0, 0                     ## dp[i-2], dp[i-1]
    for x in nums:
        two_back, one_back = one_back, max(one_back, two_back + x)
    return one_back

def rob_brute(nums):                              ## check: every non-adjacent subset
    n, best = len(nums), 0
    for mask in range(1 << n):
        if mask & (mask >> 1):
            continue
        best = max(best, sum(nums[i] for i in range(n) if mask >> i & 1))
    return best

## tests

assert rob([1, 2, 3, 1]) == 4
assert rob([2, 7, 9, 3, 1]) == 12
assert rob([]) == 0
assert rob([5]) == 5
cases = [[], [5], [2, 1, 1, 2], [6, 6, 4, 8, 4, 3, 3, 10], [1, 3, 1, 3, 100]]
for a in cases:                                   ## cross-check against brute force
    assert rob(a) == rob_brute(a)
print(rob([2, 7, 9, 3, 1]), "brute force agrees on", len(cases), "cases")
```

```
12 brute force agrees on 5 cases
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P4. House Robber II — the same, but the houses are in a circle

**Which template.** Template 3, run twice.
**The trick.** The circle adds exactly one constraint: house 0 and house `n-1` are now adjacent, so at
most one of them is robbed. Split on that. Either house 0 is excluded, which leaves the line
`nums[1:]`, or house `n-1` is excluded, which leaves the line `nums[:-1]`. Every legal circular
selection appears in at least one of those two lines, so the answer is the larger of the two linear
answers. Handle `n == 1` separately, because both slices are then empty.

```python
def rob_line(nums):
    two_back, one_back = 0, 0
    for x in nums:
        two_back, one_back = one_back, max(one_back, two_back + x)
    return one_back

def rob_circle(nums):
    if len(nums) == 1:
        return nums[0]
    ## house 0 and house n-1 are adjacent, so at most one of them is robbed
    return max(rob_line(nums[:-1]), rob_line(nums[1:]))

def rob_circle_brute(nums):                       ## check: subsets with no adjacency on a circle
    n, best = len(nums), 0
    for mask in range(1 << n):
        bits = [i for i in range(n) if mask >> i & 1]
        ok = all(bits[k + 1] - bits[k] > 1 for k in range(len(bits) - 1))
        if ok and not (n > 1 and mask & 1 and mask >> (n - 1) & 1):
            best = max(best, sum(nums[i] for i in bits))
    return best

## tests

assert rob_circle([2, 3, 2]) == 3
assert rob_circle([1, 2, 3, 1]) == 4
assert rob_circle([1, 2, 3]) == 3
assert rob_circle([7]) == 7
cases = [[2, 3, 2], [1, 2, 3, 1], [4, 1, 2, 7, 5, 3, 1], [9, 1, 1, 9]]
for a in cases:                                   ## cross-check against brute force
    assert rob_circle(a) == rob_circle_brute(a)
print(rob_circle([2, 3, 2]), rob_circle([9, 1, 1, 9]), "brute force agrees")
```

```
3 10 brute force agrees
```

**Complexity.** $O(n)$ time, $O(1)$ space, two passes.

### P5. Fibonacci with memoisation — compute the `n`-th Fibonacci number

**Which template.** Template 1, and this is the teaching case for it.
**The trick.** Nothing about the recurrence changes when you add the cache. The only change is that
each distinct argument is evaluated once. The counters below make that visible: the naive version makes
21891 calls for `n = 20`, and the memoised version makes 21 for the same `n` — the extra calls in the
printed total come from the later `fib_memo(90)`, which a naive version could not finish at all.

```python
from functools import lru_cache

calls = {"naive": 0, "memo": 0}

def fib_naive(n):
    calls["naive"] += 1
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)    ## the same subproblem again and again

@lru_cache(maxsize=None)
def fib_memo(n):
    calls["memo"] += 1
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)      ## each n is computed once

## tests

assert fib_naive(20) == 6765
assert fib_memo(20) == 6765
assert fib_memo(90) == 2880067194370816120
assert calls["memo"] < calls["naive"]
print(fib_memo(20), "calls: naive", calls["naive"], "memo", calls["memo"])
```

```
6765 calls: naive 21891 memo 91
```

**Complexity.** Naive $O(\varphi^n)$ time; memoised $O(n)$ time and $O(n)$ space for the cache.

### P6. Maximum Subarray — the largest sum over all contiguous subarrays

**Which template.** Template 4. The state is "the best sum **ending here**".
**The trick.** People call this greedy, and it is DP. The state `best_here` is the best sum of a
subarray ending exactly at the current index, and the transition has two options: start a new subarray
at this element, or extend the previous one. Take the larger. Because the state ends at `i`, the answer
is the maximum over all `i`, not the last value — that is template 4 in one line.

```python
def max_subarray(nums):
    best_here = nums[0]                           ## best sum ENDING at the current index
    best_anywhere = nums[0]
    for x in nums[1:]:
        best_here = max(x, best_here + x)         ## start fresh, or extend
        best_anywhere = max(best_anywhere, best_here)
    return best_anywhere

def max_subarray_brute(nums):                     ## check: every contiguous subarray
    n = len(nums)
    return max(sum(nums[i:j]) for i in range(n) for j in range(i + 1, n + 1))

## tests

assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([-3, -1, -2]) == -1
cases = [[-2, 1, -3, 4, -1, 2, 1, -5, 4], [1], [-3, -1, -2], [5, 4, -1, 7, 8], [-1, 0, -2]]
for a in cases:                                   ## cross-check against brute force
    assert max_subarray(a) == max_subarray_brute(a)
print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), "brute force agrees")
```

```
6 brute force agrees
```

**Complexity.** $O(n)$ time, $O(1)$ space. Initialise from `nums[0]`, not from 0, or an all-negative
array returns 0.

### P7. Maximum Product Subarray — the largest product over all contiguous subarrays

**Which template.** Template 4, with **two** states per index.
**The trick.** A negative number turns the largest product into the smallest and the smallest into the
largest. So the maximum ending here can come from the minimum ending at the previous index. Carry both.
At each step the three candidates are the element alone, the element times the previous maximum, and
the element times the previous minimum; the new maximum and minimum are the largest and smallest of
those three. Zeros are handled for free, because "the element alone" restarts the run.

```python
def max_product(nums):
    best_here, worst_here = nums[0], nums[0]      ## max and min product ENDING here
    answer = nums[0]
    for x in nums[1:]:
        candidates = (x, best_here * x, worst_here * x)
        best_here, worst_here = max(candidates), min(candidates)
        answer = max(answer, best_here)
    return answer

def max_product_brute(nums):                      ## check: every contiguous subarray
    n, best = len(nums), nums[0]
    for i in range(n):
        product = 1
        for j in range(i, n):
            product *= nums[j]
            best = max(best, product)
    return best

## tests

assert max_product([2, 3, -2, 4]) == 6
assert max_product([-2, 0, -1]) == 0
assert max_product([-2, 3, -4]) == 24
cases = [[2, 3, -2, 4], [-2, 0, -1], [-2, 3, -4], [0, -3, -2, 0, 5], [-1, -1, -1]]
for a in cases:                                   ## cross-check against brute force
    assert max_product(a) == max_product_brute(a)
print(max_product([-2, 3, -4]), "brute force agrees")
```

```
24 brute force agrees
```

**Complexity.** $O(n)$ time, $O(1)$ space. Compute both new states from the old pair in one statement,
or the updated maximum contaminates the minimum.

### P8. Best Time to Buy and Sell Stock — one buy and one later sell, maximise the profit

**Which template.** Template 4, where the state is the best profit **selling here**.
**The trick.** Selling at day `i` for the best profit means having bought at the cheapest day at or
before `i`. So one running minimum is the entire state, and the answer is the maximum over all sell
days. This is Kadane on the array of consecutive differences, and saying so is a good signal.

```python
def max_profit(prices):
    cheapest = float("inf")                       ## best buy price seen so far
    best = 0
    for price in prices:
        cheapest = min(cheapest, price)
        best = max(best, price - cheapest)        ## best profit SELLING here
    return best

def max_profit_brute(prices):                     ## check: every buy/sell pair
    n = len(prices)
    return max([0] + [prices[j] - prices[i] for i in range(n) for j in range(i, n)])

## tests

assert max_profit([7, 1, 5, 3, 6, 4]) == 5
assert max_profit([7, 6, 4, 3, 1]) == 0
assert max_profit([]) == 0
cases = [[7, 1, 5, 3, 6, 4], [7, 6, 4, 3, 1], [2, 4, 1], [1, 2, 3, 4, 5]]
for a in cases:                                   ## cross-check against brute force
    assert max_profit(a) == max_profit_brute(a)
print(max_profit([7, 1, 5, 3, 6, 4]), "brute force agrees")
```

```
5 brute force agrees
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P9. Best Time to Buy and Sell Stock II — unlimited transactions, one share at a time

**Which template.** A two-state machine, which is the smallest interesting DP on this page.
**The trick.** With no limit on transactions, every rise can be banked separately: any profitable buy
and sell pair equals the sum of the daily gains inside it, so summing the positive daily differences is
optimal. The DP form keeps two states, `hold` and `free`, and the cross-check confirms the two give the
same number. Learn the state-machine form, because the cooldown and fee variants extend it and the
greedy form does not.

```python
def max_profit_many(prices):
    total = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            total += prices[i] - prices[i - 1]    ## bank every upward step
    return total

def max_profit_many_dp(prices):                   ## the state-machine form of the same answer
    hold, free = float("-inf"), 0                 ## best value holding / not holding
    for price in prices:
        hold, free = max(hold, free - price), max(free, hold + price)
    return free

## tests

assert max_profit_many([7, 1, 5, 3, 6, 4]) == 7
assert max_profit_many([1, 2, 3, 4, 5]) == 4
assert max_profit_many([7, 6, 4, 3, 1]) == 0
cases = [[7, 1, 5, 3, 6, 4], [1, 2, 3, 4, 5], [7, 6, 4, 3, 1], [3, 3, 5, 0, 0, 3, 1, 4]]
for a in cases:                                   ## cross-check the greedy against the DP
    assert max_profit_many(a) == max_profit_many_dp(a)
print(max_profit_many([7, 1, 5, 3, 6, 4]), "greedy and state-machine DP agree")
```

```
7 greedy and state-machine DP agree
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P10. Coin Change — the fewest coins summing to `amount`, with unlimited coins

**Which template.** Template 2, minimisation, complete knapsack.
**The trick.** The state is one number: `dp[a]` is the fewest coins summing to `a`. The last coin used
is one of the denominations, so `dp[a] = 1 + min(dp[a - c])` over every coin `c` that fits. Use
`float("inf")` for unreachable amounts so the `min` needs no special case, and convert it to `-1` once
at the end. The cross-check is a breadth-first search over reachable sums, which finds the same minimum
by a completely different route.

```python
def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount            ## dp[a] = fewest coins summing to a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return -1 if dp[amount] == float("inf") else dp[amount]

def coin_change_brute(coins, amount):             ## check: breadth-first search over sums
    seen, frontier, steps = {0}, [0], 0
    while frontier:
        if amount in frontier:
            return steps
        nxt = [s + c for s in frontier for c in coins if s + c <= amount and s + c not in seen]
        seen.update(nxt)
        frontier, steps = list(set(nxt)), steps + 1
    return -1

## tests

assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
cases = [([1, 2, 5], 11), ([2], 3), ([1], 0), ([186, 419, 83, 408], 6249), ([3, 7], 20)]
for coins, amount in cases:                       ## cross-check against a BFS
    assert coin_change(coins, amount) == coin_change_brute(coins, amount)
print(coin_change([1, 2, 5], 11), coin_change([3, 7], 20), "BFS agrees")
```

```
3 4 BFS agrees
```

**Complexity.** $O(\text{amount} \times |\text{coins}|)$ time, $O(\text{amount})$ space. Greedy fails:
with coins 1, 3 and 4 and amount 6, greedy takes 4 then 1 then 1, and the answer is 3 plus 3.

### P11. Coin Change II — the number of **combinations** of coins summing to `amount`

**Which template.** Template 2, counting, complete knapsack — with the loops the other way round.
**The trick.** This is the most confusing pair in all of DP, so state the rule and the reason. Put the
**coins in the outer loop** and you count combinations. Put the **amount in the outer loop** and you
count ordered sequences. The reason is that the outer loop fixes what is decided first. With coins
outside, the table after processing coins 1 and 2 contains only ways that use coins in the order
1-then-2, so `1+2` and `2+1` are the same entry and are counted once. With the amount outside, every
amount is free to be reached by any coin last, so `1+2` and `2+1` are two different paths into
`dp[3]`. Both loops keep the capacity **ascending**, which is what allows a coin to be reused.

```python
def coin_change_combinations(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1                                     ## one way to make 0: take nothing
    for c in coins:                               ## COINS OUTSIDE: fixes the coin order
        for a in range(c, amount + 1):            ## ascending: reuse of c is allowed
            dp[a] += dp[a - c]
    return dp[amount]

def coin_change_permutations(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for a in range(1, amount + 1):                ## AMOUNT OUTSIDE: counts ordered sequences
        for c in coins:
            if c <= a:
                dp[a] += dp[a - c]
    return dp[amount]

def combinations_brute(amount, coins):            ## check: recursive non-decreasing choices
    def go(rest, start):
        if rest == 0:
            return 1
        return sum(go(rest - coins[i], i) for i in range(start, len(coins))
                   if coins[i] <= rest)
    return go(amount, 0)

## tests

assert coin_change_combinations(5, [1, 2, 5]) == 4
assert coin_change_combinations(3, [2]) == 0
assert coin_change_combinations(10, [10]) == 1
assert coin_change_permutations(4, [1, 2]) == 5   ## ordered: 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2
assert coin_change_combinations(4, [1, 2]) == 3   ## unordered: 1+1+1+1, 1+1+2, 2+2
for amount, coins in [(5, [1, 2, 5]), (11, [1, 2, 5]), (9, [2, 3, 4]), (0, [7])]:
    assert coin_change_combinations(amount, coins) == combinations_brute(amount, coins)
print(coin_change_combinations(5, [1, 2, 5]), coin_change_permutations(4, [1, 2]),
      "recursive count agrees")
```

```
4 5 recursive count agrees
```

**Complexity.** $O(\text{amount} \times |\text{coins}|)$ time, $O(\text{amount})$ space for both.

### P12. Longest Increasing Subsequence — the longest strictly increasing subsequence, not contiguous

**Which template.** Template 4 for the $O(n^2)$ version; a `bisect` on a tails array for the fast one.
**The trick.** The quadratic version is template 4 exactly: `dp[i]` is the longest run ending at `i`,
and you look back at every `j < i` with a smaller value. The patience version keeps `tails`, where
`tails[k]` is the **smallest possible tail** of an increasing subsequence of length `k+1`. That array
is sorted, so `bisect_left` finds where a new value belongs in $O(\log n)$: past the end, it extends
the longest run; inside, it replaces a tail with a smaller one, which never shortens anything and makes
future extensions easier. `tails` is not itself a valid subsequence — only its **length** is the
answer, and claiming otherwise is a common interview slip.

```python
from bisect import bisect_left

def lis_quadratic(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)                          ## dp[i] = longest subsequence ENDING at i
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)                                ## the answer is the max over all i

def lis_patience(nums):
    tails = []                                    ## tails[k] = smallest tail of a length-(k+1) run
    for x in nums:
        pos = bisect_left(tails, x)               ## first tail >= x
        if pos == len(tails):
            tails.append(x)                       ## x extends the longest run
        else:
            tails[pos] = x                        ## x makes that length cheaper to extend
    return len(tails)

## tests

assert lis_quadratic([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_patience([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_patience([7, 7, 7, 7]) == 1
assert lis_patience([]) == 0
cases = [[10, 9, 2, 5, 3, 7, 101, 18], [0, 1, 0, 3, 2, 3], [7, 7, 7, 7], [], [4, 3, 2, 1],
         [1, 3, 6, 7, 9, 4, 10, 5, 6]]
for a in cases:                                   ## cross-check the two implementations
    assert lis_quadratic(a) == lis_patience(a)
print(lis_patience([10, 9, 2, 5, 3, 7, 101, 18]),
      lis_patience([1, 3, 6, 7, 9, 4, 10, 5, 6]), "both versions agree")
```

```
4 6 both versions agree
```

**Complexity.** $O(n^2)$ time for the first, $O(n \log n)$ for the second, $O(n)$ space for both. For a
non-decreasing subsequence use `bisect_right` instead.

### P13. Word Break — can `s` be cut into a sequence of dictionary words

**Which template.** Template 2, boolean, with an OR over the transitions.
**The trick.** `dp[i]` is true when the first `i` characters split cleanly. The last word ends at `i`
and starts at some `j`, so `dp[i]` is true when any `j` has `dp[j]` true and `s[j:i]` in the
dictionary. Put the dictionary in a `set` first, because the inner test is the hot line. Break out of
the inner loop on the first success — one witness is enough for an OR.

```python
def word_break(s, word_dict):
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)                        ## dp[i] = the first i characters split cleanly
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:         ## a good prefix plus one whole word
                dp[i] = True
                break
    return dp[n]

def word_break_brute(s, word_dict):               ## check: plain exponential recursion
    words = set(word_dict)
    def go(rest):
        if rest == "":
            return True
        return any(rest.startswith(w) and go(rest[len(w):]) for w in words)
    return go(s)

## tests

assert word_break("leetcode", ["leet", "code"]) is True
assert word_break("applepenapple", ["apple", "pen"]) is True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
cases = [("leetcode", ["leet", "code"]), ("applepenapple", ["apple", "pen"]),
         ("catsandog", ["cats", "dog", "sand", "and", "cat"]), ("aaaab", ["a", "aa"])]
for s, d in cases:                                ## cross-check against the brute-force search
    assert word_break(s, d) == word_break_brute(s, d)
print(word_break("leetcode", ["leet", "code"]),
      word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]), "brute force agrees")
```

```
True False brute force agrees
```

**Complexity.** $O(n^2)$ substring tests, each $O(n)$ to hash, so $O(n^3)$ worst case and $O(n)$ space.

### P14. Decode Ways — count the decodings of a digit string where A is 1 and Z is 26

**Which template.** Template 3, counting, with two guarded transitions.
**The trick.** The last letter used either one digit or two. One digit is legal when that digit is not
`0`. Two digits are legal when the pair reads between 10 and 26 inclusive, which rules out `06` as well
as `27`. Both guards are needed; dropping either is the classic wrong answer. Zeros make this problem
sharp, so test `"06"`, `"10"` and `"100"` before you say you are done.

```python
def num_decodings(s):
    if not s or s[0] == "0":
        return 0
    two_back, one_back = 1, 1                     ## dp[i-2], dp[i-1]
    for i in range(1, len(s)):
        current = 0
        if s[i] != "0":
            current += one_back                   ## take s[i] alone
        if 10 <= int(s[i - 1:i + 1]) <= 26:
            current += two_back                   ## take s[i-1:i+1] as one letter
        if current == 0:
            return 0
        two_back, one_back = one_back, current
    return one_back

def num_decodings_brute(s):                       ## check: plain recursion over every split
    def go(i):
        if i == len(s):
            return 1
        if s[i] == "0":
            return 0
        total = go(i + 1)
        if i + 1 < len(s) and 10 <= int(s[i:i + 2]) <= 26:
            total += go(i + 2)
        return total
    return go(0) if s else 0

## tests

assert num_decodings("12") == 2
assert num_decodings("226") == 3
assert num_decodings("06") == 0
assert num_decodings("2101") == 1
for s in ["12", "226", "06", "10", "2101", "1111", "27", "100", "1201234"]:
    assert num_decodings(s) == num_decodings_brute(s)
print(num_decodings("226"), num_decodings("2101"), "brute force agrees")
```

```
3 1 brute force agrees
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P15. Partition Equal Subset Sum — can the array be split into two parts of equal sum

**Which template.** 0/1 knapsack on a boolean array, which is template 2 over a capacity axis.
**The trick.** Two equal halves means a subset summing to `total // 2`, so an odd total is an
immediate `False`. Then the state is `dp[s]`, "some subset sums to `s`", and each number is offered
once. The capacity loop runs **descending**, and that direction is the whole 0/1 versus complete
distinction: descending, `dp[s - x]` still holds the value from before `x` was offered, so `x` is used
at most once. Ascending, you would read a cell that already includes `x` and allow it twice.

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2 == 1:
        return False
    target = total // 2
    dp = [False] * (target + 1)                   ## dp[s] = some subset sums to s
    dp[0] = True
    for x in nums:                                ## each number is used AT MOST ONCE
        for s in range(target, x - 1, -1):        ## DESCENDING, so dp[s-x] is still the old row
            if dp[s - x]:
                dp[s] = True
    return dp[target]

def can_partition_brute(nums):                    ## check: every subset
    n, total = len(nums), sum(nums)
    for mask in range(1 << n):
        if 2 * sum(nums[i] for i in range(n) if mask >> i & 1) == total:
            return True
    return False

## tests

assert can_partition([1, 5, 11, 5]) is True
assert can_partition([1, 2, 3, 5]) is False
assert can_partition([2, 2]) is True
cases = [[1, 5, 11, 5], [1, 2, 3, 5], [2, 2], [1, 1, 1, 1, 2, 2], [3, 3, 3, 4, 5], [7]]
for a in cases:                                   ## cross-check against every subset
    assert can_partition(a) == can_partition_brute(a)
print(can_partition([1, 5, 11, 5]), can_partition([1, 2, 3, 5]), "brute force agrees")
```

```
True False brute force agrees
```

**Complexity.** $O(n \times \text{total})$ time, $O(\text{total})$ space.

### P16. Jump Game — can you reach the last index when `nums[i]` is the maximum jump from `i`

**Which template.** A boolean DP, then the greedy that replaces it.
**The trick.** The DP asks "is the end reachable from `i`" and scans backwards, which is $O(n^2)$. The
greedy carries one number, the furthest index reachable so far, and fails the moment the loop reaches
an index beyond it, because a gap can never be crossed later. Present the DP to show you understand
the structure, then reduce it. The two are cross-checked below.

```python
def can_jump(nums):
    furthest = 0                                  ## the rightmost index reachable so far
    for i, step in enumerate(nums):
        if i > furthest:
            return False                          ## a gap: index i cannot be reached
        furthest = max(furthest, i + step)
    return True

def can_jump_dp(nums):                            ## the same answer as a boolean DP
    n = len(nums)
    dp = [False] * n
    dp[n - 1] = True                              ## dp[i] = the end is reachable from i
    for i in range(n - 2, -1, -1):
        dp[i] = any(dp[i + k] for k in range(1, min(nums[i], n - 1 - i) + 1))
    return dp[0]

## tests

assert can_jump([2, 3, 1, 1, 4]) is True
assert can_jump([3, 2, 1, 0, 4]) is False
assert can_jump([0]) is True
cases = [[2, 3, 1, 1, 4], [3, 2, 1, 0, 4], [0], [2, 0, 0], [1, 0, 1, 0], [5, 0, 0, 0, 0, 0]]
for a in cases:                                   ## cross-check the greedy against the DP
    assert can_jump(a) == can_jump_dp(a)
print(can_jump([2, 3, 1, 1, 4]), can_jump([3, 2, 1, 0, 4]), "greedy and DP agree")
```

```
True False greedy and DP agree
```

**Complexity.** $O(n)$ time, $O(1)$ space for the greedy; $O(n^2)$ and $O(n)$ for the DP.

### P17. Jump Game II — the fewest jumps needed to reach the last index

**Which template.** A minimisation DP, reduced to a layered scan.
**The trick.** Think of the indices as levels of a breadth-first search: everything reachable in one
jump is layer one, everything reachable from layer one is layer two, and so on. The scan keeps
`current_end`, the last index of the layer being consumed, and `furthest`, the best landing found
anywhere in it. When `i` reaches `current_end` the layer is exhausted, so you spend one jump and the
next layer runs to `furthest`. Stop the loop one before the end, or you count a jump you never take.

```python
def jump(nums):
    jumps, current_end, furthest = 0, 0, 0
    for i in range(len(nums) - 1):
        furthest = max(furthest, i + nums[i])     ## best landing from anywhere in this layer
        if i == current_end:                      ## the layer is finished
            jumps += 1
            current_end = furthest
    return jumps

def jump_dp(nums):                                ## check: the O(n^2) minimum-cost DP
    n = len(nums)
    dp = [float("inf")] * n
    dp[0] = 0
    for i in range(n):
        for k in range(1, min(nums[i], n - 1 - i) + 1):
            dp[i + k] = min(dp[i + k], dp[i] + 1)
    return dp[n - 1]

## tests

assert jump([2, 3, 1, 1, 4]) == 2
assert jump([2, 3, 0, 1, 4]) == 2
assert jump([0]) == 0
cases = [[2, 3, 1, 1, 4], [2, 3, 0, 1, 4], [0], [1, 1, 1, 1], [4, 1, 1, 1, 1], [1, 2, 3]]
for a in cases:                                   ## cross-check the greedy against the DP
    assert jump(a) == jump_dp(a)
print(jump([2, 3, 1, 1, 4]), jump([1, 2, 3]), "layered greedy and O(n^2) DP agree")
```

```
2 2 layered greedy and O(n^2) DP agree
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P18. Palindromic Substrings — count the palindromic substrings of `s`, counting repeats separately

**Which template.** Expand around centres, which is the space-reduced form of the interval DP.
**The trick.** Every palindrome has a centre, and there are `2n - 1` of them: `n` single characters and
`n - 1` gaps between neighbours. Expand from each centre while the two ends match. The count is exactly
the number of successful expansions, because each expansion **is** one distinct palindrome, so no
separate bookkeeping is needed. This costs $O(1)$ space where the interval table costs $O(n^2)$.

```python
def count_substrings(s):
    total = 0
    for centre in range(len(s)):
        for left, right in ((centre, centre), (centre, centre + 1)):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                total += 1                        ## every expansion IS one palindrome
                left -= 1
                right += 1
    return total

def count_substrings_brute(s):                    ## check: test every substring
    n = len(s)
    return sum(1 for i in range(n) for j in range(i + 1, n + 1)
               if s[i:j] == s[i:j][::-1])

## tests

assert count_substrings("abc") == 3
assert count_substrings("aaa") == 6
assert count_substrings("") == 0
for s in ["abc", "aaa", "", "abba", "aabaa", "xyzzyx", "abacaba"]:
    assert count_substrings(s) == count_substrings_brute(s)
print(count_substrings("aaa"), count_substrings("abacaba"), "brute force agrees")
```

```
6 12 brute force agrees
```

**Complexity.** $O(n^2)$ time, $O(1)$ space.

### P19. Longest Palindromic Substring — the longest contiguous palindrome in `s`

**Which template.** The same centre expansion, recording indices instead of counting.
**The trick.** It is P18 with a different accumulator, and saying that is half the answer. Handle the
two centre kinds in one loop over the pairs `(c, c)` and `(c, c+1)`, so odd and even lengths share one
body. Record `best_at` and `best_len`, not a slice, and cut the string once at the end — slicing inside
the loop turns an $O(n^2)$ scan into an $O(n^3)$ one. The two-pointer expansion here is the same
outward walk as in the two-pointers chapter, applied from a centre rather than from the ends.

```python
def longest_palindrome(s):
    best_at, best_len = 0, 0
    for centre in range(len(s)):
        for left, right in ((centre, centre), (centre, centre + 1)):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                if right - left + 1 > best_len:   ## record the INDICES, slice once at the end
                    best_at, best_len = left, right - left + 1
                left -= 1
                right += 1
    return s[best_at:best_at + best_len]

def longest_palindrome_brute(s):                  ## check: test every substring
    best = ""
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            piece = s[i:j]
            if piece == piece[::-1] and len(piece) > len(best):
                best = piece
    return best

## tests

assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("") == ""
for s in ["babad", "cbbd", "", "a", "forgeeksskeegfor", "abacdfgdcaba"]:
    assert len(longest_palindrome(s)) == len(longest_palindrome_brute(s))
print(longest_palindrome("babad"), longest_palindrome("forgeeksskeegfor"), "brute force agrees")
```

```
bab geeksskeeg brute force agrees
```

**Complexity.** $O(n^2)$ time, $O(1)$ space. Ties are broken by the earliest start here, so confirm
with the interviewer that any longest palindrome is acceptable.

## Tricks and tips

**Say the state in a sentence before you write a line.** "Let `dp[i]` be the fewest coins summing to
`i`." "Let `dp[i]` be the best sum of a subarray ending at `i`." If the sentence needs a comma and a
second clause about a different index, you have a two-dimensional state and you should say so rather
than fight it. This one habit prevents more lost rounds than any coding technique on the page.

**Write the memoised recursion first.** It is easier because the base case is the only place you must
think, and the iteration order is free. Once it runs, converting to a table is mechanical: the
arguments become the indices, the base case becomes the initialisation, and the order becomes whatever
makes every read happen after its write. Many interviewers accept the memoised version as the final
answer, and offering the table as an optimisation reads better than starting with a table you cannot
justify.

**The word in the question tells you the combining operator.** "How many" means sum. "Minimum" or
"maximum" means `min` or `max`. "Is it possible" means OR. Nothing else about the recurrence changes
between the three, which is why Coin Change and Coin Change II look almost identical.

**Loop direction encodes reuse.** Over a capacity axis, ascending means an item can be used again in
the same pass, because you read a cell that already includes it — that is complete knapsack.
Descending means each item is used once, because you read a cell from before the item existed — that
is 0/1 knapsack. Write the direction down next to the state sentence, and check it with a two-element
example by hand.

**Loop nesting encodes order.** Items outside, capacity inside, counts unordered combinations. Capacity
outside, items inside, counts ordered sequences. This is the same rule that separates Coin Change II
from the staircase-style counting problems, and the fastest sanity check is amount 4 with coins 1 and
2: three combinations, five sequences.

**Initialise from real data, not from zero, whenever negatives are possible.** Maximum Subarray and
Maximum Product Subarray both start from `nums[0]`. Starting from 0 silently returns 0 on all-negative
input, and the sample test will not catch it.

**Guard the state on an empty input before anything else.** `n == 0`, an empty string, an amount of
zero, a single element. Each is one line at the top and each has appeared in the tests above.

## The bugs that cost the round

**Defining the state loosely.** "`dp[i]` is the answer up to `i`" is not a definition, because it does
not say whether the answer must use element `i`. Those are different states with different
recurrences, and confusing them is why an answer is sometimes `dp[n-1]` and sometimes `max(dp)`. If
your state contains the word "ending", scan the whole table at the end.

**Reading the wrong end of the table.** Template 4 states end at `i`, so `dp[n-1]` covers only the
subsequences that finish at the last element. Returning it instead of `max(dp)` passes the sample
input surprisingly often and then fails.

**Updating two coupled states in sequence.** In Maximum Product Subarray, computing `best_here` and
then using it to compute `worst_here` uses the new value where the old one was meant. Compute both from
the old pair, either with a tuple assignment or with two temporaries.

**The wrong capacity direction.** An ascending capacity loop in a 0/1 knapsack lets each item be taken
many times, and a descending one in a complete knapsack forbids reuse. Both produce plausible numbers
on small inputs. Check the direction against the sentence "may an item be used twice".

**Base cases that are off by one.** `dp[0] = 1` means "one way to do nothing", and it is what makes
every counting DP work. Setting it to 0 makes the whole table zero. In Min Cost Climbing Stairs the
target is index `n`, one past the last stair, and stopping at `n-1` returns the cost of standing on the
last stair rather than of leaving it.

**Missing the unreachable case.** In Coin Change, `float("inf")` marks unreachable amounts and must be
translated to `-1` at the end. Forgetting the translation returns `inf`, and forgetting the sentinel
entirely makes the `min` pick a garbage zero.

**Recursion depth.** A memoised recursion over 10000 elements exceeds the default Python limit. Say
this out loud and offer the table form; do not silently raise the limit.

## Done when

- Given a new problem, you can state the state in one sentence, the transition in one more, and the
  base case, before writing any code, in under a minute.
- You can write the memoised recursion, convert it to a bottom-up table, and then reduce that table to
  rolling variables, and explain what the reduction costs you.
- You can say why Coin Change II loops over coins on the outside while Coin Change loops over the
  amount, and demonstrate the difference on amount 4 with coins 1 and 2.
- You can look at a state definition and say immediately whether the answer is the last cell or the
  maximum over all cells.
