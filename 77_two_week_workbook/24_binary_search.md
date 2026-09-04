# Binary search: every variation

Binary search halves a search space at every step, so it finds what it is looking for in
$O(\log n)$ comparisons instead of the $O(n)$ a linear scan would need. Everyone knows that much.
The reason it still fails in interviews is that the textbook version — find an exact value in a
sorted array — is almost never the question asked. The real questions are boundary searches, such
as "the first element that is at least `x`" or "the last element that is at most `x`", and searches
over an **answer space** that is not an array at all.

The failure is never the halving. The failure is the loop invariant: which half you discard, whether
the bounds are inclusive or exclusive, and what the loop leaves behind when it ends. A candidate who
writes `lo = mid` instead of `lo = mid + 1` gets an infinite loop. A candidate who writes
`hi = len(a) - 1` with a `while lo <= hi` and then returns `lo` gets an off-by-one that only shows up
when the target is larger than every element. Both are invariant errors, not algorithm errors.

This chapter takes one position: learn ONE invariant and use it everywhere. Do not memorise four
variants that differ by an off-by-one, because under pressure you will reach for the wrong one. Learn
the half-open form, prove to yourself that it terminates, and derive every other question from it.

## Recognising it from the phrasing

| The interviewer says | They mean | Search space | What the loop returns |
|---|---|---|---|
| "find the value / does it contain" | plain search, or a boundary plus one check | array indices | `lower_bound`, then compare |
| "first / smallest index where ... becomes true" | lower bound on a predicate | array indices | the boundary itself |
| "last / largest index where ... is still true" | upper bound, then step back one | array indices | `upper_bound - 1` |
| "how many times does `x` occur" | two boundaries, subtracted | array indices | `upper - lower` |
| "minimum capacity / speed / size such that it works" | binary search on the answer | a range of **values** | smallest feasible value |
| "maximum such that it still works" | binary search on the answer, predicate flipped | a range of **values** | largest feasible value |
| "sorted array, but rotated" | find which half is sorted, then decide | array indices | the index, or `-1` |
| "two sorted arrays, find the median" | binary search on the partition point | the split of the shorter array | the correct partition |
| "find a peak element" | binary search on the local slope | array indices | any local maximum |

Before you write a line, ask one question: **what is the predicate, and is it monotone over the
search space?** Binary search needs a predicate that reads false, false, false, true, true, true —
it flips exactly once and never flips back. If you can name that predicate and confirm it never
flips back, you can binary search, whether or not the thing you are searching is an array. Sortedness
is only the most common way to get monotonicity; it is not the requirement. "Is `a[i] >= target`" is
monotone because the array is sorted. "Can Koko finish at speed `s`" is monotone because a faster
speed is never worse. "Is `nums[i] > nums[i+1]`" is monotone enough on a slope to find a peak in an
unsorted array. However, if the predicate is not monotone, binary search will still terminate and
still return some boundary — it just will not be the boundary you wanted, and no test on the sample
input will tell you.

## The templates

Every template below uses the same invariant. The interval is **half-open**, written `[lo, hi)`:
`lo` is inclusive, `hi` is exclusive, and the answer always lies inside it. The loop is
`while lo < hi`, the midpoint is `mid = lo + (hi - lo) // 2`, and each step does exactly one of
`lo = mid + 1` or `hi = mid`.

That form cannot loop forever, and the proof is two lines. Because `lo < hi`, integer division gives
`lo <= mid < hi`. So `hi = mid` strictly decreases `hi`, because `mid < hi`. And `lo = mid + 1`
strictly increases `lo`, because `mid >= lo`. The gap `hi - lo` therefore shrinks by at least one on
every iteration, so the loop ends. When it ends, `lo == hi`, and that single index is the answer —
the first position where the predicate is true. Nothing is left to check afterwards. Write
`mid = lo + (hi - lo) // 2` rather than `(lo + hi) // 2` out of habit; in Python the two are the
same, but in Java or C++ the second overflows, and interviewers notice.

**Template 1 — lower bound.** Use when you want the first index where `a[i] >= target`.

```python
def lower_bound(a, target):
    lo, hi = 0, len(a)                            ## half-open: the answer is in [lo, hi]
    while lo < hi:
        mid = lo + (hi - lo) // 2                 ## lo <= mid < hi, always
        if a[mid] < target:
            lo = mid + 1                          ## a[mid] too small: discard it and everything left
        else:
            hi = mid                              ## a[mid] is a candidate: keep it, discard the right
    return lo                                     ## first index with a[i] >= target

## tests

assert lower_bound([1, 3, 3, 5, 7], 3) == 1
assert lower_bound([1, 3, 3, 5, 7], 4) == 3
assert lower_bound([1, 3, 3, 5, 7], 0) == 0
assert lower_bound([1, 3, 3, 5, 7], 9) == 5
assert lower_bound([], 3) == 0
print(lower_bound([1, 3, 3, 5, 7], 3), lower_bound([1, 3, 3, 5, 7], 4))
```

```
1 3
```

**Template 2 — upper bound.** Use when you want the first index where `a[i] > target`. The skeleton
is character-for-character the same as template 1; only the comparison changes from `<` to `<=`.

```python
def lower_bound(a, target):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def upper_bound(a, target):
    lo, hi = 0, len(a)                            ## identical skeleton to lower_bound
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] <= target:                      ## the ONLY changed line: < becomes <=
            lo = mid + 1
        else:
            hi = mid
    return lo                                     ## first index with a[i] > target

def count_occurrences(a, target):
    return upper_bound(a, target) - lower_bound(a, target)

## tests

import bisect
a = [1, 3, 3, 5, 7]
assert upper_bound(a, 3) == 3
assert upper_bound(a, 0) == 0
assert upper_bound(a, 7) == 5
assert count_occurrences(a, 3) == 2
assert count_occurrences(a, 4) == 0
assert count_occurrences([2, 2, 2, 2], 2) == 4
assert all(upper_bound(a, t) == bisect.bisect_right(a, t) for t in range(-1, 9))
assert all(lower_bound(a, t) == bisect.bisect_left(a, t) for t in range(-1, 9))
print(upper_bound(a, 3), count_occurrences(a, 3))
```

```
3 2
```

**Template 3 — binary search on the answer.** Use when the answer is a number in a known range and
you can test a candidate cheaply. The array indices are gone; `lo` and `hi` are values.

```python
def smallest_feasible(lo, hi, feasible):
    ## the search space is VALUES in [lo, hi], not indices
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid                              ## mid works, so the answer is at most mid
        else:
            lo = mid + 1                          ## mid fails, so the answer is more than mid
    return lo                                     ## smallest value for which feasible is true

## tests

assert smallest_feasible(1, 100, lambda x: x * x >= 50) == 8
assert smallest_feasible(1, 10, lambda x: x >= 1) == 1
assert smallest_feasible(0, 10, lambda x: x >= 10) == 10
print(smallest_feasible(1, 100, lambda x: x * x >= 50))
```

```
8
```

**Template 4 — rotated sorted array.** Use when the array was sorted and then rotated, so it is not
globally sorted but one half of every split still is. This is the one template that uses a closed
range, because it must compare `nums[mid]` to the target and return that index.

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1                     ## closed range: we test nums[mid] directly
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                 ## the LEFT half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                     ## therefore the RIGHT half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1

## tests

assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert search_rotated([1], 1) == 0
assert search_rotated([], 5) == -1
assert search_rotated([3, 1], 1) == 1
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 0), search_rotated([4, 5, 6, 7, 0, 1, 2], 3))
```

```
4 -1
```

Now the point of the section. Exact search, first occurrence, last occurrence, insert position and
count are **not five algorithms**. They are five one-line wrappers around templates 1 and 2. Learn to
derive them and you never have to remember which variant returns what.

```python
def bound(a, target, inclusive):                          ## one loop, both boundaries
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        below = a[mid] <= target if inclusive else a[mid] < target
        if below:
            lo = mid + 1
        else:
            hi = mid
    return lo

def lower_bound(a, target):
    return bound(a, target, False)

def upper_bound(a, target):
    return bound(a, target, True)

def exact_search(a, target):
    i = lower_bound(a, target)
    return i if i < len(a) and a[i] == target else -1     ## one bounds check, one equality check

def first_occurrence(a, target):
    return exact_search(a, target)                        ## lower_bound IS the first occurrence

def last_occurrence(a, target):
    i = upper_bound(a, target) - 1                        ## one before the first strictly greater
    return i if i >= 0 and a[i] == target else -1

def insert_position(a, target):
    return lower_bound(a, target)                         ## the same number, renamed

def count_of(a, target):
    return upper_bound(a, target) - lower_bound(a, target)

## tests

a = [1, 2, 2, 2, 5, 9]
assert exact_search(a, 5) == 4
assert exact_search(a, 3) == -1
assert first_occurrence(a, 2) == 1
assert last_occurrence(a, 2) == 3
assert last_occurrence(a, 3) == -1
assert insert_position(a, 3) == 4
assert insert_position(a, 10) == 6
assert count_of(a, 2) == 3
assert exact_search([], 1) == -1 and count_of([], 1) == 0
print(exact_search(a, 5), first_occurrence(a, 2), last_occurrence(a, 2),
      insert_position(a, 3), count_of(a, 2))
```

```
4 1 3 4 3
```

Note where the answer is recorded in each template. In templates 1, 2 and 3 it is **not** recorded
inside the loop at all — the loop narrows the interval and `lo` is the answer when the loop ends.
That is the whole benefit of the half-open form: there is no `best = mid` line to forget, and no
question about whether the recorded value is stale. Only template 4 returns from inside the loop,
because it is looking for an exact hit rather than a boundary.

## Binary search on the answer

This is the variant people do not recognise, and it is the one that separates candidates. The
statement contains no sorted array and often no array search at all. It says something like "find the
minimum speed", "find the smallest capacity", "find the largest possible minimum distance". Nothing
in the wording says binary search. The shape you are looking for has three parts. First, the answer
is a number inside a range you can name. Second, checking one candidate answer is cheap, usually a
single linear pass. Third, feasibility is monotone: if a candidate works, then every candidate on one
side of it also works.

Answer three questions before you write any code.

**What is the search range?** Name a value that certainly fails and a value that certainly works. For
a speed, the range runs from 1 to the largest pile. For a ship capacity, from the largest single
package, because no smaller ship can carry it at all, up to the sum of all packages, which finishes
in one day. Getting the low end wrong is the usual bug: a capacity of 1 is not a valid low end when a
package weighs 10.

**What does `feasible(x)` mean?** Write it as a function that returns a boolean, and write it before
the search. It is almost always a greedy simulation: walk the input once with `x` fixed and see
whether the constraint holds.

**Does feasibility go false-to-true, or true-to-false?** For a minimisation the predicate is false
for small `x` and true for large `x`, so you want the first true and the code is template 3
unchanged. For a maximisation the predicate is true for small `x` and false for large `x`, so you
want the last true. Do not write a second template for that. Instead flip the predicate — search for
the first `x` where it **fails**, then subtract one — or negate the quantity you are searching over.

**Worked example: Koko eating bananas.** Koko has `piles` of bananas and `hours` hours. In one hour
she eats up to `speed` bananas from a single pile; if the pile has fewer, she eats it and the hour is
over. Find the smallest `speed` that lets her finish everything in time.

The range is 1 to `max(piles)`. Speed 1 is the slowest that makes any progress, and a speed above
`max(piles)` cannot help, because each pile already takes one hour at `max(piles)`. Feasible means
the total hours needed is at most `hours`, and the hours for one pile are `ceil(pile / speed)`. The
predicate is monotone because a faster speed never needs more hours for any pile: `ceil(p / s)` is
non-increasing in `s`. So the pattern of `feasible` over the range is false, false, ..., false, true,
true, ..., true, and the answer is the first true. That is template 3.

Take `piles = [3, 6, 7, 11]` and `hours = 8`. At speed 3 the hours are 1, 2, 3, 4, which is 10, so
speed 3 fails. At speed 4 they are 1, 2, 2, 3, which is 8, so speed 4 works. Speeds 1 and 2 are
slower than 3 and therefore also fail, and every speed above 4 works. The boundary is 4.

```python
def min_eating_speed(piles, hours):
    def feasible(speed):
        needed = 0
        for pile in piles:
            needed += (pile + speed - 1) // speed        ## ceiling division, no floats
        return needed <= hours
    lo, hi = 1, max(piles)                               ## 1 certainly slow, max(piles) certainly enough
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid                                     ## this speed works, try slower
        else:
            lo = mid + 1                                 ## too slow, must go faster
    return lo

## tests

assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
assert min_eating_speed([1], 1) == 1
assert min_eating_speed([1000000000], 2) == 500000000
print(min_eating_speed([3, 6, 7, 11], 8), min_eating_speed([30, 11, 23, 4, 20], 6))
```

```
4 23
```

The cost is $O(n \log(\text{range}))$, where `n` is the length of the input and the range is the span
of candidate answers. Say that carefully in an interview: the logarithm is over the **value** range,
not over the array length. For Koko with piles up to $10^9$ that is about 30 iterations of a linear
scan, which is why the method is fast even though the answer space is enormous.

Use `(pile + speed - 1) // speed` for the ceiling rather than `math.ceil(pile / speed)`. Floating
point division of large integers loses precision and will fail a hidden test with a value near
$10^9$.

## The problems

### P1. Classic binary search — return the index of `target` in a sorted array, or `-1`

**Which template.** Template 1, plus one equality check after the loop.
**The trick.** Do not write a separate exact-search loop with `while lo <= hi` and an early return.
Write `lower_bound` and check the one index it hands back. The cost is one extra comparison, and the
gain is that you now have only one loop shape to remember under pressure. The bounds check
`lo < len(nums)` comes first, because when the target is larger than everything, `lo` equals `len`.

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(nums) and nums[lo] == target:     ## the one check after the boundary
        return lo
    return -1

## tests

assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4
assert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1
assert binary_search([5], 5) == 0
assert binary_search([5], 4) == -1
assert binary_search([], 1) == -1
assert binary_search([1, 1, 1], 1) == 0
print(binary_search([-1, 0, 3, 5, 9, 12], 9), binary_search([-1, 0, 3, 5, 9, 12], 2))
```

```
4 -1
```

**Complexity.** $O(\log n)$ time, $O(1)$ space.

### P2. Search insert position — the index where `target` is, or where it would be inserted to keep the array sorted

**Which template.** Template 1, with nothing after the loop.
**The trick.** This problem is `lower_bound` with no wrapper at all. That is worth noticing, because
it explains why `hi` starts at `len(nums)` and not `len(nums) - 1`: the insert position of a target
larger than every element is `n`, which is a legal answer and must be reachable. A closed range can
never return `n`, so the half-open form is not a style choice here, it is required.

```python
def search_insert(nums, target):
    lo, hi = 0, len(nums)                         ## hi = len(nums), because n is a legal answer
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo                                     ## no post-check at all

## tests

import bisect
assert search_insert([1, 3, 5, 6], 5) == 2
assert search_insert([1, 3, 5, 6], 2) == 1
assert search_insert([1, 3, 5, 6], 7) == 4
assert search_insert([1, 3, 5, 6], 0) == 0
assert search_insert([], 3) == 0
a = [1, 3, 5, 6]
assert all(search_insert(a, t) == bisect.bisect_left(a, t) for t in range(-2, 9))
print(search_insert([1, 3, 5, 6], 5), search_insert([1, 3, 5, 6], 7))
```

```
2 4
```

**Complexity.** $O(\log n)$ time, $O(1)$ space. The final assertion checks the whole function against
`bisect.bisect_left` on every target from -2 to 8, and they agree everywhere.

### P3. First and last position of an element — the range `[first, last]` of `target`, or `[-1, -1]`

**Which template.** Template 1 run twice, at `target` and at `target + 1`.
**The trick.** You do not need a second, mirrored loop for the last position. The first index strictly
after all copies of `target` is `lower_bound(target + 1)`, so the last copy is one before it. This
works only for integers, because `target + 1` assumes the next possible value; for floats or strings
use a real `upper_bound` instead. Check absence once, on the first bound, and return early.

```python
def search_range(nums, target):
    def lower_bound(t):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo
    first = lower_bound(target)
    if first == len(nums) or nums[first] != target:
        return [-1, -1]                           ## target absent: both ends are -1
    last = lower_bound(target + 1) - 1            ## first index past the target, minus one
    return [first, last]

## tests

assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
assert search_range([], 0) == [-1, -1]
assert search_range([2, 2, 2, 2], 2) == [0, 3]
assert search_range([1], 1) == [0, 0]
print(search_range([5, 7, 7, 8, 8, 10], 8), search_range([2, 2, 2, 2], 2))
```

```
[3, 4] [0, 3]
```

**Complexity.** $O(\log n)$ time — two searches, so $2 \log n$ — and $O(1)$ space.

### P4. Count occurrences — how many times `target` appears in a sorted array

**Which template.** Templates 1 and 2, subtracted.
**The trick.** `upper_bound(t) - lower_bound(t)` is the count, and it needs no special case for
absence: when the target is missing, both bounds land on the same index and the difference is zero.
Here the two searches share one function with a flag, which makes the single differing comparison
visible on one line.

```python
def count_target(nums, target):
    def bound(t, inclusive):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = lo + (hi - lo) // 2
            below = nums[mid] <= t if inclusive else nums[mid] < t
            if below:
                lo = mid + 1
            else:
                hi = mid
        return lo
    return bound(target, True) - bound(target, False)   ## upper_bound - lower_bound

## tests

import bisect
a = [1, 2, 2, 2, 3, 5, 5]
assert count_target(a, 2) == 3
assert count_target(a, 5) == 2
assert count_target(a, 4) == 0
assert count_target([], 1) == 0
assert count_target([7, 7, 7], 7) == 3
assert all(count_target(a, t) == bisect.bisect_right(a, t) - bisect.bisect_left(a, t)
           for t in range(0, 7))
print(count_target(a, 2), count_target(a, 5), count_target(a, 4))
```

```
3 2 0
```

**Complexity.** $O(\log n)$ time, $O(1)$ space. The last assertion checks every target against
`bisect.bisect_right` minus `bisect.bisect_left`, and the two agree on all of them.

### P5. Integer square root — the largest integer `r` with `r * r <= x`

**Which template.** Template 3 on the value range, looking for the **last** true rather than the
first.
**The trick.** The predicate `mid * mid <= x` is true then false, which is the reverse of the
template's shape. Do not write a new loop. Search for the first index where the predicate fails and
subtract one. That single move converts every "largest such that" into the one template you already
know. Use integer multiplication, never `x ** 0.5`, because floats round the wrong way near perfect
squares of large numbers.

```python
def integer_sqrt(x):
    if x < 2:
        return x
    lo, hi = 1, x // 2 + 1                        ## for x >= 2 the root is at most x // 2
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid * mid <= x:                        ## mid is feasible, so the answer is at least mid
            lo = mid + 1
        else:
            hi = mid
    return lo - 1                                 ## last feasible = first infeasible, minus one

## tests

assert integer_sqrt(4) == 2
assert integer_sqrt(8) == 2
assert integer_sqrt(0) == 0
assert integer_sqrt(1) == 1
assert integer_sqrt(2147395600) == 46340
assert all(integer_sqrt(n) == int(n ** 0.5) for n in range(0, 2000))
print(integer_sqrt(8), integer_sqrt(2147395600))
```

```
2 46340
```

**Complexity.** $O(\log x)$ time, $O(1)$ space. The logarithm is over the value, not over any array.

### P6. Guess number higher or lower — a judge tells you whether your guess is too high, too low or right

**Which template.** Template 4's closed-range shape, because a hit returns immediately.
**The trick.** This is the only common problem where the closed range is genuinely simpler, because
the judge gives you a three-way answer and one of the three is "stop". The mapping to remember: the
judge returns -1 when your guess is too high, so the new interval is `[lo, mid - 1]`. Read the sign
convention out loud before coding, because the API is deliberately counter-intuitive.

```python
def guess_number(n, pick):
    def guess(num):                               ## the judge: -1 too high, 1 too low, 0 correct
        if num > pick:
            return -1
        if num < pick:
            return 1
        return 0
    lo, hi = 1, n                                 ## closed range, because a hit returns immediately
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        result = guess(mid)
        if result == 0:
            return mid
        if result < 0:
            hi = mid - 1                          ## too high: discard mid and everything above
        else:
            lo = mid + 1
    return -1

## tests

assert guess_number(10, 6) == 6
assert guess_number(1, 1) == 1
assert guess_number(2, 1) == 1
assert all(guess_number(100, p) == p for p in range(1, 101))
print(guess_number(10, 6), guess_number(2, 1))
```

```
6 1
```

**Complexity.** $O(\log n)$ time, $O(1)$ space.

### P7. First bad version — versions 1 to `n`, all after some point are bad; find the first bad one with the fewest API calls

**Which template.** Template 3, on version numbers rather than indices.
**The trick.** This problem is the purest statement of the whole pattern, so use it as your mental
model. There is no array. The search space is the integers 1 to `n`, and `is_bad` is exactly the
monotone predicate the pattern needs: once true it stays true. When the loop ends, `lo` is the
boundary and no post-check is needed, because the problem guarantees at least one bad version.

```python
def first_bad_version(n, first_bad):
    def is_bad(version):
        return version >= first_bad               ## monotone: false ... false, true ... true
    lo, hi = 1, n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if is_bad(mid):
            hi = mid                              ## mid is bad, so the first bad one is at most mid
        else:
            lo = mid + 1                          ## mid is good, so the first bad one is after mid
    return lo

## tests

assert first_bad_version(5, 4) == 4
assert first_bad_version(1, 1) == 1
assert first_bad_version(2126753390, 1702766719) == 1702766719
assert all(first_bad_version(50, b) == b for b in range(1, 51))
print(first_bad_version(5, 4), first_bad_version(2126753390, 1702766719))
```

```
4 1702766719
```

**Complexity.** $O(\log n)$ API calls, $O(1)$ space.

### P8. Peak element — return the index of any element strictly greater than both its neighbours

**Which template.** Template 1's shape, with the comparison against the neighbour instead of a
target.
**The trick.** The array is **not sorted**, and saying so is the point of the problem. Binary search
still applies because of an invariant: if `nums[mid] < nums[mid + 1]` then the right half must
contain a peak, since the values are rising and the boundary counts as negative infinity. So the
half you keep always still contains a peak, and the interval shrinks to a single index that must be
one. Treat the two ends as negative infinity, which the loop does implicitly by never letting `lo`
or `hi` step outside the array.

```python
def find_peak_element(nums):
    lo, hi = 0, len(nums) - 1                     ## a peak always exists inside [lo, hi]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:             ## uphill on the right: a peak lies to the right
            lo = mid + 1
        else:                                     ## downhill: mid itself may be the peak
            hi = mid
    return lo

## tests

def is_peak(a, i):
    left = a[i - 1] if i > 0 else float("-inf")
    right = a[i + 1] if i + 1 < len(a) else float("-inf")
    return a[i] > left and a[i] > right

assert find_peak_element([1, 2, 3, 1]) == 2
assert find_peak_element([1]) == 0
assert find_peak_element([1, 2]) == 1
assert find_peak_element([2, 1]) == 0
peak = find_peak_element([1, 2, 1, 3, 5, 6, 4])
assert peak in (1, 5)
assert is_peak([1, 2, 1, 3, 5, 6, 4], peak)
print(find_peak_element([1, 2, 3, 1]), peak)
```

```
2 5
```

**Complexity.** $O(\log n)$ time, $O(1)$ space. Any peak is acceptable, so there is no need to keep
searching once the interval collapses.

### P9. Search in a rotated sorted array — a sorted array rotated at an unknown pivot; find `target`

**Which template.** Template 4.
**The trick.** Split at `mid` and exactly one of the two halves is sorted, always. Decide which by
comparing `nums[lo]` to `nums[mid]`. Then you can test membership in the sorted half with a simple
range check, and the answer is either in it or in the other half. The comparison must be `<=`, not
`<`, because when `lo == mid` the left half is a single element and is trivially sorted; using `<`
there sends a two-element array down the wrong branch.

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                 ## the left half [lo, mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1                      ## target lies inside the sorted left half
            else:
                lo = mid + 1
        else:                                     ## therefore the right half [mid, hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1                      ## target lies inside the sorted right half
            else:
                hi = mid - 1
    return -1

## tests

assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 5) == 1
assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
assert search_rotated([1], 0) == -1
assert search_rotated([], 1) == -1
assert search_rotated([1, 2, 3, 4, 5], 4) == 3
base = [0, 1, 2, 4, 5, 6, 7]
for r in range(len(base)):
    rot = base[r:] + base[:r]
    for t in base:
        assert rot[search_rotated(rot, t)] == t
print(search_rotated([4, 5, 6, 7, 0, 1, 2], 0), search_rotated([4, 5, 6, 7, 0, 1, 2], 3))
```

```
4 -1
```

**Complexity.** $O(\log n)$ time, $O(1)$ space. The loop over all seven rotations checks every target
in every rotation, and all of them are found at the right index.

### P10. Search in a rotated sorted array with duplicates — the same question, but values may repeat; return a boolean

**Which template.** Template 4 with one extra branch.
**The trick.** Duplicates break the "one half is always sorted" argument. When
`nums[lo] == nums[mid] == nums[hi]`, as in `[1, 0, 1, 1, 1]`, the comparison tells you nothing: the
pivot could be on either side. There is no clever fix. The only correct move is to shed one element
from each end and try again. Be honest about the cost in the interview: on an array that is all one
value the algorithm degrades to $O(n)$, because each iteration removes two elements instead of half
the range. That degradation is not a bug in your code, it is a property of the problem — no
comparison-based method can do better, since an array of all 1s with a single 0 hidden anywhere
forces you to look at essentially every position.

```python
def search_rotated_dup(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:     ## cannot tell which half is sorted
            lo += 1                               ## shed one from each end and retry
            hi -= 1
        elif nums[lo] <= nums[mid]:               ## the left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                     ## the right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False

## tests

assert search_rotated_dup([2, 5, 6, 0, 0, 1, 2], 0) is True
assert search_rotated_dup([2, 5, 6, 0, 0, 1, 2], 3) is False
assert search_rotated_dup([1, 0, 1, 1, 1], 0) is True
assert search_rotated_dup([1, 1, 1, 1, 1], 2) is False
assert search_rotated_dup([], 1) is False
assert search_rotated_dup([1], 1) is True
print(search_rotated_dup([2, 5, 6, 0, 0, 1, 2], 0), search_rotated_dup([1, 0, 1, 1, 1], 0))
```

```
True True
```

**Complexity.** $O(\log n)$ average, $O(n)$ worst case, $O(1)$ space.

### P11. Find the minimum in a rotated sorted array — the smallest value, with no duplicates

**Which template.** Template 1's shape, comparing `nums[mid]` to `nums[hi]` instead of to a target.
**The trick.** Compare against the **right** end, not the left. If `nums[mid] > nums[hi]` then `mid`
sits in the higher run before the pivot, so the minimum is strictly to the right and `lo = mid + 1`.
Otherwise `mid` is in the lower run and could itself be the minimum, so `hi = mid` keeps it.
Comparing against `nums[lo]` instead needs an extra case for the already-sorted array, which is
exactly the kind of special case that gets forgotten under pressure.

```python
def find_min_rotated(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:                  ## mid is in the high run: minimum is to the right
            lo = mid + 1
        else:                                     ## mid may be the minimum: keep it
            hi = mid
    return nums[lo]

## tests

assert find_min_rotated([3, 4, 5, 1, 2]) == 1
assert find_min_rotated([4, 5, 6, 7, 0, 1, 2]) == 0
assert find_min_rotated([11, 13, 15, 17]) == 11
assert find_min_rotated([2, 1]) == 1
assert find_min_rotated([5]) == 5
base = [1, 3, 5, 7, 9, 11]
for r in range(len(base)):
    assert find_min_rotated(base[r:] + base[:r]) == 1
print(find_min_rotated([3, 4, 5, 1, 2]), find_min_rotated([4, 5, 6, 7, 0, 1, 2]))
```

```
1 0
```

**Complexity.** $O(\log n)$ time, $O(1)$ space.

### P12. Find the minimum in a rotated sorted array with duplicates — the same, with equal endpoints allowed

**Which template.** P11 with the tie split out into its own branch.
**The trick.** The two-way test of P11 becomes a three-way test, and the third case is the whole
problem. When `nums[mid] == nums[hi]` you cannot tell whether the pivot is left or right of `mid`, so
you cannot discard a half. But you can safely discard **one** element: `hi -= 1` is correct because
`nums[hi]` has an equal twin at `mid`, so removing it never removes the only copy of the minimum.
That justification is what the interviewer is listening for. Note it must be `hi -= 1` and not
`hi = mid`, because `hi = mid` would already have skipped past nothing and can stall.

```python
def find_min_rotated_dup(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1                          ## minimum is strictly to the right of mid
        elif nums[mid] < nums[hi]:
            hi = mid                              ## mid is a candidate minimum
        else:
            hi -= 1                               ## equal ends: nums[hi] has a twin at mid, drop it
    return nums[lo]

## tests

assert find_min_rotated_dup([1, 3, 5]) == 1
assert find_min_rotated_dup([2, 2, 2, 0, 1]) == 0
assert find_min_rotated_dup([3, 3, 1, 3]) == 1
assert find_min_rotated_dup([1, 1, 1, 1]) == 1
assert find_min_rotated_dup([10, 1, 10, 10, 10]) == 1
assert find_min_rotated_dup([5]) == 5
base = [1, 1, 2, 2, 3, 3]
for r in range(len(base)):
    assert find_min_rotated_dup(base[r:] + base[:r]) == 1
print(find_min_rotated_dup([2, 2, 2, 0, 1]), find_min_rotated_dup([10, 1, 10, 10, 10]))
```

```
0 1
```

**Complexity.** $O(\log n)$ average, $O(n)$ worst case when all values are equal, $O(1)$ space.

### P13. Search a 2D matrix — rows are sorted and each row starts after the previous row ends

**Which template.** Template 1, over a virtual flat array.
**The trick.** The stated property — every row begins with a value larger than the last value of the
row above — means the matrix read row by row is one sorted list. So do not write a two-stage search.
Binary search the flat index range `[0, rows * cols)` and convert with `divmod`: row is
`mid // cols`, column is `mid % cols`. One loop, one conversion line, and the 2D structure never
enters the logic.

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols                       ## flat indices 0 .. rows * cols - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        value = matrix[mid // cols][mid % cols]   ## the only line that knows the matrix is 2D
        if value < target:
            lo = mid + 1
        else:
            hi = mid
    return lo < rows * cols and matrix[lo // cols][lo % cols] == target

## tests

m = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
assert search_matrix(m, 3) is True
assert search_matrix(m, 13) is False
assert search_matrix(m, 60) is True
assert search_matrix(m, 0) is False
assert search_matrix([[1]], 1) is True
assert search_matrix([], 1) is False
assert search_matrix([[]], 1) is False
flat = [x for row in m for x in row]
assert all(search_matrix(m, v) == (v in flat) for v in range(0, 62))
print(search_matrix(m, 3), search_matrix(m, 13))
```

```
True False
```

**Complexity.** $O(\log(mn))$ time, $O(1)$ space.

### P14. Search a 2D matrix II — rows and columns are each sorted, but rows do not chain; is `target` present

**Which template.** None. This one is **not** binary search, and saying so is the answer.
**The trick.** Without the chaining property the flat array is no longer sorted, so P13's method is
wrong. Instead walk a staircase from the top-right corner. That corner is the largest in its row and
the smallest in its column, so a comparison always eliminates a whole row or a whole column. If the
value is too big, every element below it in that column is too big, so drop the column. If it is too
small, every element to its left in that row is too small, so drop the row. Each step removes one
row or one column, so the walk is $O(m + n)$, not $O(\log(mn))$. Include this next to P13 because
interviewers pair the two deliberately to see whether you notice the missing property.

```python
def search_matrix_staircase(matrix, target):
    if not matrix or not matrix[0]:
        return False
    row, col = 0, len(matrix[0]) - 1              ## start at the TOP-RIGHT corner
    while row < len(matrix) and col >= 0:
        value = matrix[row][col]
        if value == target:
            return True
        if value > target:
            col -= 1                              ## everything below in this column is bigger
        else:
            row += 1                              ## everything left in this row is smaller
    return False

## tests

m = [[1, 4, 7, 11, 15],
     [2, 5, 8, 12, 19],
     [3, 6, 9, 16, 22],
     [10, 13, 14, 17, 24],
     [18, 21, 23, 26, 30]]
assert search_matrix_staircase(m, 5) is True
assert search_matrix_staircase(m, 20) is False
assert search_matrix_staircase(m, 30) is True
assert search_matrix_staircase(m, 1) is True
assert search_matrix_staircase([], 1) is False
present = {x for row in m for x in row}
assert all(search_matrix_staircase(m, v) == (v in present) for v in range(0, 32))
print(search_matrix_staircase(m, 5), search_matrix_staircase(m, 20))
```

```
True False
```

**Complexity.** $O(m + n)$ time, $O(1)$ space. Binary searching each row separately is
$O(m \log n)$, which is worse whenever `m` is large, so the staircase is the right answer.

### P15. Koko eating bananas — the smallest eating speed that clears all piles within `hours` hours

**Which template.** Template 3, binary search on the answer.
**The trick.** Worked in full in the section above. The three answers are: the range is 1 to
`max(piles)`; `feasible(speed)` is "the summed ceiling divisions are at most `hours`"; and
feasibility runs false then true, so the first true is the answer. One pile can never be shared
across hours, which is why the hours for a pile are `ceil(pile / speed)` and not a single global
division.

```python
def min_eating_speed(piles, hours):
    def feasible(speed):
        needed = 0
        for pile in piles:
            needed += (pile + speed - 1) // speed
        return needed <= hours
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

## tests

assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
assert min_eating_speed([312884470], 968709470) == 1
assert min_eating_speed([1, 1, 1, 1], 4) == 1
print(min_eating_speed([3, 6, 7, 11], 8), min_eating_speed([30, 11, 23, 4, 20], 5))
```

```
4 30
```

**Complexity.** $O(n \log(\max(\text{piles})))$ time, $O(1)$ space.

### P16. Capacity to ship packages within D days — the smallest ship capacity that ships all packages in order within `days` days

**Which template.** Template 3, with a greedy `feasible`.
**The trick.** The low end of the range is `max(weights)`, not 1. A ship smaller than the heaviest
package can never carry it, so every capacity below that is infeasible and including them only wastes
iterations — worse, it invites an off-by-one if you then forget the packages must fit individually.
The high end is `sum(weights)`, which finishes in a single day. `feasible` is a greedy pass: load
until the next package would overflow, then start a new day. Greedy is optimal here because the
package order is fixed.

```python
def ship_within_days(weights, days):
    def feasible(capacity):
        used, load = 1, 0
        for w in weights:
            if load + w > capacity:
                used += 1                         ## start a new day
                load = 0
            load += w
        return used <= days
    lo, hi = max(weights), sum(weights)           ## low end is the HEAVIEST package, not 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

## tests

assert ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
assert ship_within_days([3, 2, 2, 4, 1, 4], 3) == 6
assert ship_within_days([1, 2, 3, 1, 1], 4) == 3
assert ship_within_days([10], 1) == 10
assert ship_within_days([1, 2, 3], 3) == 3
print(ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5), ship_within_days([3, 2, 2, 4, 1, 4], 3))
```

```
15 6
```

**Complexity.** $O(n \log(\sum w))$ time, $O(1)$ space.

### P17. Split array largest sum — split the array into `k` contiguous parts, minimising the largest part sum

**Which template.** Template 3, and it is P16 with the words changed.
**The trick.** Say the reduction out loud: "days" becomes "parts" and "capacity" becomes "largest
allowed part sum", and the code is identical. The dynamic-programming solution to this problem is
$O(n^2 k)$ and takes twenty minutes to write correctly; the binary search is twelve lines. Recognising
that "minimise the maximum" is a feasibility search, not an optimisation over splits, is the entire
answer.

```python
def split_array_largest_sum(nums, k):
    def feasible(cap):
        parts, running = 1, 0
        for x in nums:
            if running + x > cap:
                parts += 1                        ## start a new part
                running = 0
            running += x
        return parts <= k
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

## tests

assert split_array_largest_sum([7, 2, 5, 10, 8], 2) == 18
assert split_array_largest_sum([1, 2, 3, 4, 5], 2) == 9
assert split_array_largest_sum([1, 4, 4], 3) == 4
assert split_array_largest_sum([5], 1) == 5
assert split_array_largest_sum([2, 3, 1], 3) == 3
print(split_array_largest_sum([7, 2, 5, 10, 8], 2), split_array_largest_sum([1, 2, 3, 4, 5], 2))
```

```
18 9
```

**Complexity.** $O(n \log(\sum \text{nums}))$ time, $O(1)$ space.

### P18. Minimum days to make m bouquets — each bouquet needs `k` **adjacent** flowers that have bloomed

**Which template.** Template 3, over days rather than over flowers.
**The trick.** Two things. First, the impossibility check `m * k > len(bloom_day)` must come before
the search, because otherwise the loop returns `max(bloom_day)` and reports a wrong answer instead of
`-1`. Second, the adjacency requirement lives entirely in `feasible`: walk the array counting a run
of bloomed flowers, and reset the run to zero on any flower that has not bloomed. Forgetting the
reset counts non-adjacent flowers and produces an answer that is too small.

```python
def min_days_bouquets(bloom_day, m, k):
    if m * k > len(bloom_day):
        return -1                                 ## not enough flowers, on any day
    def feasible(day):
        made, run = 0, 0
        for d in bloom_day:
            if d <= day:
                run += 1
                if run == k:                      ## a full adjacent group
                    made += 1
                    run = 0
            else:
                run = 0                           ## the group must be ADJACENT
        return made >= m
    lo, hi = min(bloom_day), max(bloom_day)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

## tests

assert min_days_bouquets([1, 10, 3, 10, 2], 3, 1) == 3
assert min_days_bouquets([1, 10, 3, 10, 2], 3, 2) == -1
assert min_days_bouquets([7, 7, 7, 7, 12, 7, 7], 2, 3) == 12
assert min_days_bouquets([1, 10, 2, 9, 3, 8, 4, 7, 5, 6], 4, 2) == 9
assert min_days_bouquets([1], 1, 1) == 1
print(min_days_bouquets([1, 10, 3, 10, 2], 3, 1), min_days_bouquets([7, 7, 7, 7, 12, 7, 7], 2, 3))
```

```
3 12
```

**Complexity.** $O(n \log(\max(\text{bloom}) - \min(\text{bloom})))$ time, $O(1)$ space.

### P19. Find the smallest divisor given a threshold — the smallest `d` such that the summed ceiling divisions are at most `threshold`

**Which template.** Template 3, and it is Koko with the story removed.
**The trick.** There is no trick beyond recognition, which is why the problem is useful: it is the
same computation as P15 with `hours` renamed `threshold`. The high end is `max(nums)`, because at
that divisor every term is already 1 and the sum equals `len(nums)`, which is the smallest sum
reachable. A larger divisor cannot help.

```python
def smallest_divisor(nums, threshold):
    def total(divisor):
        s = 0
        for x in nums:
            s += (x + divisor - 1) // divisor     ## ceiling division
        return s
    lo, hi = 1, max(nums)                         ## divisor max(nums) makes every term 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if total(mid) <= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo

## tests

assert smallest_divisor([1, 2, 5, 9], 6) == 5
assert smallest_divisor([1, 2, 5, 9], 17) == 1
assert smallest_divisor([44, 22, 33, 11, 1], 5) == 44
assert smallest_divisor([2, 3, 5, 7, 11], 11) == 3
assert smallest_divisor([1], 1) == 1
print(smallest_divisor([1, 2, 5, 9], 6), smallest_divisor([44, 22, 33, 11, 1], 5))
```

```
5 44
```

**Complexity.** $O(n \log(\max(\text{nums})))$ time, $O(1)$ space.

### P20. Kth smallest element in a sorted matrix — rows and columns are sorted; find the `k`-th smallest overall

**Which template.** Template 3 on the value range, with P14's staircase inside the predicate.
**The trick.** Search values, not positions. The predicate is "at least `k` entries are less than or
equal to `x`", which is monotone in `x`, and the counting is P14's staircase walk from the bottom-left
corner in $O(n)$. Two things make the result correct rather than merely close. First, the answer
returned is always an element of the matrix: the boundary value is the smallest `x` whose count
reaches `k`, and the count only increases at values that are actually present. Second, duplicates
need no special handling, because counting "at most `x`" already absorbs them. A min-heap solution is
$O(k \log n)$ and is worse when `k` is near $n^2$.

```python
def kth_smallest_matrix(matrix, k):
    n = len(matrix)
    def count_at_most(x):
        total, row, col = 0, n - 1, 0             ## staircase from the BOTTOM-LEFT corner
        while row >= 0 and col < n:
            if matrix[row][col] <= x:
                total += row + 1                  ## the whole column above this row qualifies
                col += 1
            else:
                row -= 1
        return total
    lo, hi = matrix[0][0], matrix[n - 1][n - 1]   ## search VALUES, not positions
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if count_at_most(mid) >= k:
            hi = mid                              ## mid is large enough, try smaller
        else:
            lo = mid + 1
    return lo

## tests

m = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
assert kth_smallest_matrix(m, 8) == 13
assert kth_smallest_matrix(m, 1) == 1
assert kth_smallest_matrix(m, 9) == 15
assert kth_smallest_matrix([[-5]], 1) == -5
flat = sorted(x for row in m for x in row)
assert all(kth_smallest_matrix(m, k) == flat[k - 1] for k in range(1, 10))
print(kth_smallest_matrix(m, 8), kth_smallest_matrix(m, 1), kth_smallest_matrix(m, 9))
```

```
13 1 15
```

**Complexity.** $O(n \log(\text{value range}))$ time, $O(1)$ space.

### P21. Median of two sorted arrays — the median of the union, in logarithmic time

**Which template.** Template 4's closed range, searching over the **partition point** of the shorter
array. This is the hardest classic here, so work it slowly.
**The trick.** Do not think about merging. Think about cutting. A median splits the union into a left
part and a right part of known sizes. If you take `i` elements from `a` for the left part, then `j`
is forced: `j = half - i`, where `half = (m + n + 1) // 2`. So there is one unknown, `i`, and it
ranges over `[0, m]`. The partition is correct exactly when both cross-conditions hold:
`a[i-1] <= b[j]` and `b[j-1] <= a[i]`. If `a[i-1] > b[j]` you took too many from `a`, so move `hi`
down; otherwise you took too few, so move `lo` up. Three details decide whether this works. Search
the **shorter** array, so `j` can never fall outside `b`. Use `(m + n + 1) // 2` with the `+ 1`, so
the odd case puts the extra element on the left and the median is `max(a_left, b_left)`. And use
infinities for the four edge reads, which removes every boundary special case.

```python
def median_two_sorted(a, b):
    if len(a) > len(b):
        a, b = b, a                               ## always search the SHORTER array
    m, n = len(a), len(b)
    half = (m + n + 1) // 2                       ## size of the combined left part
    lo, hi = 0, m                                 ## i = how many of a go left, in [0, m]
    while lo <= hi:
        i = lo + (hi - lo) // 2
        j = half - i                              ## j is forced by i
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < m else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n else float("inf")
        if a_left <= b_right and b_left <= a_right:          ## the partition is correct
            if (m + n) % 2 == 1:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2.0
        if a_left > b_right:
            hi = i - 1                            ## took too many from a
        else:
            lo = i + 1                            ## took too few from a
    return 0.0

## tests

import random
assert median_two_sorted([1, 3], [2]) == 2.0
assert median_two_sorted([1, 2], [3, 4]) == 2.5
assert median_two_sorted([], [1]) == 1.0
assert median_two_sorted([], [2, 3]) == 2.5
assert median_two_sorted([1, 1, 1], [1, 1, 1]) == 1.0
assert median_two_sorted([5, 6, 7], [1, 2, 3, 4]) == 4.0
for _ in range(300):                              ## random check against a full merge
    x = sorted(random.randint(-20, 20) for _ in range(random.randint(0, 6)))
    y = sorted(random.randint(-20, 20) for _ in range(random.randint(1, 6)))
    merged = sorted(x + y)
    t = len(merged)
    want = merged[t // 2] if t % 2 else (merged[t // 2 - 1] + merged[t // 2]) / 2.0
    assert median_two_sorted(x, y) == want
print(median_two_sorted([1, 3], [2]), median_two_sorted([1, 2], [3, 4]))
```

```
2.0 2.5
```

**Complexity.** $O(\log(\min(m, n)))$ time, $O(1)$ space. The 300 random trials all agree with a full
merge, including the cases where one array is empty.

### P22. Find a value in an **unsorted** array — the problem where binary search is the wrong tool

**Which template.** None. Scan it.
**The trick.** Binary search needs a monotone predicate, and an unsorted array supplies none. The
tempting move is to sort first and then search, which is $O(n \log n) + O(\log n)$. A linear scan is
$O(n)$ and stops at the first hit. Therefore sorting to enable binary search is slower than not doing
it, for a single query. The rule to state in the interview is about the number of queries: sorting
costs $O(n \log n)$ once and then each query costs $O(\log n)$, so sorting pays for itself only when
you expect roughly $\log n$ queries or more against the same data. For one query, scan. Note also
that sorting destroys the original indices, so if the answer is an index you must sort pairs, which
costs extra space as well.

```python
def find_in_unsorted_linear(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i                              ## O(n), and it stops early on a hit
    return -1

def find_in_unsorted_by_sorting(nums, target):
    pairs = sorted((x, i) for i, x in enumerate(nums))       ## O(n log n) just to prepare
    lo, hi = 0, len(pairs)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if pairs[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(pairs) and pairs[lo][0] == target:
        return pairs[lo][1]
    return -1

## tests

nums = [9, 4, 7, 1, 8, 3]
assert find_in_unsorted_linear(nums, 7) == 2
assert find_in_unsorted_linear(nums, 5) == -1
assert find_in_unsorted_by_sorting(nums, 7) == 2
assert find_in_unsorted_by_sorting(nums, 5) == -1
assert find_in_unsorted_linear([], 1) == -1
assert find_in_unsorted_by_sorting([], 1) == -1
assert all(find_in_unsorted_linear(nums, t) == find_in_unsorted_by_sorting(nums, t)
           for t in range(0, 11))
print(find_in_unsorted_linear(nums, 7), find_in_unsorted_by_sorting(nums, 7))
```

```
2 2
```

**Complexity.** $O(n)$ time and $O(1)$ space for the scan, against $O(n \log n)$ time and $O(n)$
space for the sort-then-search. Both return the same answers here, and the second is strictly worse.

## Tricks and tips

**Write the predicate before the loop.** Before you touch `lo` and `hi`, write one line saying what
is true of every element to the right of the boundary and false of every element to its left. If you
cannot write that line, you do not yet have a binary search, and no amount of fiddling with `mid`
will produce one. Once the line exists, the loop is mechanical.

**Use half-open `[lo, hi)` everywhere you can.** `hi = len(a)`, `while lo < hi`, and the two updates
`lo = mid + 1` or `hi = mid`. It terminates by construction, it needs no post-loop adjustment, and it
can return `len(a)`, which the insert-position and lower-bound questions require. Keep the closed
range `[lo, hi]` with `while lo <= hi` for exactly two situations: when a hit returns immediately
from inside the loop, as in the rotated search and the guessing game, and when you compare `mid`
against `hi` itself, as in finding the minimum of a rotated array.

**Turn "largest such that" into "first such that" by flipping and subtracting one.** The half-open
loop finds the first true. For a last-true question, search for the first index where the predicate
**fails**, then subtract one. The integer square root in P5 is the model: the predicate
`mid * mid <= x` is a last-true question, the loop finds the first failure, and `return lo - 1` is
the whole adaptation. Writing a second mirrored template is the mistake this avoids.

**Use `divmod` arithmetic instead of nested searches for a flattened matrix.** `mid // cols` and
`mid % cols` reduce a 2D search to a 1D one and keep the loop identical to template 1. Reach for it
whenever the matrix is stated to be one long sorted sequence.

**Use integer ceiling division, never floats.** `(a + b - 1) // b` is exact for positive integers.
`math.ceil(a / b)` converts to float first and loses precision above $2^{53}$, which is inside the
range these problems use. The same rule bans `x ** 0.5` for integer square roots.

**Name the low end of an answer range by asking what certainly fails.** For a ship capacity that is
`max(weights) - 1`, so the range starts at `max(weights)`. For a divisor it is 0, so the range starts
at 1. Guessing 1 as the low end out of habit is the most common error in answer-space problems,
because the code still runs and still returns something.

**When duplicates appear, expect the guarantee to weaken.** In a rotated array with duplicates you
cannot tell which half is sorted, so the method degrades to $O(n)$. In finding the minimum you can
still shed one element per tie, which is also $O(n)$ in the worst case. Say the degradation aloud
before the interviewer asks, and give the reason: an array of identical values with one different
element hides that element from every comparison.

**Check your loop against `bisect` when practising.** `bisect.bisect_left` is `lower_bound` and
`bisect.bisect_right` is `upper_bound`. Comparing your handwritten function against them over a range
of targets, including targets below and above every element, finds off-by-one errors in seconds. In
the interview write the loop by hand, because the point is the loop, but in practice let the library
grade you.

## The bugs that cost the round

**Mixing the two range conventions.** Writing `hi = len(a) - 1` with `while lo < hi` is the classic:
the last element is never examined, and the bug appears only when the target is the maximum. Writing
`hi = len(a)` with `while lo <= hi` is the other half: `a[mid]` reads past the end. Pick the half-open
form, and when you deliberately use the closed form for a rotated search, change all three lines
together — the initial `hi`, the loop test, and the `hi = mid - 1` update.

**Writing `lo = mid` instead of `lo = mid + 1`.** This is an infinite loop, not a wrong answer. When
`hi - lo == 1` the midpoint equals `lo`, so `lo = mid` changes nothing and the loop spins forever.
The half-open form makes the rule easy to remember: the branch that keeps `mid` is always the `hi`
branch, and the branch that discards it is always the `lo` branch.

**Confusing `<` with `<=` in the boundary test.** `a[mid] < target` gives the first element greater
than or equal to the target; `a[mid] <= target` gives the first element strictly greater. One
character separates lower bound from upper bound. Say which one you want before typing the
comparison, because both compile and both return plausible indices.

**Forgetting the post-check on an exact search.** `lower_bound` returns an insertion point, not a
match. It can equal `len(a)`, and even when it does not, the element there may be a different value.
Both checks are needed, in that order: `lo < len(a)` first, then `a[lo] == target`.

**Getting the low end of an answer range wrong.** Starting a ship-capacity search at 1 rather than
`max(weights)` makes `feasible` false for a stretch of the range that should not be there. In this
particular greedy the answer still comes out right, but the same mistake in a predicate that loops
forever on an impossible capacity will hang, and you will not see it on the sample input.

**Not checking impossibility first.** In the bouquets problem, `m * k > len(bloom_day)` means no
answer exists. The binary search cannot express "impossible", so it returns the top of the range and
looks like an answer. Any answer-space problem that can be infeasible needs that guard above the
loop.

## Done when

- From a blank file, in under three minutes, you can write `lower_bound` and `upper_bound` with
  identical skeletons, and state which single character differs between them.
- You can state the termination argument out loud: `lo <= mid < hi`, so `hi = mid` strictly shrinks
  and `lo = mid + 1` strictly grows, therefore `hi - lo` decreases every iteration.
- Given "find the minimum capacity such that ...", you can name the search range, write `feasible`,
  and say whether the predicate runs false-to-true or true-to-false, before writing the loop.
- You can derive exact search, first occurrence, last occurrence, insert position and count from the
  two bound functions, and explain why the exact search needs a bounds check before its equality
  check.
