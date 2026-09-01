# Sliding window: every variation

A sliding window is a contiguous stretch of an array or string — a subarray or substring — that you
extend on the right and contract on the left, never moving either pointer backwards. Because each
pointer only walks forward once, every element enters the window once and leaves it once, so the whole
scan is $O(n)$ even though an element can be visited twice. The pattern replaces the obvious
$O(n^2)$ double loop whenever the work needed to update the window after a one-element change is
$O(1)$: a running sum, a count array, a distinct counter.

The reason the pattern is hard has nothing to do with pointers. It is that "sliding window" is not one
template but four, and they differ in the two lines that matter most — the condition you shrink on and
the point at which you record the answer. Choosing the wrong one of the four is the failure mode in an
interview, not the coding. So the work is recognition, and recognition comes from the phrasing.

## Recognising it from the phrasing

| The interviewer says | They mean | Shrink condition | Record where |
|---|---|---|---|
| "longest / maximum ... such that" | variable window, template 2 | while the window is **invalid** | after the shrink loop |
| "shortest / minimum ... containing / at least" | variable window, template 3 | while the window is **valid** | inside the shrink loop |
| "every subarray of size k", "all windows of length k" | fixed window, template 1 | while size exceeds `k` | when size equals `k` |
| "maximum / minimum in every window" | monotonic deque, template 4 | pop from the back while dominated | when `right >= k - 1` |
| "exactly k distinct / exactly k odd / sum equals goal" | the at-most-k trick | two runs of template 2 | `atMost(k) - atMost(k-1)` |
| "number of subarrays such that ..." | counting form of template 2 | while invalid | `total += right - left + 1` |
| "at most k replacements / flips / removals" | template 2 with a budget | while cost exceeds `k` | after the shrink loop |
| "remove / replace the shortest piece so that ..." | inverted window | while the **outside** is valid | inside the shrink loop |

Before writing anything, ask yourself one question: **can extending the window on the right ever turn
an invalid window back into a valid one?** For a sum of positive numbers the answer is no — adding an
element only pushes the sum up, so validity is monotone in the window and a sliding window is correct.
For a count of distinct characters the answer is no. For a product of positive numbers the answer is
no. But for a sum that may include negative numbers the answer is yes: a window that overshoots the
target can be rescued by appending a negative number, so shrinking from the left is no longer
justified and the pointer that "never goes back" has to go back. That is the signal to abandon the
window entirely and reach for prefix sums with a hash map. Problem P21 below is exactly that case, and
recognising it is worth as much as recognising the other twenty.

## The four templates

Templates 1, 2 and 3 have deliberately identical skeletons: one `for right`, one add, one `while`
shrink, one record. Only the `while` condition and the position of the record line change. Learn the
skeleton once and the three variants become two decisions rather than three programs.

**Template 1 — fixed size `k`.** Use when the problem names the window length.

```python
def max_sum_of_size_k(nums, k):
    window_sum, best = 0, float("-inf")
    left = 0
    for right in range(len(nums)):
        window_sum += nums[right]                 ## 1. the entering element
        while right - left + 1 > k:               ## 2. shrink while TOO BIG
            window_sum -= nums[left]              ##    undo the leaving element
            left += 1
        if right - left + 1 == k:                 ## 3. record once the size is exact
            best = max(best, window_sum)
    return best

## tests

assert max_sum_of_size_k([2, 1, 5, 1, 3, 2], 3) == 9
assert max_sum_of_size_k([1, 2], 3) == float("-inf")
assert max_sum_of_size_k([-1, -2, -3], 2) == -3
print(max_sum_of_size_k([2, 1, 5, 1, 3, 2], 3))
```

```
9
```

**Template 2 — longest window satisfying a condition.** Shrink **while invalid**, record **after** the
shrink loop, when the window is guaranteed valid again.

```python
def longest_with_at_most_k_distinct(nums, k):
    count, best = {}, 0
    left = 0
    for right in range(len(nums)):
        count[nums[right]] = count.get(nums[right], 0) + 1   ## 1. enter
        while len(count) > k:                                ## 2. shrink while INVALID
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                del count[nums[left]]
            left += 1
        best = max(best, right - left + 1)                   ## 3. record AFTER: window is valid
    return best

## tests

assert longest_with_at_most_k_distinct([1, 2, 1, 3, 4], 2) == 3
assert longest_with_at_most_k_distinct([1, 1, 1, 1], 1) == 4
assert longest_with_at_most_k_distinct([], 2) == 0
print(longest_with_at_most_k_distinct([1, 2, 1, 3, 4], 2))
```

```
3
```

**Template 3 — shortest window satisfying a condition.** Shrink **while valid**, and record **inside**
the shrink loop, before each contraction, because the smallest valid window is the last one seen before
validity breaks.

```python
def shortest_with_sum_at_least(nums, target):
    window_sum, best = 0, float("inf")
    left = 0
    for right in range(len(nums)):
        window_sum += nums[right]                            ## 1. enter
        while window_sum >= target:                          ## 2. shrink while VALID
            best = min(best, right - left + 1)               ## 3. record INSIDE, before shrinking
            window_sum -= nums[left]
            left += 1
    return 0 if best == float("inf") else best

## tests

assert shortest_with_sum_at_least([2, 3, 1, 2, 4, 3], 7) == 2
assert shortest_with_sum_at_least([1, 1, 1, 1], 11) == 0
assert shortest_with_sum_at_least([1, 4, 4], 4) == 1
print(shortest_with_sum_at_least([2, 3, 1, 2, 4, 3], 7))
```

```
2
```

**Template 4 — monotonic deque for an extremum over a fixed window.** Use when you need the max or min
of every window and a plain scan would cost $O(nk)$.

```python
from collections import deque

def max_of_every_window(nums, k):
    dq, out = deque(), []                          ## dq holds INDICES, their values decreasing
    for right in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[right]:
            dq.pop()                               ## a smaller earlier value can never win again
        dq.append(right)
        if dq[0] <= right - k:
            dq.popleft()                           ## the front has slid out of the window
        if right >= k - 1:
            out.append(nums[dq[0]])                ## front is the max of the current window
    return out

## tests

assert max_of_every_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_of_every_window([9, 8, 7], 1) == [9, 8, 7]
assert max_of_every_window([1, 2, 3, 4], 4) == [4]
print(max_of_every_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
```

```
[3, 3, 5, 5, 6, 7]
```

The record point is the whole difference between templates 2 and 3, and it is the line people get
wrong under pressure. In template 2 the shrink loop exists to *restore* validity, so the window is only
meaningful after the loop finishes. In template 3 the shrink loop runs *while* the window is still
valid, so the loop body is the only place a valid window exists to be measured — record after the loop
and you measure a window that has just been broken.

## The trick that unlocks a whole family: exactly K

A two-pointer window cannot count subarrays with **exactly** `k` distinct values directly, and the
reason is worth saying precisely. The counting step in every variable window is `total += right - left
+ 1`, which counts all subarrays ending at `right` whose left endpoint lies in `[left, right]`. That
step only works because the valid left endpoints form a **contiguous suffix** of the prefix: if
`[left, right]` is valid then so is every shorter window ending at `right`. "At most k distinct" has
that property. "Exactly k distinct" does not — for `right` fixed, the left endpoints giving exactly `k`
distinct values form a band in the middle, bounded on both sides, and a single `left` pointer cannot
express a band.

The fix is to write the band as the difference of two suffixes:

$$\text{exactly}(k) = \text{atMost}(k) - \text{atMost}(k-1)$$

Every subarray with at most `k` distinct values has either exactly `k`, or at most `k-1`. Subtracting
removes the second group and leaves the first. Both terms are ordinary template-2 windows, so the whole
thing is two linear scans.

**Worked example.** `nums = [1, 2, 1, 2, 3]`, `k = 2`. Counting by hand, `atMost(2) = 12`: the
subarrays ending at index 0,1,2,3,4 contribute 1, 2, 3, 4, 2 respectively. And `atMost(1) = 5`: no
two adjacent elements are equal, so the only subarrays with a single distinct value are the five
single-element ones, and each index contributes exactly 1. So
$12 - 5 = 7$, and the seven subarrays with exactly two distinct values are `[1,2]`, `[2,1]`, `[1,2]`,
`[2,3]`, `[1,2,1]`, `[2,1,2]`, `[1,2,1,2]`.

```python
def subarrays_with_k_distinct(nums, k):
    def at_most(m):
        if m < 0:
            return 0
        count, left, total = {}, 0, 0
        for right in range(len(nums)):
            count[nums[right]] = count.get(nums[right], 0) + 1
            while len(count) > m:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1
            total += right - left + 1            ## every left in [left, right] is valid
        return total
    return at_most(k) - at_most(k - 1)

## tests

assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
assert subarrays_with_k_distinct([1, 2, 1, 3, 4], 3) == 3
assert subarrays_with_k_distinct([1, 1, 1], 1) == 6
print(subarrays_with_k_distinct([1, 2, 1, 2, 3], 2))
```

```
7
```

The same subtraction works for any monotone predicate: exactly `k` odd numbers, sum exactly equal to a
goal on a binary array, exactly `k` characters of some class. Whenever you see the word "exactly" next
to "number of subarrays", write `atMost(k) - atMost(k-1)` before you write anything else.

## The problems

### P1. Maximum sum subarray of size k — the largest sum over all contiguous windows of length `k`

**Which template.** Template 1, fixed size, with a running sum.
**The trick.** Build the first window with a `sum()` and then never recompute: each step adds the
entering element and subtracts the leaving one, which is the whole reason the scan is $O(n)$ rather
than $O(nk)$.

```python
def max_sum_subarray_k(nums, k):
    if k > len(nums) or k <= 0:
        return None
    window_sum = sum(nums[:k])                       ## build the first window explicitly
    best = window_sum
    for right in range(k, len(nums)):
        window_sum += nums[right] - nums[right - k]  ## one in, one out: O(1) per step
        best = max(best, window_sum)
    return best

## tests

assert max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3) == 9
assert max_sum_subarray_k([-3, -1, -4, -2], 2) == -4
assert max_sum_subarray_k([5], 2) is None
print(max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3))
```

```
9
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P2. Find all anagrams in a string — every start index where `s` contains a permutation of `p`

**Which template.** Template 1, fixed size `len(p)`, with a 26-slot count array.
**The trick.** An anagram is exactly a window whose letter counts equal the pattern's letter counts, so
the whole problem is "slide a window of size `len(p)` and compare two count arrays". Use a list of 26,
not a dict: `have == need` on two lists is a single cheap comparison and there is no key-deletion
bookkeeping to get wrong.

```python
def find_anagrams(s, p):
    k = len(p)
    if k > len(s):
        return []
    need, have = [0] * 26, [0] * 26
    for ch in p:
        need[ord(ch) - 97] += 1
    out = []
    for right in range(len(s)):
        have[ord(s[right]) - 97] += 1
        if right >= k:                               ## evict the element leaving on the left
            have[ord(s[right - k]) - 97] -= 1
        if right >= k - 1 and have == need:          ## comparing two 26-slot lists is O(26)
            out.append(right - k + 1)
    return out

## tests

assert find_anagrams("cbaebabacd", "abc") == [0, 6]
assert find_anagrams("abab", "ab") == [0, 1, 2]
assert find_anagrams("a", "aa") == []
print(find_anagrams("cbaebabacd", "abc"))
```

```
[0, 6]
```

**Complexity.** $O(26n)$ time, which is $O(n)$, and $O(26)$ space.

### P3. Permutation in string — does `s` contain any permutation of `pattern` as a substring

**Which template.** Template 1 again, identical window, but returning a boolean at the first hit.
**The trick.** Same as P2, except this version replaces the $O(26)$ array comparison with an $O(1)$
one by maintaining a `matches` counter: how many of the 26 letters currently have `have[i] == need[i]`.
A letter's contribution flips only when its count crosses `need[i]` from either side, so each update
touches `matches` at most once. When `matches == 26` the window is a permutation.

```python
def check_inclusion(pattern, s):
    k = len(pattern)
    if k > len(s):
        return False
    need, have = [0] * 26, [0] * 26
    for ch in pattern:
        need[ord(ch) - 97] += 1
    matches = sum(1 for i in range(26) if need[i] == 0)   ## letters already balanced
    for right in range(len(s)):
        i = ord(s[right]) - 97
        have[i] += 1
        if have[i] == need[i]:       matches += 1
        elif have[i] == need[i] + 1: matches -= 1
        if right >= k:
            j = ord(s[right - k]) - 97
            have[j] -= 1
            if have[j] == need[j]:       matches += 1
            elif have[j] == need[j] - 1: matches -= 1
        if matches == 26:
            return True
    return False

## tests

assert check_inclusion("ab", "eidbaooo") is True
assert check_inclusion("ab", "eidboaoo") is False
assert check_inclusion("adc", "dcda") is True
print(check_inclusion("ab", "eidbaooo"), check_inclusion("ab", "eidboaoo"))
```

```
True False
```

**Complexity.** $O(n + 26)$ time, $O(26)$ space.

### P4. Grumpy bookstore owner — one secret technique keeps the owner calm for `minutes` consecutive minutes; maximise satisfied customers

**Which template.** Template 1 on the *gain*, not on the raw totals.
**The trick.** Split the answer into a fixed part and a windowed part. Customers served in non-grumpy
minutes are satisfied no matter what, so they are a constant `base`. The technique can only ever help
during grumpy minutes, so the window should sum `customers[i]` where `grumpy[i] == 1` and nothing else.
Inverting the objective from "total satisfied in the window" to "extra satisfied because of the window"
is what turns a confusing statement into template 1.

```python
def max_satisfied(customers, grumpy, minutes):
    base = sum(c for c, g in zip(customers, grumpy) if g == 0)   ## always-happy customers
    gained, best = 0, 0
    left = 0
    for right in range(len(customers)):
        if grumpy[right] == 1:
            gained += customers[right]                           ## only grumpy minutes are gains
        while right - left + 1 > minutes:
            if grumpy[left] == 1:
                gained -= customers[left]
            left += 1
        best = max(best, gained)
    return base + best

## tests

assert max_satisfied([1, 0, 1, 2, 1, 1, 7, 5], [0, 1, 0, 1, 0, 1, 0, 1], 3) == 16
assert max_satisfied([1], [0], 1) == 1
assert max_satisfied([4, 10, 10], [1, 1, 0], 2) == 24
print(max_satisfied([1, 0, 1, 2, 1, 1, 7, 5], [0, 1, 0, 1, 0, 1, 0, 1], 3))
```

```
16
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P5. Maximum points from cards — take exactly `k` cards, each from either end of the row, maximising the total

**Which template.** Template 1, but the window is the part you do **not** take.
**The trick.** Taking `k` cards from the two ends means leaving behind a single contiguous block of
`n - k` cards in the middle. Maximising what you take is therefore minimising what you leave, and what
you leave is an ordinary fixed window of size `n - k`. This is the cleanest example on the page of
inverting a problem to expose the window: the statement mentions two ends and no window at all.

```python
def max_score(card_points, k):
    n = len(card_points)
    total = sum(card_points)
    if k >= n:
        return total
    window = n - k                                   ## the cards you LEAVE BEHIND
    current = sum(card_points[:window])
    smallest = current
    for right in range(window, n):
        current += card_points[right] - card_points[right - window]
        smallest = min(smallest, current)
    return total - smallest

## tests

assert max_score([1, 2, 3, 4, 5, 6, 1], 3) == 12
assert max_score([2, 2, 2], 2) == 4
assert max_score([9, 7, 7, 9, 7, 7, 9], 7) == 55
print(max_score([1, 2, 3, 4, 5, 6, 1], 3))
```

```
12
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P6. Longest substring without repeating characters — the longest window with no duplicate character

**Which template.** Template 2, but with a jump instead of a shrink loop.
**The trick.** Store each character's most recent index. When the entering character is already inside
the window, `left` can jump straight past its previous position rather than stepping one at a time —
the guard `last_index[c] >= left` is what stops a stale index outside the window from dragging `left`
backwards, and forgetting it is the classic bug here.

```python
def length_of_longest_unique(s):
    last_index = {}                                  ## char -> most recent index
    left, best = 0, 0
    for right in range(len(s)):
        c = s[right]
        if c in last_index and last_index[c] >= left:
            left = last_index[c] + 1                 ## jump left past the old copy
        last_index[c] = right
        best = max(best, right - left + 1)
    return best

## tests

assert length_of_longest_unique("abcabcbb") == 3
assert length_of_longest_unique("bbbbb") == 1
assert length_of_longest_unique("pwwkew") == 3
assert length_of_longest_unique("") == 0
print(length_of_longest_unique("abcabcbb"), length_of_longest_unique("pwwkew"))
```

```
3 3
```

**Complexity.** $O(n)$ time, $O(\min(n, \Sigma))$ space for the alphabet.

### P7. Longest substring with at most k distinct characters — the longest window using no more than `k` different characters

**Which template.** Template 2, the canonical instance of it.
**The trick.** Maintain a `distinct` counter that goes up when a count rises from 0 to 1 and down when
it falls back to 0. That keeps validity an $O(1)$ integer test rather than a `len(count)` call each
step, and it is the habit that generalises to every other problem in this family.

```python
def longest_k_distinct(s, k):
    count, distinct = {}, 0
    left, best = 0, 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        if count[s[right]] == 1:
            distinct += 1                            ## keep a counter, do not call len()
        while distinct > k:                          ## shrink while INVALID
            count[s[left]] -= 1
            if count[s[left]] == 0:
                distinct -= 1
            left += 1
        best = max(best, right - left + 1)
    return best

## tests

assert longest_k_distinct("eceba", 2) == 3
assert longest_k_distinct("aa", 1) == 2
assert longest_k_distinct("abc", 0) == 0
assert longest_k_distinct("", 3) == 0
print(longest_k_distinct("eceba", 2), longest_k_distinct("aa", 1))
```

```
3 2
```

**Complexity.** $O(n)$ time, $O(k)$ space.

### P8. Fruit into baskets — two baskets, each holding one fruit type; pick the longest run you can collect

**Which template.** Template 2. This is P7 with `k = 2`, and you should say that out loud.
**The trick.** There is no trick beyond recognition. "Two baskets, each holding a single type" is a
costume for "at most two distinct values in the window". Interviewers use this problem specifically to
see whether you translate the story into the constraint, so name the reduction in one sentence before
you write anything.

```python
def total_fruit(fruits):
    count, distinct = {}, 0
    left, best = 0, 0
    for right in range(len(fruits)):
        count[fruits[right]] = count.get(fruits[right], 0) + 1
        if count[fruits[right]] == 1:
            distinct += 1
        while distinct > 2:                          ## two baskets == at most two distinct
            count[fruits[left]] -= 1
            if count[fruits[left]] == 0:
                distinct -= 1
            left += 1
        best = max(best, right - left + 1)
    return best

## tests

assert total_fruit([1, 2, 1]) == 3
assert total_fruit([0, 1, 2, 2]) == 3
assert total_fruit([1, 2, 3, 2, 2]) == 4
assert total_fruit([3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4]) == 5
print(total_fruit([3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4]))
```

```
5
```

**Complexity.** $O(n)$ time, $O(1)$ space — the map never holds more than three keys.

### P9. Max consecutive ones III — longest run of 1s if you may flip at most `k` zeros

**Which template.** Template 2 with a budget counter.
**The trick.** Do not think about which zeros to flip. The window is valid whenever it contains at most
`k` zeros, because you would simply flip all of them, so the only state you need is `zeros`. Every
"you may change at most k things" problem collapses to counting how much of the budget the current
window spends.

```python
def longest_ones(nums, k):
    zeros = 0
    left, best = 0, 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1                               ## the budget spent by this window
        while zeros > k:                             ## shrink while INVALID
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best

## tests

assert longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
assert longest_ones([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3) == 10
assert longest_ones([0, 0, 0], 0) == 0
print(longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2))
```

```
6
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P10. Longest repeating character replacement — longest window you can make uniform with at most `k` character replacements

**Which template.** Template 2, in its window-never-shrinks form.
**The trick.** A window is valid when `size - max_count <= k`, where `max_count` is the frequency of
its most common letter. The trick is that `max_count` is allowed to be stale — it is the best count
ever seen, not the best count in the current window. A stale, too-large `max_count` can only make an
invalid window look valid, and that can only produce a window at least as long as the true best, which
was already recorded when it was genuinely valid. So an `if` suffices in place of the `while`: the
window slides forward without ever shrinking, and its final size is the answer.

```python
def character_replacement(s, k):
    count = [0] * 26
    max_count = 0                                    ## best single-letter count ever seen
    left, best = 0, 0
    for right in range(len(s)):
        i = ord(s[right]) - 65
        count[i] += 1
        max_count = max(max_count, count[i])
        if (right - left + 1) - max_count > k:       ## an IF, not a while: the window only slides
            count[ord(s[left]) - 65] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best

## tests

assert character_replacement("ABAB", 2) == 4
assert character_replacement("AABABBA", 1) == 4
assert character_replacement("AAAA", 0) == 4
assert character_replacement("", 2) == 0
print(character_replacement("AABABBA", 1), character_replacement("ABAB", 2))
```

```
4 4
```

**Complexity.** $O(n)$ time, $O(26)$ space.

### P11. Longest subarray with absolute difference at most a limit — longest window where `max - min <= limit`

**Which template.** Template 2, with two monotonic deques supplying the window max and min.
**The trick.** Validity depends on two extrema at once, so you need two deques: one decreasing for the
max and one increasing for the min. Both hold indices, and when `left` advances you pop a deque's front
only if that front *is* `left` — the deque may legitimately have already discarded it. Without the
deques, recomputing `max - min` each step would cost $O(nk)$.

```python
from collections import deque

def longest_subarray_within_limit(nums, limit):
    max_dq, min_dq = deque(), deque()                ## decreasing / increasing indices
    left, best = 0, 0
    for right in range(len(nums)):
        while max_dq and nums[max_dq[-1]] <= nums[right]:
            max_dq.pop()
        max_dq.append(right)
        while min_dq and nums[min_dq[-1]] >= nums[right]:
            min_dq.pop()
        min_dq.append(right)
        while nums[max_dq[0]] - nums[min_dq[0]] > limit:   ## shrink while INVALID
            if max_dq[0] == left:
                max_dq.popleft()
            if min_dq[0] == left:
                min_dq.popleft()
            left += 1
        best = max(best, right - left + 1)
    return best

## tests

assert longest_subarray_within_limit([8, 2, 4, 7], 4) == 2
assert longest_subarray_within_limit([10, 1, 2, 4, 7, 2], 5) == 4
assert longest_subarray_within_limit([4, 2, 2, 2, 4, 4, 2, 2], 0) == 3
print(longest_subarray_within_limit([10, 1, 2, 4, 7, 2], 5))
```

```
4
```

**Complexity.** $O(n)$ time — each index is pushed and popped at most once per deque — and $O(n)$ space.

### P12. Minimum size subarray sum — shortest contiguous subarray whose sum is at least `target`

**Which template.** Template 3, the canonical instance of it.
**The trick.** All values are positive, so the sum is monotone in the window: once the window is valid,
shrinking is the only way to find something smaller, and it can only end validity. Record the size
inside the `while`, before subtracting — after the loop the window has already been broken.

```python
def min_subarray_len(target, nums):
    window_sum, best = 0, float("inf")
    left = 0
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum >= target:                  ## shrink while VALID
            best = min(best, right - left + 1)       ## record BEFORE breaking validity
            window_sum -= nums[left]
            left += 1
    return 0 if best == float("inf") else best

## tests

assert min_subarray_len(7, [2, 3, 1, 2, 4, 3]) == 2
assert min_subarray_len(4, [1, 4, 4]) == 1
assert min_subarray_len(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0
assert min_subarray_len(5, []) == 0
print(min_subarray_len(7, [2, 3, 1, 2, 4, 3]))
```

```
2
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P13. Minimum window substring — shortest window of `s` containing every character of `t`, with multiplicity

**Which template.** Template 3, and it is the hardest classic in the set. Work it slowly.
**The trick.** The whole difficulty is checking validity cheaply. Comparing two maps each step is
$O(\Sigma)$ and easy to get wrong; instead keep `required = len(need)` and `have` = how many *distinct*
characters currently meet their full quota. `have` rises only when a count crosses its quota upward
(`window[c] == need[c]`, an equality, never `>=`) and falls only when it crosses back down
(`window[d] < need[d]`, strict). The window is valid exactly when `have == required`, an $O(1)$ test.
Record the *indices* of the best window, not a slice, so you copy the string once at the end.

```python
def min_window(s, t):
    if not s or not t:
        return ""
    need = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    required, have = len(need), 0        ## have = characters whose quota is currently met
    window = {}
    left, best_len, best_at = 0, float("inf"), 0
    for right in range(len(s)):
        c = s[right]
        window[c] = window.get(c, 0) + 1
        if c in need and window[c] == need[c]:
            have += 1                    ## crossed the quota exactly once
        while have == required:          ## shrink while VALID
            if right - left + 1 < best_len:
                best_len, best_at = right - left + 1, left
            d = s[left]
            window[d] -= 1
            if d in need and window[d] < need[d]:
                have -= 1
            left += 1
    return "" if best_len == float("inf") else s[best_at:best_at + best_len]

## tests

assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
assert min_window("", "a") == ""
print(min_window("ADOBECODEBANC", "ABC"))
```

```
BANC
```

**Complexity.** $O(|s| + |t|)$ time, $O(|s| + |t|)$ space. Note that `t` may contain duplicates, which
is why `need` counts rather than a set — a set answer passes the sample and fails the hidden tests.

### P14. Smallest subarray covering all distinct elements — shortest window containing every distinct value of the array

**Which template.** Template 3, with `required = len(set(nums))`.
**The trick.** It is minimum window substring with the pattern computed from the input rather than
given. `have` counts distinct values present at least once, so it rises when a count reaches 1 and
falls when it returns to 0 — a simpler bookkeeping than P13 because every quota is exactly one.

```python
def smallest_covering_subarray(nums):
    required = len(set(nums))
    count, have = {}, 0
    left, best_len, best_at = 0, float("inf"), 0
    for right in range(len(nums)):
        count[nums[right]] = count.get(nums[right], 0) + 1
        if count[nums[right]] == 1:
            have += 1
        while have == required:                      ## shrink while VALID
            if right - left + 1 < best_len:
                best_len, best_at = right - left + 1, left
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                have -= 1
            left += 1
    return [] if best_len == float("inf") else nums[best_at:best_at + best_len]

## tests

assert smallest_covering_subarray([1, 2, 2, 3, 1, 3, 2]) == [2, 3, 1]
assert smallest_covering_subarray([1, 1, 1]) == [1]
assert smallest_covering_subarray([]) == []
print(smallest_covering_subarray([1, 2, 2, 3, 1, 3, 2]))
```

```
[2, 3, 1]
```

**Complexity.** $O(n)$ time, $O(n)$ space. Ties are broken by earliest start, which is worth confirming
with the interviewer before you code.

### P15. Subarrays with exactly k distinct integers — count subarrays containing exactly `k` different values

**Which template.** Two runs of template 2, subtracted.
**The trick.** `atMost(k) - atMost(k-1)`, for the reason given above: the left endpoints producing
*exactly* `k` distinct values form a band, not a suffix, so `total += right - left + 1` cannot count
them directly. This version carries an explicit `distinct` counter instead of calling `len(count)`,
which is the form to write from memory.

```python
def subarrays_with_k_distinct(nums, k):
    def at_most(m):
        if m < 0:
            return 0
        count, distinct, left, total = {}, 0, 0, 0
        for right in range(len(nums)):
            count[nums[right]] = count.get(nums[right], 0) + 1
            if count[nums[right]] == 1:
                distinct += 1
            while distinct > m:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    distinct -= 1
                left += 1
            total += right - left + 1
        return total
    return at_most(k) - at_most(k - 1)

## tests

assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
assert subarrays_with_k_distinct([1, 2, 1, 3, 4], 3) == 3
assert subarrays_with_k_distinct([1, 1, 1], 1) == 6
assert subarrays_with_k_distinct([1, 2, 3], 4) == 0
print(subarrays_with_k_distinct([1, 2, 1, 2, 3], 2))
```

```
7
```

**Complexity.** $O(n)$ time — two linear passes — and $O(k)$ space.

### P16. Count number of nice subarrays — count subarrays containing exactly `k` odd numbers

**Which template.** The same two runs of template 2, on a different predicate.
**The trick.** Replace "distinct values" with "count of odd numbers" and nothing else changes. Say that
in the interview: it is the identical `atMost(k) - atMost(k-1)` skeleton with `odd += nums[right] % 2`
in place of the distinct-counter update. Recognising the reuse is the whole answer.

```python
def number_of_nice_subarrays(nums, k):
    def at_most(m):
        if m < 0:
            return 0
        odd, left, total = 0, 0, 0
        for right in range(len(nums)):
            odd += nums[right] % 2                   ## the predicate: "is odd"
            while odd > m:
                odd -= nums[left] % 2
                left += 1
            total += right - left + 1
        return total
    return at_most(k) - at_most(k - 1)

## tests

assert number_of_nice_subarrays([1, 1, 2, 1, 1], 3) == 2
assert number_of_nice_subarrays([2, 4, 6], 1) == 0
assert number_of_nice_subarrays([2, 2, 2, 1, 2, 2, 1, 2, 2, 2], 2) == 16
print(number_of_nice_subarrays([2, 2, 2, 1, 2, 2, 1, 2, 2, 2], 2))
```

```
16
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P17. Binary subarrays with sum equal to a goal — count subarrays of a 0/1 array summing to `goal`

**Which template.** The at-most-k trick again, now on the sum.
**The trick.** The array is binary, so all values are non-negative and the window sum is monotone — that
is what makes `atMost` a legal sliding window at all. Note the shrink test is `while window_sum > g`,
strictly greater, because a window summing to exactly `g` is still "at most `g`" and must be kept.
Using `>=` here is the single most common bug in this problem.

```python
def num_subarrays_with_sum(nums, goal):
    def at_most(g):
        if g < 0:
            return 0
        window_sum, left, total = 0, 0, 0
        for right in range(len(nums)):
            window_sum += nums[right]
            while window_sum > g:                    ## strict >, because g itself is allowed
                window_sum -= nums[left]
                left += 1
            total += right - left + 1
        return total
    return at_most(goal) - at_most(goal - 1)

## tests

assert num_subarrays_with_sum([1, 0, 1, 0, 1], 2) == 4
assert num_subarrays_with_sum([0, 0, 0, 0, 0], 0) == 15
assert num_subarrays_with_sum([1, 1, 1], 1) == 3
print(num_subarrays_with_sum([1, 0, 1, 0, 1], 2), num_subarrays_with_sum([0, 0, 0, 0, 0], 0))
```

```
4 15
```

**Complexity.** $O(n)$ time, $O(1)$ space. The zeros case matters: `[0,0,0,0,0]` with goal 0 has all
15 subarrays valid, and a solution that forgets the empty-sum windows gets it wrong.

### P18. Subarray product less than k — count subarrays whose product is strictly less than `k`

**Which template.** Template 2, counting form, no subtraction needed.
**The trick.** All values are positive integers, so the product is **monotone** in the window:
extending multiplies by something at least 1, which can only push the product up, and contracting can
only bring it down. That monotonicity is exactly the property the pattern requires, and it is why this
problem is a genuine sliding window while the sum-with-negatives problem below is not. Say the word
"positive" out loud — it is the licence to use the window.

```python
def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0                                     ## no positive product is below 1
    product, left, total = 1, 0, 0
    for right in range(len(nums)):
        product *= nums[right]
        while product >= k:                          ## shrink while INVALID
            product //= nums[left]                   ## exact: nums[left] divides the product
            left += 1
        total += right - left + 1                    ## all windows ending at right are valid
    return total

## tests

assert num_subarray_product_less_than_k([10, 5, 2, 6], 100) == 8
assert num_subarray_product_less_than_k([1, 2, 3], 0) == 0
assert num_subarray_product_less_than_k([1, 1, 1], 2) == 6
print(num_subarray_product_less_than_k([10, 5, 2, 6], 100))
```

```
8
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P19. Sliding window maximum — the maximum of every window of size `k`, as a list

**Which template.** Template 4, the monotonic deque.
**The trick.** The deque holds indices whose values are decreasing. Two eviction rules and they are
different: pop from the **back** while the incoming value dominates (a smaller, older element can never
be the maximum of any future window), and pop from the **front** when its index has expired out of the
window. Storing indices rather than values is what makes the expiry check possible at all.

```python
from collections import deque

def max_sliding_window(nums, k):
    dq, out = deque(), []                            ## indices, values strictly decreasing
    for right in range(len(nums)):
        while dq and nums[dq[-1]] <= nums[right]:
            dq.pop()                                 ## dominated: smaller AND older
        dq.append(right)
        if dq[0] <= right - k:
            dq.popleft()                             ## expired: index left the window
        if right >= k - 1:
            out.append(nums[dq[0]])
    return out

## tests

assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([7, 2, 4], 2) == [7, 4]
assert max_sliding_window([], 3) == []
print(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
```

```
[3, 3, 5, 5, 6, 7]
```

**Complexity.** $O(n)$ time — each index is pushed once and popped at most once, so the inner `while`
is amortised $O(1)$ — and $O(k)$ space. A heap solution is $O(n \log k)$ and needs lazy deletion; say
that you considered it and chose the deque.

### P20. Replace the substring for a balanced string — shortest substring you can rewrite so that Q, W, E and R each occur `n/4` times

**Which template.** Template 3, but the condition lives **outside** the window. This one confuses
almost everyone, so slow down.
**The trick.** You are allowed to rewrite the window into anything you like, so the window's own
contents are irrelevant. What matters is the counts of the characters *not* covered by the window: if
every character occurs at most `n/4` times outside, then the window has exactly enough room to top each
one up to its quota, and the rewrite succeeds. So maintain `outside`, initialised to the full string's
counts, decrement when an index enters the window and increment when it leaves. Then it is an ordinary
template 3: shrink while the outside is still fixable, recording the size before each contraction.
The mental move is to stop asking "is the window valid" and start asking "is the complement valid".

```python
def balanced_string(s):
    n, quota = len(s), len(s) // 4
    outside = {c: 0 for c in "QWER"}
    for ch in s:
        outside[ch] += 1                             ## start: everything is outside the window
    if all(outside[c] == quota for c in "QWER"):
        return 0
    left, best = 0, n
    for right in range(n):
        outside[s[right]] -= 1                       ## s[right] joins the replaceable window
        while left <= right and all(outside[c] <= quota for c in "QWER"):
            best = min(best, right - left + 1)       ## record while the OUTSIDE is fixable
            outside[s[left]] += 1                    ## s[left] returns to the outside
            left += 1
    return best

## tests

assert balanced_string("QWER") == 0
assert balanced_string("QQWE") == 1
assert balanced_string("QQQW") == 2
assert balanced_string("QQQQ") == 3
print(balanced_string("QQWE"), balanced_string("QQQW"), balanced_string("QQQQ"))
```

```
1 2 3
```

**Complexity.** $O(4n)$ time, which is $O(n)$, and $O(1)$ space.

### P21. Subarray sum equals k, negatives allowed — count subarrays summing to exactly `k` when values may be negative

**Which template.** None. This is the problem where the window is the wrong answer.
**The trick.** The failed reasoning is worth stating precisely: a sliding window shrinks from the left
the moment the sum overshoots `k`, which assumes an overshoot can never be repaired by extending
further right. With negative numbers it can — in `[3, 4, 7, 2, -3, 1, 4, 2]` the window `[7, 2]`
already exceeds 7, yet `[7, 2, -3, 1]` sums to exactly 7 — so `left` would have to move backwards and
the linear-time guarantee is gone. The correct pattern is prefix sums in a hash map: a subarray
`(i, j]` sums to `k` exactly when `prefix[j] - prefix[i] == k`, so as you scan you look up how many
earlier prefixes equal `running - k`. Seed the map with `{0: 1}` for the empty prefix, or every
subarray starting at index 0 is missed.

```python
def subarray_sum_equals_k(nums, k):
    prefix_counts = {0: 1}                           ## the empty prefix has sum 0
    running, total = 0, 0
    for x in nums:
        running += x
        total += prefix_counts.get(running - k, 0)   ## every earlier prefix that closes a match
        prefix_counts[running] = prefix_counts.get(running, 0) + 1
    return total

def sliding_window_attempt(nums, k):
    window_sum, left, total = 0, 0, 0
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum > k and left <= right:      ## assumes overshoot is permanent -- it is not
            window_sum -= nums[left]
            left += 1
        if window_sum == k:
            total += 1
    return total

## tests

assert subarray_sum_equals_k([1, 1, 1], 2) == 2
assert subarray_sum_equals_k([1, 2, 3], 3) == 2
assert subarray_sum_equals_k([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
assert sliding_window_attempt([3, 4, 7, 2, -3, 1, 4, 2], 7) != 4
print(subarray_sum_equals_k([3, 4, 7, 2, -3, 1, 4, 2], 7),
      sliding_window_attempt([3, 4, 7, 2, -3, 1, 4, 2], 7))
```

```
4 2
```

**Complexity.** $O(n)$ time, $O(n)$ space — you trade the window's constant space for a map, and that
trade is the price of allowing negative numbers.

## Tricks and tips

**Use a fixed count array, not a hash map, when the alphabet is small.** For lowercase letters use
`[0] * 26` and index with `ord(ch) - 97`; for arbitrary ASCII use `[0] * 128`. It is faster, it never
raises a `KeyError`, and — the real win — two arrays can be compared with a single `==`, which is what
makes the anagram problem a four-line window. A dict forces you to delete keys when a count hits zero,
because `{'a': 0}` and `{}` compare unequal, and forgetting that deletion is a bug you will not see on
the sample input.

**Keep a running `distinct` counter instead of calling `len()`.** Increment it when a count rises from
0 to 1, decrement it when it returns to 0. `len(count)` is $O(1)$ in CPython so this is not really a
speed matter; it is that the counter forces you to write the two crossing conditions explicitly, and
those two lines are where the bugs live. The same habit scales up to the `have`/`need` pair in minimum
window substring, where the check genuinely would be $O(\Sigma)$ otherwise.

**For "longest" problems the window never has to shrink more than once per step, so an `if` works.**
Take Longest Repeating Character Replacement. The answer is the largest window with
`size - max_count <= k`. When a new character arrives the window grows by one, so at most one
contraction is ever needed to restore a size you have already achieved — you are not looking for the
largest valid window at each step, only for the largest valid window overall. Replace the `while` with
an `if` and the window's size becomes non-decreasing: it grows when it can and slides when it cannot.
Be honest about what this costs. The window at the end is not guaranteed to be valid; it sits at the
best-ever size. That is fine when the question asks only for the size, and wrong the moment it asks
you to return the substring itself, in which case go back to the `while` and track indices.

**A sum with negative numbers is not monotone, so the window breaks.** Extending can lower the sum,
which means overshooting the target is not permanent and `left` would have to move backwards. Prefix
sums with a hash map is the correct pattern there, and the same applies to products that may include
zero or a negative factor.

**When the question is about the part outside the window, invert it.** "Remove the shortest subarray so
the rest is sorted", "replace the shortest substring so the string is balanced", "take k cards from the
ends" — all three become "keep the longest, or find the shortest, window such that the complement
satisfies the property". Maintain the counts of the complement rather than of the window and the
templates apply unchanged.

**Two conditions can need two deques.** Anything of the form `max - min <= limit` needs a decreasing
deque and an increasing one, advanced together and popped from the front only when their front index
equals `left`.

**Write the shrink condition first.** Before the loop, before the counters, write the one line that
says when the window is unacceptable. That line defines the problem, and once it is on the page the
choice between template 2 and template 3 is just whether you shrink on its truth or its negation.

## The bugs that cost the round

**Recording the answer at the wrong point.** This is the single most common failure and it differs
between the two variable templates. In the "longest" template the shrink loop restores validity, so the
answer is recorded *after* the loop; record inside it and you measure invalid windows. In the
"shortest" template the shrink loop runs while the window is still valid, so the answer is recorded
*inside*, before each contraction; record after the loop and you measure a window that has just been
broken. If you take one thing from this page, take the pairing of shrink condition and record point.

**Shrinking with `if` when the problem needs `while`.** The `if` shortcut is legitimate only for
size-reporting "longest" problems, as explained above. Anywhere you must return the actual window, or
where several elements can leave at once, an `if` leaves the window invalid and silently returns a
wrong answer on the second test case.

**Forgetting to undo the state when `left` moves.** Every `left += 1` must be preceded by the exact
inverse of what the `right` step did: decrement the count, subtract from the sum, divide out of the
product, decrement `distinct` if the count reached zero. Write the two as a pair, immediately, before
writing anything else.

**Off-by-one in the window size.** It is `right - left + 1`, always, with both ends inclusive. A window
of size `k` starts at `right - k + 1`, and the first complete window ends at `right == k - 1`.

**`>=` where `>` is needed on the shrink condition.** In "at most `g`" counting, a window summing to
exactly `g` is valid, so the shrink test is `while window_sum > g`. Using `>=` throws away the very
windows you are counting.

**Empty input.** `n == 0`, `k == 0`, `k > n`, and a pattern longer than the string. Each of the four
should be one line at the top of the function, and each has appeared in the tests above.

## Done when

- Given a problem statement you have not seen, you can name which of the four templates it needs, and
  say the shrink condition and the record point, in under 30 seconds and before writing any code.
- You can write templates 1 through 3 from a blank file with identical skeletons, and explain why the
  record line moves between them.
- You can derive `atMost(k) - atMost(k-1)` out loud, including why `total += right - left + 1` counts
  a suffix of left endpoints and why "exactly k" is a band that a single `left` pointer cannot express.
- You can look at a problem involving a sum and say immediately whether negatives are possible, and
  switch to prefix sums with a hash map when they are.
