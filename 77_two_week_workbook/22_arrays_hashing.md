# Arrays and hash tables: every variation

A hash table answers "have I seen this value" in $O(1)$ instead of $O(n)$, so it turns a scan inside a
loop into a lookup inside a loop. That single substitution collapses the obvious $O(n^2)$ double loop
to $O(n)$, and the price is $O(n)$ memory. Almost every array problem that looks quadratic is really a
question about membership, about pairing, or about counting, and all three are lookups.

The reason the pattern is hard has nothing to do with the hash table. A dictionary is three lines and
nobody fails on the syntax. The hard part is deciding **what to key on**, and that decision is
different in every problem. Two Sum keys on the complement `target - x`, not on the value. Group
Anagrams keys on a canonical form of the word, so `"eat"` and `"tea"` collapse to the same key.
Subarray Sum Equals K keys on a running prefix sum, not on any element. Longest Consecutive Sequence
keys on nothing at all beyond set membership, and the whole solution is one `in` test in the right
place. The interview tests whether you can invent the key. So the work is not learning the data
structure; it is learning to look at a question and name the one value that makes the answer a single
lookup.

## Recognising it from the phrasing

| The interviewer says | Key on | Structure | Record where |
|---|---|---|---|
| "a pair that sums to a target" | the **complement** `target - x` | dict value → index | on the lookup, before inserting `x` |
| "have I seen this", "any duplicate" | the value itself | a `set` | on the lookup, before adding |
| "group these together" | a **canonical form**: sorted letters, or a 26-count tuple | dict key → list | after the whole pass |
| "count of subarrays summing to k" | the **prefix sum** | dict sum → how many times | inside the loop, on each lookup |
| "top k frequent / most common" | the value, counted | dict, then buckets or a heap | after counting, walking buckets down |
| "the missing number", "the duplicate number, in place" | the **index**: value `v` belongs at slot `v-1` | the array itself | in a second pass over the slots |
| "product of all elements except self" | nothing — prefix and suffix products | two passes, one output array | as you build the second pass |
| "does the order of the answer matter" | insertion order of a dict | dict, Python 3.7+ | never depend on it for correctness |

Before writing any code, ask one question: **what single value, computed from the current element or
from the prefix ending at it, would answer the question in one lookup?** If you can name that value,
the problem is a hash-map problem and the rest writes itself — you build the map of that value as you
scan, and at each element you look up the one thing that would complete an answer. If you cannot name
it, the problem is probably not a hash-map problem. Two Sum on a sorted array has no useful key,
because sorting already gives you order and two pointers do it in $O(1)$ space. Range queries have no
useful key, because a hash table cannot answer "how many values lie between 10 and 20" at all. Those
are the cases where sorting, binary search or two pointers wins, and reaching for a dict there costs
you memory for nothing.

## The templates

Four skeletons cover the whole family. The first two are the same loop with a different lookup, and
the third is the same loop again with the key changed from the element to the running prefix. Learn
the shape once: **scan, look up what would complete the answer, then insert the current thing.** The
order of those last two steps is the bug that appears most often, so it is marked in every template.

**Template 1 — the seen-set.** Use when the question is existence, membership or duplication.

```python
def has_duplicate(nums):
    seen = set()                                     ## every value met so far
    for x in nums:
        if x in seen:                                ## O(1) lookup replaces an O(n) scan
            return True
        seen.add(x)                                  ## add AFTER the test, never before
    return False

## tests

assert has_duplicate([1, 2, 3, 1]) is True
assert has_duplicate([1, 2, 3]) is False
assert has_duplicate([]) is False
print(has_duplicate([1, 2, 3, 1]), has_duplicate([1, 2, 3]))
```

```
True False
```

**Template 2 — the complement map.** Use when the question asks for a pair with a fixed relation.

```python
def two_sum(nums, target):
    index_of = {}                                    ## value -> the index where it was seen
    for i, x in enumerate(nums):
        complement = target - x                      ## the ONE value that completes the pair
        if complement in index_of:
            return [index_of[complement], i]         ## record on the LOOKUP, not on the insert
        index_of[x] = i                              ## insert after, so x cannot pair with itself
    return []

## tests

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 3], 6) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
assert two_sum([1, 2], 100) == []
print(two_sum([2, 7, 11, 15], 9))
```

```
[0, 1]
```

**Template 3 — the prefix-sum counter map.** Use when the question counts subarrays by their sum. It is
template 2 with the key changed from the element to the running prefix sum.

```python
def count_subarrays_summing_to(nums, k):
    counts = {0: 1}                                  ## the empty prefix has sum 0 and exists once
    running, total = 0, 0
    for x in nums:
        running += x                                 ## running = prefix sum up to and including x
        total += counts.get(running - k, 0)          ## earlier prefixes that close a match here
        counts[running] = counts.get(running, 0) + 1  ## record THIS prefix for later positions
    return total

## tests

assert count_subarrays_summing_to([1, 1, 1], 2) == 2
assert count_subarrays_summing_to([1, 2, 3], 3) == 2
assert count_subarrays_summing_to([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
print(count_subarrays_summing_to([1, 2, 3], 3))
```

```
2
```

The seed `counts = {0: 1}` is the single most misunderstood line in this family, and the next section
explains it properly. Do not write this template without it.

**Template 4 — frequency map plus bucket sort.** Use for "top k" when you want $O(n)$ and not
$O(n \log k)$.

```python
def top_k_frequent(nums, k):
    freq = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    buckets = [[] for _ in range(len(nums) + 1)]      ## bucket i holds values seen i times
    for value, count in freq.items():
        buckets[count].append(value)
    out = []
    for count in range(len(nums), 0, -1):             ## walk from the highest count downwards
        for value in buckets[count]:
            out.append(value)
            if len(out) == k:                         ## record as soon as k are collected
                return out
    return out

## tests

assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([7], 1) == [7]
assert sorted(top_k_frequent([4, 4, 5, 5, 6], 3)) == [4, 5, 6]
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
```

```
[1, 2]
```

Bucket sort is linear because a count can never exceed `n`, so the counts themselves are array indices.
That bound is what removes the $\log$ factor a heap would cost.

## The trick that unlocks a whole family: prefix sums in a hash map

Write `prefix[i]` for the sum of the first `i` elements, with `prefix[0] = 0`. The sum of the subarray
covering indices `i` to `j-1` is then `prefix[j] - prefix[i]`. Asking for that sum to equal `k` gives

$$\text{prefix}[j] - \text{prefix}[i] = k \quad\Longleftrightarrow\quad \text{prefix}[i] = \text{prefix}[j] - k$$

and the rearranged form is the whole trick. At each position `j` you already know `prefix[j]`, so the
question "how many subarrays end here and sum to `k`" becomes "how many earlier prefixes equalled
`prefix[j] - k`". That is one lookup in a map from prefix value to how many times it has occurred. One
pass, $O(n)$.

**Worked example.** Take `nums = [3, 4, 7, 2, -3, 1, 4, 2]` and `k = 7`. Scan left to right, and at each
step look up `running - 7` in the map before inserting `running`.

| index | value | running | look up | found | total | map after inserting |
|---|---|---|---|---|---|---|
| start | — | 0 | — | — | 0 | `{0: 1}` |
| 0 | 3 | 3 | -4 | 0 | 0 | `{0: 1, 3: 1}` |
| 1 | 4 | 7 | 0 | 1 | 1 | `{0: 1, 3: 1, 7: 1}` |
| 2 | 7 | 14 | 7 | 1 | 2 | `{0: 1, 3: 1, 7: 1, 14: 1}` |
| 3 | 2 | 16 | 9 | 0 | 2 | `..., 16: 1` |
| 4 | -3 | 13 | 6 | 0 | 2 | `..., 13: 1` |
| 5 | 1 | 14 | 7 | 1 | 3 | `..., 14: 2` |
| 6 | 4 | 18 | 11 | 0 | 3 | `..., 18: 1` |
| 7 | 2 | 20 | 13 | 1 | 4 | `..., 20: 1` |

The four subarrays are `[3, 4]`, `[7]`, `[7, 2, -3, 1]` and `[-3, 1, 4, 2]`. Note the hit at index 5:
the map held `7` from index 1, so the earlier prefix and the current one differ by exactly 7. Note also
that `14` reaches a count of 2, and that count is what lets a later position claim both matches at
once with a single lookup.

Now the seed. `counts = {0: 1}` says that the empty prefix — the prefix of length zero, whose sum is 0
— has been seen once, before the scan starts. It matters at index 1 in the table above: `running` is 7,
so the lookup is for 0, and the only prefix with sum 0 is the empty one. That empty prefix is what
represents the subarray `[3, 4]`, which starts at index 0. Without the seed, every subarray that starts
at index 0 is missed, and the tests that hide the bug are the ones whose answer happens to lie in the
middle of the array. Seed it with `{0: 1}`, not `{0: 0}` and not `{}`.

```python
def subarray_sum_equals_k(nums, k):
    counts = {0: 1}
    running, total = 0, 0
    for x in nums:
        running += x
        total += counts.get(running - k, 0)          ## look up BEFORE inserting
        counts[running] = counts.get(running, 0) + 1
    return total

## tests

assert subarray_sum_equals_k([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
assert subarray_sum_equals_k([1, 1, 1], 2) == 2
assert subarray_sum_equals_k([1], 0) == 0
assert subarray_sum_equals_k([0, 0, 0], 0) == 6
print(subarray_sum_equals_k([3, 4, 7, 2, -3, 1, 4, 2], 7))
```

```
4
```

This is the pattern that takes over the moment a sliding window fails. A window shrinks from the left
as soon as the sum overshoots the target, which assumes an overshoot is permanent. With negative
numbers it is not: in the array above, `[7, 2]` already exceeds 7, yet `[7, 2, -3, 1]` hits it exactly.
So `left` would have to move backwards and the linear guarantee is gone. See `21_sliding_window.md`,
problem P21, for the same array worked from the window side. The rule is short: positive values only,
use a window; negatives possible, use prefix sums in a hash map.

## The problems

### P1. Two sum — return the indices of the two numbers that add to `target`

**Which template.** Template 2, the complement map, and it is the canonical instance of it.
**The trick.** Do not key on the value you are holding. Key on `target - x`, which is the one value
that would complete a pair with `x`. Then a single pass suffices, because by the time you reach the
second member of the pair the first is already in the map. Insert `x` after the lookup, so an element
cannot pair with itself when `target == 2 * x`.

```python
def two_sum(nums, target):
    index_of = {}
    for i, x in enumerate(nums):
        if target - x in index_of:                   ## the complement is the key, not x
            return [index_of[target - x], i]
        index_of[x] = i                              ## insert after the test
    return []

## tests

assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
assert two_sum([3, 3], 6) == [0, 1]
assert two_sum([1, 5, 9], 3) == []
print(two_sum([3, 2, 4], 6))
```

```
[1, 2]
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P2. Two sum on a sorted array — same question, but the input is sorted, and indices are 1-based

**Which template.** None of the four. Sorting has already done the work a hash map would pay memory
for, so two pointers win.
**The trick.** Compare `numbers[left] + numbers[right]` to the target. If the sum is too small, only a
larger left value can help, so `left += 1`. If it is too large, only a smaller right value can help, so
`right -= 1`. Every step eliminates one candidate permanently, so the scan is linear. Put P1 and P2
side by side: same question, same $O(n)$ time, but $O(n)$ space against $O(1)$ space. When the input
arrives sorted, or when the interviewer asks for constant space, the hash map is the wrong answer.

```python
def two_sum_sorted(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        total = numbers[left] + numbers[right]
        if total == target:
            return [left + 1, right + 1]             ## the problem asks for 1-based indices
        if total < target:
            left += 1                                ## only a larger left value can help
        else:
            right -= 1                               ## only a smaller right value can help
    return []

## tests

assert two_sum_sorted([2, 7, 11, 15], 9) == [1, 2]
assert two_sum_sorted([2, 3, 4], 6) == [1, 3]
assert two_sum_sorted([-1, 0], -1) == [1, 2]
assert two_sum_sorted([1, 2, 3], 100) == []
print(two_sum_sorted([2, 7, 11, 15], 9), two_sum_sorted([2, 3, 4], 6))
```

```
[1, 2] [1, 3]
```

**Complexity.** $O(n)$ time, $O(1)$ space. Sorting an unsorted input first costs $O(n \log n)$, so on
unsorted input P1 is faster and P2 is smaller.

### P3. Contains duplicate — does the array hold any value twice

**Which template.** Template 1, the seen-set.
**The trick.** The one-liner `len(set(nums)) < len(nums)` is correct and fine to say out loud, but the
explicit loop returns as soon as it finds the first repeat, so it is faster on inputs whose duplicate
is early and it uses less memory in that case. Say which one you are writing and why.

```python
def contains_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False

def contains_duplicate_short(nums):
    return len(set(nums)) < len(nums)                ## same answer, but always scans everything

## tests

assert contains_duplicate([1, 2, 3, 1]) is True
assert contains_duplicate([1, 2, 3, 4]) is False
assert contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) is True
assert contains_duplicate_short([1, 2, 3, 1]) is True
print(contains_duplicate([1, 2, 3, 1]), contains_duplicate([1, 2, 3, 4]))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P4. Contains duplicate within distance k — is there a repeated value whose two indices differ by at most `k`

**Which template.** Template 1, but the set becomes a map to the most recent index.
**The trick.** Store only the latest index of each value and overwrite it every time. An earlier copy
is never more useful than a later one, because a later copy is closer to whatever comes next. That
overwrite is what keeps the map at one entry per distinct value. A sliding window of size `k` holding a
set is the other correct solution; the map is shorter to write and harder to get wrong.

```python
def contains_nearby_duplicate(nums, k):
    last_index = {}                                  ## value -> the most recent index of it
    for i, x in enumerate(nums):
        if x in last_index and i - last_index[x] <= k:
            return True                              ## the nearest earlier copy is the best chance
        last_index[x] = i                            ## overwrite: only the latest index matters
    return False

## tests

assert contains_nearby_duplicate([1, 2, 3, 1], 3) is True
assert contains_nearby_duplicate([1, 0, 1, 1], 1) is True
assert contains_nearby_duplicate([1, 2, 3, 1, 2, 3], 2) is False
assert contains_nearby_duplicate([1], 0) is False
print(contains_nearby_duplicate([1, 2, 3, 1], 3), contains_nearby_duplicate([1, 2, 3, 1, 2, 3], 2))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(d)$ space for `d` distinct values.

### P5. Valid anagram — do two strings contain the same letters with the same multiplicities

**Which template.** A frequency map, and for a lowercase alphabet a 26-slot list beats a dict.
**The trick.** Count up on the first string and down on the second in one shared array, and bail out
the moment a count goes negative. That single array replaces the two maps most people write, and the
negative test means you do not need a final comparison pass. The length check at the top is what makes
the negative test sufficient: with equal lengths, no count can be left positive if none went negative.

```python
def is_anagram(s, t):
    if len(s) != len(t):
        return False                                 ## a length check kills most inputs at once
    counts = [0] * 26
    for ch in s:
        counts[ord(ch) - 97] += 1
    for ch in t:
        counts[ord(ch) - 97] -= 1
        if counts[ord(ch) - 97] < 0:                 ## t has a letter s does not have enough of
            return False
    return True

## tests

assert is_anagram("anagram", "nagaram") is True
assert is_anagram("rat", "car") is False
assert is_anagram("a", "ab") is False
assert is_anagram("", "") is True
print(is_anagram("anagram", "nagaram"), is_anagram("rat", "car"))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(26)$ space, which is $O(1)$.

### P6. Group anagrams — partition a word list into groups that are anagrams of each other

**Which template.** A map from a canonical key to a list, and the key is the whole problem.
**The trick.** Two words belong together when they share a canonical form. Sorting the letters gives
one, `"".join(sorted(word))`, and costs $O(L \log L)$ per word. A tuple of 26 counts gives another and
costs $O(L)$. Use the count tuple: it must be a `tuple`, because a list is not hashable and a dict key
must be hashable. Say the sorted-string version first, then offer the count tuple as the improvement.

```python
def group_anagrams(words):
    groups = {}
    for word in words:
        counts = [0] * 26
        for ch in word:
            counts[ord(ch) - 97] += 1
        key = tuple(counts)                          ## canonical form, built in O(len(word))
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())

## tests

out = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
assert sorted(sorted(g) for g in out) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
assert group_anagrams([""]) == [[""]]
assert group_anagrams(["a"]) == [["a"]]
print(out)
```

```
[['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

**Complexity.** $O(N L)$ time for `N` words of length up to `L`, and $O(N L)$ space.

### P7. Top k frequent elements — the `k` values that appear most often

**Which template.** Template 4, frequency map plus bucket sort.
**The trick.** A count can never exceed `n`, so counts are valid indices into an array of length
`n + 1`. Put each value into the bucket named by its count and then walk the buckets downwards, taking
values until you have `k`. There is no sort and no heap, so the whole thing is $O(n)$. The heap
alternative, `heapq.nlargest(k, freq, key=freq.get)`, is $O(n \log k)$ and is the right answer to give
when `k` is small and memory matters.

```python
def top_k_frequent(nums, k):
    freq = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    buckets = [[] for _ in range(len(nums) + 1)]     ## index = count, so index <= n
    for value, count in freq.items():
        buckets[count].append(value)
    out = []
    for count in range(len(nums), 0, -1):
        for value in buckets[count]:
            out.append(value)
            if len(out) == k:
                return out
    return out

## tests

assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert sorted(top_k_frequent([5, 5, 4, 4, 3, 3, 2], 3)) == [3, 4, 5]
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
```

```
[1, 2]
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P8. Product of array except self — `out[i]` is the product of every element other than `nums[i]`, without division

**Which template.** None. This is the problem that looks like it needs a map and does not.
**The trick.** The answer at `i` is (everything to the left of `i`) times (everything to the right of
`i`). Build the left products in a forward pass, writing each into `out[i]` *before* multiplying
`nums[i]` in, then run a backward pass doing the same with a suffix accumulator. Two passes, one output
array, no division, so a zero anywhere in the input is handled with no special case. The division
solution breaks on a single zero and gives the wrong answer on two zeros.

```python
def product_except_self(nums):
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):
        out[i] = prefix                              ## product of everything strictly left of i
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix                             ## multiply in everything strictly right of i
        suffix *= nums[i]
    return out

## tests

assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
assert product_except_self([2, 3]) == [3, 2]
assert product_except_self([0, 0]) == [0, 0]
print(product_except_self([1, 2, 3, 4]), product_except_self([-1, 1, 0, -3, 3]))
```

```
[24, 12, 8, 6] [0, 0, 9, 0, 0]
```

**Complexity.** $O(n)$ time, $O(1)$ extra space if the output array is not counted.

### P9. Longest consecutive sequence — the length of the longest run of consecutive integers, in any order

**Which template.** Template 1, a set, and one extra test that is the entire problem.
**The trick.** Put every value in a set, then for each value walk upwards while `x + 1` is present. On
its own that is $O(n^2)$, because a run of length `L` gets walked from all `L` of its members. The fix
is one line: **only start walking when `x - 1` is not in the set**, so each run is walked exactly once,
from its first element. The total work over all runs is then the sum of the run lengths, which is at
most `n`. That single `continue` is what makes the solution linear, and it is the thing the
interviewer is checking.

```python
def longest_consecutive(nums):
    values = set(nums)
    best = 0
    for x in values:
        if x - 1 in values:                          ## x is not the start of a run, so skip it
            continue
        length = 1
        while x + length in values:                  ## walk the run upwards from its first element
            length += 1
        best = max(best, length)
    return best

## tests

assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
assert longest_consecutive([]) == 0
assert longest_consecutive([1, 1, 1]) == 1
print(longest_consecutive([100, 4, 200, 1, 3, 2]), longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))
```

```
4 9
```

**Complexity.** $O(n)$ time and $O(n)$ space. Sorting also solves it in $O(n \log n)$ and $O(1)$ extra
space, so name that trade if the interviewer asks for less memory.

### P10. Subarray sum equals k — count them, then find the longest one

**Which template.** Template 3, the prefix-sum map. The counting version is in the section above; this
is the same idea with the map's value changed.
**The trick.** Change what the map stores and you change the question it answers. Store *how many
times* each prefix sum occurred and you count subarrays. Store the *first index* at which each prefix
sum occurred and you get the longest subarray, because the earliest match gives the widest span. So
never overwrite an existing key in this version — an earlier index is always at least as good. Seed
with `{0: -1}` here, not `{0: 1}`: the empty prefix ends just before index 0, so its index is `-1` and
the length arithmetic `i - (-1)` comes out right.

```python
def longest_subarray_sum_k(nums, k):
    first_index = {0: -1}                            ## prefix sum 0 ends just before index 0
    running, best = 0, 0
    for i, x in enumerate(nums):
        running += x
        if running - k in first_index:               ## a match: keep the EARLIEST such prefix
            best = max(best, i - first_index[running - k])
        if running not in first_index:               ## do not overwrite: earlier means longer
            first_index[running] = i
    return best

## tests

assert longest_subarray_sum_k([1, -1, 5, -2, 3], 3) == 4
assert longest_subarray_sum_k([-2, -1, 2, 1], 1) == 2
assert longest_subarray_sum_k([1, 2, 3], 100) == 0
assert longest_subarray_sum_k([0, 0, 0], 0) == 3
print(longest_subarray_sum_k([1, -1, 5, -2, 3], 3), longest_subarray_sum_k([-2, -1, 2, 1], 1))
```

```
4 2
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P11. Continuous subarray sum divisible by k — is there a subarray of length at least 2 whose sum is a multiple of `k`

**Which template.** Template 3, with the key changed from the prefix sum to the prefix sum **modulo k**.
**The trick.** A subarray sums to a multiple of `k` exactly when its two bounding prefix sums leave the
same remainder modulo `k`, because `prefix[j] - prefix[i]` is divisible by `k` if and only if
`prefix[j] % k == prefix[i] % k`. So the key is the remainder, and the map holds at most `k` entries
rather than `n`. Store the first index of each remainder and require the gap to be at least 2. Python's
`%` returns a non-negative result for a positive `k`, so negative values need no special handling.

```python
def check_subarray_sum(nums, k):
    first_index = {0: -1}                            ## remainder 0 before the array starts
    running = 0
    for i, x in enumerate(nums):
        running = (running + x) % k                  ## key on the REMAINDER, not on the sum
        if running in first_index:
            if i - first_index[running] >= 2:        ## the problem needs length at least 2
                return True
        else:
            first_index[running] = i                 ## keep the earliest index of each remainder
    return False

## tests

assert check_subarray_sum([23, 2, 4, 6, 7], 6) is True
assert check_subarray_sum([23, 2, 6, 4, 7], 13) is False
assert check_subarray_sum([1, 0], 2) is False
assert check_subarray_sum([0, 0], 7) is True
print(check_subarray_sum([23, 2, 4, 6, 7], 6), check_subarray_sum([23, 2, 6, 4, 7], 13))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(\min(n, k))$ space.

### P12. Find all duplicates in an array — values are in `1..n` and each appears once or twice; return the repeats, in $O(1)$ space

**Which template.** Index-as-hash. The array is its own hash table, because the values and the indices
have the same range.
**The trick.** Value `v` belongs at index `v - 1`. Mark a slot as visited by negating what is stored
there. When you reach a value whose home slot is already negative, that value has been seen before, so
it is a duplicate. Always read `abs(x)`, because an earlier mark may have flipped the sign of the very
cell you are reading. Restore the signs at the end unless the interviewer says the input may be
destroyed.

```python
def find_duplicates(nums):
    out = []
    for x in nums:
        slot = abs(x) - 1                            ## value v belongs at index v-1
        if nums[slot] < 0:                           ## already marked, so v was seen before
            out.append(abs(x))
        else:
            nums[slot] = -nums[slot]                 ## mark slot v-1 as visited
    for i in range(len(nums)):
        nums[i] = abs(nums[i])                       ## restore the array before returning
    return out

## tests

assert sorted(find_duplicates([4, 3, 2, 7, 8, 2, 3, 1])) == [2, 3]
assert find_duplicates([1, 1, 2]) == [1]
assert find_duplicates([1]) == []
data = [4, 3, 2, 7, 8, 2, 3, 1]
print(find_duplicates(data), data)
```

```
[2, 3] [4, 3, 2, 7, 8, 2, 3, 1]
```

**Complexity.** $O(n)$ time, $O(1)$ extra space. Sign marking needs strictly positive values, so it
fails if zero is allowed; use cyclic sort then, as in P13.

### P13. First missing positive — the smallest positive integer absent from the array, in $O(n)$ time and $O(1)$ space

**Which template.** Cyclic sort, the strongest form of index-as-hash.
**The trick.** The answer must lie in `1..n+1`, because `n` values cannot cover more than `n` of the
positives. So put every value `v` in range at index `v - 1` by swapping, then scan once for the first
index whose content is wrong. The swap loop is a `while`, not an `if`, because the value that arrives
at position `i` may itself need to move. The guard `nums[nums[i] - 1] != nums[i]` compares values, not
indices, and it is what stops the loop spinning forever on duplicates — a swap that would change
nothing is refused.

```python
def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            target = nums[i] - 1                     ## send nums[i] home to index nums[i]-1
            nums[i], nums[target] = nums[target], nums[i]
    for i in range(n):
        if nums[i] != i + 1:                         ## the first slot holding the wrong value
            return i + 1
    return n + 1

## tests

assert first_missing_positive([1, 2, 0]) == 3
assert first_missing_positive([3, 4, -1, 1]) == 2
assert first_missing_positive([7, 8, 9, 11, 12]) == 1
assert first_missing_positive([]) == 1
assert first_missing_positive([1, 1]) == 2
print(first_missing_positive([3, 4, -1, 1]), first_missing_positive([7, 8, 9, 11, 12]))
```

```
2 1
```

**Complexity.** $O(n)$ time — each swap puts one value permanently home, so there are at most `n` of
them — and $O(1)$ space.

### P14. Single number — every value appears twice except one; find it

**Which template.** None. XOR replaces the hash table entirely.
**The trick.** XOR has three properties that together solve the problem: `x ^ x == 0`, `x ^ 0 == x`,
and it is commutative and associative, so the order of the array does not matter. XOR the whole array
and every pair cancels, leaving the lonely value. A set gives the same answer in $O(n)$ space; XOR
gives it in $O(1)$, and that is the reason the question is asked.

```python
def single_number(nums):
    result = 0
    for x in nums:
        result ^= x                                  ## equal values cancel, order does not matter
    return result

## tests

assert single_number([2, 2, 1]) == 1
assert single_number([4, 1, 2, 1, 2]) == 4
assert single_number([1]) == 1
assert single_number([0, 0, -5]) == -5
print(single_number([4, 1, 2, 1, 2]))
```

```
4
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P15. Majority element — the value that appears more than `n/2` times

**Which template.** Boyer-Moore voting. A frequency map solves it in $O(n)$ space; voting solves it in
$O(1)$.
**The trick.** Hold one candidate and a counter. Every matching element is a vote for the candidate,
every other element is a vote against, and when the counter reaches zero the candidate is replaced. The
reason it works: each cancellation removes one majority element and one non-majority element, and
because the majority strictly exceeds `n/2` it cannot be exhausted before the others are. Whatever
survives is the answer. If the problem only promises a *plurality* rather than a majority, this is
wrong — add a second pass that counts the candidate and verifies it.

```python
def majority_element(nums):
    candidate, count = None, 0
    for x in nums:
        if count == 0:
            candidate = x                            ## no survivor left, so start a new candidate
        count += 1 if x == candidate else -1         ## a vote for, or a cancelling vote against
    return candidate

## tests

assert majority_element([3, 2, 3]) == 3
assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
assert majority_element([1]) == 1
assert majority_element([6, 5, 5]) == 5
print(majority_element([2, 2, 1, 1, 1, 2, 2]))
```

```
2
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P16. Intersection of two arrays — the values present in both

**Which template.** Template 1, with the set built from one side and the lookup done from the other.
**The trick.** Build the set from the **smaller** array, so the space is $O(\min(n, m))$ rather than
$O(n)$. Then ask what the answer means. If the result must be distinct values, collect into a set. If
the result must respect multiplicity — each value appears as often as it occurs in both inputs — count
instead of setting, and decrement as you spend each copy. Ask the interviewer which one they want,
because the two answers differ on the very first example.

```python
def intersection(nums1, nums2):
    smaller, larger = (nums1, nums2) if len(nums1) <= len(nums2) else (nums2, nums1)
    lookup = set(smaller)                            ## build from the SMALLER side
    out = set()
    for x in larger:
        if x in lookup:
            out.add(x)
    return list(out)

def intersection_with_counts(nums1, nums2):
    counts = {}
    for x in nums1:
        counts[x] = counts.get(x, 0) + 1
    out = []
    for x in nums2:
        if counts.get(x, 0) > 0:                     ## multiplicity version: spend one copy
            out.append(x)
            counts[x] -= 1
    return out

## tests

assert sorted(intersection([1, 2, 2, 1], [2, 2])) == [2]
assert sorted(intersection([4, 9, 5], [9, 4, 9, 8, 4])) == [4, 9]
assert sorted(intersection_with_counts([1, 2, 2, 1], [2, 2])) == [2, 2]
assert intersection_with_counts([1], [2]) == []
print(sorted(intersection([4, 9, 5], [9, 4, 9, 8, 4])), intersection_with_counts([1, 2, 2, 1], [2, 2]))
```

```
[4, 9] [2, 2]
```

**Complexity.** $O(n + m)$ time, $O(\min(n, m))$ space.

### P17. Isomorphic strings — can the letters of `s` be renamed consistently to give `t`

**Which template.** Two maps, and the second one is the point of the question.
**The trick.** One map from `s` to `t` is not enough. It accepts `"badc"` against `"baba"`, because
each letter of `s` maps to exactly one letter of `t` there — but `d` and `b` both map onto `b`, so the
renaming is not reversible. A renaming must be a one-to-one correspondence, therefore you need the
map in both directions and a conflict in either one is a rejection. Write both checks before either
assignment, or the second check reads a value you have just written.

```python
def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    forward, backward = {}, {}                       ## s->t and t->s, both needed
    for a, b in zip(s, t):
        if a in forward and forward[a] != b:
            return False                             ## a already maps somewhere else
        if b in backward and backward[b] != a:
            return False                             ## b is already claimed by another letter
        forward[a] = b
        backward[b] = a
    return True

## tests

assert is_isomorphic("egg", "add") is True
assert is_isomorphic("foo", "bar") is False
assert is_isomorphic("badc", "baba") is False
assert is_isomorphic("paper", "title") is True
print(is_isomorphic("egg", "add"), is_isomorphic("badc", "baba"))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(\Sigma)$ space.

### P18. Word pattern — does a sentence follow a letter pattern, with one word per letter

**Which template.** P17 exactly, with characters on one side and words on the other.
**The trick.** It is the same two-map bijection, so recognise it and reuse the code. Two things are
new. First, `zip` stops at the shorter sequence, so a pattern and a sentence of different lengths would
silently pass; the explicit length check is a real test case, not defensive noise. Second, split on
whitespace with `sentence.split()` and not `split(" ")`, because the latter produces empty strings from
repeated spaces.

```python
def word_pattern(pattern, sentence):
    words = sentence.split()
    if len(pattern) != len(words):
        return False                                 ## the length check is a real test case
    forward, backward = {}, {}
    for ch, word in zip(pattern, words):
        if ch in forward and forward[ch] != word:
            return False
        if word in backward and backward[word] != ch:
            return False
        forward[ch] = word
        backward[word] = ch
    return True

## tests

assert word_pattern("abba", "dog cat cat dog") is True
assert word_pattern("abba", "dog cat cat fish") is False
assert word_pattern("aaaa", "dog cat cat dog") is False
assert word_pattern("abba", "dog dog dog dog") is False
assert word_pattern("a", "dog dog") is False
print(word_pattern("abba", "dog cat cat dog"), word_pattern("abba", "dog dog dog dog"))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P19. Ransom note — can the note be built from the letters of the magazine, each used once

**Which template.** A frequency map spent down, which is the counting form of template 1.
**The trick.** Count the supply, then walk the demand and spend one copy per character. A missing key
and an exhausted key are the same failure, so `available.get(ch, 0) == 0` covers both in one test and
there is no `KeyError` to guard. The length check at the top is free and rejects a whole class of
inputs before any counting.

```python
def can_construct(note, magazine):
    if len(note) > len(magazine):
        return False
    available = {}
    for ch in magazine:
        available[ch] = available.get(ch, 0) + 1
    for ch in note:
        if available.get(ch, 0) == 0:                ## missing, or already spent
            return False
        available[ch] -= 1                           ## spend one copy
    return True

## tests

assert can_construct("a", "b") is False
assert can_construct("aa", "aab") is True
assert can_construct("aa", "ab") is False
assert can_construct("", "anything") is True
print(can_construct("aa", "aab"), can_construct("aa", "ab"))
```

```
True False
```

**Complexity.** $O(n + m)$ time, $O(\Sigma)$ space.

### P20. Substring with concatenated words — every start index where `s` holds all of `words` joined in some order, each once

**Which template.** A frequency map compared against a window, but the window moves in whole words.
**The trick.** All words have the same length `k`, so treat `s` as a sequence of `k`-character chunks.
There are `k` different ways to align the chunk grid, so run `k` independent scans, one per starting
offset, and inside each scan slide a window measured in words rather than characters. Shrink from the
left while any word is over its quota, and reset the whole window when a chunk is not a word at all.
Every character is visited once per offset, so the cost is $O(k \cdot n)$ and not $O(n \cdot m)$.

```python
def find_substring(s, words):
    k, m = len(words[0]), len(words)
    need = {}
    for w in words:
        need[w] = need.get(w, 0) + 1
    out = []
    for offset in range(k):                          ## k independent chunk alignments
        window, count, left = {}, 0, offset
        for right in range(offset, len(s) - k + 1, k):
            word = s[right:right + k]
            if word not in need:
                window.clear()                       ## an unusable word resets everything
                count, left = 0, right + k
                continue
            window[word] = window.get(word, 0) + 1
            count += 1
            while window[word] > need[word]:         ## too many copies: drop from the left
                gone = s[left:left + k]
                window[gone] -= 1
                count -= 1
                left += k
            if count == m:
                out.append(left)
    return out

## tests

assert sorted(find_substring("barfoothefoobarman", ["foo", "bar"])) == [0, 9]
assert find_substring("wordgoodgoodgoodbestword", ["word", "good", "best", "word"]) == []
assert sorted(find_substring("barfoofoobarthefoobarman", ["bar", "foo", "the"])) == [6, 9, 12]
print(sorted(find_substring("barfoothefoobarman", ["foo", "bar"])))
```

```
[0, 9]
```

**Complexity.** $O(k \cdot n)$ time, $O(m k)$ space.

### P21. Four sum count — how many tuples `(i, j, p, q)` from four lists satisfy `a[i] + b[j] + c[p] + d[q] == 0`

**Which template.** Template 2, the complement map, applied to **pairs** instead of elements.
**The trick.** Four nested loops cost $O(n^4)$. Split the four lists into two halves. Build a map from
every sum `a[i] + b[j]` to how many pairs produce it — that is $O(n^2)$ entries. Then for every pair
from the other two lists, look up the complement `-(c[p] + d[q])` and add its count. The key is a pair
sum, not an element, and that generalisation is what takes the problem from $O(n^4)$ to $O(n^2)$. The
same meet-in-the-middle idea appears whenever a problem splits cleanly into two independent halves.

```python
def four_sum_count(a, b, c, d):
    pair_sums = {}
    for x in a:
        for y in b:
            pair_sums[x + y] = pair_sums.get(x + y, 0) + 1   ## all n^2 sums of the first two lists
    total = 0
    for z in c:
        for w in d:
            total += pair_sums.get(-(z + w), 0)              ## look up the complement
    return total

## tests

assert four_sum_count([1, 2], [-2, -1], [-1, 2], [0, 2]) == 2
assert four_sum_count([0], [0], [0], [0]) == 1
assert four_sum_count([1], [1], [1], [1]) == 0
print(four_sum_count([1, 2], [-2, -1], [-1, 2], [0, 2]))
```

```
2
```

**Complexity.** $O(n^2)$ time, $O(n^2)$ space.

### P22. Counting values in a range — how many elements lie between `low` and `high` inclusive

**Which template.** None, and that is the lesson. This is the problem where a hash map is the wrong
answer.
**The trick.** A hash table answers equality and nothing else. It has no order, so it cannot answer
"which values are less than 7" without visiting every key. The map version below is correct but its
cost grows with the **width of the range**, so a query over `1..10**9` is hopeless even on three
elements. Sorting once and using binary search answers any range in $O(\log n)$, and the sort is paid
back after the second query. The general rule: equality and membership go to a hash map, order and
ranges go to a sorted array with `bisect`, or to a balanced tree. If the interviewer says "many
queries", they are asking for the sorted version.

```python
from bisect import bisect_left, bisect_right

def count_in_range(nums, low, high):
    ordered = sorted(nums)                           ## pay O(n log n) once, then O(log n) per query
    return bisect_right(ordered, high) - bisect_left(ordered, low)

def count_in_range_with_a_map(nums, low, high):
    seen = {}
    for x in nums:
        seen[x] = seen.get(x, 0) + 1
    total = 0
    for value in range(low, high + 1):               ## the map forces a scan of the whole range
        total += seen.get(value, 0)
    return total

## tests

data = [5, 1, 9, 3, 7, 3]
assert count_in_range(data, 3, 7) == 4
assert count_in_range(data, 0, 100) == 6
assert count_in_range(data, 4, 4) == 0
assert count_in_range_with_a_map(data, 3, 7) == 4
print(count_in_range(data, 3, 7), count_in_range(data, 0, 100))
```

```
4 6
```

**Complexity.** $O(n \log n)$ to sort, then $O(\log n)$ per query and $O(n)$ space. The map version is
$O(n + (high - low))$ per query, which is unusable for a wide range.

## Tricks and tips

**Say the key out loud before you write anything.** "I will key on the complement." "I will key on the
sorted letters." "I will key on the running prefix sum modulo k." That sentence is the solution, and
the code after it is mechanical. If you cannot finish the sentence, you have the wrong pattern, and the
usual replacements are sorting, two pointers or binary search.

**Look up before you insert.** In every template the current element is inserted into the map only
after the lookup that uses the map. Insert first and an element pairs with itself, so Two Sum returns
`[0, 0]` on `[3, 2, 4]` with target 6. This is one line in the wrong order and it is the most common
single-character bug in the whole family.

**Use `collections.Counter` and `collections.defaultdict` when they help, but know what they cost.**
`Counter(s) == Counter(t)` is a complete Valid Anagram in one line, and it is a fine answer. However,
`defaultdict(int)` inserts a key on every read, so `if x in d` after a read of `d[x]` is always true,
and that turns a membership test into a silent yes. If you use a `defaultdict`, never test membership
on it. `dict.get(key, 0)` avoids the whole issue and works everywhere.

**Delete a key when its count reaches zero, or compare counts instead of maps.** `{'a': 0}` and `{}`
are not equal, so a map that keeps zero entries breaks any `==` comparison and any `len()` used as a
distinct-value count. Either delete on reaching zero, or switch to a fixed-size list when the alphabet
is small: `[0] * 26` compares with a single `==` and has no keys to delete.

**Make the key hashable and canonical.** A list cannot be a dict key; a tuple can. `tuple(counts)` and
`"".join(sorted(word))` are both canonical, and the tuple is faster because it avoids the sort. For a
pair of coordinates use `(row, col)`, never a string built with a separator, because the separator can
appear in the data.

**When the values are a permutation of `1..n`, the array is already a hash table.** Index `v - 1`
belongs to value `v`, so you can mark by negating or place by swapping, and the space drops to $O(1)$.
Sign marking needs strictly positive values; cyclic sort tolerates anything, because out-of-range
values are simply left where they are.

**Two directions, not one, for any correspondence.** Isomorphic strings and Word Pattern both need a
map each way, because a renaming must be reversible. One map accepts a collapse of two symbols onto
one.

**Insertion order is a Python detail, not a guarantee of the problem.** A `dict` preserves insertion
order in Python 3.7 and later, so `list(groups.values())` comes out in first-seen order. That is
convenient for reading the output, and it is never a correctness argument. If the problem needs a
specific order, sort explicitly and say so.

## The bugs that cost the round

**Inserting before looking up.** Covered above and worth repeating, because it fails only on the inputs
where an element could pair with itself, which is rarely the first example.

**Forgetting the `{0: 1}` seed on a prefix-sum map.** Without it every subarray that starts at index 0
is missed. The classic symptom is that `[1, 2, 3]` with `k = 3` returns 1 instead of 2 — the subarray
`[3]` is found, `[1, 2]` is not. If a prefix-sum answer is exactly one too small, this is why.

**Seeding with the wrong value for the wrong variant.** The counting form seeds `{0: 1}` because it
stores occurrence counts. The longest-subarray form seeds `{0: -1}` because it stores indices, and the
empty prefix ends just before index 0. Mixing the two gives lengths that are off by one, or a
`TypeError` when a count is subtracted from an index.

**Overwriting a first index that should be kept.** When the map stores indices for a longest-span
answer, an existing key must never be updated, because an earlier index gives a wider span. When the
map stores the most recent index, as in P4, it must always be updated. Decide which one the question
needs and write the `if x not in map` guard, or its absence, deliberately.

**Mutating the input and then reading it as if it were unchanged.** Sign marking flips values in place,
so every later read must be `abs(x)`. Cyclic sort permutes the array. Restore or warn.

**Using a hash map where order is the question.** Ranges, nearest values, "the smallest greater than
x", and top-k by value rather than by count are all order questions, and a hash map cannot answer any
of them without a full scan. Reach for `sorted` plus `bisect`, or for a heap.

**Assuming the alphabet.** `ord(ch) - 97` is wrong the moment uppercase, digits or Unicode appear. Ask
what the input alphabet is, and use a dict if the answer is anything but lowercase ASCII.

## Done when

- Given a problem statement you have not seen, you can name the key — complement, canonical form,
  prefix sum, remainder, or index — in under 30 seconds, and say why the other four are wrong.
- You can write the prefix-sum counter map from a blank file in two minutes, explain the `{0: 1}` seed
  by naming the empty prefix, and convert it to the `{0: -1}` longest-subarray form without hesitating.
- You can write Two Sum, Group Anagrams and Top K Frequent from memory in five minutes total, with the
  lookup before the insert in the first and a hashable canonical key in the second.
- You can state, for any array problem, whether a hash map beats sorting, and give the memory cost of
  choosing it — including the cases in P2, P13 and P22 where sorting or the array itself wins.
