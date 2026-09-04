# Two pointers: every variation

Two pointers replaces a double loop with a single scan. You hold two indices into the data, move each
one forward only, and never restart either, so the whole pass costs $O(n)$ time and $O(1)$ extra space
where the brute force costs $O(n^2)$.

The pattern is hard because "two pointers" names three genuinely different techniques that share only
the name. The first is **converging**: one pointer starts at each end and they walk towards each other.
It needs a **sorted** array, and it works because moving one pointer changes the quantity in a known
direction. The second is **same-direction fast and slow** on an array, which is really in-place
partitioning: a `read` pointer scans every element and a `write` pointer records the ones you keep. The
third is **fast and slow on a linked list**, which is cycle detection and midpoint finding, and shares
no code with the other two.

Converging is the variant interviewers mean by default. It is correct only when the array is sorted, or
when the quantity is monotone for some other reason, because that monotonicity is what justifies
discarding a whole side of the search space in one step. Choose it on an unsorted array and it will
silently miss answers.

A sliding window is also a same-direction two-pointer technique, specialised to contiguous ranges.
`21_sliding_window.md` covers that case in full, so this chapter concentrates on the converging and the
fast-and-slow forms. When the problem asks about a contiguous subarray or substring, go there instead.

## Recognising it from the phrasing

| The interviewer says | The variant | Where the pointers start | What justifies the move |
|---|---|---|---|
| "sorted array, find a pair / triple summing to target" | converging | both ends | the array is sorted, so the sum is monotone in each pointer |
| "palindrome check", "reverse in place" | converging | both ends | each step compares or swaps one matched pair |
| "container with most water", "trapping rain water" | converging with a greedy discard rule | both ends | the shorter wall caps the answer, so it can be discarded |
| "remove / move elements in place, keep order" | fast and slow write pointer | both at index 0 | `write` never overtakes `read`, so nothing unread is overwritten |
| "dedupe a sorted array in place" | fast and slow write pointer | `read` at 1, `write` at 1 | equal values are adjacent once sorted |
| "linked list cycle", "middle node", "nth from the end" | fast and slow on node pointers | both at the head | a fixed speed ratio or a fixed gap gives the position |
| "merge two sorted things" | two pointers over two sequences | index 0 of each | the smaller head is always the next output |
| "is one string a subsequence of another" | two pointers over two sequences | index 0 of each | greedy earliest match never loses a solution |

Before writing a converging solution, ask one question: **what happens to the quantity when you move
the left pointer right, and what happens when you move the right pointer left?** If one move can only
increase the quantity and the other can only decrease it, then a comparison against the target tells you
which side cannot contain the answer, the discard is justified, and the technique is correct. Two Sum on
a sorted array passes this test: moving `left` right raises the sum, moving `right` left lowers it. If
neither move is monotone, converging pointers will skip the answer without any error, because you will
throw away a side that still contained it. On an unsorted array the fix is to sort first, which costs
$O(n \log n)$, or to abandon pointers and use a hash map, which costs $O(n)$ time and $O(n)$ space. Ask
this question before the first line of code, because a wrong answer from a converging scan looks exactly
like a right one.

## The templates

Templates 1 and 2 are the two array forms and they look nothing alike, so learn them as separate
skeletons. Template 3 is template 1 applied to two sequences instead of one. Template 4 is the linked
list form.

**Template 1 — converging pointers on a sorted array.** Use when the input is sorted and you look for a
pair with a target property.

```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]                  ## 1. hit: record here and stop
        elif total < target:
            left += 1                             ## 2. too small: only a bigger left can help
        else:
            right -= 1                            ## 3. too big: only a smaller right can help
    return [-1, -1]

## tests

assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]
assert two_sum_sorted([2, 3, 4], 6) == [0, 2]
assert two_sum_sorted([1, 2], 99) == [-1, -1]
print(two_sum_sorted([2, 7, 11, 15], 9))
```

```
[0, 1]
```

**Template 2 — fast and slow write pointer for in-place filtering.** Use when you must keep some
elements, drop the rest, preserve order, and use no extra array.

```python
def keep_if(nums, predicate):
    write = 0                                     ## write = number of kept elements so far
    for read in range(len(nums)):
        if predicate(nums[read]):
            nums[write] = nums[read]              ## copy a keeper forward
            write += 1
    return write                                  ## the kept elements are nums[:write]

## tests

a = [3, 1, 4, 1, 5, 9, 2, 6]
kept = keep_if(a, lambda x: x % 2 == 1)
assert kept == 5 and a[:kept] == [3, 1, 1, 5, 9]
assert keep_if([2, 4, 6], lambda x: x % 2 == 1) == 0
assert keep_if([1, 3, 5], lambda x: x % 2 == 1) == 3
print(kept, a[:kept])
```

```
5 [3, 1, 1, 5, 9]
```

At the end, `write` is the count of kept elements and `nums[:write]` is the answer. That is where the
answer is recorded in this template, and it is the part people forget: the function returns a length,
not an array, and everything at or after index `write` is garbage that the caller must ignore.

**Template 3 — two pointers over two sequences.** Use for merging, for subsequence checks, and for
intersections of sorted lists.

```python
def merge_two_sorted(a, b):
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:                          ## <= keeps the merge stable
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])                             ## one list is exhausted; drain the other
    out.extend(b[j:])
    return out

## tests

assert merge_two_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_two_sorted([], [1, 2]) == [1, 2]
assert merge_two_sorted([1, 1], [1]) == [1, 1, 1]
print(merge_two_sorted([1, 3, 5], [2, 4, 6]))
```

```
[1, 2, 3, 4, 5, 6]
```

**Template 4 — Floyd's cycle detection, both phases.** Use for any linked list question about a cycle.
This block also defines the `ListNode` class and the `build` and `to_list` helpers that every linked
list problem below reuses.

```python
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]          ## close a cycle, for the tests only
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def cycle_start(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next                          ## 1 step
        fast = fast.next.next                     ## 2 steps
        if slow is fast:                          ## phase 1: a meeting point inside the cycle
            walker = head
            while walker is not slow:             ## phase 2: both move ONE step
                walker = walker.next
                slow = slow.next
            return walker                         ## they meet exactly at the cycle entry
    return None

## tests

assert to_list(build([1, 2, 3])) == [1, 2, 3]
assert cycle_start(build([3, 2, 0, -4], cycle_at=1)).val == 2
assert cycle_start(build([1, 2], cycle_at=0)).val == 1
assert cycle_start(build([1, 2, 3])) is None
print(cycle_start(build([3, 2, 0, -4], cycle_at=1)).val)
```

```
2
```

The answer in template 4 is recorded at the end of phase 2, not at the meeting point. The meeting point
is somewhere inside the cycle and carries no useful information on its own. Phase 2 turns it into the
entry node, and the reason is arithmetic: if the tail before the cycle has length `a` and the meeting
point is `b` steps into a cycle of length `c`, then the fast pointer has travelled twice as far as the
slow one, which forces `a` and `b` to satisfy `a = c - b` modulo `c`. So a walker starting at the head
and the slow pointer starting at the meeting point, both moving one step at a time, arrive at the entry
node together.

## The trick that unlocks a whole family: k-sum reduction

Every k-sum problem is the same problem. Sort the array once. Then 2Sum is a single converging scan.
3Sum is 2Sum run inside one loop that fixes the first element: for each `i`, run the converging scan on
the rest of the array with target `-nums[i]`. 4Sum is 3Sum inside another loop. The general rule is one
extra loop per extra term, so k-sum costs $O(n^{k-1})$ time after an $O(n \log n)$ sort. Learn one
converging scan and you have the whole family.

The part that fails people in interviews is not the reduction. It is duplicate removal. The question
asks for unique triples, and a sorted array puts equal values next to each other, so there are exactly
two places a duplicate can enter. First, at the fixed index: if `nums[i] == nums[i-1]` then every
triple starting at `i` was already produced starting at `i-1`, so skip it. Second, after recording a
hit: the moving pointers must step past any repeat of the values just used, on both sides. The three
skip lines are:

```
if i > 0 and nums[i] == nums[i - 1]: continue         ## repeated fixed value
while left < right and nums[left] == nums[left - 1]: left += 1
while left < right and nums[right] == nums[right + 1]: right -= 1
```

Note the direction of each comparison. The fixed-index skip looks **backwards** at `i-1`, because `i`
has not been used yet. The two pointer skips look at `left - 1` and `right + 1`, which are the positions
the pointers just left after the `left += 1` and `right -= 1` that follow a recorded hit. Get either
index wrong and you either emit duplicates or skip real answers.

**Worked example.** Take `nums = [-1, 0, 1, 2, -1, -4]`. Sorted it is `[-4, -1, -1, 0, 1, 2]`.

At `i = 0` the fixed value is `-4` and the scan over `[-1, -1, 0, 1, 2]` finds nothing, because the
largest possible pair sum is `1 + 2 = 3` and the target is `4`. At `i = 1` the fixed value is `-1` and
the target is `1`. The pointers start at `-1` and `2`, sum `1`, a hit, so `[-1, -1, 2]` is recorded.
Both pointers now move inward, to index 3 holding `0` and index 4 holding `1`. The left skip compares
`nums[3]` with `nums[2]`, which is `0` against `-1`, so no skip happens; the right skip compares
`nums[4]` with `nums[5]`, which is `1` against `2`, so no skip happens either. The pair `0 + 1 = 1` is a
second hit, so `[-1, 0, 1]` is recorded. At `i = 2` the fixed value is `-1`
again and `nums[2] == nums[1]`, so the whole index is skipped — without that skip, both triples would be
emitted a second time. At `i = 3` the fixed value is `0`, which is not positive, so the scan runs over
`[1, 2]` with target `0` and finds nothing. The result is two triples.

```python
def three_sum(nums):
    nums.sort()
    out = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue                              ## skip a repeated FIXED value
        if nums[i] > 0:
            break                                 ## sorted: no triple of positives sums to 0
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                out.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1                     ## skip repeats on the LEFT pointer
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1                    ## skip repeats on the RIGHT pointer
    return out

## tests

assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
assert three_sum([1, 2, 3]) == []
assert three_sum([-2, -1, -1, 0, 1, 1, 2]) == [[-2, 0, 2], [-2, 1, 1], [-1, -1, 2], [-1, 0, 1]]
print(three_sum([-1, 0, 1, 2, -1, -4]))
```

```
[[-1, -1, 2], [-1, 0, 1]]
```

## The problems

### P1. Valid palindrome — is a string a palindrome when non-alphanumeric characters are ignored and case is dropped

**Which template.** Template 1, converging, with a skip loop on each side.
**The trick.** Two inner `while` loops advance each pointer past junk before any comparison. Both inner
loops need the `left < right` guard as well, because a string of pure punctuation would otherwise run a
pointer off the end. Compare `lower()` on both sides, never on one.

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1                             ## skip junk from the left
        while left < right and not s[right].isalnum():
            right -= 1                            ## skip junk from the right
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

## tests

assert is_palindrome("A man, a plan, a canal: Panama") is True
assert is_palindrome("race a car") is False
assert is_palindrome(" ") is True
assert is_palindrome(".,") is True
print(is_palindrome("A man, a plan, a canal: Panama"), is_palindrome("race a car"))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(1)$ space. Building a filtered copy first is $O(n)$ space and is the
answer the interviewer is testing you against.

### P2. Reverse a string in place — reverse a list of characters without allocating a second list

**Which template.** Template 1, converging, swapping instead of comparing.
**The trick.** The loop condition is `left < right`, not `left <= right`. With an odd length the middle
character is its own mirror, so swapping it with itself is harmless but pointless, and with `<=` on an
even length the two pointers cross and undo the swap you just made.

```python
def reverse_string(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return chars

## tests

a = list("hello")
assert reverse_string(a) == list("olleh")
assert reverse_string(list("ab")) == list("ba")
assert reverse_string(list("x")) == list("x")
assert reverse_string([]) == []
print("".join(reverse_string(list("hello"))))
```

```
olleh
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P3. Two sum on a sorted array — return the indices of the two values that add to `target`

**Which template.** Template 1 exactly, unmodified.
**The trick.** The discard argument is the whole answer, so say it out loud. If the current sum is below
the target, then `nums[left]` paired with anything at or below `right` is also below the target, because
the array is sorted. Therefore no pair using `left` can work, and `left` is discarded in one step. The
symmetric argument applies on the right. Each step removes one index permanently, so the scan is linear.

```python
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1                             ## discard left: no pair with it can reach target
        else:
            right -= 1                            ## discard right: every pair with it overshoots
    return [-1, -1]

## tests

assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]
assert two_sum_sorted([-3, -1, 0, 4], 3) == [1, 3]
assert two_sum_sorted([1, 2], 99) == [-1, -1]
print(two_sum_sorted([-3, -1, 0, 4], 3))
```

```
[1, 3]
```

**Complexity.** $O(n)$ time, $O(1)$ space. On an unsorted array use a hash map instead, which is $O(n)$
time and $O(n)$ space, because sorting to enable the pointers would destroy the original indices.

### P4. 3Sum — all unique triples that sum to zero

**Which template.** Template 1 inside one loop, with the three duplicate skips.
**The trick.** Sort, fix the first element, and run a converging scan for the remaining pair. The
duplicate handling is the graded part, and there are exactly three skips as set out above. The
`nums[i] > 0` break is a small extra: once the fixed value is positive every later value is positive
too, so no triple can reach zero.

```python
def three_sum(nums):
    nums.sort()
    out = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue                              ## skip a repeated FIXED value
        if nums[i] > 0:
            break
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                out.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
    return out

## tests

assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
assert three_sum([1, 2, 3]) == []
assert three_sum([-2, 0, 1, 1, 2]) == [[-2, 0, 2], [-2, 1, 1]]
print(three_sum([-2, 0, 1, 1, 2]))
```

```
[[-2, 0, 2], [-2, 1, 1]]
```

**Complexity.** $O(n^2)$ time — one loop times one linear scan — plus $O(n \log n)$ for the sort, so
$O(n^2)$ overall, and $O(1)$ space beyond the output.

### P5. 3Sum closest — the triple sum nearest to `target`

**Which template.** Template 1 inside one loop, with no duplicate skipping at all.
**The trick.** You return a number, not a set of triples, so duplicates cost nothing and the skip lines
are dead weight. The move rule is unchanged, because `total < target` still means only a larger left
value can help. Record `best` before the move test, and return early on an exact hit because nothing can
beat a distance of zero.

```python
def three_sum_closest(nums, target):
    nums.sort()
    n = len(nums)
    best = nums[0] + nums[1] + nums[2]
    for i in range(n - 2):
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if abs(total - target) < abs(best - target):
                best = total                      ## strictly closer, so replace
            if total == target:
                return total                      ## cannot beat an exact hit
            elif total < target:
                left += 1
            else:
                right -= 1
    return best

## tests

assert three_sum_closest([-1, 2, 1, -4], 1) == 2
assert three_sum_closest([0, 0, 0], 1) == 0
assert three_sum_closest([1, 1, 1, 0], -100) == 2
print(three_sum_closest([-1, 2, 1, -4], 1))
```

```
2
```

**Complexity.** $O(n^2)$ time, $O(1)$ space.

### P6. 4Sum — all unique quadruples summing to `target`

**Which template.** Template 1 inside two loops. This is the reduction stated in full.
**The trick.** The second loop needs its own duplicate skip, and the guard is `j > i + 1`, not `j > 0`.
Using `j > 0` would skip the first inner value whenever it happens to equal `nums[i]`, which silently
loses quadruples such as `[2, 2, 2, 2]`. Precompute `need = target - nums[i] - nums[j]` so the inner
scan is an ordinary two-sum.

```python
def four_sum(nums, target):
    nums.sort()
    n, out = len(nums), []
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue                              ## repeated OUTER fixed value
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue                          ## repeated INNER fixed value
            left, right = j + 1, n - 1
            need = target - nums[i] - nums[j]
            while left < right:
                total = nums[left] + nums[right]
                if total < need:
                    left += 1
                elif total > need:
                    right -= 1
                else:
                    out.append([nums[i], nums[j], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
    return out

## tests

assert four_sum([1, 0, -1, 0, -2, 2], 0) == [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
assert four_sum([2, 2, 2, 2, 2], 8) == [[2, 2, 2, 2]]
assert four_sum([1, 2, 3], 6) == []
print(four_sum([1, 0, -1, 0, -2, 2], 0))
```

```
[[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
```

**Complexity.** $O(n^3)$ time, $O(1)$ space beyond the output. The general k-sum is $O(n^{k-1})$.

### P7. Container with most water — pick two lines so the water they hold between them is greatest

**Which template.** Template 1, converging, with a greedy discard rule instead of a sum comparison.
**The trick.** Start at the two ends, which is the widest possible container, and always move the
**shorter** wall inward. Here is why that is safe. The area is `width * min(left_height, right_height)`,
so it is capped by the shorter wall. Consider the shorter wall and any partner to its inside. That
container is narrower, because the width shrank, and its height is still at most the shorter wall,
because the minimum cannot exceed either wall. Therefore every remaining container using the shorter
wall is no better than the one just measured, and the shorter wall can be discarded with nothing lost.
Moving the taller wall has no such guarantee, which is why the rule is one-directional.

```python
def max_area(height):
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        width = right - left
        best = max(best, width * min(height[left], height[right]))
        if height[left] < height[right]:
            left += 1                             ## move the SHORTER wall inward
        else:
            right -= 1
    return best

## tests

assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
assert max_area([1, 1]) == 1
assert max_area([4, 3, 2, 1, 4]) == 16
assert max_area([1, 2, 1]) == 2
print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))
```

```
49
```

**Complexity.** $O(n)$ time, $O(1)$ space. The brute force over all pairs is $O(n^2)$.

### P8. Trapping rain water — total water held by an elevation map

**Which template.** Template 1, converging, carrying a running maximum on each side.
**The trick.** Water above column `i` is `min(max_to_the_left, max_to_the_right) - height[i]`. The
easiest way to derive that is two prefix arrays: one pass right to build `left_max`, one pass left to
build `right_max`, then one pass to sum the differences. That version is correct and easy to explain,
but it uses $O(n)$ space. The two-pointer version removes the arrays by noticing that you only need the
**smaller** of the two maxima, and you always know which side that is. If `left_max <= right_max`, then
whatever lies to the right can only be higher, so `left_max` is already the binding wall for the left
column, and the water there is fixed. Process that column and move `left` in. The mirror case moves
`right`. So the running maxima replace the arrays entirely.

```python
def trap(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max <= right_max:                 ## the left side is the binding wall
            left += 1
            left_max = max(left_max, height[left])
            water += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += right_max - height[right]
    return water

## tests

assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
assert trap([3, 3, 3]) == 0
assert trap([]) == 0
print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), trap([4, 2, 0, 3, 2, 5]))
```

```
6 9
```

**Complexity.** $O(n)$ time, $O(1)$ space. Derive the $O(n)$-space prefix-max version first if you need
to, then say you can remove the arrays, and then do it.

### P9. Remove duplicates from a sorted array — keep one copy of each value, in place, and return the new length

**Which template.** Template 2, the write-pointer form.
**The trick.** Compare against the last **kept** value, `nums[write - 1]`, not against the previous read
value `nums[read - 1]`. The two agree here because the array is sorted, but the `write - 1` form is the
one that generalises to "keep at most two copies", where you compare against `nums[write - 2]`. Start
`write` at 1, because the first element is always kept.

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    write = 1                                     ## nums[0] is always kept
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:         ## compare against the last KEPT value
            nums[write] = nums[read]
            write += 1
    return write

## tests

a = [1, 1, 2]
k = remove_duplicates(a)
assert k == 2 and a[:k] == [1, 2]
b = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k2 = remove_duplicates(b)
assert k2 == 5 and b[:k2] == [0, 1, 2, 3, 4]
assert remove_duplicates([]) == 0
print(k2, b[:k2])
```

```
5 [0, 1, 2, 3, 4]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P10. Remove element — delete every occurrence of `val` in place and return the new length

**Which template.** Template 2, in its purest form.
**The trick.** This is `keep_if` with the predicate `x != val`. Nothing else is going on, and saying
that in one sentence is the whole answer. Because `write <= read` always holds, the write never lands on
an element that has not been read yet, so a plain copy is safe and no swap is needed.

```python
def remove_element(nums, val):
    write = 0
    for read in range(len(nums)):
        if nums[read] != val:                     ## keep everything that is not val
            nums[write] = nums[read]
            write += 1
    return write

## tests

a = [3, 2, 2, 3]
k = remove_element(a, 3)
assert k == 2 and a[:k] == [2, 2]
b = [0, 1, 2, 2, 3, 0, 4, 2]
k2 = remove_element(b, 2)
assert k2 == 5 and b[:k2] == [0, 1, 3, 0, 4]
assert remove_element([1, 1], 1) == 0
print(k2, b[:k2])
```

```
5 [0, 1, 3, 0, 4]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P11. Move zeroes — push every zero to the end while keeping the order of the other values

**Which template.** Template 2, but with a swap in place of a copy.
**The trick.** A copy would leave stale values in the tail, and the question requires the tail to be
zeros. Swapping puts the zero that `write` was holding into the slot `read` just vacated, so the zeros
accumulate at the back for free and no second pass is needed. Everything else is identical to P10.

```python
def move_zeroes(nums):
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]   ## swap, do not overwrite
            write += 1
    return nums

## tests

assert move_zeroes([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0]
assert move_zeroes([0]) == [0]
assert move_zeroes([1, 2, 3]) == [1, 2, 3]
assert move_zeroes([0, 0, 1]) == [1, 0, 0]
print(move_zeroes([0, 1, 0, 3, 12]))
```

```
[1, 3, 12, 0, 0]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P12. Sort colours — sort an array of 0s, 1s and 2s in one pass, in place

**Which template.** Template 2 extended to three pointers: the Dutch national flag partition.
**The trick.** Keep three indices. Everything before `low` is 0, everything from `low` to `mid - 1` is
1, everything after `high` is 2, and the region from `mid` to `high` is unexamined. The asymmetry is the
graded detail: after swapping a 0 into place you advance `mid`, because the value swapped in came from
the already-scanned 1-region and must be a 1. After swapping a 2 to the back you do **not** advance
`mid`, because the value swapped in came from the unexamined region and has not been looked at. The loop
condition is `mid <= high`, with the equality, because the element at `high` is still unexamined.

```python
def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1                              ## the swapped-in value is already scanned
        elif nums[mid] == 1:
            mid += 1                              ## 1s belong where they are
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1                             ## do NOT advance mid: the new value is unseen
    return nums

## tests

assert sort_colors([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2]
assert sort_colors([2, 0, 1]) == [0, 1, 2]
assert sort_colors([0]) == [0]
assert sort_colors([2, 2, 2, 0, 0]) == [0, 0, 2, 2, 2]
print(sort_colors([2, 0, 2, 1, 1, 0]))
```

```
[0, 0, 1, 1, 2, 2]
```

**Complexity.** $O(n)$ time, one pass, $O(1)$ space. Counting sort takes two passes and is also
acceptable, but the interviewer usually asks for one.

### P13. Merge sorted array in place — merge `nums2` into `nums1`, which has `n` empty slots at the end

**Which template.** Template 3 over two sequences, run **backwards**.
**The trick.** Walking forwards would overwrite entries of `nums1` that have not been merged yet.
Walking backwards fixes that completely, because the write pointer starts at the very last slot and the
free space is always in front of it: `write` is never smaller than `i`, so it can never clobber an
unread value. That single reversal is the entire problem. Loop while `j >= 0` only, because once
`nums2` is exhausted the remaining entries of `nums1` are already in their correct places.

```python
def merge_in_place(nums1, m, nums2, n):
    i, j, write = m - 1, n - 1, m + n - 1         ## all three walk BACKWARDS
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[write] = nums1[i]
            i -= 1
        else:
            nums1[write] = nums2[j]
            j -= 1
        write -= 1
    return nums1                                  ## nums1[:i+1] is untouched and already correct

## tests

assert merge_in_place([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3) == [1, 2, 2, 3, 5, 6]
assert merge_in_place([1], 1, [], 0) == [1]
assert merge_in_place([0], 0, [1], 1) == [1]
assert merge_in_place([4, 5, 6, 0, 0, 0], 3, [1, 2, 3], 3) == [1, 2, 3, 4, 5, 6]
print(merge_in_place([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3))
```

```
[1, 2, 2, 3, 5, 6]
```

**Complexity.** $O(m + n)$ time, $O(1)$ space.

### P14. Is subsequence — can `s` be obtained from `t` by deleting characters

**Which template.** Template 3 over two sequences, greedy.
**The trick.** Advance `j` on every step and advance `i` only on a match. The greedy choice of matching
each character of `s` as early as possible in `t` never loses a solution, because taking an earlier match
leaves a longer suffix of `t` for the rest of `s`, and a longer suffix contains every subsequence a
shorter one contains. The answer is `i == len(s)`, which reads as "all of `s` was consumed".

```python
def is_subsequence(s, t):
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1                                ## matched one character of s
        j += 1                                    ## t always advances
    return i == len(s)

## tests

assert is_subsequence("abc", "ahbgdc") is True
assert is_subsequence("axc", "ahbgdc") is False
assert is_subsequence("", "anything") is True
assert is_subsequence("abc", "") is False
print(is_subsequence("abc", "ahbgdc"), is_subsequence("axc", "ahbgdc"))
```

```
True False
```

**Complexity.** $O(|s| + |t|)$ time, $O(1)$ space. If many different `s` are checked against one fixed
`t`, precompute for each position of `t` the next occurrence of each letter and answer each query in
$O(|s| \log |t|)$ or better.

### P15. Backspace string compare — do two strings become equal after `#` deletes the previous character

**Which template.** Template 3 over two sequences, both walked from the **back**.
**The trick.** A `#` deletes the character before it, so the effect of a backspace is only known once you
have seen it, which means forwards is the wrong direction. From the back you can carry a `skips` counter:
a `#` increases it, and any ordinary character is deleted while `skips` is positive. The helper returns
the index of the next surviving character, or `-1`. The comparison then needs both an equality check and
a length check, because one string running out before the other must be a mismatch.

```python
def backspace_compare(s, t):
    def prev_valid(string, index):
        skips = 0
        while index >= 0:
            if string[index] == "#":
                skips += 1                        ## one more character to delete
            elif skips > 0:
                skips -= 1                        ## this character is deleted
            else:
                return index                      ## a surviving character
            index -= 1
        return -1

    i, j = len(s) - 1, len(t) - 1
    while i >= 0 or j >= 0:
        i = prev_valid(s, i)
        j = prev_valid(t, j)
        if i >= 0 and j >= 0 and s[i] != t[j]:
            return False
        if (i >= 0) != (j >= 0):                  ## one string ran out before the other
            return False
        i -= 1
        j -= 1
    return True

## tests

assert backspace_compare("ab#c", "ad#c") is True
assert backspace_compare("ab##", "c#d#") is True
assert backspace_compare("a#c", "b") is False
assert backspace_compare("bxj##tw", "bxo#j##tw") is True
print(backspace_compare("ab#c", "ad#c"), backspace_compare("a#c", "b"))
```

```
True False
```

**Complexity.** $O(|s| + |t|)$ time, $O(1)$ space. Building both strings with a stack is $O(n)$ space
and is the version to mention before you give this one.

### P16. Squares of a sorted array — square every value of a sorted array and return the result sorted

**Which template.** Template 1, converging, writing into the output from the back.
**The trick.** The array is sorted but may contain negatives, so the largest square is at one end or the
other, never in the middle. Compare absolute values at the two ends, take the bigger, and place it at
the last free slot of the output. Filling from the back is what makes one pass enough: the largest
element is found first, so it must be written last.

```python
def sorted_squares(nums):
    n = len(nums)
    out = [0] * n
    left, right = 0, n - 1
    for write in range(n - 1, -1, -1):            ## fill from the BACK, largest first
        if abs(nums[left]) > abs(nums[right]):
            out[write] = nums[left] * nums[left]
            left += 1
        else:
            out[write] = nums[right] * nums[right]
            right -= 1
    return out

## tests

assert sorted_squares([-4, -1, 0, 3, 10]) == [0, 1, 9, 16, 100]
assert sorted_squares([-7, -3, 2, 3, 11]) == [4, 9, 9, 49, 121]
assert sorted_squares([1]) == [1]
assert sorted_squares([]) == []
print(sorted_squares([-4, -1, 0, 3, 10]))
```

```
[0, 1, 9, 16, 100]
```

**Complexity.** $O(n)$ time, $O(n)$ space for the output. Squaring and calling `sort()` is
$O(n \log n)$ and is what you are being asked to beat.

### P17. Linked list cycle — does a linked list contain a cycle

**Which template.** Template 4, phase 1 only. The `ListNode`, `build` and `to_list` helpers come from
that template block.
**The trick.** Move `slow` one node and `fast` two nodes per step. If there is no cycle, `fast` reaches
the end and the loop exits. If there is a cycle, both pointers end up inside it, and the gap between them
shrinks by exactly one node per step, so it must reach zero and they must meet. That "gap shrinks by one"
argument is the proof, and it is short enough to say in the interview. Guard both `fast` and `fast.next`
before the double step.

```python
## prelude: the linked-list helpers from the templates section
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next                          ## 1 step
        fast = fast.next.next                     ## 2 steps
        if slow is fast:                          ## the gap closes by 1 each step, so they meet
            return True
    return False

## tests

assert has_cycle(build([3, 2, 0, -4], cycle_at=1)) is True
assert has_cycle(build([1, 2], cycle_at=0)) is True
assert has_cycle(build([1, 2, 3])) is False
assert has_cycle(build([])) is False
print(has_cycle(build([3, 2, 0, -4], cycle_at=1)), has_cycle(build([1, 2, 3])))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(1)$ space. A visited-set solution is $O(n)$ space and is the version
this problem exists to rule out.

### P18. Find the start of the cycle — return the node where the cycle begins

**Which template.** Template 4, both phases.
**The trick.** Phase 2 is the part to memorise. After the meeting, put one pointer back at the head,
leave the other at the meeting point, and advance both **one** node at a time. They meet at the cycle
entry. Use `is` for the comparison, not `==`, because you compare node identity and two different nodes
may hold equal values.

```python
## prelude: the linked-list helpers from the templates section
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def detect_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:                          ## phase 1: find any meeting point
            walker = head
            while walker is not slow:             ## phase 2: both move ONE step
                walker = walker.next
                slow = slow.next
            return walker                         ## they meet at the cycle entry
    return None

## tests

assert detect_cycle(build([3, 2, 0, -4], cycle_at=1)).val == 2
assert detect_cycle(build([1, 2], cycle_at=0)).val == 1
assert detect_cycle(build([1], cycle_at=0)).val == 1
assert detect_cycle(build([1, 2, 3])) is None
print(detect_cycle(build([3, 2, 0, -4], cycle_at=1)).val)
```

```
2
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P19. Middle of a linked list — return the middle node in one pass

**Which template.** Template 4's speed ratio, without the cycle logic.
**The trick.** `fast` covers twice the distance of `slow`, so when `fast` reaches the end `slow` is at
the halfway point. Confirm the tie-breaking rule with the interviewer before coding: as written, an
even-length list returns the **second** middle, because `fast` takes one more step. Starting `fast` at
`head.next` instead returns the first middle, and that variant is what P21 needs.

```python
## prelude: the linked-list helpers from the templates section
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def middle_node(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow                                   ## fast covers 2x, so slow lands at the middle

## tests

assert middle_node(build([1, 2, 3, 4, 5])).val == 3
assert middle_node(build([1, 2, 3, 4, 5, 6])).val == 4      ## second middle when even
assert middle_node(build([1])).val == 1
assert middle_node(build([])) is None
print(middle_node(build([1, 2, 3, 4, 5])).val, middle_node(build([1, 2, 3, 4, 5, 6])).val)
```

```
3 4
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P20. Remove the nth node from the end — delete the nth-from-last node in one pass

**Which template.** Two same-direction pointers at a **fixed gap**, not a fixed speed ratio.
**The trick.** Advance `lead` by `n` nodes first, then advance both until `lead.next` is `None`. The gap
of `n` guarantees that `trail` stops on the node just **before** the one to delete, which is what a
single-link deletion needs. Start both at a `dummy` node in front of the head, because removing the head
itself is otherwise a special case with its own bug.

```python
## prelude: the linked-list helpers from the templates section
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)                     ## dummy handles "remove the head"
    lead, trail = dummy, dummy
    for _ in range(n):
        lead = lead.next                          ## open a gap of exactly n
    while lead.next:
        lead = lead.next
        trail = trail.next                        ## trail stops BEFORE the target
    trail.next = trail.next.next
    return dummy.next

## tests

assert to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
assert to_list(remove_nth_from_end(build([1]), 1)) == []
assert to_list(remove_nth_from_end(build([1, 2]), 2)) == [2]
assert to_list(remove_nth_from_end(build([1, 2]), 1)) == [1]
print(to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)))
```

```
[1, 2, 3, 5]
```

**Complexity.** $O(n)$ time, one pass, $O(1)$ space.

### P21. Palindrome linked list — is a singly linked list a palindrome, in $O(1)$ space

**Which template.** Template 4 to find the middle, then an in-place reversal, then template 3 to
compare. It is the only problem here that needs all three moves.
**The trick.** You cannot walk a singly linked list backwards, so you make a backwards half instead:
find the middle with fast and slow, reverse from the middle onward, then walk the original head and the
reversed tail together. Drive the comparison loop off the **reversed** half, because on an odd length
that half is the shorter one and the extra middle node is correctly ignored. The triple assignment
`slow.next, prev, slow = prev, slow, slow.next` is the whole reversal; write it as one line and check
the right-hand side is evaluated before any assignment happens.

```python
## prelude: the linked-list helpers from the templates section
class ListNode:
    def __init__(self, val, nxt=None):
        self.val, self.next = val, nxt

def build(values, cycle_at=-1):
    head, nodes = None, []
    for v in reversed(values):
        head = ListNode(v, head)
    node = head
    while node:
        nodes.append(node)
        node = node.next
    if cycle_at >= 0 and nodes:
        nodes[-1].next = nodes[cycle_at]
    return head

def to_list(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out

def is_palindrome_list(head):
    slow, fast = head, head
    while fast and fast.next:                     ## 1. find the middle
        slow = slow.next
        fast = fast.next.next
    prev = None
    while slow:                                   ## 2. reverse the second half
        slow.next, prev, slow = prev, slow, slow.next
    first, second = head, prev
    while second:                                 ## 3. compare; the reversed half is shorter
        if first.val != second.val:
            return False
        first = first.next
        second = second.next
    return True

## tests

assert is_palindrome_list(build([1, 2, 2, 1])) is True
assert is_palindrome_list(build([1, 2, 3, 2, 1])) is True
assert is_palindrome_list(build([1, 2])) is False
assert is_palindrome_list(build([1])) is True
assert is_palindrome_list(build([])) is True
print(is_palindrome_list(build([1, 2, 2, 1])), is_palindrome_list(build([1, 2])))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(1)$ space. The list is left modified, so say that you would reverse the
second half back before returning if the caller needs the list intact.

### P22. Partition an array around a pivot — rearrange so every value below `pivot` comes first

**Which template.** Template 2, with a swap, and it is the partition step of quicksort.
**The trick.** This is `move_zeroes` with the predicate `x < pivot` instead of `x != 0`. The return value
is `write`, the count of values below the pivot, which is also the index where the second region starts.
The order **within** each region is not preserved, because swapping moves an element from the far side
into the near one. Say that out loud: if the question wants stable partitioning, this solution is wrong
and you need $O(n)$ extra space.

```python
def partition(nums, pivot):
    write = 0
    for read in range(len(nums)):
        if nums[read] < pivot:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
    return write                                  ## nums[:write] < pivot <= nums[write:]

## tests

a = [9, 12, 3, 5, 14, 10, 10]
k = partition(a, 10)
assert k == 3 and all(x < 10 for x in a[:k]) and all(x >= 10 for x in a[k:])
assert partition([1, 2, 3], 0) == 0
assert partition([1, 2, 3], 99) == 3
print(k, a)
```

```
3 [9, 3, 5, 12, 14, 10, 10]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P23. Longest palindromic substring — the longest substring that reads the same both ways

**Which template.** Converging pointers run **outward** instead of inward, from every centre.
**The trick.** This is the contrast worth learning. Everywhere else on this page two pointers start apart
and close; here they start together and open. A palindrome is defined by a centre, so try all of them.
There are `2n - 1` centres, not `n`, because a palindrome of even length has its centre between two
characters — that is what the second `expand(centre, centre + 1)` call covers, and forgetting it is why
`"cbbd"` returns `"b"` instead of `"bb"`. When expansion stops, the last valid span is
`[left + 1, right - 1]`, so its length is `right - left - 1`.

```python
def longest_palindrome(s):
    if not s:
        return ""
    best_start, best_len = 0, 1

    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1                             ## grow OUTWARD: the mirror of converging
            right += 1
        return left + 1, right - left - 1         ## start and length of the last valid span

    for centre in range(len(s)):
        for start, length in (expand(centre, centre), expand(centre, centre + 1)):
            if length > best_len:
                best_start, best_len = start, length
    return s[best_start:best_start + best_len]

## tests

assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("a") == "a"
assert longest_palindrome("") == ""
assert longest_palindrome("forgeeksskeegfor") == "geeksskeeg"
print(longest_palindrome("babad"), longest_palindrome("forgeeksskeegfor"))
```

```
bab geeksskeeg
```

**Complexity.** $O(n^2)$ time — `n` centres times $O(n)$ expansion — and $O(1)$ space. Manacher's
algorithm is $O(n)$; name it, then write this one.

## Tricks and tips

**Sort first, and say why.** Converging pointers need monotonicity, and sorting is the cheapest way to
create it. The sort costs $O(n \log n)$, which is usually free next to the $O(n^2)$ that a k-sum needs
anyway. However, sorting destroys the original indices. If the answer must be indices rather than
values, as in unsorted Two Sum, sorting is not available and a hash map is the correct pattern.

**`while left < right`, never `<=`, for a converging pair.** With `<=` a pair problem will pair an
element with itself, and a swap problem will undo its own last swap. The only converging loop that uses
`<=` is the Dutch national flag partition, and it does so because `high` marks unexamined territory
rather than a partner index.

**In the write-pointer template, the answer is the length.** The function returns `write`, and the
caller reads `nums[:write]`. Everything from `write` onward is stale. State that in one sentence when you
finish coding, because the interviewer is checking that you know the tail is not cleared.

**Copy when you overwrite only stale data; swap when the displaced value matters.** Remove Element can
copy, because a value at or before `write` has already been read and either kept or rejected. Move
Zeroes must swap, because the zeros it displaces are required output. Partition must swap for the same
reason. Ask which case you are in before you type the line.

**When forwards would clobber unread data, go backwards.** Merge Sorted Array is the standard example:
the free space sits at the end, so the write pointer must start there. Squares of a Sorted Array uses the
same reversal for a different reason — the largest value is found first, so it must be written last.
Backspace String Compare goes backwards because the meaning of a character depends on what follows it.
Whenever a forward pass fights the data, try the reverse before adding an auxiliary array.

**A fixed gap and a fixed speed ratio are different tools.** A gap of `n` locates the nth node from the
end. A 2:1 speed ratio locates the middle and detects cycles. Both are same-direction pointers, and
mixing them up produces a solution that almost works.

**Use `is`, not `==`, on nodes.** Cycle detection compares identity. Two distinct nodes can hold the
same value, and `==` on them would report a cycle that does not exist.

**A dummy head removes a whole class of bugs.** Any linked list problem that can delete or replace the
first node should start with `dummy = ListNode(0, head)` and return `dummy.next`. It costs one line and
removes every "what if the target is the head" branch.

**Name the brute force and its cost before you start.** For every problem here the brute force is a
double loop, or a sort, or an auxiliary array. Say what it costs, then say which property of the input —
sortedness, positivity, a known alphabet — lets two pointers beat it. That sentence is what separates a
memorised solution from an understood one.

## The bugs that cost the round

**Running converging pointers on unsorted data.** This produces no error and no crash. It returns a
wrong answer that looks plausible, and it passes the first small test case often enough to survive until
the interviewer supplies a second one. Before writing a converging loop, confirm the array is sorted, or
sort it yourself.

**Getting the 3Sum duplicate skips wrong.** There are three of them and each has a different index. The
fixed-index skip compares `nums[i]` with `nums[i - 1]` and looks backwards, because `i` has not been used
yet. The two pointer skips run **after** a recorded hit and after both pointers have moved, so they
compare `nums[left]` with `nums[left - 1]` and `nums[right]` with `nums[right + 1]`. Skipping before the
move, or skipping on the wrong side, either emits duplicate triples or loses real ones.

**Forgetting `left < right` inside a skip loop.** In Valid Palindrome the inner loops that skip
punctuation must carry the same bound as the outer loop. Without it, a string of pure punctuation walks a
pointer past the end and raises an `IndexError`.

**Advancing `mid` after a swap with `high` in the Dutch national flag.** The value swapped in came from
the unexamined region, so it must be inspected. Advancing past it drops a 0 or a 1 into the 2-region.

**Merging two sorted arrays forwards when the target is one of them.** The write pointer overtakes the
read pointer and destroys values that have not been merged. Walk backwards.

**No `fast.next` guard before `fast.next.next`.** A list with an even number of nodes and no cycle will
raise an `AttributeError` on `None`. The condition is `while fast and fast.next`, both parts, every time.

**Testing only odd or only even lengths.** Middles, palindromes and reversals all behave differently on
the two parities. Every function on this page has at least one test of each, and that habit catches more
bugs than any amount of rereading.

## Done when

- Given a problem statement you have not seen, you can say in under 30 seconds which of the three
  techniques it needs — converging, write pointer, or fast and slow on nodes — and name the property of
  the input that makes it correct.
- You can write 3Sum from a blank file in five minutes, including all three duplicate skips, and explain
  why each compares the index it does.
- You can state the container-with-most-water discard argument and the trapping-rain-water binding-wall
  argument out loud, without code, in two sentences each.
- You can write Floyd's cycle detection with both phases from memory, and say why the second phase starts
  one pointer at the head and moves both by one node.
