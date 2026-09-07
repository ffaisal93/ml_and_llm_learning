# Heap and priority queue: every variation

A heap is a binary tree kept in an array, with the rule that every parent compares less than or equal to
its children. That one rule is enough to give you the smallest item in $O(1)$ and to remove it or add a
new item in $O(\log n)$. It does not give you the second smallest, or the median, or any position in the
middle. Therefore a heap is the correct structure whenever you need the extreme item again and again,
and never need the full sorted order.

The thing that makes heaps hard is a counter-intuitive inversion. To keep the k **largest** items you
use a **min**-heap of size k. The reason is that the item you must throw away is the smallest of the
ones you are currently keeping, so the root must be that smallest item. To keep the k **smallest** items
you use a **max**-heap of size k, for the mirror reason. Getting this backwards is the single most
common failure with heaps, so say the rule out loud before you type: the heap holds the survivors, and
its root is the next survivor to be evicted.

One more practical fact. Python's `heapq` is a min-heap only. You get a max-heap by pushing negated
values and negating again on the way out. That works, and it is also a real source of sign bugs, so
negate at exactly two points and nowhere else.

## Recognising it from the phrasing

| The interviewer says | They mean | Structure | Cost |
|---|---|---|---|
| "kth largest / kth smallest" | a heap of size k, or quickselect | min-heap of size k (or max-heap for kth smallest) | $O(n \log k)$, or $O(n)$ average |
| "top k frequent / most common k" | count first, then select | heap of size k, or bucket sort | $O(n \log k)$, or $O(n)$ |
| "merge k sorted lists / arrays / streams" | a heap of the k current heads | min-heap of size k | $O(N \log k)$ |
| "median of a stream", "running median" | two heaps | max-heap of the low half, min-heap of the high half | $O(\log n)$ per insert |
| "schedule", "always take the most urgent / largest" | greedy simulation | max-heap | $O(n \log n)$ |
| "k closest points / k nearest" | keep the k best by distance | max-heap of size k, keyed on distance | $O(n \log k)$ |
| "minimum cost to combine / connect / merge" | repeatedly join the two cheapest | min-heap, the Huffman shape | $O(n \log n)$ |
| "kth smallest in a sorted matrix / sorted sums" | heap over the frontier | min-heap of candidates | $O(k \log n)$ |

Before writing anything, ask one question: **do I need the full sorted order, or only the extreme item,
repeatedly, as the data changes?** If the data is fixed and you need every element in order, sort it
once for $O(n \log n)$ and stop. A heap wins in two situations only. The first is when items arrive over
time, because sorting cannot start until the data stops arriving, while a heap absorbs each new item in
$O(\log n)$. The second is when k is much smaller than n, because a size-k heap costs $O(n \log k)$
against $O(n \log n)$ for the full sort. Put numbers on it: for $n = 10^6$ and $k = 10$, $\log_2 k$ is
about 3.3 and $\log_2 n$ is about 20, so the heap does about six times less comparison work and holds
ten items instead of a million. For $k$ close to $n$ the two costs are the same and sorting is simpler,
so say so and sort.

## The templates

Templates 1 and 2 share one skeleton: push, then pop if the heap is too big. Only the sign changes.
Learn the skeleton once.

**Template 1 — min-heap of size k, for the k largest.** Use when you want the k largest items, or the
kth largest as the final root.

```python
import heapq

def k_largest(nums, k):
    heap = []                                    ## MIN-heap: the root is the weakest survivor
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)                  ## evict the smallest of the ones we keep
    return sorted(heap, reverse=True)            ## heap[0] alone is the kth largest

## tests

assert k_largest([3, 2, 1, 5, 6, 4], 2) == [6, 5]
assert k_largest([7, 7, 7], 2) == [7, 7]
assert k_largest([1], 5) == [1]
print(k_largest([3, 2, 1, 5, 6, 4], 2))
```

```
[6, 5]
```

**Template 2 — max-heap by negation, for the k smallest or for greedy "take the largest".** Negate on
the way in and on the way out, at those two points only.

```python
import heapq

def k_smallest(nums, k):
    heap = []                                    ## MAX-heap, simulated with negated values
    for x in nums:
        heapq.heappush(heap, -x)                 ## negation point 1: entering
        if len(heap) > k:
            heapq.heappop(heap)                  ## evict the largest of the ones we keep
    return sorted(-y for y in heap)              ## negation point 2: leaving

## tests

assert k_smallest([3, 2, 1, 5, 6, 4], 2) == [1, 2]
assert k_smallest([-1, -5, 0], 2) == [-5, -1]
assert k_smallest([4], 1) == [4]
print(k_smallest([3, 2, 1, 5, 6, 4], 2))
```

```
[1, 2]
```

**Template 3 — two heaps for a running median.** The low half in a max-heap, the high half in a
min-heap, kept within one element of each other. The answer is read from the two roots.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.low, self.high = [], []             ## low is negated (max-heap), high is a min-heap
    def add(self, x):
        heapq.heappush(self.low, -x)             ## always enter through low
        heapq.heappush(self.high, -heapq.heappop(self.low))   ## hand the largest to high
        if len(self.high) > len(self.low):       ## keep low the same size or one bigger
            heapq.heappush(self.low, -heapq.heappop(self.high))
    def median(self):
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2.0

## tests

mf = MedianFinder()
for value, expected in [(1, 1.0), (2, 1.5), (3, 2.0), (4, 2.5)]:
    mf.add(value)
    assert mf.median() == expected
print(mf.median())
```

```
2.5
```

**Template 4 — heap of heads, for merging k sorted sequences.** Push one entry per sequence, pop the
smallest, push that sequence's next item. Push the tuple `(value, index, item)` so that ties never
compare the items themselves.

```python
import heapq

def merge_sorted_lists(lists):
    heap, out = [], []
    for i, seq in enumerate(lists):
        if seq:
            heapq.heappush(heap, (seq[0], i, 0))       ## (value, list index, position)
    while heap:
        value, i, pos = heapq.heappop(heap)
        out.append(value)
        if pos + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][pos + 1], i, pos + 1))
    return out

## tests

assert merge_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]
assert merge_sorted_lists([[], []]) == []
assert merge_sorted_lists([[2], [1]]) == [1, 2]
print(merge_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]]))
```

```
[1, 1, 2, 3, 4, 4, 5, 6]
```

The second element of that tuple is not decoration. Python compares tuples element by element, so when
two values tie it moves on to the next element. If the tuple were `(value, node)` and `node` were a
linked-list node or any object without an ordering, the tie would raise
`TypeError: '<' not supported between instances of 'ListNode' and 'ListNode'`. The list index is a
unique integer, so the comparison always stops there and never reaches the object. Put the index in
every heap tuple whose payload is not a number, every time.

## The two-heap median

This is the highest-value heap trick, because a heap gives you an extreme and the median is the exact
opposite of an extreme. The idea is to split the data at the median and keep each half in the heap that
puts the median-side end at its root.

Keep a max-heap `low` holding the smaller half, and a min-heap `high` holding the larger half. Two
invariants hold at all times. Every item in `low` is at most every item in `high`. And `len(low)` equals
`len(high)` or is exactly one more. Under those two invariants the median is `low[0]` when the total
count is odd, and the average of `low[0]` and `high[0]` when it is even.

The insert is three lines and always the same three. Push the new item into `low`. Move `low`'s root
into `high`, which enforces the ordering invariant because whatever is now leaving `low` is the largest
of the low side. Then, if `high` has grown bigger than `low`, move `high`'s root back into `low`, which
enforces the size invariant. Doing it in that fixed order means you never need a case analysis on where
the new value belongs.

**Worked example.** Insert 5, 15, 1, 3, 8 in that order. Heaps are written as sets with the root first.

| Step | Action | `low` (max-heap) | `high` (min-heap) | Median |
|---|---|---|---|---|
| 1 | add 5 | `[5]` | `[]` | 5 |
| 2 | add 15 | `[5]` | `[15]` | 10.0 |
| 3 | add 1 | `[5, 1]` | `[15]` | 5 |
| 4 | add 3 | `[3, 1]` | `[5, 15]` | 4.0 |
| 5 | add 8 | `[5, 1, 3]` | `[8, 15]` | 5 |

Follow step 3 in detail. Before it, `low = [5]` and `high = [15]`. Push 1 into `low`, giving root 5.
Move that root 5 into `high`, giving `low = [1]` and `high = [5, 15]`. Now `high` is bigger, so move its
root 5 back into `low`, giving `low = [5, 1]` and `high = [15]`. Three items, `low` is the bigger heap,
so the median is its root, 5. That is correct: the sorted data is 1, 5, 15.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.low, self.high = [], []
    def addNum(self, num):
        heapq.heappush(self.low, -num)                        ## 1. always enter through low
        heapq.heappush(self.high, -heapq.heappop(self.low))   ## 2. ordering invariant
        if len(self.high) > len(self.low):                    ## 3. size invariant
            heapq.heappush(self.low, -heapq.heappop(self.high))
    def findMedian(self):
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2.0

## tests

mf = MedianFinder()
answers = []
for value in [5, 15, 1, 3, 8]:
    mf.addNum(value)
    answers.append(mf.findMedian())
assert answers == [5.0, 10.0, 5.0, 4.0, 5.0]
print(answers)
```

```
[5.0, 10.0, 5.0, 4.0, 5.0]
```

Step 2 is the even case: two items, one in each heap, so the median is the average of the two roots,
`(5 + 15) / 2 = 10.0`. Every even step reads both roots and every odd step reads only `low[0]`.

## The problems

### P1. Kth Largest Element in an Array — return the kth largest value, counting duplicates

**Which template.** Template 1, a min-heap of size k. Quickselect is the alternative and is worth
writing too.
**The trick.** The heap holds the k largest seen so far, so its root is the kth largest. Quickselect
instead partitions around a pivot and recurses into one side only, which gives $O(n)$ on average
because the work halves each round: $n + n/2 + n/4 + \dots = 2n$. Its worst case is $O(n^2)$ on a bad
pivot sequence, so a random pivot is part of the answer.

```python
import heapq
import random

def kth_largest_heap(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)                  ## the root is the smallest survivor
    return heap[0]

def kth_largest_quickselect(nums, k):
    values = list(nums)
    target = len(values) - k                     ## kth largest is index n-k when sorted ascending
    left, right = 0, len(values) - 1
    while True:
        pivot = values[random.randint(left, right)]
        low = [v for v in values[left:right + 1] if v < pivot]
        mid = [v for v in values[left:right + 1] if v == pivot]
        high = [v for v in values[left:right + 1] if v > pivot]
        values[left:right + 1] = low + mid + high
        if target < left + len(low):
            right = left + len(low) - 1
        elif target < left + len(low) + len(mid):
            return pivot
        else:
            left = left + len(low) + len(mid)

## tests

for nums, k, want in [([3, 2, 1, 5, 6, 4], 2, 5), ([3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4), ([1], 1, 1)]:
    assert kth_largest_heap(nums, k) == want
    assert kth_largest_quickselect(nums, k) == want
print(kth_largest_heap([3, 2, 1, 5, 6, 4], 2), kth_largest_quickselect([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))
```

```
5 4
```

**Complexity.** Heap: $O(n \log k)$ time, $O(k)$ space. Quickselect: $O(n)$ average and $O(n^2)$ worst
time, $O(n)$ space as written here.

### P2. Kth Largest Element in a Stream — a class that reports the kth largest after each new value

**Which template.** Template 1, held as class state.
**The trick.** This is the problem that proves why the size-k min-heap is right. The heap never grows
past k, so each `add` is $O(\log k)$ and the answer is always `heap[0]` with no search. A sorted list
would need $O(n)$ per insert, and a max-heap would put the wrong item at the root.

```python
import heapq

class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = list(nums)
        heapq.heapify(self.heap)                 ## heapify is O(n), better than n pushes
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

## tests

kl = KthLargest(3, [4, 5, 8, 2])
assert [kl.add(v) for v in [3, 5, 10, 9, 4]] == [4, 5, 5, 8, 8]
kl2 = KthLargest(1, [])
assert kl2.add(-3) == -3
print([KthLargest(3, [4, 5, 8, 2]).add(v) for v in [3]])
```

```
[4]
```

**Complexity.** $O(n)$ to build, $O(\log k)$ per `add`, $O(k)$ space.

### P3. Last Stone Weight — smash the two heaviest stones together until at most one remains

**Which template.** Template 2, a max-heap by negation.
**The trick.** "Always take the two largest" is the plain signal for a max-heap. Each smash removes two
stones and puts back at most one, so the loop runs at most n times. The only care needed is the sign:
push `-stone`, pop and negate, push back `-(difference)`.

```python
import heapq

def last_stone_weight(stones):
    heap = [-s for s in stones]                  ## negate once, on the way in
    heapq.heapify(heap)
    while len(heap) > 1:
        first = -heapq.heappop(heap)             ## heaviest
        second = -heapq.heappop(heap)            ## second heaviest
        if first != second:
            heapq.heappush(heap, -(first - second))
    return -heap[0] if heap else 0

## tests

assert last_stone_weight([2, 7, 4, 1, 8, 1]) == 1
assert last_stone_weight([1]) == 1
assert last_stone_weight([2, 2]) == 0
assert last_stone_weight([]) == 0
print(last_stone_weight([2, 7, 4, 1, 8, 1]))
```

```
1
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space.

### P4. K Closest Points to Origin — the k points with the smallest Euclidean distance to the origin

**Which template.** Template 1 inverted: a **max**-heap of size k keyed on distance, because you keep
the k smallest distances and evict the largest.
**The trick.** Do not take a square root. The square root is monotone, so ordering by
$x^2 + y^2$ gives the same answer and avoids floating-point noise entirely. Say this in the interview;
it is the detail interviewers listen for.

```python
import heapq

def k_closest(points, k):
    heap = []                                    ## MAX-heap of (negated distance, x, y)
    for x, y in points:
        distance = x * x + y * y                 ## no sqrt: it is monotone, so it changes nothing
        heapq.heappush(heap, (-distance, x, y))
        if len(heap) > k:
            heapq.heappop(heap)                  ## evict the farthest survivor
    return [[x, y] for _, x, y in heap]

## tests

assert sorted(k_closest([[1, 3], [-2, 2]], 1)) == [[-2, 2]]
assert sorted(k_closest([[3, 3], [5, -1], [-2, 4]], 2)) == [[-2, 4], [3, 3]]
assert k_closest([[0, 1]], 5) == [[0, 1]]
print(sorted(k_closest([[3, 3], [5, -1], [-2, 4]], 2)))
```

```
[[-2, 4], [3, 3]]
```

**Complexity.** $O(n \log k)$ time, $O(k)$ space.

### P5. Top K Frequent Elements — the k values that occur most often

**Which template.** Count with a dictionary, then template 1 on the counts.
**The trick.** The heap step is a min-heap of `(count, value)` of size k, so the root is the least
frequent survivor. However, there is a strictly better answer for this specific problem: bucket sort. A
count can never exceed n, so an array of n+1 buckets indexed by count holds every value, and reading it
from the top down gives the k most frequent in $O(n)$. That version appears in the arrays chapter; give
the heap first because it generalises, then offer the bucket version as the improvement.

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    counts = Counter(nums)
    heap = []                                    ## MIN-heap of (count, value), size k
    for value, count in counts.items():
        heapq.heappush(heap, (count, value))
        if len(heap) > k:
            heapq.heappop(heap)                  ## drop the least frequent survivor
    return [value for count, value in sorted(heap, reverse=True)]

## tests

assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert sorted(top_k_frequent([4, 4, 5, 5, 6], 2)) == [4, 5]
print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
```

```
[1, 2]
```

**Complexity.** $O(n + m \log k)$ time for m distinct values, $O(m)$ space. The bucket-sort version is
$O(n)$ time and $O(n)$ space.

### P6. Merge K Sorted Lists — merge k sorted linked lists into one sorted list

**Which template.** Template 4, a heap of the k current heads.
**The trick.** The heap holds one node per list, never more, so it stays at size k while the output
grows to N items. Each of the N pops costs $O(\log k)$. The tuple must be
`(value, unique_index, node)`, because two nodes with equal values would otherwise be compared directly
and raise a `TypeError`. The nodes are represented here as Python lists so the block runs standalone.

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def merge_k_lists(heads):
    heap = []
    for i, node in enumerate(heads):
        if node:
            heapq.heappush(heap, (node.val, i, node))     ## i breaks ties before nodes compare
    dummy = tail = ListNode()
    while heap:
        value, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next

## tests

assert to_list(merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])) == [1, 1, 2, 3, 4, 4, 5, 6]
assert to_list(merge_k_lists([])) == []
assert to_list(merge_k_lists([build([]), build([1])])) == [1]
print(to_list(merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])))
```

```
[1, 1, 2, 3, 4, 4, 5, 6]
```

**Complexity.** $O(N \log k)$ time for N total nodes, $O(k)$ space. See the linked-list chapter for the
divide-and-conquer merge, which has the same time bound and no heap.

### P7. Find Median from Data Stream — support `addNum` and `findMedian` on a growing stream

**Which template.** Template 3, the two heaps.
**The trick.** Explained in full above. The one line people get wrong is the rebalance test: it is
`len(high) > len(low)`, not `!=`, because `low` is allowed to be one larger and that is exactly what
makes the odd case readable off `low[0]`.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.low, self.high = [], []             ## low: max-heap (negated). high: min-heap
    def addNum(self, num):
        heapq.heappush(self.low, -num)
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):       ## >, not !=: low may be one bigger
            heapq.heappush(self.low, -heapq.heappop(self.high))
    def findMedian(self):
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2.0

## tests

mf = MedianFinder()
mf.addNum(1); assert mf.findMedian() == 1.0
mf.addNum(2); assert mf.findMedian() == 1.5
mf.addNum(3); assert mf.findMedian() == 2.0
mf2 = MedianFinder()
for v in [6, 10, 2, 6, 5, 0, 6, 3]:
    mf2.addNum(v)
assert mf2.findMedian() == 5.5
print(mf.findMedian(), mf2.findMedian())
```

```
2.0 5.5
```

**Complexity.** $O(\log n)$ per `addNum`, $O(1)$ per `findMedian`, $O(n)$ space.

### P8. Task Scheduler — least time to run all tasks when equal tasks must be `n` apart

**Which template.** Template 2, a max-heap, plus a cooling queue.
**The trick.** At every tick, run the task with the most remaining copies, because leaving a
high-count task for later is what forces idle time at the end. A task that has just run is not
available again until `time + n`, so park it in a plain queue holding `(ready_time, remaining_count)`
and move it back to the heap when the clock reaches `ready_time`. The heap decides *what* to run and
the queue decides *when* it may return.

```python
import heapq
from collections import Counter, deque

def least_interval(tasks, n):
    heap = [-c for c in Counter(tasks).values()]      ## max-heap of remaining counts
    heapq.heapify(heap)
    cooling = deque()                                 ## (ready_time, remaining_count)
    time = 0
    while heap or cooling:
        time += 1
        if heap:
            remaining = heapq.heappop(heap) + 1       ## one copy is used, count is negative
            if remaining < 0:
                cooling.append((time + n, remaining))
        if cooling and cooling[0][0] == time:
            heapq.heappush(heap, cooling.popleft()[1])
    return time

## tests

assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
assert least_interval(["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"], 2) == 10
assert least_interval(["A"], 3) == 1
print(least_interval(["A", "A", "A", "B", "B", "B"], 2))
```

```
8
```

**Complexity.** $O(T \log m)$ time for T total ticks and m distinct tasks, $O(m)$ space.

### P9. Reorganize String — rearrange the letters so that no two adjacent characters are equal

**Which template.** Template 2, a max-heap on the letter counts, popping **two** at a time.
**The trick.** Take the two most frequent remaining letters and place them next to each other. Because
they are different letters, the pair is safe, and because you always spend the largest count first, no
letter is ever left stranded at the end unless it was impossible from the start. The impossibility test
is `max_count > (len(s) + 1) // 2`, and it falls out of the algorithm anyway when one letter is left
over with a count above one.

```python
import heapq
from collections import Counter

def reorganize_string(s):
    heap = [(-c, ch) for ch, c in Counter(s).items()]
    heapq.heapify(heap)
    out = []
    while len(heap) > 1:
        count_a, char_a = heapq.heappop(heap)             ## the two most frequent remaining
        count_b, char_b = heapq.heappop(heap)
        out.append(char_a)
        out.append(char_b)
        if count_a + 1 < 0:
            heapq.heappush(heap, (count_a + 1, char_a))
        if count_b + 1 < 0:
            heapq.heappush(heap, (count_b + 1, char_b))
    if heap:
        count, char = heapq.heappop(heap)
        if count < -1:
            return ""                                     ## more than one copy left: impossible
        out.append(char)
    return "".join(out)

## tests

assert reorganize_string("aab") in ("aba",)
assert reorganize_string("aaab") == ""
assert reorganize_string("a") == "a"
result = reorganize_string("aaabbc")
assert all(result[i] != result[i + 1] for i in range(len(result) - 1))
print(reorganize_string("aab"), reorganize_string("aaab") == "", result)
```

```
aba True ababac
```

**Complexity.** $O(n \log \Sigma)$ time for alphabet size $\Sigma$, $O(\Sigma)$ space.

### P10. Design Twitter — post tweets, follow users, and return the 10 most recent tweets in a feed

**Which template.** Template 1, a min-heap of size 10 keyed on timestamp.
**The trick.** The feed is a merge of several time-sorted lists, so it is P6 in a costume. Keep a global
counter as the timestamp so ordering never depends on wall-clock time. Only the last 10 tweets of each
followed user can matter, so slice `[-10:]` before pushing and the work per call stays bounded by the
number of people followed.

```python
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)               ## user -> [(time, tweet_id), ...]
        self.following = defaultdict(set)
    def postTweet(self, user_id, tweet_id):
        self.time += 1
        self.tweets[user_id].append((self.time, tweet_id))
    def getNewsFeed(self, user_id):
        heap = []                                     ## MIN-heap of size 10 keeps the 10 NEWEST
        for person in self.following[user_id] | {user_id}:
            for stamp, tweet_id in self.tweets[person][-10:]:
                heapq.heappush(heap, (stamp, tweet_id))
                if len(heap) > 10:
                    heapq.heappop(heap)
        return [tweet_id for stamp, tweet_id in sorted(heap, reverse=True)]
    def follow(self, follower_id, followee_id):
        self.following[follower_id].add(followee_id)
    def unfollow(self, follower_id, followee_id):
        self.following[follower_id].discard(followee_id)

## tests

tw = Twitter()
tw.postTweet(1, 5)
assert tw.getNewsFeed(1) == [5]
tw.follow(1, 2)
tw.postTweet(2, 6)
assert tw.getNewsFeed(1) == [6, 5]
tw.unfollow(1, 2)
assert tw.getNewsFeed(1) == [5]
print(tw.getNewsFeed(1))
```

```
[5]
```

**Complexity.** $O(1)$ for `postTweet`, `follow` and `unfollow`. $O(f \log 10)$ for `getNewsFeed` with f
people followed, which is $O(f)$.

### P11. Minimum Cost to Connect Sticks — repeatedly join two sticks at a cost equal to their combined length

**Which template.** Template 1 as a plain min-heap, with no size limit. This is the Huffman shape.
**The trick.** Always join the two shortest sticks. The reason is that every join adds the combined
length to the total, and a stick joined early is counted again in every later join, so the sticks you
touch most often must be the short ones. A greedy over a min-heap is therefore optimal, and the same
argument is the whole of Huffman coding.

```python
import heapq

def connect_sticks(sticks):
    heap = list(sticks)
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        first = heapq.heappop(heap)                   ## the two SHORTEST
        second = heapq.heappop(heap)
        total += first + second                       ## the cost of this join
        heapq.heappush(heap, first + second)
    return total

## tests

assert connect_sticks([2, 4, 3]) == 14
assert connect_sticks([1, 8, 3, 5]) == 30
assert connect_sticks([5]) == 0
assert connect_sticks([]) == 0
print(connect_sticks([2, 4, 3]), connect_sticks([1, 8, 3, 5]))
```

```
14 30
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space.

### P12. Sort Characters By Frequency — rewrite the string with the most frequent characters first

**Which template.** Template 2, a max-heap of `(-count, char)`, drained completely.
**The trick.** There is no size-k trimming here, because you need every character, so the heap is
simply a sorting device. Say out loud that `sorted(counts.items(), key=...)` is the same $O(m \log m)$
and shorter; the heap is asked for because the interviewer wants the drain loop, not because it is
faster.

```python
import heapq
from collections import Counter

def frequency_sort(s):
    heap = [(-count, ch) for ch, count in Counter(s).items()]
    heapq.heapify(heap)
    out = []
    while heap:
        count, ch = heapq.heappop(heap)               ## most frequent remaining
        out.append(ch * (-count))                     ## negate once, here
    return "".join(out)

## tests

assert frequency_sort("tree") in ("eert", "eetr")
assert frequency_sort("cccaaa") in ("cccaaa", "aaaccc")
assert frequency_sort("Aabb") == "bbAa"
assert frequency_sort("") == ""
print(frequency_sort("cccaaa"), frequency_sort("Aabb"))
```

```
aaaccc bbAa
```

**Complexity.** $O(n + m \log m)$ time for m distinct characters, $O(n)$ space.

### P13. Ugly Number II — the nth number whose only prime factors are 2, 3 and 5

**Which template.** Template 1 as a growing min-heap over a frontier, with a `seen` set.
**The trick.** Every ugly number is some earlier ugly number multiplied by 2, 3 or 5. Therefore pop the
smallest number produced so far and push its three children. The `seen` set is required, not optional:
6 is reachable as 2 times 3 and as 3 times 2, so without it the heap fills with duplicates and the
count is wrong.

```python
import heapq

def nth_ugly_number(n):
    heap = [1]
    seen = {1}                                        ## required: 6 arrives by two routes
    value = 1
    for _ in range(n):
        value = heapq.heappop(heap)
        for factor in (2, 3, 5):
            child = value * factor
            if child not in seen:
                seen.add(child)
                heapq.heappush(heap, child)
    return value

## tests

assert nth_ugly_number(1) == 1
assert nth_ugly_number(10) == 12
assert nth_ugly_number(11) == 15
assert nth_ugly_number(1690) == 2123366400
print(nth_ugly_number(10), nth_ugly_number(11))
```

```
12 15
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space. The three-pointer dynamic-programming version is
$O(n)$ and is the follow-up they want.

### P14. Kth Smallest Element in a Sorted Matrix — each row and each column is sorted; find the kth smallest

**Which template.** Template 4, a heap of the row heads.
**The trick.** Push the first element of every row as `(value, row, column)`. Pop k times, and after
each pop push the next element of that same row. Only the row frontier is ever in the heap, so it holds
at most n entries. The tuple's row and column entries are also what let you advance the right frontier
after a tie. Binary search on the value range is the alternative, at
$O(n \log(\text{max} - \text{min}))$ with $O(1)$ space; it is covered in the binary-search chapter.

```python
import heapq

def kth_smallest(matrix, k):
    n = len(matrix)
    heap = [(matrix[r][0], r, 0) for r in range(min(n, k))]   ## one head per row
    heapq.heapify(heap)
    value = None
    for _ in range(k):
        value, row, col = heapq.heappop(heap)
        if col + 1 < len(matrix[row]):
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    return value

## tests

assert kth_smallest([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8) == 13
assert kth_smallest([[-5]], 1) == -5
assert kth_smallest([[1, 2], [1, 3]], 2) == 1
assert kth_smallest([[1, 2], [1, 3]], 4) == 3
print(kth_smallest([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8))
```

```
13
```

**Complexity.** $O(k \log n)$ time, $O(n)$ space.

### P15. Single-Threaded CPU — process tasks by shortest processing time among those already available

**Which template.** Two structures at once: the tasks sorted by arrival time, and a max-heap-free
min-heap of `(processing_time, index)` over the available ones.
**The trick.** The two orderings are different and both are needed. Arrival order decides *when* a task
becomes a candidate, so sort by enqueue time and walk a pointer through it. Priority decides *which*
candidate runs, so the heap is keyed on `(processing_time, original_index)` with the index as the
documented tie-break. When the heap is empty the CPU idles, and you must jump the clock forward to the
next arrival rather than stepping one unit at a time.

```python
import heapq

def get_order(tasks):
    order = sorted(range(len(tasks)), key=lambda i: tasks[i][0])   ## by enqueue time
    heap, out = [], []
    time, pointer = 0, 0
    while len(out) < len(tasks):
        while pointer < len(order) and tasks[order[pointer]][0] <= time:
            i = order[pointer]
            heapq.heappush(heap, (tasks[i][1], i))                 ## (duration, index) tie-break
            pointer += 1
        if not heap:
            time = tasks[order[pointer]][0]                        ## idle: jump to the next arrival
            continue
        duration, i = heapq.heappop(heap)
        time += duration
        out.append(i)
    return out

## tests

assert get_order([[1, 2], [2, 4], [3, 2], [4, 1]]) == [0, 2, 3, 1]
assert get_order([[7, 10], [7, 12], [7, 5], [7, 4], [7, 2]]) == [4, 3, 2, 0, 1]
assert get_order([[1, 1]]) == [0]
print(get_order([[1, 2], [2, 4], [3, 2], [4, 1]]))
```

```
[0, 2, 3, 1]
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space.

### P16. Sliding Window Median — the median of every window of size k

**Which template.** Template 3, the two heaps, plus lazy deletion. This is the case where a heap alone
is not enough.
**The trick.** A heap supports removing the root, and nothing else. The sliding window needs you to
remove `nums[i - k]`, which sits somewhere in the middle, and no heap can do that in $O(\log n)$
without an index-aware structure. The repair is **lazy deletion**: do not remove the item, record that
it is owed a removal in a `delayed` counter, and discard it later when it happens to reach a root. Keep
the logical sizes `low_size` and `high_size` yourself, because `len(low)` now over-counts by the
delayed entries still sitting inside. Prune both roots before you read them, and the medians are
correct even though the heaps hold junk.

```python
import heapq
from collections import defaultdict

def median_sliding_window(nums, k):
    low, high = [], []                       ## low: max-heap (negated). high: min-heap
    delayed = defaultdict(int)
    low_size = high_size = 0
    out = []
    def prune(heap, sign):                   ## drop root entries that are owed a deletion
        while heap and delayed[sign * heap[0]] > 0:
            delayed[sign * heap[0]] -= 1
            heapq.heappop(heap)
    for i, x in enumerate(nums):
        if not low or x <= -low[0]:
            heapq.heappush(low, -x); low_size += 1
        else:
            heapq.heappush(high, x); high_size += 1
        if i >= k:
            old = nums[i - k]
            delayed[old] += 1                ## mark it, do not search for it
            if old <= -low[0]: low_size -= 1
            else: high_size -= 1
        while low_size > high_size + 1:
            prune(low, -1); heapq.heappush(high, -heapq.heappop(low))
            low_size -= 1; high_size += 1
        while low_size < high_size:
            prune(high, 1); heapq.heappush(low, -heapq.heappop(high))
            high_size -= 1; low_size += 1
        prune(low, -1); prune(high, 1)
        if i >= k - 1:
            out.append(float(-low[0]) if k % 2 else (-low[0] + high[0]) / 2.0)
    return out

## tests

assert median_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
assert median_sliding_window([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
assert median_sliding_window([5], 1) == [5.0]
print(median_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3))
```

```
[1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space. The heaps can hold up to n entries because deleted
items linger, which is the price of lazy deletion. A balanced tree or a `SortedList` gives $O(n \log k)$
and is the right answer if the library is allowed.

## Tricks and tips

**Say the inversion out loud before you type.** k largest means a min-heap of size k. k smallest means a
max-heap of size k. The heap holds the survivors and its root is the next one to be evicted, so the root
must be the weakest survivor. Every size-k heap problem is the same four lines: push, then pop if the
size exceeds k, and the answer is `heap[0]` or the whole heap at the end.

**Negate at exactly two points.** Python has no max-heap, so you push `-x` and you negate again when you
read the value out. Do it anywhere else and the signs stop tracking. When the payload is a tuple, negate
only the sort key: `(-count, char)`, never `(-count, -char)`.

**Put a unique integer in every heap tuple whose payload is an object.** `(value, index, node)` never
raises. `(value, node)` raises `TypeError` the first time two values tie, and equal values are exactly
what the test cases contain. The index costs nothing and removes a whole class of crash.

**`heapify` is $O(n)$, n pushes is $O(n \log n)$.** When you already hold the whole list, call
`heapq.heapify(items)` rather than looping. It is a real difference and interviewers notice it.

**`heapq.nsmallest` and `nlargest` exist and are legitimate.** They are $O(n \log k)$ internally and
implemented with exactly the size-k heap above. Use them to state your intent, then write the loop out
if the interviewer wants the mechanism.

**Two heaps whenever the question asks for a middle rather than an end.** Running median, "the k-th
element of a stream around the centre", and balanced partition problems all take a max-heap of the low
half against a min-heap of the high half. The invariants to state are the ordering one and the size one,
in that order.

**Lazy deletion is how a heap survives arbitrary removals.** Keep a `delayed` counter of items owed a
deletion, keep the logical sizes yourself because `len(heap)` now lies, and prune the root before you
read it. It is the standard repair, and knowing its name is worth saying.

**A heap is not always the answer.** Top k frequent is $O(n)$ with bucket sort. Kth largest is $O(n)$
average with quickselect. Kth smallest in a sorted matrix is $O(n \log(\text{range}))$ with binary
search on the value. Ugly numbers are $O(n)$ with three pointers. Give the heap first because it is
always correct and always writable under pressure, then name the better bound; that sequence reads as
strength, and jumping straight to a clever bound you cannot finish does not.

## The bugs that cost the round

**The wrong heap direction.** Using a max-heap for "k largest" gives you the largest at the root, so you
evict the biggest items and end with the k smallest. It passes the single-element test and fails
everything else. Check the direction by asking what you want to throw away, not what you want to keep.

**Forgetting to negate on the way out.** `heapq.heappop(heap)` on a negated heap returns a negative
number. Returning it directly gives an answer with the right magnitude and the wrong sign, which is the
easiest bug to spot in review and the easiest to write under time pressure.

**A `TypeError` on tied values.** Pushing `(distance, point)` or `(value, node)` crashes as soon as two
distances or two values are equal. Add the unique index.

**Comparing sizes with `!=` in the two-heap rebalance.** The invariant is that `low` may be one bigger
than `high`, so the test is `len(high) > len(low)`. With `!=` the heaps ping-pong and the odd-count
median is read from the wrong heap.

**Popping from an empty heap.** `heapq.heappop([])` raises `IndexError`. Guard the drain loop with
`while len(heap) > 1` when you pop two at a time, and handle the leftover single item after the loop.

**Trusting `len(heap)` under lazy deletion.** Once you defer deletions, the heap contains entries that
are logically gone. Track the sizes in your own integers and prune before every read of a root.

**Integer division on the median.** The even case is `(a + b) / 2.0`, not `// 2`. An integer median of
1 and 2 gives 1 rather than 1.5, and a single test case exposes it.

## Done when

- Given "k largest" or "k smallest" you can say which heap direction you need, and why, in one sentence
  and without a diagram.
- You can write the size-k heap template, the negation max-heap, the two-heap median and the k-way merge
  from a blank file, each in under three minutes.
- You can explain when a heap loses: to sorting when k is close to n, to quickselect for a single kth
  element, to bucket sort for top-k-frequent, and to binary search on the value for a sorted matrix.
- You can describe lazy deletion, say why `len(heap)` becomes unreliable, and write the prune step.
