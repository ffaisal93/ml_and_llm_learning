# LeetCode / NeetCode 150 — Pattern Recognition & Intuitive Approaches

> Big-tech and frontier-lab coding-round preparation, organized by **pattern recognition**, not by individual problems.
> Built around the 18 NeetCode 150 categories. The goal is: read a problem, recognize the pattern in 30 seconds, write the right template.

The single biggest jump in coding-interview competence is moving from "let me try to solve this problem" to "this is a sliding-window problem; here's the template; let me adapt it." This chapter trains the second pattern.

---

## Table of contents

1. The 30-second triage — how to read any problem
2. Arrays & Hashing
3. Two Pointers
4. Sliding Window
5. Stack
6. Binary Search
7. Linked List
8. Trees
9. Tries
10. Heap / Priority Queue
11. Backtracking
12. Graphs
13. Advanced Graphs (MST, topo sort, Dijkstra, Bellman-Ford, Floyd-Warshall)
14. 1-D Dynamic Programming
15. 2-D Dynamic Programming
16. Greedy
17. Intervals
18. Math & Geometry
19. Bit Manipulation
20. Problem-to-pattern transformation tricks
21. Complexity reasoning cheatsheet
22. Common interview traps
23. The 5-step problem-solving protocol
24. Drilling routine

---

## 1. The 30-second triage

When a problem lands, ask in this order:

**(a) What's the input shape?**
- Single array → arrays/hashing, two pointers, sliding window, sorting, prefix sum.
- Sorted array / monotonic property → binary search, two pointers.
- 2D grid → BFS/DFS/DP/backtracking.
- Tree / graph → recursion, BFS/DFS, topo, union-find.
- String → hashing, two pointers, sliding window, DP, trie.
- Number / bits → math, bit manipulation, DP.
- Multiple inputs / intervals → sorting, sweep line, intervals.

**(b) What's the output shape?**
- Yes/no, count, exists → often DP / hashing / sliding window.
- Min/max value → DP, greedy, heap, binary search on answer.
- All combinations / subsets / permutations → backtracking.
- Single index / pair → two pointers, hashing.
- Path / order → BFS, DFS, topo sort.

**(c) Are there constraints that hint at complexity?**
- $n \le 20$ → exponential OK; backtracking, bitmask DP.
- $n \le 100-500$ → $O(n^3)$ OK; 2-D DP, Floyd-Warshall.
- $n \le 10^4$ → $O(n^2)$ OK; standard DP.
- $n \le 10^5-10^6$ → $O(n \log n)$; sort + binary search, heap.
- $n \le 10^7$ → $O(n)$; hashing, sliding window, two pointers.
- $n \le 10^9$ → $O(\log n)$ or math; binary search on answer, formulas.

**(d) Are there words / structural cues?**
- "Subarray" / "contiguous" → sliding window or prefix sum.
- "Subsequence" → DP.
- "Sorted" → binary search or two pointers.
- "Cycle" → fast-slow pointers (linked lists), DFS visited (graphs).
- "Top-K / smallest-K / median" → heap.
- "Connected" → union-find or DFS.
- "Shortest path" → BFS (unweighted), Dijkstra (weighted), Bellman-Ford (negative), Floyd-Warshall (all-pairs).
- "Balanced parentheses / next greater" → stack.
- "All combinations" → backtracking.
- "Number of ways" / "min cost to reach" → DP.

**(e) Can you reduce to a simpler known problem?**
- Reverse the string and you have a known pattern.
- Sort and the structure becomes obvious.
- Build a graph from the input and BFS/DFS works.
- Memoize the recursion and you have DP.

This 30-second triage is 80% of competence. State it explicitly during a real interview.

> **Saying it out loud.** Before I write anything: let me make sure I've got the problem right — the input's a single array, I need a count back, and n goes up to ten to the fifth. That last number is doing most of the work here, because it rules out anything quadratic and tells me I'm looking for linear or n-log-n. Contiguous means I should be thinking sliding window rather than subsequence DP. Let me say the brute force first so we agree on correctness — nested loops, O of n squared — and then I'll talk about how to get it down. The thing I'm doing here on purpose is narrating the triage instead of silently pattern-matching, because if I'm wrong about the pattern you can stop me in ten seconds instead of twenty minutes.

---

## 2. Arrays & Hashing

### Recognition signals

- "Find duplicates / pairs / sum / count of X."
- "Group by some key."
- $O(n)$ or $O(n \log n)$ allowed.
- Need fast lookup or counting.

### Mental model

A hash map / set converts $O(n)$ scans into $O(1)$ lookups. Use it whenever you'd otherwise nest a loop.

### Templates

**Counting / frequency:**

```python
from collections import Counter
freq = Counter(arr)
for x, c in freq.items():
    ...
```

**Index lookup (Two Sum-style):**

```python
seen = {}
for i, x in enumerate(arr):
    if target - x in seen:
        return [seen[target - x], i]
    seen[x] = i
```

**Group by canonical form (anagrams):**

```python
groups = defaultdict(list)
for s in strs:
    key = tuple(sorted(s))    # or freq tuple
    groups[key].append(s)
return list(groups.values())
```

### Representative problems

- **Two Sum.** Hash index → $O(n)$.
- **Group Anagrams.** Hash by sorted tuple → $O(n \cdot k \log k)$.
- **Top K Frequent Elements.** Counter + heap of size $k$ → $O(n \log k)$. Or bucket sort → $O(n)$.
- **Product of Array Except Self.** Prefix and suffix products → $O(n)$, no division.
- **Valid Sudoku.** Three sets per (row, col, box), validate uniqueness.
- **Encode/Decode Strings.** Length-prefix encoding.
- **Longest Consecutive Sequence.** Hash set; for each `n`, only start counting from `n` if `n-1` not in set → $O(n)$ amortized.

### Common traps

- Forgetting that hashing strings has hidden $O(L)$ cost where $L$ is string length.
- Modifying a dict while iterating it.
- Assuming hash maps preserve insertion order (Python 3.7+ does; Java's HashMap doesn't).
- Using a list as a key (not hashable) — use a tuple.

> **Saying it out loud.** Here's my thinking — the brute force is checking every pair, which is O of n squared. But I don't actually need to *search* for the complement, I can *remember* what I've already seen. So I'll do one pass with a hash map from value to index, and at each element ask whether the complement is already in the map. One pass, O of n time, O of n extra space — I'm trading memory for time, which is the whole move for this pattern. Let me code it and I'll narrate as I go... and the edge case I want to handle explicitly is duplicates: I insert the current element *after* I check for its complement, so an element never matches with itself.

---

## 3. Two Pointers

### Recognition signals

- "Sorted array."
- "Find a pair / triplet / window with property X."
- "Move toward each other" or "fast/slow."
- Want to avoid an $O(n^2)$ nested loop.

### Mental model

Two indices into the array, advancing based on a comparison. Each pointer moves $O(n)$ times total → linear time.

### Templates

**Opposing pointers (sorted array, target sum):**

```python
l, r = 0, len(arr) - 1
while l < r:
    s = arr[l] + arr[r]
    if s == target: return [l, r]
    if s < target:  l += 1
    else:           r -= 1
```

**Same-direction (fast/slow, in-place modification):**

```python
slow = 0
for fast in range(len(arr)):
    if condition(arr[fast]):
        arr[slow] = arr[fast]
        slow += 1
return slow
```

### Representative problems

- **Two Sum II (sorted).** Opposing pointers.
- **3Sum.** Fix one element; two-pointer the rest. $O(n^2)$.
- **Container With Most Water.** Opposing pointers; move the shorter side.
- **Trapping Rain Water.** Two pointers + running max from each side.
- **Valid Palindrome.** Skip non-alphanumeric, compare from outside in.
- **Remove Duplicates from Sorted Array.** Slow/fast in place.

### Common traps

- Forgetting to skip duplicates in 3Sum (gives wrong answer with duplicate triplets).
- Off-by-one in loop bounds.
- Using two pointers on unsorted data (usually wrong).

> **Saying it out loud.** The array's sorted, and that's the hint I want to use — sorted means I can throw away half the search space with each comparison instead of scanning. So: pointer at each end. If the sum is too big, the only way to shrink it is to move the right pointer in; if it's too small, move the left in. Each pointer only ever moves one direction, so this is O of n with O of 1 extra space — better than the hash map version because I don't need the map at all. Let me state the invariant out loud since that's what makes it correct: the answer, if it exists, is always inside the window between the two pointers. Edge case I'll check: fewer than two elements, and duplicates if the problem wants unique pairs.

---

## 4. Sliding Window

### Recognition signals

- "Subarray" / "contiguous substring."
- "Longest / shortest / count of X with property Y."
- $O(n)$ target.
- The property is *monotonic* in window size (extending the window only adds; shrinking only removes).

### Mental model

Maintain a window $[l, r]$. Expand $r$ to include new elements; when constraint violated, shrink $l$. Each pointer moves $O(n)$ → linear time.

### Templates

**Variable-size window (longest with property):**

```python
l = 0
state = init()
best = 0
for r, x in enumerate(arr):
    state = add(state, x)
    while not valid(state):
        state = remove(state, arr[l])
        l += 1
    best = max(best, r - l + 1)
return best
```

**Fixed-size window (size $k$):**

```python
state = init()
for r in range(k):
    state = add(state, arr[r])
ans = state.value()
for r in range(k, len(arr)):
    state = add(state, arr[r])
    state = remove(state, arr[r - k])
    ans = better(ans, state.value())
return ans
```

**Counting / "at most $k$" trick:**

```python
def at_most_k(k):
    # number of subarrays with at-most-k of property X
    ...
# count exactly k = at_most_k(k) - at_most_k(k - 1)
```

### Representative problems

- **Best Time to Buy and Sell Stock.** Track running min, look for max diff. $O(n)$.
- **Longest Substring Without Repeating Characters.** Hash window of seen chars; shrink on duplicate.
- **Longest Repeating Character Replacement.** Track max char count in window; valid if window_size − max_count ≤ k.
- **Minimum Window Substring.** Counter of needed chars; shrink while valid; track minimum.
- **Permutation in String.** Fixed window of size = pattern length; compare counters.
- **Sliding Window Maximum.** Monotonic deque (also see §5).

### Common traps

- Wrong shrink condition (when do you contract the window?).
- Forgetting to update state on both add and remove.
- Counting versus length tracking.
- Treating subsequence problems as sliding-window (don't — those are DP).

> **Saying it out loud.** The word "contiguous" is what makes me reach for sliding window. Instead of re-examining every subarray from scratch, I expand the right edge one element at a time, and whenever the window becomes invalid I shrink from the left until it's valid again. The key insight for the complexity: each index enters the window once and leaves once, so even though there's a nested loop it's O of n total, not n squared — that's the amortized argument and it's worth saying out loud because it looks quadratic on the page. I'll keep a hash map of counts inside the window as my validity check. Off-by-one is the classic bug here, so let me trace a two-element example before I call it done.

---

## 5. Stack

### Recognition signals

- "Matching parentheses / brackets."
- "Next greater element / nearest smaller."
- "Evaluate expression / parse."
- "Histogram / rectangle."
- "Undo / backtrack."

### Mental model

Last-in-first-out. Use when each item's processing depends on a recently-seen item that hasn't been resolved yet.

### Templates

**Matching:**

```python
pairs = {')': '(', ']': '[', '}': '{'}
stack = []
for c in s:
    if c in '([{':
        stack.append(c)
    else:
        if not stack or stack.pop() != pairs[c]: return False
return not stack
```

**Monotonic stack (next greater):**

```python
res = [-1] * len(arr)
stack = []  # indices, decreasing values
for i, x in enumerate(arr):
    while stack and arr[stack[-1]] < x:
        res[stack.pop()] = x
    stack.append(i)
return res
```

### Representative problems

- **Valid Parentheses.** Standard stack matching.
- **Min Stack.** Two stacks or store `(val, current_min)` tuples.
- **Evaluate Reverse Polish Notation.** Stack of operands; pop two on operator.
- **Generate Parentheses.** Backtracking — stack-of-thinking, not literal stack.
- **Daily Temperatures.** Monotonic decreasing stack of indices.
- **Car Fleet.** Sort by position desc; stack of times-to-target.
- **Largest Rectangle in Histogram.** Monotonic stack; key insight: for each bar, find left+right boundaries via stack.

### Common traps

- Pushing/popping order wrong.
- Forgetting to handle leftover stack items at the end (e.g., remaining bars in histogram).
- Confusing monotonic increasing vs decreasing stacks.

> **Saying it out loud.** The signal I'm picking up on is that I need to match each element with something earlier that I haven't resolved yet — that's a stack. For the monotonic version: I keep the stack decreasing, and whenever a new element is bigger than the top, that new element is the "next greater" for everything I pop. Every element gets pushed once and popped at most once, so it's O of n even though there's a while loop inside a for loop — same amortized argument as sliding window. Space is O of n in the worst case, when the input's already sorted the wrong way. The edge case to handle is what's left on the stack at the end: those elements never found a match, so they get the default answer, usually negative one.

---

## 6. Binary Search

### Recognition signals

- "Sorted" array.
- $O(\log n)$ target.
- "Find / first / last / smallest X such that property holds."
- A *monotonic* property — once it flips, it doesn't flip back.
- Constraints up to $10^9$ on the answer space → "binary search on the answer."

### Mental model

Each step halves the search space. Use whenever you can bisect a monotonic predicate.

### Templates

**Standard binary search:**

```python
l, r = 0, len(arr) - 1
while l <= r:
    m = (l + r) // 2
    if arr[m] == target: return m
    if arr[m] < target:  l = m + 1
    else:                r = m - 1
return -1
```

**Find first / leftmost match (predicate-based):**

```python
l, r = 0, len(arr)
while l < r:
    m = (l + r) // 2
    if pred(arr[m]):  r = m
    else:             l = m + 1
return l   # first index where pred is True
```

**Binary search on the answer:**

```python
def feasible(x): ...   # True iff answer x is feasible
l, r = lo, hi
while l < r:
    m = (l + r) // 2
    if feasible(m):  r = m
    else:            l = m + 1
return l
```

### Representative problems

- **Binary Search.** Standard.
- **Search a 2D Matrix.** Treat 2D as flattened sorted 1D.
- **Koko Eating Bananas.** Binary search on speed; predicate = "can finish in $h$ hours."
- **Find Minimum in Rotated Sorted Array.** Binary search comparing $\text{mid}$ to $\text{right}$.
- **Search in Rotated Sorted Array.** Binary search with rotation handling.
- **Time Based Key-Value Store.** Per-key sorted timestamps; bisect.
- **Median of Two Sorted Arrays.** Binary search on partition position. $O(\log \min(m, n))$.

### Common traps

- Off-by-one between `l <= r` and `l < r` patterns. Pick one and stick with it.
- Integer overflow on `(l + r) // 2` in languages with fixed int (use `l + (r - l) // 2`).
- Predicate not monotonic → binary search doesn't apply.
- Forgetting which boundary to return.

> **Saying it out loud.** Sorted input and a log-n target complexity is basically an announcement that this is binary search. The version I want to write is the one that finds a boundary rather than an exact match, because that generalizes to "first element greater than or equal to x" without extra cases. I'll use the half-open convention — low inclusive, high exclusive, mid computed as low plus high-minus-low over two — and I'll state my loop invariant, which is that the answer always lives in the range low to high. That invariant is the thing that keeps me out of the classic infinite loop where mid never advances. Edge cases: empty array, target smaller than everything, target bigger than everything. It's O of log n time and O of 1 space.

---

## 7. Linked List

### Recognition signals

- Input is a linked list head pointer.
- "Reverse / merge / cycle / kth-from-end."
- $O(1)$ space target → can't dump to array.

### Mental model

Pointer manipulation. Always use a sentinel (dummy) node to simplify head edge cases.

### Templates

**Reverse:**

```python
prev, curr = None, head
while curr:
    nxt = curr.next
    curr.next = prev
    prev = curr
    curr = nxt
return prev
```

**Cycle detection (Floyd / fast-slow):**

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast: return True   # cycle
return False
```

**Cycle entry (Floyd):**

```python
# After fast-slow meet, restart slow at head; advance both 1 step.
slow = head
while slow != fast:
    slow = slow.next
    fast = fast.next
return slow
```

**Merge two sorted:**

```python
dummy = ListNode()
tail = dummy
while l1 and l2:
    if l1.val <= l2.val: tail.next = l1; l1 = l1.next
    else:                tail.next = l2; l2 = l2.next
    tail = tail.next
tail.next = l1 or l2
return dummy.next
```

### Representative problems

- **Reverse Linked List.** Standard.
- **Merge Two Sorted Lists.** Standard with dummy.
- **Reorder List.** Find middle (fast-slow); reverse second half; merge alternating.
- **Remove Nth From End.** Two pointers; advance fast by n; then both.
- **Copy List with Random Pointer.** Hash old→new, two passes; or interleave trick.
- **Add Two Numbers.** Carry-based with dummy.
- **LRU Cache.** Doubly linked list + hash map for $O(1)$ get/put.
- **Merge K Sorted Lists.** Heap of $k$ heads, $O(N \log k)$.
- **Reverse Nodes in k-Group.** Reverse $k$, recurse.
- **Find the Duplicate Number.** Floyd's cycle on the implicit "index → arr[index]" linked list.

### Common traps

- Losing the head — always use a dummy.
- Forgetting to update `prev.next` and `curr.next` in the right order.
- Off-by-one in "Nth from end" (n vs n+1 advance).
- Cycle in input → infinite loop if not detecting.

> **Saying it out loud.** With linked lists I can't index, so almost everything reduces to two pointer tricks. Fast and slow for anything about the middle or about cycles: if there's a cycle, the fast pointer laps the slow one and they meet, which is Floyd's algorithm — O of n time, O of 1 space, and that constant space is why we prefer it to a visited set. Two pointers offset by k for anything about "kth from the end." And I'll add a dummy head node before I write any code, because it makes deleting or inserting at the front behave exactly like the middle and eliminates a whole class of null checks. The edge cases I always run: empty list, single node, and operating on the head itself.

---

## 8. Trees (Binary Trees, BST)

### Recognition signals

- Input is a tree root.
- "DFS / BFS / level order / serialize."
- "Path / depth / diameter / lowest common ancestor."
- BST: "search / insert / delete / inorder is sorted."

### Mental model

Trees naturally decompose: solve for left subtree, solve for right subtree, combine. Recursion is the dominant tool. BST adds the ordering invariant.

### Templates

**DFS (postorder, with return value):**

```python
def dfs(node):
    if not node: return base_case
    left  = dfs(node.left)
    right = dfs(node.right)
    return combine(left, right, node)
```

**BFS (level order):**

```python
from collections import deque
q = deque([root])
levels = []
while q:
    level = []
    for _ in range(len(q)):
        node = q.popleft()
        level.append(node.val)
        if node.left:  q.append(node.left)
        if node.right: q.append(node.right)
    levels.append(level)
return levels
```

**BST property exploitation:**

```python
def search(node, target):
    if not node or node.val == target: return node
    if target < node.val: return search(node.left,  target)
    else:                 return search(node.right, target)
```

### Representative problems

- **Invert Binary Tree.** Swap children recursively.
- **Maximum Depth.** Recursion: 1 + max(depth(left), depth(right)).
- **Diameter of Binary Tree.** DFS returning depth; track max(left+right) seen.
- **Balanced Binary Tree.** DFS returning height or -1 (unbalanced sentinel).
- **Same Tree / Subtree.** Recursive structural compare.
- **Lowest Common Ancestor.**
  - BST: walk down; first node where p ≤ node ≤ q.
  - General tree: DFS; return non-null whenever both sides are non-null.
- **Binary Tree Right Side View.** BFS, last node per level. Or DFS prioritizing right.
- **Construct Tree from Preorder + Inorder.** Recursive; partition inorder around preorder root.
- **Validate BST.** Recursion with (min, max) bounds.
- **Kth Smallest in BST.** Inorder traversal; stop at $k$.
- **Serialize/Deserialize Binary Tree.** Preorder with null markers.

### Common traps

- BFS without level tracking when level matters.
- Validate BST: comparing only with parent (wrong) vs with bounds (right).
- Forgetting null base case → AttributeError.
- LCA on BST without exploiting BST property → unnecessary $O(n)$.
- Recursion depth limit on skewed trees in Python (`sys.setrecursionlimit`).

> **Saying it out loud.** Two questions decide the shape of my code: does each node need answers from its children, or does it need context from its parent? If it's children-up, that's post-order recursion where I return something and combine. If it's parent-down, I pass state as an argument — for a BST validity check, the running min and max bounds. And if the problem is about levels or shortest depth, that's BFS with a queue, not recursion. Complexity is O of n because I visit every node once; space is O of h for the recursion stack, which is log n on a balanced tree and n on a degenerate one. That worst case is the thing to name — a linked-list-shaped tree will blow Python's default recursion limit of a thousand.

---

## 9. Tries

### Recognition signals

- Multiple string lookups / prefix queries.
- "Word search / autocomplete / all words with prefix X."
- Set of strings + repeated queries.

### Mental model

A 26-ary tree (or 256-ary, or hash-based) where each path from root spells a string. Insert/lookup are $O(L)$ where $L$ is word length.

### Templates

```python
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, word):
        node = self.root
        for c in word:
            node = node.setdefault(c, {})
        node['$'] = True
    def search(self, word):
        node = self.root
        for c in word:
            if c not in node: return False
            node = node[c]
        return '$' in node
    def starts_with(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node: return False
            node = node[c]
        return True
```

### Representative problems

- **Implement Trie.** Standard.
- **Design Add and Search Word.** Trie + DFS for `.` wildcard.
- **Word Search II.** Build trie of words; DFS the board; prune when no trie path matches.

### Common traps

- Using a class-per-node when a dict-of-dicts works fine.
- Forgetting the end-of-word marker.
- Heavy memory use (consider compressed/Patricia trie for large dictionaries).

> **Saying it out loud.** A trie is what I use when the queries are about prefixes, because a hash set can tell me a word exists but can't tell me anything shares a prefix. Each node is a map from character to child node plus a flag marking end-of-word. Insert and search are O of L in the length of the word, completely independent of how many words are stored — that independence is the whole reason to build one. The tradeoff to name is memory: you're paying a node per character in the worst case, so it's a real space cost compared to a set, and you'd only take it if prefix queries are actually in the requirements. The edge cases are the empty string and making sure a word that's a prefix of another still has its end flag set.

---

## 10. Heap / Priority Queue

### Recognition signals

- "Top K / Kth largest / Kth smallest."
- "Median of stream."
- "Merge K sorted."
- "Schedule / next event."
- Want $O(\log n)$ insert/extract.

### Mental model

Heap maintains the smallest (min-heap) or largest (max-heap) at the root in $O(\log n)$ ops. Two heaps balance to give a streaming median.

### Templates

**Top K largest (min-heap of size k):**

```python
import heapq
h = []
for x in arr:
    heapq.heappush(h, x)
    if len(h) > k:
        heapq.heappop(h)
return h           # k largest
```

**Streaming median (two heaps):**

```python
class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (negated)
        self.hi = []  # min-heap
    def add(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def median(self):
        if len(self.lo) > len(self.hi): return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

**Merge K sorted lists:**

```python
h = [(node.val, i, node) for i, node in enumerate(lists) if node]
heapq.heapify(h)
dummy = ListNode(); tail = dummy
while h:
    val, i, node = heapq.heappop(h)
    tail.next = node; tail = node
    if node.next:
        heapq.heappush(h, (node.next.val, i, node.next))
return dummy.next
```

### Representative problems

- **Kth Largest Element.** Heap of size $k$ → $O(n \log k)$. Or quickselect → $O(n)$ average.
- **Last Stone Weight.** Max-heap; pop two; push diff.
- **K Closest Points to Origin.** Heap of size $k$.
- **Task Scheduler.** Counter + heap by frequency; CPU cycles with cool-down.
- **Find Median from Data Stream.** Two heaps.
- **Merge K Sorted Lists.** Heap of heads.
- **Top K Frequent.** Counter + heap, or bucket sort.
- **Design Twitter.** Heap of latest tweets per follower.

### Common traps

- Python's `heapq` is min-heap only; use negation for max-heap.
- Forgetting to handle ties when heap stores tuples.
- Time complexity of `heapify` is $O(n)$, not $O(n \log n)$.

> **Saying it out loud.** "Top k" is the trigger. I could sort — that's n log n — but I only need the k largest, so I'll keep a min-heap of size k: push everything, and pop whenever the heap exceeds k. The smallest of my current best k is always sitting at the top where I can compare against it cheaply. That's O of n log k, which is a real win when k is small relative to n, and O of k space. One Python detail I'll say out loud because it bites people: heapq is a min-heap only, so for a max-heap I push negated values and negate on the way out. Edge cases: k bigger than the array, and ties.

---

## 11. Backtracking

### Recognition signals

- "All subsets / permutations / combinations."
- "Find all paths / valid configurations."
- Constraint satisfaction (Sudoku, N-Queens).
- Small $n$ (≤ 20) — exponential is OK.

### Mental model

DFS through the choice tree. At each node, try each option; recurse; undo (the "backtrack"). Prune branches that can't lead to a valid answer.

### Templates

**Standard backtrack:**

```python
def backtrack(state, choices):
    if is_complete(state):
        result.append(copy(state))
        return
    for c in choices:
        if not is_valid(state, c): continue
        apply(state, c)
        backtrack(state, next_choices(choices, c))
        undo(state, c)
```

**Subsets:**

```python
def dfs(i, current):
    if i == len(arr):
        result.append(current[:])
        return
    # exclude arr[i]
    dfs(i + 1, current)
    # include arr[i]
    current.append(arr[i])
    dfs(i + 1, current)
    current.pop()
```

**Permutations:**

```python
def dfs(remaining, current):
    if not remaining:
        result.append(current[:])
        return
    for i in range(len(remaining)):
        current.append(remaining[i])
        dfs(remaining[:i] + remaining[i+1:], current)
        current.pop()
```

### Representative problems

- **Subsets / Subsets II.** Include/exclude. For II, sort and skip duplicates at the same level.
- **Combination Sum / II / III.** Backtrack with running sum and start index.
- **Permutations / II.** Swap-in-place or used-array.
- **Word Search.** DFS the grid with visited markers; backtrack on return.
- **Palindrome Partitioning.** Try each prefix; if palindrome, recurse on suffix.
- **N-Queens.** Place queens row by row with column / diagonal sets for $O(1)$ check.
- **Letter Combinations of Phone Number.** Backtrack by digit.
- **Restore IP Addresses.** Try 1-3 chars, validate, recurse.
- **Sudoku Solver.** Try 1-9 in each empty cell with row/col/box sets.

### Common traps

- Forgetting to undo state (the "backtrack" step).
- Appending mutable state without copying — append `current[:]` not `current`.
- Wrong start index in combinations vs permutations (combinations: `i+1`; permutations: include all).
- Not handling duplicates when input has dups (sort + skip).

> **Saying it out loud.** The output is "all combinations," and n is small — twenty or under — so exponential is expected and I shouldn't be trying to be clever. This is backtracking: choose, recurse, un-choose. I'll pass a start index so I never revisit an earlier element, which is what stops me generating the same combination in different orders. Complexity is roughly O of n times two to the n for subsets, or n factorial for permutations — I'll say which and why. Two things I'll be careful about: I append a *copy* of the path when I hit a leaf, because the path list gets mutated after; and to handle duplicates I sort first and skip an element if it equals the previous one and the previous one wasn't used at this level.

---

## 12. Graphs

### Recognition signals

- "Connected" / "shortest path" / "all paths" / "cycle."
- Adjacency list / matrix or implicit grid.
- "Number of islands / connected components."
- BFS for unweighted shortest, DFS for connectivity.

### Mental model

A graph is nodes + edges. Visit each node once with a `visited` set; recurse / queue neighbors; combine results.

### Templates

**DFS (recursive):**

```python
visited = set()
def dfs(u):
    visited.add(u)
    for v in graph[u]:
        if v not in visited: dfs(v)
```

**BFS (queue):**

```python
from collections import deque
q = deque([(start, 0)])
visited = {start}
while q:
    u, d = q.popleft()
    if u == target: return d
    for v in graph[u]:
        if v not in visited:
            visited.add(v); q.append((v, d + 1))
return -1
```

**Grid traversal:**

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def dfs(r, c):
    if not (0 <= r < R and 0 <= c < C): return
    if grid[r][c] != target: return
    grid[r][c] = '#'   # mark visited in-place (or use set)
    for dr, dc in DIRS:
        dfs(r + dr, c + dc)
```

**Union-Find (DSU):**

```python
parent = list(range(n))
rank = [0] * n
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path compression
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return False
    if rank[ra] < rank[rb]: ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]: rank[ra] += 1
    return True
```

### Representative problems

- **Number of Islands.** DFS/BFS each '1' cell, mark visited.
- **Clone Graph.** DFS/BFS with old→new map.
- **Max Area of Island.** DFS with returned area.
- **Pacific Atlantic Water Flow.** BFS from each ocean's edge cells; intersection.
- **Surrounded Regions.** DFS from edge 'O' to mark "safe"; flip rest.
- **Rotting Oranges.** Multi-source BFS; track minutes.
- **Walls and Gates.** Multi-source BFS from gates.
- **Course Schedule.** Topological sort (next section).
- **Redundant Connection.** Union-Find; the edge that creates a cycle.
- **Number of Connected Components.** DFS-count or Union-Find.
- **Graph Valid Tree.** $n$ nodes, $n-1$ edges, no cycle, connected — Union-Find or DFS.
- **Word Ladder.** BFS on word graph, transform one letter at a time.

### Common traps

- Forgetting the visited set → infinite recursion.
- BFS without level tracking when distance matters.
- Mutating the input grid without restoring (some interviewers care).
- Stack overflow on huge DFS in Python — convert to iterative or `setrecursionlimit`.

> **Saying it out loud.** The moment a problem mentions connectivity or reachability I'm building a graph, even if it's handed to me as a grid — a grid is just a graph where the neighbors are up, down, left, right. Then the choice is simple: BFS if I need the shortest path in an unweighted graph, DFS if I just need to explore or count components. Either way it's O of V plus E, and the single most important line is marking a node visited *when I enqueue it*, not when I dequeue it, or the same node gets queued many times and the complexity blows up. Edge cases: disconnected components, so I loop over all starting nodes; self-loops; and an empty grid.

---

## 13. Advanced Graphs

Algorithms that build on the basics.

### 13.1 Topological sort

**Signal:** "Order tasks subject to prerequisites." DAG. Course Schedule, Build Order.

**Kahn's algorithm (BFS):**

```python
indeg = [0] * n
for u, v in edges:
    indeg[v] += 1
q = deque([i for i in range(n) if indeg[i] == 0])
order = []
while q:
    u = q.popleft()
    order.append(u)
    for v in graph[u]:
        indeg[v] -= 1
        if indeg[v] == 0: q.append(v)
return order if len(order) == n else []   # cycle if not all included
```

**DFS-based:** post-order DFS; reverse the result. Detect cycle via "in-stack" marker.

### 13.2 Dijkstra (single-source shortest path, non-negative weights)

```python
import heapq
dist = [float('inf')] * n; dist[src] = 0
h = [(0, src)]
while h:
    d, u = heapq.heappop(h)
    if d > dist[u]: continue
    for v, w in graph[u]:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            heapq.heappush(h, (dist[v], v))
return dist
```

$O((V + E) \log V)$.

### 13.3 Bellman-Ford (handles negative weights)

```python
dist = [float('inf')] * n; dist[src] = 0
for _ in range(n - 1):
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
# Detect negative cycle: one more pass; if any update, there's a cycle.
```

$O(VE)$. Use when negative weights exist (Cheapest Flights Within K Stops).

### 13.4 Floyd-Warshall (all-pairs)

```python
for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i][k] + dist[k][j] < dist[i][j]:
                dist[i][j] = dist[i][k] + dist[k][j]
```

$O(V^3)$. Use when $V$ small (~500) and you need all-pairs.

### 13.5 Minimum Spanning Tree (MST)

**Kruskal's** — sort edges; Union-Find to add if not creating cycle.

```python
edges.sort(key=lambda e: e[2])
uf = UnionFind(n)
mst_weight = 0
for u, v, w in edges:
    if uf.union(u, v):
        mst_weight += w
```

**Prim's** — Dijkstra-like, but distance is from "any tree node," not from source.

### 13.6 Representative problems

- **Course Schedule / II.** Topological sort.
- **Network Delay Time.** Dijkstra.
- **Cheapest Flights Within K Stops.** Bellman-Ford with at most K relaxations.
- **Min Cost to Connect All Points.** MST (Kruskal or Prim).
- **Swim in Rising Water.** Dijkstra with max-along-path metric, or binary search + BFS.
- **Reconstruct Itinerary.** Hierholzer's (Eulerian path) or DFS with sorted destinations.
- **Alien Dictionary.** Build precedence graph; topological sort.
- **Foreign Dictionary / Find Order.** Same family.

### 13.7 Common traps

- Dijkstra with negative weights → wrong answer.
- Topological sort on cyclic graph → not all nodes appear → detect.
- Using BFS instead of Dijkstra when edges have non-uniform weights.
- Forgetting `if d > dist[u]: continue` in Dijkstra → unnecessary relaxations.

> **Saying it out loud.** These four are really "which constraint did you add." Dependencies and ordering means topological sort — Kahn's algorithm with in-degrees, O of V plus E, and if the queue empties before I've output every node there's a cycle, which is often what the problem was actually testing. Weighted with non-negative edges is Dijkstra, a heap-based BFS, E log V. Negative edges break Dijkstra's greedy assumption entirely, so that's Bellman-Ford at V times E, which also detects negative cycles with one extra pass. All-pairs on a small dense graph is Floyd-Warshall, three nested loops, n cubed. The mistake to name out loud: reaching for Dijkstra on an unweighted graph, where plain BFS gets the same answer without the log factor.

---

## 14. 1-D Dynamic Programming

### Recognition signals

- "Number of ways," "min/max value," subsequence.
- Subproblems overlap; optimal substructure.
- Brute force is exponential but state space is polynomial.
- State can be expressed by an index `i` (sometimes plus a small flag).

### Mental model

Define `dp[i]` = answer to subproblem ending at / using index $i$. Express `dp[i]` in terms of earlier `dp[j]`. Base case + transition + final answer.

### Templates

**Top-down memoization:**

```python
from functools import lru_cache
@lru_cache(None)
def f(i):
    if base(i): return ...
    return combine(f(i-1), f(i-2), ...)
```

**Bottom-up tabulation:**

```python
dp = [0] * (n + 1)
dp[0] = base
for i in range(1, n + 1):
    dp[i] = combine(dp[i-1], dp[i-2], ...)
return dp[n]
```

**Space-optimized (only last few values):**

```python
prev2, prev1 = ..., ...
for i in range(2, n + 1):
    curr  = combine(prev1, prev2, ...)
    prev2 = prev1
    prev1 = curr
return prev1
```

### Representative problems

- **Climbing Stairs / Fib.** $dp[i] = dp[i-1] + dp[i-2]$.
- **Min Cost Climbing Stairs.** Like Fib with cost.
- **House Robber.** $dp[i] = \max(dp[i-1], dp[i-2] + nums[i])$.
- **House Robber II.** Two passes (skip first or skip last).
- **Longest Palindromic Substring.** Expand around centers, $O(n^2)$. Or Manacher's $O(n)$.
- **Palindromic Substrings.** Expand around centers; count.
- **Decode Ways.** $dp[i]$ = ways to decode prefix of length $i$.
- **Coin Change.** $dp[\text{amount}] = \min(dp[\text{amount} - c] + 1)$ over coins. Unbounded knapsack.
- **Maximum Product Subarray.** Track both max and min ending at $i$ (negative flips).
- **Word Break.** $dp[i]$ = can break prefix of length $i$.
- **Longest Increasing Subsequence.** $O(n^2)$ DP, or $O(n \log n)$ with patience sorting.
- **Partition Equal Subset Sum.** Subset sum to total/2; 0/1 knapsack.

### Common traps

- Confusing subarray (contiguous) with subsequence (non-contiguous).
- Off-by-one in indexing (e.g., `dp[0]` vs `dp[1]` as base case).
- Forgetting to initialize the base case.
- Using $O(n)$ space when $O(1)$ suffices.
- Coin Change unbounded vs 0/1 — different recurrence.

> **Saying it out loud.** Let me define the state before I write anything, because that's where DP goes wrong: dp of i is the answer for the first i elements. Then the recurrence — at position i I either take this element and add dp of i minus two, or skip it and keep dp of i minus one — so it's the max of those. Base cases are dp of zero and dp of one, and getting those wrong is the usual bug. That's O of n time and O of n space, and since I only ever look back two positions I can collapse it to two variables and get O of 1 space, which I'll do if you want it. The way I got here, and this is the general recipe, is: write the brute-force recursion, notice the overlapping subproblems, then memoize and flip it to a bottom-up loop.

---

## 15. 2-D Dynamic Programming

### Recognition signals

- Two indices in the state (string i, string j; row, col; range $[l, r]$).
- "Edit distance / longest common subsequence / interleaving."
- Grid path counting / min cost.
- Range queries.

### Mental model

`dp[i][j]` for two-string or grid problems. Often the recurrence is "match or don't match" or "go right or down."

### Templates

**LCS:**

```python
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if a[i-1] == b[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
return dp[m][n]
```

**Grid path:**

```python
dp[0][0] = grid[0][0]
for i in range(m):
    for j in range(n):
        if i == 0 and j == 0: continue
        a = dp[i-1][j] if i > 0 else inf
        b = dp[i][j-1] if j > 0 else inf
        dp[i][j] = grid[i][j] + min(a, b)
return dp[m-1][n-1]
```

**Range DP:**

```python
for L in range(2, n + 1):                   # window length
    for i in range(n - L + 1):
        j = i + L - 1
        dp[i][j] = combine_over(dp[i][k], dp[k+1][j] for k in range(i, j))
```

### Representative problems

- **Unique Paths / II.** Grid path count; II has obstacles.
- **Longest Common Subsequence.** Standard LCS.
- **Edit Distance.** Insert/delete/replace recurrence.
- **Distinct Subsequences.** Count of t in s.
- **Interleaving String.** Boolean DP.
- **Best Time to Buy/Sell with Cooldown / Fee / K Transactions.** State = (i, holding, last action).
- **Target Sum.** Reduces to subset sum count.
- **Burst Balloons.** Range DP.
- **Regular Expression Matching / Wildcard Matching.** $dp[i][j]$ for prefix matches.
- **Longest Increasing Path in a Matrix.** DFS with memoization.
- **Maximal Rectangle.** Histogram per row + Largest Rectangle in Histogram.

### Common traps

- Forgetting to handle the "first row" and "first column" of a grid DP.
- Range DP loop order (must process smaller ranges first).
- Confusing "subsequence" with "substring" — the recurrence differs.
- LCS variants: longest common substring is different (resets on mismatch).

> **Saying it out loud.** Two sequences or a grid means two indices in the state, so dp of i and j is the answer for the first i of one string and the first j of the other. The recurrence splits on whether the characters match: if they do, I extend the diagonal, dp of i minus one, j minus one, plus one; if they don't, I take the best of dropping a character from either side. Fill the base row and column first, then sweep. That's O of m times n time and space. Since each row only depends on the row above it, I can drop to one-dimensional and get O of n space — worth mentioning even if I don't do it. Edge cases: empty strings, which is exactly what the zero row and column encode.

---

## 16. Greedy

### Recognition signals

- "Minimum / maximum X with simple rule."
- A *local optimal choice leads to global optimal.*
- Often involves sorting first.
- Can prove correctness via exchange argument.

### Mental model

At each step, pick the locally best option without considering future. Hard to know when greedy works without proof — common in interval / scheduling problems.

### Templates

**Sort + sweep:**

```python
arr.sort(key=...)
result = init
for x in arr:
    if condition(x, result):
        result = update(result, x)
return result
```

### Representative problems

- **Maximum Subarray (Kadane's).** Track current sum; reset to 0 if negative. $O(n)$.
- **Jump Game / II.** Greedy: track the farthest reachable.
- **Gas Station.** Skip to next station after a deficit; one pass.
- **Hand of Straights.** Sort + use Counter; greedily form groups starting from min.
- **Merge Triplets to Form Target.** Filter triplets that don't exceed target; check max in each position.
- **Partition Labels.** Compute last index per char; sweep.
- **Valid Parenthesis String.** Track range of possible open counts.

### Common traps

- Assuming greedy works without proof — it often doesn't on problems that look greedy. (Coin change with arbitrary denominations is *not* greedy; with US coins, it is.)
- Wrong sort key.
- Edge cases at boundaries.

> **Saying it out loud.** I want to be careful here, because greedy is the pattern that looks right and is wrong. My claim is that taking the locally best choice at each step gives the global optimum, and I should justify that rather than assume it — usually with an exchange argument: if some optimal solution differs from my greedy choice, I can swap in the greedy choice without making it worse. Sorting first is almost always part of the setup, so it's n log n dominated by the sort, then a linear sweep. If I can't articulate why greedy is safe here, that's my cue to say so and fall back to DP, which is slower but provably correct. Getting caught with an unjustified greedy is much worse than shipping a correct n squared.

---

## 17. Intervals

### Recognition signals

- Input is a list of intervals `[start, end]`.
- "Merge overlapping / insert / count overlapping / minimum rooms."

### Mental model

Sort intervals (usually by start). Sweep. Track running state (current end, active count, etc.).

### Templates

**Merge overlapping:**

```python
intervals.sort()
result = [intervals[0]]
for s, e in intervals[1:]:
    if s <= result[-1][1]:
        result[-1][1] = max(result[-1][1], e)
    else:
        result.append([s, e])
return result
```

**Sweep line / minimum rooms:**

```python
events = []
for s, e in intervals:
    events.append((s, 1))   # start
    events.append((e, -1))  # end
events.sort()
active = max_active = 0
for _, delta in events:
    active += delta
    max_active = max(max_active, active)
return max_active
```

### Representative problems

- **Insert Interval.** Three phases: before, overlap, after.
- **Merge Intervals.** Sort + sweep.
- **Non-overlapping Intervals.** Sort by end; greedy keep earliest-ending.
- **Meeting Rooms / II.** Sort + sweep / heap.
- **Minimum Interval to Include Each Query.** Sort intervals + queries; heap of active intervals.

### Common traps

- Sort by start vs by end matters.
- "Touch" vs "overlap" (`<=` vs `<`).
- Empty input.

> **Saying it out loud.** Interval problems are almost all the same problem after you sort. Sort by start time, then sweep: if the current interval starts before the previous one ended, they overlap, so I merge by extending the end to the max of the two; otherwise I close out the previous one and start fresh. That's n log n for the sort and then linear, so n log n overall, O of n space for the output. The one variant worth flagging: for "maximum non-overlapping intervals" you sort by *end* time, not start, because finishing early leaves the most room for everything after. Edge cases: intervals that merely touch at an endpoint — I'd ask you whether [1,2] and [2,3] count as overlapping, because the problem statements differ.

---

## 18. Math & Geometry

### Recognition signals

- Number theory, modular arithmetic, GCD/LCM.
- Geometry: rotation, reflection.
- "Without using the obvious O(n) data structure."

### Mental model

Often there's a closed-form or a clever invariant.

### Representative problems

- **Rotate Image.** Transpose + reverse rows.
- **Spiral Matrix.** Layered traversal with shrinking bounds.
- **Set Matrix Zeroes.** Use first row/col as markers; $O(1)$ extra.
- **Happy Number.** Floyd cycle detection on the digit-sum-of-squares function.
- **Plus One.** Carry propagation.
- **Pow(x, n).** Fast exponentiation in $O(\log n)$.
- **Multiply Strings.** Schoolbook with carry, build result in reverse.
- **Detect Squares.** Hash points; brute force pairs.

### Common traps

- Off-by-one in spiral.
- Negative exponent in Pow → 1/Pow(x, -n) and handle int min.
- Integer overflow in languages with fixed int.

> **Saying it out loud.** These are usually about spotting an invariant rather than an algorithm. Rotating a matrix in place is transpose then reverse each row — two simple passes instead of trying to reason about a four-way cyclic swap, and it's O of n squared time with O of 1 space. The spiral is four boundary pointers that close inward, and the only tricky part is the check that stops you double-traversing the middle row or column when the boundaries meet. Set-matrix-zeroes is the one where the interesting version uses the first row and column as your own marker storage to get O of 1 space. Edge cases here are non-square matrices and empty input, and I'd rather trace a three-by-three by hand than trust the indices.

---

## 19. Bit Manipulation

### Recognition signals

- "Without using extra space."
- "Single number / pair where everyone else appears twice."
- Constraints involving powers of 2 or binary representations.

### Mental model

XOR has cancellation. AND extracts. OR sets. Shift moves. Many problems reduce to clever combinations.

### Useful identities

- `x ^ x = 0`, `x ^ 0 = x` → XOR cancels.
- `x & (x - 1)` clears lowest set bit.
- `x & -x` isolates lowest set bit.
- Popcount: Brian Kernighan's loop or `bin(x).count('1')`.
- Toggle bit $k$: `x ^ (1 << k)`.
- Set bit $k$: `x | (1 << k)`.
- Clear bit $k$: `x & ~(1 << k)`.
- Test bit $k$: `(x >> k) & 1`.

### Representative problems

- **Single Number.** XOR all → leftover is the unique. $O(n)$ time, $O(1)$ space.
- **Number of 1 Bits.** Brian Kernighan loop.
- **Counting Bits (0..n).** $\text{count}(i) = \text{count}(i >> 1) + (i \& 1)$.
- **Reverse Bits.** Bit-by-bit reverse, or divide-and-conquer in halves.
- **Missing Number.** XOR 0..n with arr.
- **Sum of Two Integers without `+`.** Loop with AND-shift carry, XOR sum.

### Common traps

- Negative numbers in two's complement.
- Bit-shift on signed integers.
- Off-by-one on bit indexing.

> **Saying it out loud.** The two identities that carry most of these: XOR with itself is zero and XOR with zero is identity, so XOR-ing an entire array cancels every element that appears twice and leaves the one that doesn't — O of n time, O of 1 space, no hash set needed. And n AND n-minus-one clears the lowest set bit, so counting set bits runs in as many iterations as there are ones rather than thirty-two. I'll say what n AND n-minus-one does out loud rather than just typing it, because otherwise it reads as a magic incantation. The gotcha to name: Python integers are arbitrary precision and negatives don't wrap, so any problem involving 32-bit overflow needs an explicit mask.

---

## 20. Problem-to-pattern transformation tricks

These are the "ah-ha" moves that turn a hard problem into a known pattern.

- **Sort first.** Often turns chaos into an ordered sweep / two-pointer / greedy.
- **Build a graph.** Many problems become BFS/DFS once you express dependencies.
- **Reverse direction.** Sometimes processing from right or end-to-start exposes structure.
- **Prefix sums.** Range sum queries → prefix array, then differences.
- **Difference array.** Range update → mark deltas at endpoints; sweep.
- **Bucketize.** Continuous values → buckets → counting / sliding window.
- **Coordinate compression.** Replace original values by their rank when only ordering matters.
- **Two passes.** Left-to-right + right-to-left and combine. Trapping Rain Water, Product Except Self.
- **Inversion / negation.** Find the smallest such that NOT property holds, etc.
- **Binary search on the answer.** Numerical answer space + monotonic feasibility.
- **Hash by canonical form.** Anagrams → sorted tuple. Isomorphic strings → first-occurrence index.
- **Count instead of construct.** Often you just need a number, not the actual object.
- **Memoize the recursion.** Brute-force recursion + memoization = DP.
- **Floyd cycle on implicit graph.** Find the Duplicate Number; Happy Number.
- **Treat 2D as 1D.** Search a 2D Matrix → flatten via row*ncols + col.
- **Split into halves.** Median of Two Sorted Arrays. Quickselect partitioning.

> **Saying it out loud.** If I'm stuck, I say so and then start listing transformations out loud rather than going quiet — that's worth real points on its own. What if I sort first, does that turn this into a sweep? Can I phrase the dependencies as a graph and just BFS it? Would prefix sums turn these range queries into subtractions? Is the answer a number in a monotonic range, in which case I can binary search on the answer itself rather than compute it? Can I count instead of construct, since the problem only wants a count? The one to reach for when nothing else lands is the brute-force recursion plus memoization, because that's how every DP solution actually gets discovered.

---

## 21. Complexity reasoning cheatsheet

- $O(\log n)$ — binary search; balanced BST ops; heap push/pop.
- $O(\sqrt{n})$ — primality; Mo's algorithm.
- $O(n)$ — single pass; hashing; counting; sliding window.
- $O(n \log n)$ — sorting; heap on $n$ elements; balanced BST.
- $O(n^2)$ — nested loops; basic DP on pairs; LCS.
- $O(n^3)$ — Floyd-Warshall; certain range DP.
- $O(n \cdot 2^n)$ — bitmask DP over subsets.
- $O(2^n)$ — brute-force subsets; certain backtracking.
- $O(n!)$ — permutations.

If $n = 10^5$, you need $O(n \log n)$ or better. If $n = 10^4$, $O(n^2)$ might work. If $n \le 20$, exponential is fine.

> **Saying it out loud.** I read complexity off the constraints before I pick an approach, and I say that reasoning out loud. n up to ten to the fifth means I need n log n or better, so sorting is fine and nested loops are not. n around ten thousand means n squared will probably pass. n at twenty or below is an explicit invitation to do something exponential — subsets or bitmask DP — and trying to find a polynomial solution there is usually wasted time. That reasoning is a genuine signal to the interviewer, because it shows I'm choosing an approach against the actual constraints rather than reciting the fastest algorithm I know for the problem shape.

---

## 22. Common interview traps

- **Solving silently.** Always think aloud. Interviewer is rating *your reasoning*, not your final code.
- **Diving into code too fast.** Spend 2-5 minutes restating, examples, brute force, optimal idea, complexity, *then* code.
- **Wrong data structure.** Not using a hash map → $O(n^2)$ when $O(n)$ is expected.
- **Off-by-one.** Especially in binary search and sliding window.
- **Uninitialized state.** DP base case; visited set.
- **Mutating input without permission.** Some interviewers care.
- **Edge cases ignored.** Empty input, single element, all duplicates, negatives.
- **No complexity analysis.** Always state time and space at the end.
- **No testing.** Trace through your code with a small example before declaring done.
- **Premature optimization.** Brute force first if you're stuck.
- **Stuck silently.** When stuck, narrate: "If I sort first, then..." or "What if I tried two pointers here?"
- **Not asking clarifying questions.** Constraints, duplicates, negatives, output format.
- **Memorizing without understanding.** Interviewer asks a variant; you can't adapt.
- **Recursion depth.** Python defaults to 1000; for huge inputs, iterate or `sys.setrecursionlimit`.
- **Integer overflow.** Not in Python, but in Java/C++ — multiply two `int`s often overflows.

> **Saying it out loud.** The biggest one isn't technical: it's going silent. The interviewer is grading your reasoning, and a quiet candidate who produces correct code often scores below a talkative one who doesn't finish. So when I get stuck I narrate the stuck-ness — "the brute force is n squared, and the thing I want to avoid is re-scanning; let me think about whether sorting first buys me anything." The second biggest is coding before agreeing on the problem. I spend two to five minutes restating, making up an edge case, stating the brute force and the optimal idea and both complexities, and only then type. And I always trace a small example through the finished code before saying I'm done — that's where the off-by-one shows up.

---

## 23. The 5-step problem-solving protocol

Use this verbatim in every interview.

**Step 1: Understand.** Read aloud. Ask about constraints, duplicates, negatives, edge cases, output format.

**Step 2: Examples.** Walk through the given examples + 1-2 you make up (edge cases).

**Step 3: Brute force.** State the brute-force approach and complexity. ("I could try every pair, $O(n^2)$.")

**Step 4: Optimize.** Identify the pattern. ("This looks like sliding window because...") State the optimal approach and complexity.

**Step 5: Code.** Write clean code. State invariants. Trace through an example. State final time/space complexity.

If stuck at step 4, return to step 3 and code the brute force; many interviews accept it.

> **Saying it out loud.** "Let me make sure I understand — the input is X, I return Y, and n can be up to this. Are there duplicates? Can values be negative?" Then: "Let me walk your example, plus one of my own — an empty input." Then: "The brute force is checking every pair, O of n squared. That works but we can do better." Then: "This looks like sliding window, because the constraint is on a contiguous range — that gets us to O of n." Then code, narrating the invariant as I write it, and finish by tracing an example and restating time and space. That's the whole script, and it's worth using verbatim because if you run out of time at step four, an interviewer who watched you reason still has plenty to write down.

---

## 24. Drilling routine

Two months out from a coding round:

- **Week 1-2:** Arrays, hashing, two pointers, sliding window. Solve 5 per pattern.
- **Week 3-4:** Stack, binary search, linked list. 5 per pattern.
- **Week 5-6:** Trees, tries, heap. 5 per pattern.
- **Week 7:** Backtracking, graphs, advanced graphs. 5 per pattern.
- **Week 8:** 1D and 2D DP. 5+ per pattern.
- **Week 9:** Greedy, intervals, math, bits. 3 per pattern.
- **Last week:** Mock interviews, mixed problems on shuffle, one set per day.

Daily routine:
1. Pick a problem.
2. Try 30 minutes solo (use the 5-step protocol).
3. Check editorial / discussion.
4. If new pattern, take 10 minutes to add to your notes.
5. Re-attempt next day from scratch.

---

## How to use this chapter

1. Read once, top to bottom. Note which patterns feel weak.
2. For weak patterns, drill 5+ problems each.
3. Re-read §1 (triage), §20 (transformations), §22 (traps), §23 (protocol) every Sunday.
4. Pair with `INTERVIEW_GRILL.md` — pattern-recognition active recall.
5. Practice on NeetCode 150 / Blind 75 / Grind 75 — the curated lists.

The single sentence to remember: **for any problem, do the 30-second triage by input shape, output shape, constraints, and structural cues; then match to one of 18 patterns; then deploy the template.**
