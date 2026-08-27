# The algorithms round

Sixty minutes, one or two problems, and a grader who cares as much about how you get there as about whether you finish. The material is a small set of patterns, not a list of problems, so the preparation that works is recognising which pattern a phrasing implies and then writing that pattern correctly from memory. Candidates lose this round two ways: they start coding before they have stated the approach, and they go silent when stuck. Both are avoidable with a fixed routine. Learn eight patterns properly rather than three hundred problems badly.

## The equations

**Pattern complexity table.** $n$ is the input size, $k$ the parameter in the problem (window size, number of results, alphabet size), $V$ and $E$ the vertex and edge counts, $W$ the range of the search space.

| Pattern | Time | Space | Recognition cue |
|---|---|---|---|
| Two pointers on a sorted array | $O(n)$ after $O(n \log n)$ sort | $O(1)$ | pair or triple summing to a target |
| Sliding window | $O(n)$ | $O(k)$ | longest or shortest contiguous subarray or substring |
| Hash map counting | $O(n)$ | $O(n)$ | "have I seen", frequency, anagram, complement |
| Binary search on a sorted array | $O(\log n)$ | $O(1)$ | sorted input, find position or boundary |
| Binary search on the answer | $O(n \log W)$ | $O(1)$ | "minimum maximum", "smallest capacity that works" |
| BFS on a graph | $O(V + E)$ | $O(V)$ | fewest steps, shortest path, unweighted |
| DFS on a graph | $O(V + E)$ | $O(V)$ | connectivity, cycles, all paths, backtracking |
| Topological sort | $O(V + E)$ | $O(V)$ | prerequisites, ordering, dependencies |
| Union-find | $O(E\,\alpha(V))$ | $O(V)$ | connected components, merging groups online |
| Heap for top-k | $O(n \log k)$ | $O(k)$ | k largest, k closest, running median |
| Interval sweep | $O(n \log n)$ | $O(n)$ | merge, overlap, meeting rooms |
| 1-D dynamic programming | $O(n \cdot t)$ | $O(n)$ | count the ways, best over choices, no greedy proof |
| 2-D dynamic programming | $O(nm)$ | $O(nm)$ or $O(\min(n,m))$ | two sequences compared |

**Cost of a dynamic program**

$$T = (\text{number of distinct states}) \times (\text{work per state})$$

This is the only complexity formula you need for DP; count the states from the memo key and the work from the transition loop.

**Amortised inverse Ackermann**

$$\alpha(V) \le 4 \text{ for any } V \text{ that fits in memory}$$

Union-find with path compression and union by rank is effectively constant time per operation, so say "effectively constant, formally inverse Ackermann".

**Halving recurrence**

$$T(n) = T(n/2) + O(1) \implies T(n) = O(\log n)$$

Any loop that discards half the remaining search space each iteration is logarithmic, which is what makes binary search on the answer cheap even when the check is expensive.

**Recursion depth**

$$\text{stack space} = O(\text{depth})$$

A recursion over $n$ elements that recurses one step at a time costs $O(n)$ stack, and Python's default limit near 1000 frames is the reason a correct memoised solution can still crash on a large input.

## Code from memory

**1. Binary search — the boundary form, and binary search on the answer.** The exclusive-`hi` invariant is the part people get wrong.

```python
def lower_bound(a, target):
    # first index i with a[i] >= target; returns len(a) if none
    lo, hi = 0, len(a)            # hi is EXCLUSIVE, so len(a) is a valid answer
    while lo < hi:                # strict <, not <=, with an exclusive hi
        mid = lo + (hi - lo) // 2 # no overflow, and mid < hi always
        if a[mid] < target:
            lo = mid + 1          # a[mid] cannot be the answer
        else:
            hi = mid              # a[mid] might be the answer, so keep it
    return lo

def min_days_to_ship(weights, days):
    # binary search on the ANSWER: smallest capacity that finishes in `days`
    def ok(cap):
        used, load = 1, 0
        for w in weights:
            if load + w > cap:
                used, load = used + 1, 0
            load += w
        return used <= days
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid): hi = mid
        else:       lo = mid + 1
    return lo

a = [1, 3, 3, 5, 8]
print([lower_bound(a, t) for t in [0, 3, 4, 8, 9]])   # -> [0, 1, 3, 4, 5]
print(min_days_to_ship([1,2,3,4,5,6,7,8,9,10], 5))    # -> 15
```

Output:

```
[0, 1, 3, 4, 5]
15
```

Note `lower_bound(a, 3)` returns 1, the first of the two 3s, and `lower_bound(a, 9)` returns 5, which is past the end. Both are correct and both are the cases that a `lo <= hi` version with inclusive `hi` gets wrong.

**2. Sliding window with a hash map.** One expanding pointer, one contracting pointer, a map of state inside the window.

```python
def longest_unique(s):
    # longest substring with no repeated character
    last_seen = {}          # char -> most recent index
    left, best, best_at = 0, 0, 0
    for right in range(len(s)):
        c = s[right]
        # if c is already inside the window, move left past its old position
        if c in last_seen and last_seen[c] >= left:
            left = last_seen[c] + 1
        last_seen[c] = right
        if right - left + 1 > best:
            best, best_at = right - left + 1, left
    return best, s[best_at:best_at + best]

def min_window_k_distinct(s, k):
    # shortest window containing at least k distinct characters
    count, distinct, left, best = {}, 0, 0, None
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        if count[s[right]] == 1: distinct += 1
        while distinct >= k:                      # shrink while still valid
            if best is None or right - left + 1 < len(best):
                best = s[left:right + 1]
            count[s[left]] -= 1
            if count[s[left]] == 0: distinct -= 1
            left += 1
    return best
```

```python
print(longest_unique("abcabcbb"), longest_unique("bbbbb"), longest_unique(""))
print(min_window_k_distinct("aaabcd", 3), min_window_k_distinct("aaa", 2))
```

Output:

```
(3, 'abc') (1, 'b') (0, '')
abc None
```

The two shapes are different and worth separating in memory. A longest-valid window jumps `left` forward directly; a shortest-valid window shrinks with a `while` loop.

**3. BFS with a visited set and parent reconstruction.**

```python
from collections import deque

def bfs_shortest(graph, start, goal):
    # BFS gives the fewest-edges path on an unweighted graph
    visited = {start}                 # mark on ENQUEUE, not on dequeue
    parent = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal:
            path, cur = [], node      # walk the parent chain back
            while cur is not None:
                path.append(cur); cur = parent[cur]
            return path[::-1]
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                q.append(nb)
    return None

G = {"A": ["B", "C"], "B": ["D"], "C": ["D", "E"], "D": ["F"], "E": ["F"], "F": [], "Z": []}
print(bfs_shortest(G, "A", "F"))    # -> ['A', 'B', 'D', 'F'] (3 edges)
print(bfs_shortest(G, "A", "Z"))    # -> None, unreachable
```

Output:

```
['A', 'B', 'D', 'F']
None
```

Mark visited on enqueue, never on dequeue. Marking on dequeue lets the same node enter the queue several times before it is processed, which turns $O(V+E)$ into something much worse on dense graphs.

**4. Dynamic programming, memoised recursion and bottom-up table side by side.** Same problem, same state, both directions.

```python
def coins_memo(coins, amount, memo=None):
    # state: fewest coins to make exactly `amount`. Answer is f(amount).
    if memo is None: memo = {}
    if amount == 0: return 0
    if amount < 0:  return float("inf")
    if amount in memo: return memo[amount]
    best = float("inf")
    for c in coins:
        best = min(best, 1 + coins_memo(coins, amount - c, memo))
    memo[amount] = best
    return best

def coins_table(coins, amount):
    # same state, filled small-to-large so no recursion is needed
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount]

coins = [1, 3, 4, 7]
print(coins_memo(coins, 11), coins_table(coins, 11))      # -> 2 2  (4+7)
print(coins_memo([2], 3), coins_table([2], 3))            # -> inf inf
same = all(coins_memo(coins, n) == coins_table(coins, n) for n in range(60))
print("memo and table agree for amounts 0..59:", same)
```

Output:

```
2 2
inf inf
memo and table agree for amounts 0..59: True
```

Both are $O(\text{amount} \times |\text{coins}|)$ time and $O(\text{amount})$ space. The memoised version is easier to derive because you write the recurrence directly; the table version has no stack limit and is the one to convert to when the input is large.

## Questions

### Q1. How do you recognise which pattern a problem needs from its phrasing?

I read for four signals. First, the word "contiguous" or "substring" or "subarray" plus "longest" or "shortest" means sliding window. Second, "sorted" in the input description means two pointers or binary search, and if the phrasing is "minimum maximum" or "smallest value that works", it is binary search on the answer. Third, "shortest path", "fewest steps", or "minimum number of moves" on an unweighted structure means BFS; "is it reachable", "count components", or "all possible" means DFS or backtracking. Fourth, "count the number of ways", "maximum over a sequence of choices", or a problem where a greedy rule almost works but I can construct a counterexample means dynamic programming. If two patterns fit, I say both out loud and pick by complexity: sliding window at $O(n)$ beats DP at $O(n^2)$ when both are valid. That commentary is itself worth marks, because it shows the choice was reasoned rather than remembered.

> **Say it.** I read for four signals. Contiguous plus longest or shortest means sliding window. Sorted input means two pointers or binary search, and minimum-maximum phrasing means binary search on the answer. Fewest steps on an unweighted graph means BFS; reachability or all-paths means DFS. Count the ways, or best over a sequence of choices where greedy has a counterexample, means dynamic programming. If two fit I say both out loud and pick by complexity. That reasoning is worth marks by itself.

### Q2. When do you use two pointers and when a sliding window?

Two pointers usually means one pointer at each end of a sorted array moving inward, and it works when the sortedness gives you a monotone decision rule: if the current sum is too large, only moving the right pointer left can help, so the left pointer never needs to revisit. That is what makes it $O(n)$ rather than $O(n^2)$. A sliding window is two pointers moving in the same direction over a sequence, maintaining a window that satisfies some property; it works when the property is monotone in the window, so extending can only break validity and shrinking can only restore it. The window does not need sorted input, but it does need that monotonicity. Both are $O(n)$ with $O(1)$ or $O(k)$ extra space. The test I apply is whether advancing a pointer permanently rules out a region of the search space. If it does, one of the two applies. If it does not, I need a different pattern.

> **Say it.** Two pointers is usually opposite ends of a sorted array moving inward, and it works because sortedness makes the decision monotone — if the sum is too big only the right pointer moving left can help, so the left one never backtracks. Sliding window is two pointers moving the same direction, keeping a window valid, and it needs the validity property to be monotone in the window. Both are linear. My test is whether advancing a pointer permanently rules out part of the search space.

### Q3. What are the standard hash map patterns?

Four cover most of what appears. Complement lookup: for two-sum, store each value's index as you scan and look up `target - x` before inserting, which turns $O(n^2)$ into $O(n)$ in one pass. Frequency counting: count occurrences and compare counts, which handles anagrams, duplicates, and "does one string contain the letters of another". Canonical key grouping: map each item to a normalised key — the sorted characters of a word, or a shape signature — and append to a list under that key, which is the group-anagrams pattern. Prefix-sum with a map: store how many times each running sum has occurred, so a subarray summing to $k$ becomes a lookup for `running - k`, which is the trick that makes "count subarrays summing to k" linear even with negative numbers. All four trade $O(n)$ space for removing a nested loop. The mistake to avoid is inserting before looking up in the complement pattern, which lets an element pair with itself.

> **Say it.** Four patterns. Complement lookup — store as you scan, look up target minus x before inserting, which is two-sum in one pass. Frequency counting, for anagrams and duplicates. Canonical key grouping, where you map each item to a normalised key like its sorted characters. And prefix sums in a map, where you store counts of each running total so a subarray summing to k becomes a lookup for running minus k — that one handles negative numbers, which sliding window cannot. All of them trade linear space for dropping a nested loop.

### Q4. What is binary search on the answer, and how do you spot it?

It is binary search over the space of possible answers rather than over an array. It applies when three things hold: the answer is an integer or a bounded real in a known range, there is a predicate `ok(x)` that says whether $x$ is feasible, and that predicate is monotone — if $x$ works then every larger $x$ works too. Then the feasible region is a suffix of the range and you binary search for its first element. The phrasings that signal it are "minimum maximum", "maximum minimum", "smallest capacity such that", "minimum speed to finish in time", and "split into k parts minimising the largest part". The complexity is $O(n \log W)$ where $W$ is the width of the answer range and each check costs $O(n)$. The shipping-capacity function above is the canonical example: `ok(cap)` greedily packs and asks whether the day count fits, and the search finds the smallest capacity for which it does. The step people miss is proving monotonicity of `ok` out loud before coding.

> **Say it.** You binary search over possible answers rather than over an array. It needs three things: a bounded answer range, a feasibility check, and monotonicity — if a value works, every larger value works. Then the feasible set is a suffix and you search for its first element. The giveaway phrasings are minimum maximum, maximum minimum, or smallest capacity that finishes in time. Cost is n log W, where the check is linear. The step people skip is stating out loud why the predicate is monotone before writing it.

### Q5. BFS or DFS — when is each correct?

BFS explores by distance from the source, so it finds the fewest-edges path on an unweighted graph, and that guarantee is the reason to choose it. DFS explores one branch to exhaustion, so it is the right shape for connectivity, cycle detection, topological ordering, and any backtracking search over all configurations. Both are $O(V+E)$ time. The difference that matters in practice is memory: BFS holds a whole frontier, which on a wide graph or a large grid can be $O(V)$ and large, while DFS holds one path, which is $O(\text{depth})$ but risks a stack overflow when the depth is large — in Python the recursion limit is near 1000 frames. Two correctness points. BFS is only shortest-path for unweighted or unit-weight edges; with weights you need Dijkstra. And you must mark nodes visited when you enqueue them, not when you dequeue them, or duplicates pile up in the queue.

> **Say it.** BFS explores by distance, so it gives the fewest-edges path on an unweighted graph, and that guarantee is why I pick it. DFS goes deep, so it suits connectivity, cycle detection, topological sort, and backtracking. Both are V plus E. The real difference is memory: BFS holds a whole frontier, DFS holds one path but can blow the stack past about a thousand frames in Python. Two traps: BFS is only shortest-path without weights, and you mark visited on enqueue, not on dequeue.

### Q6. How do you spot a dynamic programming problem and define the state?

I suspect DP when the problem asks to count the ways, or to optimise over a sequence of choices, and when a greedy rule is tempting but I can construct a counterexample. The coin problem above is exactly that: greedy largest-first fails on coins 1, 3, 4 for amount 6, because greedy gives 4+1+1 and the optimum is 3+3. To define the state I ask what minimal set of facts I need to decide the rest of the problem, and the answer is the memo key. For coins it is the remaining amount. For two sequences it is a pair of indices. For a knapsack it is an index plus remaining capacity. Then I write the recurrence as a choice over the first decision, plus the answer to the smaller problem, then name the base cases. The complexity follows immediately: number of distinct states times work per state. I write the memoised version first because the recurrence transcribes directly, and convert to a table only if I need the speed or the stack safety.

> **Say it.** I suspect DP when the problem counts ways or optimises over a sequence of choices, and when greedy is tempting but I can build a counterexample — coins one, three, four for amount six breaks greedy. To define the state I ask what minimal facts decide the rest of the problem; that is the memo key. Remaining amount for coins, a pair of indices for two sequences, index plus capacity for knapsack. Then the recurrence is a choice over the first decision plus the smaller subproblem. Complexity is states times work per state.

### Q7. How do you convert a memoised recursion into a bottom-up table?

Three mechanical steps. First, the memo key becomes the table index, so a single-integer key gives a one-dimensional array and a pair of indices gives a two-dimensional one. Second, the base cases become the table's initial values — `dp[0] = 0` for coins, an infinity fill for unreachable states. Third, choose an iteration order that guarantees every value the recurrence reads is already written. For coins that is increasing amount, because `dp[a]` reads `dp[a - c]` for positive `c`. For a two-sequence problem it is usually increasing in both indices, and for a knapsack the capacity loop sometimes has to run downward to prevent reusing an item. The reasons to convert are that the table has no recursion depth limit and no per-call overhead, and that it often exposes a space reduction, because if row $i$ reads only row $i-1$ you can keep two rows instead of the whole grid. I verify the conversion by asserting both versions agree over a range of inputs, which is what the code above does for amounts 0 through 59.

> **Say it.** Three steps. The memo key becomes the table index. The base cases become the initial fill. And I pick an iteration order where every value the recurrence reads is already written — increasing amount for coins, sometimes a downward capacity loop for knapsack so items are not reused. I convert for the stack limit, the lower overhead, and because the table often exposes a space reduction when row i only reads row i minus one. Then I assert the two versions agree over a range of inputs.

### Q8. When do you reach for a heap?

When I need the k best of something and k is much smaller than n, or when I need repeated access to the current extreme of a changing set. For top-k, I keep a min-heap of size k, push each element, and pop when the size exceeds k. That is $O(n \log k)$ time and $O(k)$ space, which beats sorting everything at $O(n \log n)$ when k is small, and it works on a stream where sorting cannot. The counter-intuitive part worth saying out loud is that top-k largest uses a min-heap, because the root is the weakest survivor and is the element you evict. Other uses: merging k sorted lists by holding one element from each; the running median with two heaps, a max-heap for the lower half and a min-heap for the upper half, rebalanced so their sizes differ by at most one; and Dijkstra, where the heap holds frontier nodes by tentative distance. In Python `heapq` is a min-heap, so for maximum behaviour I push negated values.

> **Say it.** When I need the k best and k is much smaller than n, or repeated access to the current extreme of a changing set. Top-k largest uses a min-heap of size k — the root is the weakest survivor, so it is the one you evict. That is n log k against n log n for a full sort, and it works on a stream. Also merging k sorted lists, the running median with two heaps rebalanced to differ by at most one, and Dijkstra's frontier. Python's heapq is a min-heap, so I negate for maximum.

### Q9. How do you handle interval problems?

Almost always sort first, then sweep once. For merging overlapping intervals, sort by start, then walk through keeping a current interval: if the next start is at most the current end, extend the end to the maximum of the two ends, otherwise emit the current interval and start a new one. That is $O(n \log n)$ for the sort and $O(n)$ for the sweep. For "how many rooms do I need", the cleaner form is an event sweep: emit a plus-one event at each start and a minus-one at each end, sort all events by time, and track the running count; its maximum is the answer. Ties matter and you must state your convention — if an interval ending at time 5 and another starting at time 5 do not conflict, the end event must sort before the start event. For "does a new interval overlap an existing set", keep the set sorted and binary search. The two decisions to say out loud are what you sort by and how you break ties.

> **Say it.** Sort first, then sweep once. To merge, sort by start and walk through: if the next start is at most the current end, extend the end to the max, otherwise emit and start fresh. For meeting rooms I use an event sweep — plus one at each start, minus one at each end, sort by time, and the running maximum is the answer. Tie handling is the part to state explicitly: if an interval ending at five does not conflict with one starting at five, the end event sorts first. n log n for the sort, linear for the sweep.

### Q10. Implement or explain topological sort.

Kahn's algorithm is the one I write, because it is iterative and it detects cycles for free. Compute the in-degree of every node by scanning all edges. Put every node with in-degree zero into a queue. Repeatedly pop a node, append it to the output order, and for each of its out-neighbours decrement the in-degree and enqueue that neighbour if it reaches zero. When the queue is empty, if the output holds fewer than $V$ nodes then the remaining nodes are in a cycle and no valid ordering exists — that is the course-schedule answer. The complexity is $O(V + E)$ time and $O(V)$ space. The DFS alternative pushes each node onto a stack after all its descendants are finished and reverses the result at the end, which needs a three-colour marking to detect back edges as cycles. I prefer Kahn's under pressure because the cycle check is a single length comparison rather than a colour scheme I might get wrong.

> **Say it.** I write Kahn's algorithm. Compute in-degrees, queue every node with in-degree zero, then repeatedly pop a node into the output and decrement its neighbours' in-degrees, enqueueing any that hit zero. If the output has fewer than V nodes at the end, the rest form a cycle and no ordering exists — that is the course-schedule answer for free. V plus E time. The DFS version pushes nodes after their descendants finish and reverses, but it needs three-colour marking for cycles, and under pressure Kahn's length check is safer.

### Q11. Explain union-find at the level you can implement it.

Two arrays, `parent` and `rank`, both indexed by node. `find(x)` follows parent pointers to the root and applies path compression by pointing every node it passed directly at the root. `union(a, b)` finds both roots; if they are equal nothing happens, otherwise it attaches the shallower tree under the deeper one, using rank as the depth estimate. With both optimisations the amortised cost per operation is $O(\alpha(V))$, inverse Ackermann, which is at most 4 for any input that fits in memory, so I describe it as effectively constant. I use it when components merge incrementally and I need to ask about connectivity as I go: counting connected components, detecting a cycle while adding edges to an undirected graph, Kruskal's minimum spanning tree, and grouping accounts or emails. The comparison with DFS is that DFS answers connectivity on a fixed graph in one pass, while union-find handles edges arriving one at a time without rebuilding.

> **Say it.** Two arrays, parent and rank. Find walks to the root and path-compresses by repointing everything it passed at the root. Union finds both roots and hangs the shallower tree under the deeper one by rank. With both, it is amortised inverse Ackermann per operation — effectively constant, at most four for anything that fits in memory. I use it when components merge incrementally: counting components, cycle detection while adding edges, Kruskal's MST, grouping accounts. DFS handles a fixed graph; union-find handles edges arriving online.

### Q12. How do you reason about time and space complexity out loud while coding?

I state the complexity of the approach before I write it, so the interviewer can stop me if it is not what they want. Then I annotate as I go: "this loop is over n, this lookup is amortised constant, so this block is $O(n)$". At the end I give the total and the space separately, and I name what dominates. Three habits keep it honest. I count the sort explicitly rather than forgetting it, because $O(n \log n)$ from a sort often dominates a linear scan. I count the space that the output and the recursion stack take, not only the auxiliary structures. And I say amortised where it is amortised — a hash lookup is $O(1)$ average and $O(n)$ worst case, and a dynamic array append is amortised constant. If the interviewer asks whether I can do better, the useful reply is to name the lower bound: you cannot beat $O(n)$ if you must read every element, so any improvement must be in the constant or in the space.

> **Say it.** I state the complexity of the approach before writing it, so they can redirect me early. Then I annotate as I go — this loop is over n, this lookup is amortised constant, so this block is linear. At the end I give time and space separately and name what dominates. I count the sort, which is often the dominant term, and I count the recursion stack as space. I say amortised where it is amortised. If asked to do better, I name the lower bound: you cannot beat linear if you must read every element.

### Q13. What do you do when you are stuck for two minutes?

I say that I am stuck, out loud, and I say precisely where. Silence is the worst outcome, because the interviewer cannot tell whether I am thinking productively or lost, and they cannot give me the hint they are usually willing to give. Then I run a fixed escalation. First, work a small concrete example by hand and write the intermediate state down; most stuck moments are a missing invariant that becomes obvious on a five-element input. Second, state the brute force and its complexity, and get it coded if nothing better arrives, because a working $O(n^2)$ solution scores far more than an unfinished $O(n)$ one. Third, ask what makes the brute force slow, since the answer names the structure to add — repeated lookups mean a hash map, repeated scanning of a range means a heap or a prefix sum, repeated subproblems mean memoisation. Fourth, ask a direct question: "I am considering a heap or sorting here, do you have a preference?" That is not a failure, it is normal collaboration.

> **Say it.** I say I am stuck and exactly where, because silence is the worst outcome — they cannot tell if I am thinking or lost, and they cannot give me the hint they are usually happy to give. Then a fixed escalation. Work a five-element example by hand and write the state down. State the brute force and code it if nothing better arrives, because a working n-squared beats an unfinished linear. Ask what makes the brute force slow, because the answer names the structure to add. Then ask a direct question.

### Q14. How do you communicate so that a partial solution still scores?

I follow a fixed sequence. Restate the problem and confirm the input and output types, the size bounds, and one edge case, so we agree on the task. State the approach and its complexity, and get agreement before writing code. Write the solution top-down: name the helper functions I will need, write the main structure, then fill in the bodies, so at any moment there is a visible skeleton of a complete answer rather than one perfect function and nothing else. Narrate each block in one sentence as I write it. Then trace a small example by hand through the finished code, out loud, because that finds off-by-one errors faster than staring. Finally state the complexity and name the edge cases I handled and the ones I would add tests for. If time runs out, this ordering means the interviewer has seen the full approach, the complexity analysis, and working partial code, all of which are scored separately from whether the last line compiled.

> **Say it.** A fixed sequence. Restate the problem, confirm the types, the size bounds, and one edge case. State the approach and complexity and get agreement before coding. Write top-down, naming helpers first, so there is always a visible skeleton of the whole answer. Narrate each block in one sentence. Then trace a small example by hand through the finished code, which finds off-by-one errors faster than staring. Then state the complexity and the edge cases. If time runs out they have still seen the approach and the analysis, and those are scored separately.

## Done when

- You can write `lower_bound` with the exclusive-`hi` invariant from memory and explain why `hi = mid` rather than `mid - 1` on the else branch.
- Given a problem statement you have not seen, you can name the pattern and its time and space complexity in under 60 seconds, and justify it out loud.
- You can write BFS with parent reconstruction, Kahn's topological sort, and union-find with path compression from a blank file in under 10 minutes each, and all three run on your own test case first time.
- You can take any memoised recursion you write and produce the bottom-up table version, then assert the two agree over a range of inputs.
