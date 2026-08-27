# LeetCode / NeetCode 150 — Interview Grill (Pattern Recognition)

> 150+ pattern-recognition questions — read the problem statement, name the pattern in <30 seconds. Pair with [`LEETCODE_PATTERNS_DEEP_DIVE.md`](LEETCODE_PATTERNS_DEEP_DIVE.md).

The point of these questions is *not* solving — it's **identifying which template to deploy**. Speed of recognition is the leverage skill.

---

## Section A — The triage (Q1–10)

1. List the 7 questions in the 30-second triage in order.
2. For each input shape, name two candidate patterns: single array, sorted array, 2D grid, tree, graph, string, multiple intervals.
3. For each output shape, name two patterns: count, min/max, all combinations, single index/pair.
4. $n \le 20$ — what algorithmic complexity is acceptable?
5. $n \le 10^5$ — what's your target complexity?
6. "Subarray" / "contiguous" — which pattern?
7. "Subsequence" — which pattern?
8. "Top-K" — which pattern?
9. "Connected" — which pattern(s)?
10. "Shortest path, weighted, no negatives" — which?

> **Saying it out loud.** *(What the triage sounds like at the whiteboard.)* "Okay — input's a single array, output's a count, n goes up to ten to the fifth. That constraint's the loudest signal in the problem: it kills anything quadratic, so I'm aiming for linear or n-log-n. And the word 'contiguous' means subarray, not subsequence, which points me at sliding window rather than DP." Thirty seconds, said out loud, and now the interviewer can correct me before I've written a line. The number to anchor on: n at ten to the fifth means n log n or better, n at ten thousand means n squared is probably fine, and n at twenty or under is an explicit invitation to go exponential.

## Section B — Pattern recognition: arrays / hashing / two pointers / sliding window (Q11–25)

11. "Find pair summing to target in sorted array" → pattern?
12. "Find pair summing to target in unsorted array" → pattern?
13. "Group anagrams" → pattern + key idea?
14. "Top K frequent elements" → two patterns + complexities?
15. "Longest substring without repeating chars" → pattern?
16. "Min window substring containing all chars in T" → pattern?
17. "Longest repeating character replacement (k changes)" → pattern + invariant?
18. "Container with most water" → pattern + which side moves?
19. "3Sum" → pattern + complexity?
20. "Trapping rain water" → two-pass approach + alternative?
21. "Best time to buy/sell stock (one transaction)" → pattern + state?
22. "Maximum subarray sum" → name the algorithm.
23. "Product of array except self" → pattern + technique?
24. "Longest consecutive sequence" → pattern + key trick?
25. "Permutation in string" → pattern?

> **Saying it out loud.** *(Narrating the sorted-versus-unsorted fork.)* "The brute force is every pair, n squared. If the array's sorted I don't need extra memory at all — two pointers from the ends, and the sortedness tells me which one to move, so it's O of n time and O of 1 space. If it's unsorted, I trade space for time instead: one pass with a hash map from value to index, checking for the complement as I go. Still O of n, but now O of n space." That's the fork, and saying which one you're in and why is the actual scored moment. The trap to name out loud is inserting into the map before you check, which lets an element pair with itself.

## Section C — Stack / monotonic stack (Q26–33)

26. "Valid parentheses" → pattern?
27. "Min stack" — two ways to design?
28. "Daily temperatures" → pattern + monotonic-which-way?
29. "Next greater element" → pattern?
30. "Largest rectangle in histogram" → pattern + key insight?
31. "Car fleet" → preprocessing + pattern?
32. "Evaluate Reverse Polish Notation" → pattern?
33. "Generate parentheses" → pattern (this is a trap)?

> **Saying it out loud.** *(Justifying the amortized bound before they ask.)* "I know this looks quadratic because there's a while loop inside the for loop, but each element gets pushed exactly once and popped at most once — so it's O of n total, amortized." Say that unprompted; it's the single line that separates someone who memorized the monotonic stack template from someone who understands it. Then name the invariant: the stack stays decreasing, so anything I pop has just found its next greater element. And handle the leftovers explicitly — whatever's still on the stack at the end never found a match and gets the default, usually negative one.

## Section D — Binary search (Q34–43)

34. When does binary search apply? Two requirements.
35. "Find min in rotated sorted array" → pattern + key comparison?
36. "Search in rotated sorted array" → pattern?
37. "Search 2D matrix" → key trick?
38. "Koko eating bananas" → pattern + what to BS on?
39. "Median of two sorted arrays" → pattern + complexity?
40. "Time-based key-value store" → pattern + per-key structure?
41. "Capacity to ship packages within D days" → pattern?
42. "Find peak element" → pattern + invariant?
43. Difference between `l <= r` and `l < r` loop forms — when each?

> **Saying it out loud.** *(Keeping yourself out of the infinite loop.)* "I'm going to write the boundary-finding version rather than exact-match, because it generalizes to 'first element at least x' without extra branches. Low inclusive, high exclusive, mid is low plus high-minus-low over two so I don't overflow in languages where that matters. My invariant is that the answer always lives in the range low to high — that's what guarantees the range shrinks every iteration." Then the harder variant to be ready for: binary search on the *answer*, where the array isn't the search space at all and you're bisecting a numeric range with a monotonic feasibility check. O of log n either way, O of 1 space.

## Section E — Linked list (Q44–53)

44. "Reverse linked list" → recursive + iterative templates?
45. "Detect cycle" → algorithm name?
46. "Find cycle entry" → algorithm + key step?
47. "Merge two sorted lists" → pattern + dummy node?
48. "Reorder list" → 3-step decomposition?
49. "Remove Nth from end" → pattern?
50. "Copy list with random pointer" → two approaches?
51. "LRU cache" → data structures + complexities?
52. "Merge K sorted lists" → two approaches + complexities?
53. "Find the duplicate number (Floyd cycle on implicit graph)" → pattern + insight?

> **Saying it out loud.** *(Two habits, said before you type.)* "I'll add a dummy head node first — it makes inserting or deleting at the front behave identically to the middle, which kills a whole family of null checks." And for anything about the middle, cycles, or the kth-from-end: two pointers, either at different speeds or at a fixed offset. Floyd's cycle detection is the one to be able to justify — the fast pointer gains one position per step on the slow one, so inside a cycle it must eventually land on it. O of n time, O of 1 space, and that constant space is the whole reason to prefer it over a visited set. Edge cases: empty list, single node, and modifying the head itself.

## Section F — Trees (Q54–67)

54. DFS template (recursive postorder, with return).
55. BFS level-order template.
56. "Invert binary tree" → recursion?
57. "Maximum depth" — recurrence?
58. "Diameter of binary tree" — what to track during recursion?
59. "Balanced binary tree" — sentinel pattern?
60. "Same tree" / "subtree" — pattern?
61. "LCA in BST" — pattern + key BST property?
62. "LCA in general tree" — DFS pattern + return condition?
63. "Validate BST" — pattern + key parameter?
64. "Right side view" — DFS or BFS approach?
65. "Construct from preorder + inorder" — pattern?
66. "Kth smallest in BST" — pattern?
67. "Serialize / deserialize" — encoding choice?

> **Saying it out loud.** *(The question that picks your traversal.)* "Does this node need information from its children, or from its parent? Children-up is post-order — I recurse, get values back, and combine. Parent-down is passing state as an argument, like the min-max bounds for validating a BST. And anything about levels or minimum depth is BFS with a queue, not recursion at all." That one question decides the code shape and it's worth asking out loud. Complexity is O of n since every node gets visited once; space is O of h for the stack. Name the worst case: a degenerate, list-shaped tree makes h equal n, which will blow Python's default thousand-frame recursion limit.

## Section G — Tries (Q68–72)

68. When to reach for a trie?
69. Insert / search / starts_with — complexities?
70. "Add and search word with `.` wildcard" → pattern + DFS technique?
71. "Word search II" → pattern + what's the trie used for?
72. Trade-offs between trie and hash set.

> **Saying it out loud.** *(Justifying the data structure choice.)* "A hash set can answer 'does this word exist' in O of L, but it can't answer 'does anything start with this prefix' without scanning everything. That's what the trie buys — insert and search are O of L in word length and completely independent of how many words are stored." Then name the price you're paying, because interviewers want the tradeoff: memory, roughly a node per character in the worst case, which is much heavier than a set. So you build one only when prefix queries are genuinely in the requirements. Edge cases: the empty string, and a stored word that's a proper prefix of another still needing its end-of-word flag.

## Section H — Heap / priority queue (Q73–82)

73. "Kth largest element in array" → two approaches + complexities?
74. "Last stone weight" → pattern?
75. "K closest points to origin" → pattern + heap size?
76. "Task scheduler" → pattern + key data structures?
77. "Find median from data stream" → pattern + invariant?
78. "Merge K sorted lists" → heap formulation?
79. "Top K frequent elements" → two approaches?
80. "Design Twitter" → pattern?
81. Python `heapq` — min or max heap by default?
82. `heapify` complexity?

> **Saying it out loud.** *(Why not just sort.)* "I could sort and take the last k, that's n log n. But I only need the top k, so I'll hold a min-heap capped at size k — push everything, pop whenever it exceeds k — and the weakest of my current best k is always right on top where comparing is cheap. That's O of n log k, which is a real win when k is small, and O of k space." Then the language detail worth saying so the interviewer knows you've actually written it: Python's heapq is min-only, so a max-heap means pushing negated values. Edge cases: k larger than the array, and ties at the boundary.

## Section I — Backtracking (Q83–93)

83. When to reach for backtracking?
84. "Subsets" — include/exclude template?
85. "Permutations" — swap-in-place vs used-array?
86. "Combination sum" — start index + when to recurse?
87. "Combination sum II (with duplicates)" — sort + skip-when?
88. "Word search" — DFS + visited marker pattern?
89. "Palindrome partitioning" — what do you try at each step?
90. "N-queens" — what state to track for $O(1)$ check?
91. "Letter combinations of phone number" — recursion vs iterative?
92. "Sudoku solver" — what 3 sets per cell?
93. Common bug in backtracking when storing solutions?

> **Saying it out loud.** *(Giving yourself permission to be exponential.)* "n is twenty or under and the output is 'all of them,' so exponential is the intended answer and I shouldn't waste time hunting for something polynomial." Then narrate the shape: choose, recurse, un-choose, with a start index so I never revisit an earlier element and generate the same set in a different order. State the complexity as roughly n times two-to-the-n for subsets or n factorial for permutations. Two bugs to call out before they happen: append a *copy* of the path at the leaf, because you're about to mutate it; and for duplicate inputs, sort first and skip an element equal to its predecessor at the same level.

## Section J — Graphs (Q94–106)

94. DFS template. BFS template.
95. "Number of islands" → pattern?
96. "Clone graph" → pattern + key data structure?
97. "Pacific Atlantic water flow" → pattern (multi-source BFS)?
98. "Surrounded regions" → key trick?
99. "Rotting oranges" → pattern + state to track?
100. "Walls and gates" → pattern?
101. "Course schedule" → pattern + algorithm?
102. "Course schedule II" → return what?
103. "Word ladder" → pattern + how to build the graph?
104. "Number of connected components" — two methods?
105. "Graph valid tree" — three conditions?
106. "Redundant connection" → pattern?

> **Saying it out loud.** *(The one line that fixes the complexity.)* "Connectivity means graph, and a grid is just a graph whose neighbors are the four adjacent cells. BFS if I need shortest path in an unweighted graph, DFS if I only need to explore or count components — both O of V plus E. The thing I'm being careful about: I mark a node visited when I *enqueue* it, not when I dequeue it." That distinction is worth saying explicitly, because marking on dequeue lets the same node get queued many times over and quietly wrecks the bound. Edge cases: disconnected components, so loop over every possible start; self-loops; and an empty grid.

## Section K — Advanced graphs (Q107–115)

107. Topological sort — Kahn's vs DFS-based?
108. Dijkstra — when applicable + complexity?
109. Bellman-Ford — when over Dijkstra + complexity?
110. Floyd-Warshall — when + complexity?
111. MST — Kruskal vs Prim?
112. "Network delay time" → pattern?
113. "Cheapest flights within K stops" → pattern?
114. "Min cost to connect all points" → pattern?
115. "Reconstruct itinerary" → algorithm name?

> **Saying it out loud.** *(Naming which constraint you're reacting to.)* "Dependencies and an ordering means topological sort — Kahn's with in-degrees, O of V plus E, and if the queue drains before I've emitted every node there's a cycle, which is usually what the problem was really asking. Weighted with non-negative edges is Dijkstra, a heap-driven BFS at E log V. Negative edges break Dijkstra's greedy assumption outright, so that's Bellman-Ford at V times E, which also detects negative cycles with one extra relaxation pass." The mistake worth pre-empting out loud is reaching for Dijkstra on an unweighted graph — plain BFS gets the identical answer without the log factor.

## Section L — 1D DP (Q116–127)

116. When to reach for 1D DP?
117. State definition: `dp[i]` for "Climbing Stairs".
118. Coin Change — recurrence?
119. Coin Change II (number of ways) — what changes?
120. House Robber — recurrence?
121. House Robber II — how to handle the circular constraint?
122. Decode Ways — how to handle "0"?
123. Word Break — recurrence + dictionary lookup?
124. LIS — $O(n^2)$ recurrence + $O(n \log n)$ trick name?
125. Maximum Product Subarray — what state to track?
126. Partition Equal Subset Sum — reduces to which classical?
127. Longest palindromic substring — expand-around-centers vs DP?

> **Saying it out loud.** *(Defining the state before writing anything.)* "Let me define dp of i as the answer for the first i elements — I want to say that before I write code, because a fuzzy state definition is where DP goes wrong. The recurrence: at i I either take this element and add dp of i minus two, or skip it and keep dp of i minus one, so dp of i is the max. Base cases are dp of zero and dp of one, and that's where the bug usually lives." O of n time and O of n space, then offer the improvement: it only ever looks back two positions, so it collapses to two variables and O of 1 space. And name the discovery path — brute-force recursion, spot the overlapping subproblems, memoize, flip to bottom-up.

## Section M — 2D DP (Q128–138)

128. When to reach for 2D DP?
129. LCS — recurrence on equal vs unequal characters?
130. Edit Distance — three operation costs.
131. Distinct Subsequences — what's `dp[i][j]`?
132. Interleaving String — boolean DP, what's the recurrence?
133. Buy/sell with cooldown — state dimension?
134. Best Time IV (k transactions) — state?
135. Target Sum — reduces to subset sum count?
136. Burst Balloons — why range DP, not interval-greedy?
137. Regular Expression Matching — recurrence on `*`?
138. Longest Increasing Path in matrix — DFS + memo or pure DP?

> **Saying it out loud.** *(Two sequences means two indices.)* "dp of i, j is the answer for the first i characters of one string against the first j of the other. If the characters match I extend the diagonal — dp of i minus one, j minus one, plus one. If they don't, I take the better of dropping a character from either side. The zero row and column encode 'one string is empty,' which is exactly my base case." O of m times n in both time and space, and then offer the follow-up before they ask: each row only depends on the row above, so it rolls down to O of n space. The edge case that's already handled if you set the base row correctly is empty input.

## Section N — Greedy, intervals, math, bits (Q139–155)

139. When does greedy work? How do you prove it?
140. Maximum subarray (Kadane) — running sum logic?
141. Jump Game II — greedy invariant?
142. Gas station — pattern?
143. Hand of Straights — pattern + data structure?
144. Partition Labels — preprocessing + sweep?
145. Insert Interval — three phases?
146. Merge Intervals — sort by what?
147. Non-overlapping intervals — greedy + sort by what?
148. Meeting rooms II — sweep-line + heap method?
149. Min interval to include each query — pattern?
150. Rotate image — algorithm in two steps?
151. Spiral matrix — boundary trick?
152. Pow(x, n) — algorithm?
153. Single Number — pattern?
154. Counting Bits 0..n — recurrence?
155. Sum of Two Integers without `+` — bit logic?

> **Saying it out loud.** *(The greedy caveat, said unprompted.)* "Greedy is the pattern that looks right and is wrong, so let me justify it rather than assume it — with an exchange argument: if some optimal solution differs from my greedy choice, I can swap my choice in without making it worse. If I can't make that argument, I'll say so and fall back to DP." For intervals: sort by start and sweep to merge, n log n dominated by the sort — but sort by *end* time for maximum non-overlapping, because finishing early leaves the most room. For bits, say what the identity does out loud: n AND n-minus-one clears the lowest set bit, XOR cancels pairs. And flag that Python ints are arbitrary precision, so 32-bit problems need an explicit mask.

## Section O — Senior signals (Q156–165)

156. Show me the 5-step problem-solving protocol.
157. What do you do in the first 2-5 minutes of a problem?
158. What if you're stuck after 10 minutes?
159. How do you state complexity properly?
160. How do you test your code in an interview?
161. How do you handle edge cases out loud?
162. How do you communicate while coding?
163. How do you ask clarifying questions without sounding lost?
164. What's a common bug pattern in your code that you watch for?
165. How do you decide between brute force vs optimal in an interview?

> **Saying it out loud.** *(The whole script, in order.)* "Let me make sure I understand — input's X, I return Y, n goes to this. Duplicates? Negatives?" Then walk their example plus one of your own, ideally an empty one. Then: "Brute force is every pair, n squared — that's correct but we can do better." Then: "This looks like sliding window because the constraint is on a contiguous range, so O of n." Then code while narrating the invariant, and finish by tracing a small input and restating time and space. The failure mode this whole script exists to prevent is going silent — the interviewer is scoring your reasoning, and a quiet candidate with working code routinely loses to a talkative one who didn't finish.

---

## Self-grading

- 130+ correct: ready for big-tech / frontier-lab coding rounds.
- 95–129: re-read the deep dive and drill weak patterns.
- 60–94: spend a week on full deep dive + 5 problems per weak pattern.
- <60: build pattern foundations from §2–§19; one section per day for two weeks.

## 8-week drill plan (mapped to NeetCode 150)

- **Week 1:** Arrays & Hashing, Two Pointers, Sliding Window. Drill A, B.
- **Week 2:** Stack, Binary Search, Linked List. Drill C, D, E.
- **Week 3:** Trees, Tries, Heap. Drill F, G, H.
- **Week 4:** Backtracking, Graphs. Drill I, J.
- **Week 5:** Advanced Graphs. Drill K.
- **Week 6:** 1D DP. Drill L.
- **Week 7:** 2D DP. Drill M.
- **Week 8:** Greedy, Intervals, Math, Bits + mock interviews. Drill N, O.

Daily: 1 problem solo (30 min) → check editorial → re-attempt next day from scratch.
