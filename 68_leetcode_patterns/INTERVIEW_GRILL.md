# LeetCode / NeetCode 150 — Interview Grill (Pattern Recognition)

> 150+ pattern-recognition questions — read the problem statement, name the pattern in <30 seconds. Pair with `LEETCODE_PATTERNS_DEEP_DIVE.md`.

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

## Section C — Stack / monotonic stack (Q26–33)

26. "Valid parentheses" → pattern?
27. "Min stack" — two ways to design?
28. "Daily temperatures" → pattern + monotonic-which-way?
29. "Next greater element" → pattern?
30. "Largest rectangle in histogram" → pattern + key insight?
31. "Car fleet" → preprocessing + pattern?
32. "Evaluate Reverse Polish Notation" → pattern?
33. "Generate parentheses" → pattern (this is a trap)?

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

## Section G — Tries (Q68–72)

68. When to reach for a trie?
69. Insert / search / starts_with — complexities?
70. "Add and search word with `.` wildcard" → pattern + DFS technique?
71. "Word search II" → pattern + what's the trie used for?
72. Trade-offs between trie and hash set.

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
