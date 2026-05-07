# Topic 68: LeetCode / NeetCode 150 — Pattern Recognition & Intuition

> The single biggest jump in coding-interview competence is moving from "let me try to solve this problem" to "this is a sliding-window problem; here's the template." This folder trains that move.

> 🔥 **Read these first:**
> - **`LEETCODE_PATTERNS_DEEP_DIVE.md`** — The 30-second triage; all 18 NeetCode 150 pattern categories (Arrays & Hashing, Two Pointers, Sliding Window, Stack, Binary Search, Linked List, Trees, Tries, Heap, Backtracking, Graphs, Advanced Graphs, 1D DP, 2D DP, Greedy, Intervals, Math/Geometry, Bit Manipulation), each with: recognition signals, mental model, code templates, 5-10 representative problems with quick walkthroughs, common variants, and traps. Plus problem-to-pattern transformation tricks, complexity reasoning cheatsheet, common interview traps, the 5-step problem-solving protocol, and a 9-week drilling routine.
> - **`INTERVIEW_GRILL.md`** — 165 pattern-recognition active-recall questions across A–O. Read a problem statement, name the pattern in <30 seconds. 8-week drill plan mapped to NeetCode 150.
>
> **Worked-solutions files** (simplest-optimal Python, intuitive explanations, memory hooks, pressure-tested templates — 135 problems total):
> - **`SOLUTIONS_ARRAYS_HASHING_SLIDING_WINDOW.md`** — 9 Arrays & Hashing + 6 Sliding Window problems.
> - **`SOLUTIONS_TWO_POINTERS_STACK.md`** — 5 Two Pointers + 7 Stack problems.
> - **`SOLUTIONS_BINARY_SEARCH_LINKED_LIST.md`** — 7 Binary Search + 11 Linked List problems.
> - **`SOLUTIONS_TREES.md`** — 15 Tree problems.
> - **`SOLUTIONS_HEAP_BACKTRACKING.md`** — 7 Heap/Priority Queue + 10 Backtracking problems.
> - **`SOLUTIONS_GRAPHS_ADVANCED.md`** — 13 Graph + 6 Advanced Graph problems.
> - **`SOLUTIONS_DP.md`** — 12 1-D DP + 11 2-D DP problems.
> - **`SOLUTIONS_GREEDY_MATH_GEOMETRY.md`** — 8 Greedy + 8 Math & Geometry problems.

## Why this folder

NeetCode 150 / Blind 75 / Grind 75 are problem lists. Solving them one-by-one without **organizing them by pattern** is grinding. The 18 categories in NeetCode are signal: they tell you the underlying templates the lists are testing. If you can recognize "this is sliding window" in 30 seconds, you've already done the hardest part of the interview.

## What this folder gives you

- **The 30-second triage.** Input shape → output shape → constraints → structural cues → pattern.
- **Per-pattern playbook.** Recognition signals, mental model, copy-pasteable templates in Python, representative problems, traps.
- **Transformation tricks.** "Sort first," "build a graph," "binary-search on the answer," "treat 2D as 1D" — the moves that turn hard problems into known patterns.
- **Complexity cheatsheet.** Map $n$ to acceptable algorithm classes.
- **The 5-step protocol** to deploy in any coding interview: Understand → Examples → Brute force → Optimize → Code.
- **9-week drilling plan** with daily routine.
- **165-question grill** focused on pattern recognition.

## Core insight

> For any problem, do the 30-second triage by **input shape, output shape, constraints, and structural cues**; then match to one of 18 patterns; then deploy the template.

## Cross-references

- **`50_ml_coding_interview_patterns/`** — ML coding (numpy / from scratch ML algorithms), complementary.
- **`58_whiteboard_derivations/`** — math derivations (different rounds).
- **`67_frontier_intuitive_questions/`** — probabilistic reasoning questions (different round type).

## How to use

1. Read `LEETCODE_PATTERNS_DEEP_DIVE.md` once cover to cover. Note weak patterns.
2. For each weak pattern, drill 5+ problems on NeetCode / LeetCode.
3. Drill `INTERVIEW_GRILL.md` weekly until you hit 130+/165.
4. Re-read §1 (triage), §20 (transformations), §22 (traps), §23 (protocol) every Sunday.
5. Mock interviews in the last 2 weeks; mixed problems on shuffle.

## Suggested external resources

- **NeetCode 150** — neetcode.io/practice — the curated list this folder is organized around.
- **NeetCode YouTube** — the gold standard for pattern walkthroughs.
- **Blind 75** — LeetCode discuss; smaller curated list.
- **Grind 75** — same author as Blind 75, slightly extended.
- **CSES Problem Set** — for competitive-programming depth.

The single sentence to remember: **read fast, triage fast, name the pattern out loud, deploy the template, then code.**
