# Dynamic programming II: grids and strings

Two-dimensional dynamic programming is the same method as the previous chapter with a state that needs
two indices instead of one. Nothing else changes: you define the state in a sentence, write the
transition, find the base case, and choose an iteration order that fills every cell before it is read.
The only new work is that the order is now a choice in two directions rather than one.

The load-bearing claim of this chapter is about the two-string problems, and it is worth learning as a
sentence. **Almost every problem about two strings has the same shape: `dp[i][j]` is the answer for the
first `i` characters of one string and the first `j` characters of the other, and the transition splits
on whether the current characters match.** On a match you take the diagonal cell, because both
characters are consumed together. On a mismatch you combine the cell above and the cell to the left,
because one character must be dropped and you do not yet know which. Once you see that, longest common
subsequence, edit distance and distinct subsequences stop being three problems. They are one table with
three different combination rules, and the section below draws it three ways.

Grid problems are the easy half. The cell is the state, the moves in the statement are the transition,
and the sweep order is forced by the moves.

## Recognising it from the phrasing

| The interviewer says | The state | The fill order |
|---|---|---|
| "two strings, compare them" | `dp[i][j]` over both prefixes | rows then columns |
| "a grid, move right or down" | `dp[r][c]` from the top and the left | top to bottom, left to right |
| "a grid with obstacles" | the same, blocked cells forced to zero | top to bottom, left to right |
| "a knapsack with a capacity" | `dp[item][capacity]` | items outer, capacity either way |
| "an interval", "a range `i` to `j`" | `dp[i][j]` over the range | by **increasing length** |
| "with a cooldown", "you may hold one" | `dp[i][state]`, a small state machine | left to right |
| "the longest path in a grid" | memoised DFS from each cell | no sweep order exists |
| "count the ways through a grid" | `dp[r][c]`, sums | top to bottom, left to right |

Ask two questions before writing anything. First: what do the two indices mean? Say it as one sentence
with both indices named, for example "the edit distance between the first `i` characters of `a` and the
first `j` characters of `b`". Second, and this is the question people skip: **in what order must the
table be filled so that every value I read is already written?** For prefix problems the answer is left
to right and top to bottom, because each cell reads the cell above, the cell to the left and the
diagonal. For interval problems the answer is by increasing interval length, because `dp[i][j]` reads
shorter intervals inside itself, and those are neither above nor to the left in any simple sweep. That
is the order people get wrong, and it fails silently by reading zeros. Say the fill order out loud
before you write the loops and you will avoid the most common two-dimensional DP bug.

## The templates

**Template 1 — the two-string prefix table.** Use whenever the input is two sequences. The skeleton is
always these two nested loops with a match branch and a mismatch branch; only the two branch bodies
change between problems.

```python
def longest_common_subsequence(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]    ## dp[i][j]: first i of a, first j of b
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1   ## MATCH: take the diagonal and add one
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])   ## MISMATCH: drop one character
    return dp[n][m]

## tests

assert longest_common_subsequence("abcde", "ace") == 3
assert longest_common_subsequence("abc", "abc") == 3
assert longest_common_subsequence("abc", "def") == 0
assert longest_common_subsequence("", "abc") == 0
print(longest_common_subsequence("abcde", "ace"))
```

```
3
```

**Template 2 — the grid sweep.** Use when the moves are right and down. The answer is recorded in the
bottom-right cell, and the two edges are the base cases because they have only one incoming direction.

```python
def min_path_sum(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]        ## dp[r][c]: best cost to reach (r, c)
    for r in range(rows):                         ## top to bottom, left to right
        for c in range(cols):
            if r == 0 and c == 0:
                dp[r][c] = grid[r][c]             ## base case: the start cell
            elif r == 0:
                dp[r][c] = dp[r][c - 1] + grid[r][c]
            elif c == 0:
                dp[r][c] = dp[r - 1][c] + grid[r][c]
            else:
                dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
    return dp[rows - 1][cols - 1]

## tests

assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
assert min_path_sum([[1, 2, 3], [4, 5, 6]]) == 12
assert min_path_sum([[7]]) == 7
print(min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]))
```

```
7
```

**Template 3 — interval DP, filled by increasing length.** Use when the state is a range of one
sequence. The outer loop is the length, never the index, and the answer sits at `dp[0][n-1]`.

```python
def longest_palindromic_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]              ## dp[i][j]: answer for s[i..j] inclusive
    for i in range(n):
        dp[i][i] = 1                              ## base case: every single character
    for length in range(2, n + 1):                ## BY INCREASING LENGTH, not by index
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2   ## shorter interval, already computed
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1] if n else 0

## tests

assert longest_palindromic_subsequence("bbbab") == 4
assert longest_palindromic_subsequence("cbbd") == 2
assert longest_palindromic_subsequence("") == 0
print(longest_palindromic_subsequence("bbbab"))
```

```
4
```

**Template 4 — memoised DFS on a grid.** Use when no sweep order exists, because the moves can go in
all four directions. The recursion supplies the order and the dictionary supplies the memoisation.

```python
def longest_increasing_path(matrix):
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    memo = {}
    def best(r, c):                               ## longest strictly increasing path FROM (r, c)
        if (r, c) in memo:
            return memo[(r, c)]
        length = 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                length = max(length, 1 + best(nr, nc))
        memo[(r, c)] = length
        return length
    return max(best(r, c) for r in range(rows) for c in range(cols))

## tests

assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
assert longest_increasing_path([[3, 4, 5], [3, 2, 6], [2, 2, 1]]) == 4
assert longest_increasing_path([[1]]) == 1
print(longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]))
```

```
4
```

Templates 1, 2 and 3 are all bottom-up tables and differ only in the fill order. Template 4 exists
because some problems have no fill order at all: a path in a grid may go up, down, left or right, so no
row-by-row sweep can guarantee that a neighbour is ready. The values strictly increase along a path, so
the recursion cannot cycle, and that is exactly the property that licenses the memo.

## The two-string table, drawn three ways

Take `a = "abc"` and `b = "abd"` under longest common subsequence. The table has one extra row and one
extra column for the empty prefixes, and the whole of row 0 and column 0 is zero, because the longest
common subsequence with an empty string is empty.

|  | (empty) | a | b | d |
|---|---|---|---|---|
| **(empty)** | 0 | 0 | 0 | 0 |
| **a** | 0 | 1 | 1 | 1 |
| **b** | 0 | 1 | 2 | 2 |
| **c** | 0 | 1 | 2 | 2 |

Read two cells to see the rule. At row `a`, column `a`, the characters match, so the value is the
diagonal cell above and to the left, which is 0, plus one: **1**. At row `c`, column `d`, the
characters differ, so the value is the larger of the cell above, 2, and the cell to the left, 2, which
is **2**. Every cell in the table is one of those two moves. The answer is the bottom-right cell, 2,
which is the length of `"ab"`.

Now change one rule at a time and you get the other two classics.

**Edit distance** uses the same table with **three** neighbours instead of two. The base row and column
are no longer zero: turning the first `i` characters into an empty string costs `i` deletions, so
`dp[i][0] = i` and `dp[0][j] = j`. On a match the cost is the diagonal unchanged, because no operation
is needed. On a mismatch you pay one and take the cheapest of the diagonal, which is a replacement, the
cell above, which is a deletion, and the cell to the left, which is an insertion. That is the only
difference: two neighbours become three, and `max` becomes `1 + min`.

**Distinct subsequences** uses the same table with a **different combination rule**. It counts, so it
sums instead of taking a maximum. `dp[i][0] = 1`, because there is exactly one way to match an empty
target: take nothing. The cell above is always available, because you may always skip `s[i-1]`. When
the characters match you also have the option of using `s[i-1]` to cover `t[j-1]`, which is the
diagonal cell, and since these are two disjoint groups of ways, you **add** them rather than choosing
between them.

```python
def lcs_table(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1              ## diagonal + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])   ## max of up and left
    return dp

def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i                                         ## delete every character
    for j in range(m + 1):
        dp[0][j] = j                                         ## insert every character
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]                  ## free: no operation
            else:                                            ## replace, delete, insert
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def distinct_subsequences(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1                                         ## one way to match the empty t
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j]                          ## always: skip s[i-1]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]                 ## SUM, not max: also use it
    return dp[n][m]

## tests

table = lcs_table("abc", "abd")
assert table[3][3] == 2
assert edit_distance("abc", "abd") == 1
assert edit_distance("horse", "ros") == 3
assert distinct_subsequences("rabbbit", "rabbit") == 3
assert distinct_subsequences("babgbag", "bag") == 5
for row in table:
    print(row)
```

```
[0, 0, 0, 0]
[0, 1, 1, 1]
[0, 1, 2, 2]
[0, 1, 2, 2]
```

The printed table is the same one drawn above. Three problems, one grid of cells, three rules for
filling it. When a two-string problem you have not seen arrives, draw this four-by-four table on the
whiteboard first and ask only two questions: what goes in row 0 and column 0, and what happens on a
match. The rest of the solution follows.

## The problems

### P1. Unique Paths — count the paths from the top-left to the bottom-right moving only right or down

**Which template.** Template 2, counting, reduced to a single row.
**The trick.** Every path into a cell arrives from above or from the left, so `dp[r][c]` is the sum of
those two, and the whole top row and left column are 1. A single row suffices, because updating
`row[c] += row[c-1]` in place reads the old value of `row[c]`, which is the cell above, and the new
value of `row[c-1]`, which is the cell to the left. The cross-check uses the closed form: a path is a
choice of which of the `r+c-2` moves go down.

```python
from math import comb

def unique_paths(rows, cols):
    row = [1] * cols                              ## dp[0][c] = 1: only one way along the top
    for _ in range(1, rows):
        for c in range(1, cols):
            row[c] += row[c - 1]                  ## from above (old row[c]) plus from the left
    return row[cols - 1]

## tests

assert unique_paths(3, 7) == 28
assert unique_paths(3, 2) == 3
assert unique_paths(1, 1) == 1
for r in range(1, 9):                             ## cross-check against the binomial formula
    for c in range(1, 9):
        assert unique_paths(r, c) == comb(r + c - 2, r - 1)
print(unique_paths(3, 7), unique_paths(3, 2), "binomial formula agrees")
```

```
28 3 binomial formula agrees
```

**Complexity.** $O(rc)$ time, $O(c)$ space.

### P2. Unique Paths II — the same count when some cells are blocked

**Which template.** Template 2, with a forced zero.
**The trick.** A blocked cell is reachable in zero ways, so set it to 0 and let the rest of the sweep
run unchanged. That single line is the whole difference from P1, and it works because the recurrence
sums incoming counts: a zero contributes nothing downstream. Check the start cell itself, since a
blocked start makes the answer 0.

```python
def unique_paths_with_obstacles(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                dp[r][c] = 0                      ## a blocked cell is reachable in zero ways
            elif r == 0 and c == 0:
                dp[r][c] = 1
            else:
                above = dp[r - 1][c] if r > 0 else 0
                left = dp[r][c - 1] if c > 0 else 0
                dp[r][c] = above + left
    return dp[rows - 1][cols - 1]

## tests

assert unique_paths_with_obstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2
assert unique_paths_with_obstacles([[0, 1], [0, 0]]) == 1
assert unique_paths_with_obstacles([[1]]) == 0
assert unique_paths_with_obstacles([[0, 0], [1, 1], [0, 0]]) == 0
print(unique_paths_with_obstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
      unique_paths_with_obstacles([[0, 0], [1, 1], [0, 0]]))
```

```
2 0
```

**Complexity.** $O(rc)$ time, $O(rc)$ space, or $O(c)$ with a single row.

### P3. Minimum Path Sum — the cheapest path from the top-left to the bottom-right, moving right or down

**Which template.** Template 2, minimisation.
**The trick.** Identical to P1 with `min` in place of `+`, plus the cost of the current cell. The edges
are the only fiddly part: the top row can only come from the left and the left column can only come
from above, so write those two cases explicitly rather than guarding with sentinels.

```python
def min_path_sum(grid):
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]        ## dp[r][c] = cheapest cost to reach (r, c)
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                dp[r][c] = grid[r][c]
            elif r == 0:
                dp[r][c] = dp[r][c - 1] + grid[r][c]
            elif c == 0:
                dp[r][c] = dp[r - 1][c] + grid[r][c]
            else:
                dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
    return dp[rows - 1][cols - 1]

def min_path_sum_brute(grid):                     ## check: recursion over every path
    rows, cols = len(grid), len(grid[0])
    def go(r, c):
        if r == rows - 1 and c == cols - 1:
            return grid[r][c]
        options = []
        if r + 1 < rows:
            options.append(go(r + 1, c))
        if c + 1 < cols:
            options.append(go(r, c + 1))
        return grid[r][c] + min(options)
    return go(0, 0)

## tests

assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
assert min_path_sum([[1, 2, 3], [4, 5, 6]]) == 12
grids = [[[1, 3, 1], [1, 5, 1], [4, 2, 1]], [[1, 2, 3], [4, 5, 6]], [[7]],
         [[5, 9, 2, 4], [1, 1, 8, 3], [6, 2, 2, 7]]]
for g in grids:                                   ## cross-check against every path
    assert min_path_sum(g) == min_path_sum_brute(g)
print(min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]), "brute force agrees")
```

```
7 brute force agrees
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P4. Longest Common Subsequence — the longest subsequence present in both strings

**Which template.** Template 1, the reference instance of it.
**The trick.** On a match, both characters are consumed together, so the value is the diagonal plus one
and there is nothing to decide. On a mismatch, at least one of the two characters is not in the answer,
but you do not know which, so you try both by taking the maximum of the cell above and the cell to the
left. The cross-check enumerates every subsequence of the first string.

```python
def longest_common_subsequence(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def lcs_brute(a, b):                              ## check: every subsequence of a
    n, best = len(a), 0
    for mask in range(1 << n):
        pick = "".join(a[i] for i in range(n) if mask >> i & 1)
        j = 0
        for ch in b:                              ## is pick a subsequence of b?
            if j < len(pick) and pick[j] == ch:
                j += 1
        if j == len(pick):
            best = max(best, len(pick))
    return best

## tests

assert longest_common_subsequence("abcde", "ace") == 3
assert longest_common_subsequence("abc", "def") == 0
assert longest_common_subsequence("", "abc") == 0
pairs = [("abcde", "ace"), ("abc", "def"), ("bsbininm", "jmjkbkjkv"), ("oxcpqrsvwf", "shmtulqrypy")]
for a, b in pairs:                                ## cross-check against every subsequence
    assert longest_common_subsequence(a, b) == lcs_brute(a, b)
print(longest_common_subsequence("abcde", "ace"), "brute force agrees")
```

```
3 brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space, reducible to $O(\min(n, m))$ with two rows.

### P5. Edit Distance — the fewest insertions, deletions and replacements turning `a` into `b`

**Which template.** Template 1 with three neighbours.
**The trick.** Name each neighbour as an operation while you write it and the recurrence stops being
symbols. The diagonal is a replacement, the cell above is a deletion from `a`, and the cell to the left
is an insertion into `a`. On a match you move diagonally for free. The base row and column are not
zero: they are `i` and `j`, the cost of deleting or inserting everything.

```python
def min_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i                              ## delete i characters
    for j in range(m + 1):
        dp[0][j] = j                              ## insert j characters
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]       ## free
            else:                                 ## replace / delete / insert
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def min_distance_brute(a, b):                     ## check: plain recursion, no table
    def go(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        if a[i - 1] == b[j - 1]:
            return go(i - 1, j - 1)
        return 1 + min(go(i - 1, j - 1), go(i - 1, j), go(i, j - 1))
    return go(len(a), len(b))

## tests

assert min_distance("horse", "ros") == 3
assert min_distance("intention", "execution") == 5
assert min_distance("", "abc") == 3
for a, b in [("horse", "ros"), ("abc", "abd"), ("", "abc"), ("sunday", "saturday")]:
    assert min_distance(a, b) == min_distance_brute(a, b)
print(min_distance("horse", "ros"), min_distance("intention", "execution"),
      "brute force agrees")
```

```
3 5 brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space.

### P6. Distinct Subsequences — count the subsequences of `s` that equal `t`

**Which template.** Template 1, counting, so the branches add rather than choose.
**The trick.** Two options exist at every cell and they are disjoint. You may always skip `s[i-1]`,
which is the cell directly above. When `s[i-1] == t[j-1]` you may additionally match it against
`t[j-1]`, which is the diagonal. Since no arrangement is counted in both groups, the value is their
**sum**. The base column is 1, not 0: taking nothing is one valid way to build the empty target, and
setting it to 0 makes the entire table zero.

```python
def num_distinct(s, t):
    n, m = len(s), len(t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1                              ## one way to match the empty t: take nothing
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j]               ## always available: skip s[i-1]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]      ## SUM, because both choices are counted
    return dp[n][m]

def num_distinct_brute(s, t):                     ## check: every subset of positions in s
    n, total = len(s), 0
    for mask in range(1 << n):
        if "".join(s[i] for i in range(n) if mask >> i & 1) == t:
            total += 1
    return total

## tests

assert num_distinct("rabbbit", "rabbit") == 3
assert num_distinct("babgbag", "bag") == 5
assert num_distinct("abc", "") == 1
for s, t in [("rabbbit", "rabbit"), ("babgbag", "bag"), ("abc", ""), ("aaaa", "aa"), ("abc", "d")]:
    assert num_distinct(s, t) == num_distinct_brute(s, t)
print(num_distinct("rabbbit", "rabbit"), num_distinct("babgbag", "bag"), "brute force agrees")
```

```
3 5 brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space, reducible to one row scanned right to left.

### P7. Interleaving String — can `s3` be formed by interleaving `s1` and `s2` keeping both orders

**Which template.** Template 1, boolean, over the prefixes of `s1` and `s2`.
**The trick.** The third string does not need a third index. If you have used `i` characters of `s1`
and `j` of `s2`, you have used exactly `i + j` characters of `s3`, so the position in `s3` is
determined. That collapse from three indices to two is the entire insight. The first check is on
lengths: if they do not add up, return `False` immediately.

```python
def is_interleave(s1, s2, s3):
    n, m = len(s1), len(s2)
    if n + m != len(s3):
        return False                              ## the lengths must add up
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True                               ## dp[i][j]: s1[:i] and s2[:j] make s3[:i+j]
    for i in range(n + 1):
        for j in range(m + 1):
            if i > 0 and dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]:
                dp[i][j] = True                   ## the last character came from s1
            if j > 0 and dp[i][j - 1] and s2[j - 1] == s3[i + j - 1]:
                dp[i][j] = True                   ## the last character came from s2
    return dp[n][m]

def is_interleave_brute(s1, s2, s3):              ## check: recursion over both sources
    def go(i, j):
        if i + j == len(s3):
            return i == len(s1) and j == len(s2)
        if i < len(s1) and s1[i] == s3[i + j] and go(i + 1, j):
            return True
        return j < len(s2) and s2[j] == s3[i + j] and go(i, j + 1)
    return len(s1) + len(s2) == len(s3) and go(0, 0)

## tests

assert is_interleave("aabcc", "dbbca", "aadbbcbcac") is True
assert is_interleave("aabcc", "dbbca", "aadbbbaccc") is False
assert is_interleave("", "", "") is True
cases = [("aabcc", "dbbca", "aadbbcbcac"), ("aabcc", "dbbca", "aadbbbaccc"), ("", "", ""),
         ("a", "b", "ab"), ("ab", "ab", "abab"), ("a", "", "a")]
for s1, s2, s3 in cases:
    assert is_interleave(s1, s2, s3) == is_interleave_brute(s1, s2, s3)
print(is_interleave("aabcc", "dbbca", "aadbbcbcac"),
      is_interleave("aabcc", "dbbca", "aadbbbaccc"), "brute force agrees")
```

```
True False brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space.

### P8. Longest Palindromic Subsequence — the longest subsequence of `s` that reads the same both ways

**Which template.** Template 1, applied to `s` and its reverse. The trick is worth naming.
**The trick.** A palindromic subsequence of `s` is a subsequence that also appears in `s` reversed, so
the answer is `LCS(s, reversed(s))`. This turns a new problem into one you have already solved, and
naming the reduction in one sentence is a strong signal in an interview. The interval DP of template 3
computes the same number directly, and the cross-check confirms the two agree.

```python
def longest_palindrome_subseq(s):
    a, b = s, s[::-1]                             ## the trick: LCS of s with its reverse
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][n]

def lps_interval(s):                              ## check: the direct interval DP
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = (dp[i + 1][j - 1] if length > 2 else 0) + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1] if n else 0

## tests

assert longest_palindrome_subseq("bbbab") == 4
assert longest_palindrome_subseq("cbbd") == 2
assert longest_palindrome_subseq("") == 0
for s in ["bbbab", "cbbd", "", "a", "agbdba", "character", "abcdefg"]:
    assert longest_palindrome_subseq(s) == lps_interval(s)
print(longest_palindrome_subseq("bbbab"), longest_palindrome_subseq("character"),
      "interval DP agrees")
```

```
4 5 interval DP agrees
```

**Complexity.** $O(n^2)$ time, $O(n^2)$ space for either form.

### P9. Regular Expression Matching — does the pattern with `.` and `*` match the whole string

**Which template.** Template 1, boolean, and the hardest classic here. Work it slowly.
**The trick.** A `*` is never alone: it always binds to the character before it, so the unit is the
**pair** `p[j-2] p[j-1]`. That pair has exactly two behaviours. Use it zero times, which skips two
pattern characters and gives `dp[i][j-2]`. Or use it one more time, which is legal only when the
preceding pattern character matches `s[i-1]`, and which consumes one text character while leaving the
pattern where it is, giving `dp[i-1][j]`. Write those two lines and the rest is the ordinary match
branch. The row-0 initialisation matters as much: a pattern like `"a*b*"` matches the empty string, so
`dp[0][j] = dp[0][j-2]` whenever `p[j-1]` is a star.

```python
def is_match(s, p):
    n, m = len(s), len(p)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True                               ## empty pattern matches empty text
    for j in range(2, m + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]               ## "a*" can vanish
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 2]           ## use the pair ZERO times
                if p[j - 2] == s[i - 1] or p[j - 2] == ".":
                    dp[i][j] = dp[i][j] or dp[i - 1][j]   ## or ONE MORE time
            elif p[j - 1] == "." or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]       ## one character each, consumed together
    return dp[n][m]

def is_match_brute(s, p):                         ## check: plain recursion, no table
    def go(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return go(i, j + 2) or (first and go(i + 1, j))
        return first and go(i + 1, j + 1)
    return go(0, 0)

## tests

assert is_match("aa", "a") is False
assert is_match("aa", "a*") is True
assert is_match("ab", ".*") is True
assert is_match("mississippi", "mis*is*p*.") is False
cases = [("aa", "a"), ("aa", "a*"), ("ab", ".*"), ("aab", "c*a*b"), ("mississippi", "mis*is*p*."),
         ("", "a*"), ("", ".*"), ("abc", "a.c"), ("aaa", "a*a"), ("bbbba", ".*a*a")]
for s, p in cases:                                ## cross-check against the recursion
    assert is_match(s, p) == is_match_brute(s, p)
print(is_match("aa", "a*"), is_match("mississippi", "mis*is*p*."), "brute force agrees")
```

```
True False brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space.

### P10. Wildcard Matching — the same question when `*` matches any sequence and `?` any single character

**Which template.** Template 1 again, and it is simpler than P9. Say why.
**The trick.** Here `*` stands alone, so there is no pair to look back at. It matches the empty
sequence, which is `dp[i][j-1]`, or it absorbs one more text character and stays available, which is
`dp[i-1][j]`. That is the whole rule. Compare it aloud with P9: the star in a regular expression binds
to the previous character and therefore skips **two** pattern positions; the wildcard star binds to
nothing and skips **one**.

```python
def is_match_wildcard(s, p):
    n, m = len(s), len(p)
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True
    for j in range(1, m + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 1]               ## a leading run of stars matches nothing
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if p[j - 1] == "*":                   ## star alone: empty, or eat one character
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == "?" or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[n][m]

def wildcard_brute(s, p):                         ## check: plain recursion
    def go(i, j):
        if j == len(p):
            return i == len(s)
        if p[j] == "*":
            return go(i, j + 1) or (i < len(s) and go(i + 1, j))
        if i < len(s) and (p[j] == "?" or p[j] == s[i]):
            return go(i + 1, j + 1)
        return False
    return go(0, 0)

## tests

assert is_match_wildcard("aa", "a") is False
assert is_match_wildcard("aa", "*") is True
assert is_match_wildcard("cb", "?a") is False
assert is_match_wildcard("adceb", "*a*b") is True
cases = [("aa", "a"), ("aa", "*"), ("cb", "?a"), ("adceb", "*a*b"), ("acdcb", "a*c?b"),
         ("", "***"), ("", "?"), ("abc", "a*c")]
for s, p in cases:
    assert is_match_wildcard(s, p) == wildcard_brute(s, p)
print(is_match_wildcard("adceb", "*a*b"), is_match_wildcard("acdcb", "a*c?b"),
      "brute force agrees")
```

```
True False brute force agrees
```

**Complexity.** $O(nm)$ time, $O(nm)$ space.

### P11. Target Sum — count the ways to put a plus or minus before each number to reach `target`

**Which template.** A 0/1 knapsack in disguise, reduced to one row.
**The trick.** Split the numbers into the set `P` that gets a plus and the set `M` that gets a minus.
Then `sum(P) - sum(M) = target` and `sum(P) + sum(M) = total`, so `sum(P) = (total + target) / 2`. The
problem becomes: how many subsets sum to that number. If the division is not exact, or the target
exceeds the total in absolute value, the answer is 0. Each number is used once, so the capacity loop
runs descending.

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if (total + target) % 2 or abs(target) > total:
        return 0
    capacity = (total + target) // 2              ## the subset given a plus sign
    dp = [0] * (capacity + 1)
    dp[0] = 1
    for x in nums:                                ## 0/1 knapsack: capacity DESCENDING
        for c in range(capacity, x - 1, -1):
            dp[c] += dp[c - x]
    return dp[capacity]

def target_sum_brute(nums, target):               ## check: every assignment of signs
    n, total = len(nums), 0
    for mask in range(1 << n):
        value = sum(nums[i] if mask >> i & 1 else -nums[i] for i in range(n))
        total += value == target
    return total

## tests

assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5
assert find_target_sum_ways([1], 1) == 1
assert find_target_sum_ways([1], 2) == 0
cases = [([1, 1, 1, 1, 1], 3), ([1], 1), ([1], 2), ([0, 0, 1], 1), ([2, 3, 5, 1], 1)]
for nums, target in cases:                        ## cross-check against every sign assignment
    assert find_target_sum_ways(nums, target) == target_sum_brute(nums, target)
print(find_target_sum_ways([1, 1, 1, 1, 1], 3), find_target_sum_ways([0, 0, 1], 1),
      "brute force agrees")
```

```
5 4 brute force agrees
```

**Complexity.** $O(n \times \text{total})$ time, $O(\text{total})$ space. Zeros count twice, once with
each sign, which the case `[0, 0, 1]` checks.

### P12. Coin Change II — count the combinations of coins summing to `amount`, as a full table first

**Which template.** The `dp[item][capacity]` table, then its one-row reduction.
**The trick.** Writing the two-dimensional version first removes the mystery from the loop order in the
one-row version. `dp[i][a]` means "using only the first `i` coin types, the number of ways to make
`a`". Skipping coin `i` reads the **previous row**. Using coin `i` reads the **same row** at `a -
coins[i-1]`, and reading the same row is exactly what permits unlimited reuse. Collapse the rows and
that same-row read becomes the ascending capacity loop. The item axis is what forbids counting `1+2`
and `2+1` separately, because a row is only ever entered once per coin type.

```python
def change_2d(amount, coins):
    k = len(coins)
    dp = [[0] * (amount + 1) for _ in range(k + 1)]
    for i in range(k + 1):
        dp[i][0] = 1                              ## one way to make 0 with any coin prefix
    for i in range(1, k + 1):
        for a in range(amount + 1):
            dp[i][a] = dp[i - 1][a]               ## do not use coin i at all
            if coins[i - 1] <= a:
                dp[i][a] += dp[i][a - coins[i - 1]]   ## use it again: SAME row, unlimited
    return dp[k][amount]

def change_1d(amount, coins):                     ## the space-reduced form of the same table
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]

## tests

assert change_2d(5, [1, 2, 5]) == 4
assert change_2d(3, [2]) == 0
assert change_2d(10, [10]) == 1
for amount, coins in [(5, [1, 2, 5]), (11, [1, 2, 5]), (9, [2, 3, 4]), (0, [7]), (100, [1, 5, 10])]:
    assert change_2d(amount, coins) == change_1d(amount, coins)
print(change_2d(5, [1, 2, 5]), change_2d(100, [1, 5, 10]), "the one-row form agrees")
```

```
4 121 the one-row form agrees
```

**Complexity.** $O(k \times \text{amount})$ time; $O(k \times \text{amount})$ space for the table and
$O(\text{amount})$ after the reduction.

### P13. 0/1 Knapsack — maximum value of items fitting in a capacity, each item used at most once

**Which template.** The `dp[item][capacity]` table. Every named knapsack problem is a costume for this.
**The trick.** Write the table form once, by hand, and the disguises become obvious: Partition Equal
Subset Sum is this with all values equal to the weights and a boolean answer, and Target Sum is this
with a shifted capacity. The take branch reads `dp[i-1][...]`, the **previous** row, which is what
enforces "each item at most once". Collapsing to one row therefore needs the capacity loop to run
**descending**, so that `dp[c - w]` still holds a value from before this item was offered.

```python
def knapsack_2d(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]               ## skip item i
            if weights[i - 1] <= c:               ## take it: read the PREVIOUS row
                dp[i][c] = max(dp[i][c], dp[i - 1][c - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]

def knapsack_1d(weights, values, capacity):       ## the same table, one row, capacity descending
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]

def knapsack_brute(weights, values, capacity):    ## check: every subset of items
    n, best = len(weights), 0
    for mask in range(1 << n):
        w = sum(weights[i] for i in range(n) if mask >> i & 1)
        if w <= capacity:
            best = max(best, sum(values[i] for i in range(n) if mask >> i & 1))
    return best

## tests

assert knapsack_2d([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
assert knapsack_1d([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
cases = [([1, 3, 4, 5], [1, 4, 5, 7], 7), ([2, 2, 3], [3, 3, 5], 4), ([5], [10], 4),
         ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 6)]
for w, v, c in cases:                             ## cross-check both forms against every subset
    assert knapsack_2d(w, v, c) == knapsack_1d(w, v, c) == knapsack_brute(w, v, c)
print(knapsack_2d([1, 3, 4, 5], [1, 4, 5, 7], 7), "one-row form and brute force agree")
```

```
9 one-row form and brute force agree
```

**Complexity.** $O(n \times \text{capacity})$ time; $O(n \times \text{capacity})$ or
$O(\text{capacity})$ space.

### P14. Best Time to Buy and Sell Stock with Cooldown, and with a Transaction Fee — two state machines

**Which template.** `dp[i][state]`, with the states carried as plain variables.
**The trick.** Draw the machine before writing code. The cooldown version has three states: HOLD, SOLD
on this day, and REST. You may buy only from REST, because SOLD is the cooldown day, and SOLD becomes
REST on the following day. That single missing edge, SOLD to HOLD, **is** the cooldown. The fee version
needs no third state: it is the ordinary two-state machine with the fee subtracted once per sale. Update
all states in one tuple assignment so that each reads the previous day's values.

```python
def max_profit_cooldown(prices):
    hold, sold, rest = float("-inf"), float("-inf"), 0
    for price in prices:
        hold, sold, rest = (max(hold, rest - price),   ## buy only from REST, never from SOLD
                            hold + price,              ## selling today puts you in SOLD
                            max(rest, sold))           ## SOLD becomes REST after one day
    return max(sold, rest)

def max_profit_fee(prices, fee):
    hold, free = float("-inf"), 0                 ## the same machine, no cooldown state
    for price in prices:
        hold, free = max(hold, free - price), max(free, hold + price - fee)
    return free

def brute_cooldown(prices):                       ## check: recursion over buy/sell/wait
    def go(i, holding, blocked):
        if i == len(prices):
            return 0
        best = go(i + 1, holding, False)          ## wait
        if holding:
            best = max(best, prices[i] + go(i + 1, False, True))
        elif not blocked:
            best = max(best, -prices[i] + go(i + 1, True, False))
        return best
    return go(0, False, False)

## tests

assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
assert max_profit_cooldown([1]) == 0
assert max_profit_fee([1, 3, 2, 8, 4, 9], 2) == 8
assert max_profit_fee([1, 3, 7, 5, 10, 3], 3) == 6
for a in [[1, 2, 3, 0, 2], [1], [6, 1, 3, 2, 4, 7], [2, 1, 4], [1, 4, 2, 7]]:
    assert max_profit_cooldown(a) == brute_cooldown(a)
print(max_profit_cooldown([1, 2, 3, 0, 2]), max_profit_fee([1, 3, 2, 8, 4, 9], 2),
      "brute force agrees on the cooldown machine")
```

```
3 8 brute force agrees on the cooldown machine
```

**Complexity.** $O(n)$ time, $O(1)$ space for both.

### P15. Longest Increasing Path in a Matrix — the longest strictly increasing path moving in four directions

**Which template.** Template 4, memoised DFS, because no sweep order exists.
**The trick.** State why the memo is safe: the path values strictly increase, so the recursion follows
a directed acyclic graph and can never revisit a cell within one call. That is what allows a plain
dictionary with no visited set. Define the state as the longest path **starting** at a cell rather than
ending at it, so the recursion moves forwards and the answer is the maximum over all cells.

```python
def longest_increasing_path(matrix):
    if not matrix or not matrix[0]:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    memo = {}
    def best(r, c):                               ## longest increasing path STARTING at (r, c)
        if (r, c) in memo:
            return memo[(r, c)]
        length = 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                length = max(length, 1 + best(nr, nc))
        memo[(r, c)] = length                     ## no cycles: values strictly increase
        return length
    return max(best(r, c) for r in range(rows) for c in range(cols))

def brute_path(matrix):                           ## check: the same search with no memo
    rows, cols = len(matrix), len(matrix[0])
    def go(r, c):
        length = 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                length = max(length, 1 + go(nr, nc))
        return length
    return max(go(r, c) for r in range(rows) for c in range(cols))

## tests

assert longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]) == 4
assert longest_increasing_path([[3, 4, 5], [3, 2, 6], [2, 2, 1]]) == 4
assert longest_increasing_path([[1]]) == 1
grids = [[[9, 9, 4], [6, 6, 8], [2, 1, 1]], [[3, 4, 5], [3, 2, 6], [2, 2, 1]], [[1]],
         [[1, 2, 3], [6, 5, 4], [7, 8, 9]]]
for g in grids:                                   ## cross-check against the unmemoised search
    assert longest_increasing_path(g) == brute_path(g)
print(longest_increasing_path([[9, 9, 4], [6, 6, 8], [2, 1, 1]]),
      longest_increasing_path([[1, 2, 3], [6, 5, 4], [7, 8, 9]]), "unmemoised search agrees")
```

```
4 9 unmemoised search agrees
```

**Complexity.** $O(rc)$ time, because each cell is computed once, and $O(rc)$ space.

### P16. Maximal Square — the area of the largest square of 1s in a binary matrix

**Which template.** Template 2, with the state ending at a cell.
**The trick.** Let `dp[r][c]` be the side of the largest square whose **bottom-right corner** is at
`(r, c)`. A square of side `k+1` ending here requires squares of side at least `k` ending at the cell
above, the cell to the left and the cell diagonally up-left, so the side is one plus the **minimum** of
those three. The minimum is the whole idea: the smallest of the three neighbours is the binding
constraint. Return the best side squared, not the best side.

```python
def maximal_square(matrix):
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]        ## dp[r][c] = side of the square ENDING here
    best = 0
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:                             ## three neighbours limit the square
                    dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1])
                best = max(best, dp[r][c])
    return best * best

def maximal_square_brute(matrix):                 ## check: try every square directly
    rows, cols, best = len(matrix), len(matrix[0]), 0
    for r in range(rows):
        for c in range(cols):
            side = 1
            while r + side <= rows and c + side <= cols:
                if all(matrix[i][j] == 1 for i in range(r, r + side)
                       for j in range(c, c + side)):
                    best = max(best, side)
                side += 1
    return best * best

## tests

grid = [[1, 0, 1, 0, 0], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 1, 0]]
assert maximal_square(grid) == 4
assert maximal_square([[0, 1], [1, 0]]) == 1
assert maximal_square([[0]]) == 0
for g in [grid, [[0, 1], [1, 0]], [[0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]:
    assert maximal_square(g) == maximal_square_brute(g)
print(maximal_square(grid), maximal_square([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
      "brute force agrees")
```

```
4 9 brute force agrees
```

**Complexity.** $O(rc)$ time, $O(rc)$ space, reducible to one row.

### P17. Burst Balloons — burst every balloon, earning the product of a balloon and its two live neighbours

**Which template.** Template 3, interval DP, and the reframing is the entire problem.
**The trick.** Do not ask which balloon is burst **first**. That choice splits the row into two halves
that are not independent, because the halves become neighbours once the middle balloon is gone. Ask
which balloon is burst **last** inside the open range `(i, j)`. When balloon `last` is burst last,
everything strictly inside `(i, last)` and `(last, j)` is already gone, so at that moment the live
neighbours of `last` are exactly the walls `i` and `j`. The gain is therefore
`balloons[i] * balloons[last] * balloons[j]`, a fixed number, and the two sub-ranges are genuinely
independent. Pad the array with a 1 at each end so the walls always exist.

```python
def max_coins(nums):
    balloons = [1] + list(nums) + [1]             ## virtual 1s at both ends
    n = len(balloons)
    dp = [[0] * n for _ in range(n)]              ## dp[i][j] = best over the OPEN range (i, j)
    for length in range(2, n):                    ## by increasing gap between the walls
        for i in range(n - length):
            j = i + length
            for last in range(i + 1, j):          ## which balloon is burst LAST inside (i, j)
                gain = balloons[i] * balloons[last] * balloons[j]
                dp[i][j] = max(dp[i][j], dp[i][last] + gain + dp[last][j])
    return dp[0][n - 1]

def max_coins_brute(nums):                        ## check: burst in every possible order
    def go(rest):
        if not rest:
            return 0
        best = 0
        for k in range(len(rest)):
            left = rest[k - 1] if k > 0 else 1
            right = rest[k + 1] if k + 1 < len(rest) else 1
            best = max(best, left * rest[k] * right + go(rest[:k] + rest[k + 1:]))
        return best
    return go(tuple(nums))

## tests

assert max_coins([3, 1, 5, 8]) == 167
assert max_coins([1, 5]) == 10
assert max_coins([]) == 0
for a in [[3, 1, 5, 8], [1, 5], [], [7], [2, 4, 3, 5], [9, 76, 64]]:
    assert max_coins(a) == max_coins_brute(a)
print(max_coins([3, 1, 5, 8]), max_coins([9, 76, 64]), "brute force agrees")
```

```
167 44416 brute force agrees
```

**Complexity.** $O(n^3)$ time — a length loop, a start loop and a split loop — and $O(n^2)$ space.

### P18. Palindrome Partitioning II — the fewest cuts so that every piece of `s` is a palindrome

**Which template.** An interval table for the palindrome test, then a one-dimensional cut table on top.
**The trick.** Two tables, computed in the right order. First `is_pal[i][j]`, filled by increasing
length, because `s[i..j]` is a palindrome when the ends match and the inside already is. Then
`cuts[i]`, the fewest cuts for the first `i` characters: try every last piece `s[j..i-1]` that is a
palindrome and take `cuts[j] + 1`. Set `cuts[0] = -1`, not 0, so that a whole string which is itself a
palindrome costs zero cuts — with `cuts[0] = 0` the answer for `"aa"` comes out as 1.

```python
def min_cut(s):
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for length in range(1, n + 1):                ## intervals by increasing length
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length <= 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
    cuts = [0] * (n + 1)
    cuts[0] = -1                                  ## -1 so a whole-string palindrome costs 0 cuts
    for i in range(1, n + 1):
        cuts[i] = i - 1                           ## worst case: cut before every character
        for j in range(i):
            if is_pal[j][i - 1]:                  ## s[j..i-1] is a palindrome piece
                cuts[i] = min(cuts[i], cuts[j] + 1)
    return cuts[n] if n else 0

def min_cut_brute(s):                             ## check: recursion over every first piece
    def go(i):
        if i == len(s):
            return 0
        best = len(s)
        for j in range(i + 1, len(s) + 1):
            piece = s[i:j]
            if piece == piece[::-1]:
                best = min(best, 1 + go(j))
        return best
    return max(go(0) - 1, 0)

## tests

assert min_cut("aab") == 1
assert min_cut("a") == 0
assert min_cut("ab") == 1
assert min_cut("") == 0
for s in ["aab", "a", "ab", "abacdc", "cabababcbc", "leet", "aaaa"]:
    assert min_cut(s) == min_cut_brute(s)
print(min_cut("aab"), min_cut("cabababcbc"), "brute force agrees")
```

```
1 3 brute force agrees
```

**Complexity.** $O(n^2)$ time, $O(n^2)$ space.

## Tricks and tips

**Say both indices out loud, then say the fill order.** "`dp[i][j]` is the edit distance between the
first `i` characters of `a` and the first `j` of `b`, filled top to bottom and left to right." Two
sentences, ten seconds, and they prevent the two failures that matter: a state that does not determine
its own subproblems, and a loop order that reads unwritten cells.

**Prefix tables need the extra row and column.** Index `0` means the empty prefix, so the table is
`(n+1)` by `(m+1)` and the character at table position `i` is `a[i-1]`. Sizing the table `n` by `m`
forces awkward special cases everywhere and is the most common structural mistake in this family.

**The base row and column carry the whole problem type.** Zero for longest common subsequence, `i` and
`j` for edit distance, 1 for distinct subsequences. Fill them before the double loop and check one
value by hand. If the whole table comes out as zeros, the base is wrong, not the recurrence.

**Interval problems iterate by length.** `dp[i][j]` for a range reads intervals strictly inside itself,
and no row-by-row sweep guarantees those are ready. Write `for length in range(2, n+1)` and derive `j`
from `i` and `length`. An index-by-index loop compiles, runs, and reads zeros.

**When the order is impossible, memoise instead.** Four-directional grid movement has no sweep order at
all, so recursion plus a dictionary is not a shortcut, it is the correct tool. Check that the recursion
cannot cycle — in Longest Increasing Path the strictly increasing values guarantee it — and say that
out loud, because it is the part an interviewer will probe.

**Interval problems often invert the choice.** Burst Balloons is unsolvable while you think about the
first balloon and easy once you think about the last, because "last" is what makes the two sides
independent. Whenever splitting on the first move leaves the halves coupled, try splitting on the last
one.

**Reduce two rows to one only after the table works.** A two-string table needs the previous row and
the current row, so two rows suffice. Collapsing to one row is possible for the counting variants but
requires scanning in the direction that preserves the value you still need — right to left for
distinct subsequences, descending capacity for 0/1 knapsack. Get the full table correct first, then
reduce, and be ready to say that the reduction costs you the ability to reconstruct the answer.

**A third string does not always need a third index.** In Interleaving String the position in `s3` is
`i + j`. Look for that collapse whenever a problem seems to need three indices; usually one of them is
a function of the other two.

## The bugs that cost the round

**Filling an interval table in index order.** `for i` then `for j` reads `dp[i+1][j-1]`, a shorter
interval that has not been computed, and gets 0. The code runs and the answer is quietly too small.
Loop over the length.

**The wrong table size.** A prefix table must be `(n+1)` by `(m+1)`. Using `n` by `m` and then indexing
`a[i]` instead of `a[i-1]` is a mismatch that produces plausible values on short inputs.

**Base cases left at zero.** Distinct Subsequences with `dp[i][0] = 0` returns 0 for everything, and
Edit Distance without the `i` and `j` edges returns nonsense for any pair where one string is a prefix
of the other.

**Off-by-one in Palindrome Partitioning II.** `cuts[0]` must be `-1`, because the recurrence counts
pieces and the answer counts cuts. With 0 the answer for a whole-string palindrome comes out one too
high, and every sample where the answer is not zero still passes.

**Updating a state machine sequentially.** In the cooldown machine, computing `hold` first and then
using the new `hold` to compute `sold` sells a share bought on the same day. Assign all three states in
one tuple assignment from the previous day's values.

**Confusing the two kinds of star.** In a regular expression the star binds to the preceding character,
so skipping the pair means `j - 2`. In wildcard matching the star stands alone, so skipping it means
`j - 1`. Writing `j - 2` in the wildcard solution reads a pattern position that does not belong to the
star and matches the wrong strings.

**Returning the wrong quantity in Maximal Square.** The table holds the side, and the question asks for
the area. Square the best side once at the end.

## Done when

- Given a two-string problem, you can draw the `(n+1)` by `(m+1)` table, state what row 0 and column 0
  hold, and state the match and mismatch branches, before writing code.
- You can convert longest common subsequence into edit distance and into distinct subsequences by
  changing only the base row, the neighbours used, and the combination rule.
- You can say the fill order for a prefix table, a grid sweep and an interval table, and explain why an
  interval table must go by increasing length.
- You can recognise when no sweep order exists and reach for a memoised depth-first search instead, and
  justify why the recursion cannot cycle.
