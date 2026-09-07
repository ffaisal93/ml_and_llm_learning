# Backtracking and tries: every variation

Backtracking is depth-first search over the space of partial solutions, with one addition: you undo
your choice on the way out. You make a choice, you recurse, and then you put the state back exactly as
you found it. That undo is what lets a single mutable `path` list serve every branch of the tree, so
the space cost is the depth of the recursion and not the number of answers. It beats brute force
because it prunes: a partial solution that already violates a constraint is abandoned before its whole
subtree is built.

The thing that makes backtracking hard is not the recursion. It is deciding the **shape of the choice**
at each level. You either choose "which element comes next" — that is a permutation — or you choose
"do I include this element or not" — that is a subset. That single decision determines the loop, the
start index and the duplicate handling. Everything else is bookkeeping.

The second organising idea is that every backtracking solution answers three questions. What is the
state? What are the choices at this state? When do I stop and record? If you can say those three out
loud before you type, the code writes itself. If you cannot, no amount of debugging the recursion will
save you.

## Recognising it from the phrasing

| The interviewer says | Choice shape | The mechanism |
|---|---|---|
| "all subsets", "the power set" | include-or-exclude | two calls per element, or a loop with a start index |
| "all permutations", "every ordering" | which element comes next | loop over every element, `used` array |
| "combinations summing to a target, numbers may be reused" | which element comes next, no going back | loop with a start index, recurse on `i` |
| "combinations, each number used once" | include-or-exclude over an index | loop with a start index, recurse on `i + 1` |
| "the input has duplicates and the output must not" | same shape, fewer branches | sort first, skip a repeat at the same depth |
| "partition a string into pieces" | where to cut | choose an end index, recurse on the remainder |
| "place items on a board" | which cell for this row | keep constraint sets, test in $O(1)$ |
| "find a word in a grid" | which neighbour to step to | DFS, mark the cell, unmark on the way out |
| "find many words in one grid" | which neighbour, guided by a prefix | a trie over the word list plus DFS |

Before writing any code, ask one question: **what does a partial solution look like halfway through,
and what does a single choice add to it?** If the choice is "which item goes next", the answer has a
permutation shape and you need a `used` set, because any unused element can come next. If the choice is
"is this item in or out", the answer has a subset shape and you need a start index, because you walk
the input once from left to right and never look back. Getting this wrong produces duplicates or
missing answers. A subset problem written with a `used` array gives you every ordering of every subset;
a permutation problem written with a start index gives you only the subsets. Neither failure is a bug
in the recursion, so debugging the recursion will not fix it.

## The templates

**Template 1 — subsets by include-or-exclude.** Use when you want the choice shape stated as literally
as possible: two branches per element, one that takes it and one that does not.

```python
def subsets_include_exclude(nums):
    out = []
    path = []
    def dfs(i):
        if i == len(nums):                    ## no choices left: record
            out.append(path[:])               ## COPY, because path keeps changing
            return
        path.append(nums[i])                  ## choice A: include nums[i]
        dfs(i + 1)
        path.pop()                            ## undo
        dfs(i + 1)                            ## choice B: exclude nums[i]
    dfs(0)
    return out

## tests

assert subsets_include_exclude([1, 2]) == [[1, 2], [1], [2], []]
assert len(subsets_include_exclude([1, 2, 3])) == 8
assert subsets_include_exclude([]) == [[]]
print(subsets_include_exclude([1, 2, 3]))
```

```
[[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]
```

**Template 2 — subsets by a loop with a start index.** Use this form for everything else, because the
loop is what carries the duplicate skip and the pruning break.

```python
def subsets_start_index(nums):
    out = []
    path = []
    def dfs(start):
        out.append(path[:])                   ## EVERY node is an answer here
        for i in range(start, len(nums)):
            path.append(nums[i])              ## choose nums[i] as the next element
            dfs(i + 1)                        ## i + 1: each element used at most once
            path.pop()                        ## undo
    dfs(0)
    return out

## tests

assert subsets_start_index([1, 2]) == [[], [1], [1, 2], [2]]
assert len(subsets_start_index([1, 2, 3])) == 8
assert sorted(map(sorted, subsets_start_index([1, 2, 3]))) == sorted(map(sorted, [[1,2,3],[1,2],[1,3],[1],[2,3],[2],[3],[]]))
print(subsets_start_index([1, 2, 3]))
```

```
[[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

Templates 1 and 2 produce the **same set of subsets in a different order**, and you should know both.
Template 1 records only at the leaves, so the recursion tree has exactly $2^n$ leaves and every answer
sits at depth `n`. Template 2 records at every node, so the tree has $2^n$ nodes and answers appear at
every depth. Template 1 is the clearer statement of the include-or-exclude idea and the easier one to
explain out loud. Template 2 is the one that generalises, because a loop gives you somewhere to put
`continue` for duplicates and `break` for pruning.

**Template 3 — permutations with a `used` array.** Use when the choice is "which element comes next".
There is no start index, because every unused element is a legal next choice at every level.

```python
def permutations(nums):
    out = []
    path = []
    used = [False] * len(nums)
    def dfs():
        if len(path) == len(nums):            ## a full-length path is a permutation
            out.append(path[:])
            return
        for i in range(len(nums)):            ## NO start index: every position is open
            if used[i]:
                continue                      ## each element appears exactly once
            used[i] = True
            path.append(nums[i])
            dfs()
            path.pop()                        ## undo both parts of the choice
            used[i] = False
    dfs()
    return out

## tests

assert permutations([1, 2, 3]) == [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
assert len(permutations([1, 2, 3, 4])) == 24
assert permutations([]) == [[]]
print(permutations([1, 2, 3]))
```

```
[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

**Template 4 — the duplicate skip.** Use when the input contains repeated values and the output must
not repeat. Sort first, then add one line to the loop of template 2.

```python
def subsets_with_duplicates(nums):
    nums = sorted(nums)                       ## equal values must be adjacent
    out = []
    path = []
    def dfs(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                      ## same value, same depth: already tried
            path.append(nums[i])
            dfs(i + 1)
            path.pop()
    dfs(0)
    return out

## tests

assert subsets_with_duplicates([1, 2, 2]) == [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
assert subsets_with_duplicates([2, 2, 2]) == [[], [2], [2, 2], [2, 2, 2]]
assert subsets_with_duplicates([]) == [[]]
print(subsets_with_duplicates([1, 2, 2]))
```

```
[[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
```

The guard is `i > start`, and not `i > 0`, and the difference decides whether the function is correct.
`i > start` means "this is not the first candidate considered at this level", so the skip fires only
when the same value has already been tried **at the same depth**, which is exactly the branch that
would produce a duplicate answer. Take `[2, 2]`. At depth 0 the loop runs with `start = 0`: `i = 0`
takes the first 2, and `i = 1` is skipped because `i > start` and the values are equal. Inside that
first branch the loop runs with `start = 1`: now `i = 1` and `i > start` is **false**, so the second 2
is taken and `[2, 2]` is produced. With `i > 0` the second 2 would be skipped there too, and `[2, 2]`
would never be built. So `i > 0` deletes real answers, while `i > start` deletes only repeats. Say the
distinction as "skip a repeat sideways, never downwards".

**Template 5 — a trie node with insert and search.** Use when the problem holds a set of words and you
need prefix questions answered.

```python
class TrieNode:
    def __init__(self):
        self.children = {}                    ## character -> TrieNode
        self.is_word = False                  ## a word ENDS at this node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True                   ## mark only the last node

    def _walk(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_word

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

## tests

t = Trie()
for w in ["apple", "app"]:
    t.insert(w)
assert t.search("apple") is True
assert t.search("appl") is False
assert t.starts_with("appl") is True
assert t.starts_with("bad") is False
print(t.search("apple"), t.search("appl"), t.starts_with("appl"))
```

```
True False True
```

## Pruning: the difference between passing and timing out

A correct backtracking solution and a fast one differ by a few lines. Both explore the same tree; the
fast one refuses to enter subtrees that cannot contain an answer. Three techniques cover almost every
interview problem.

**Sort the candidates, then `break` instead of `continue`.** If the candidate list is sorted ascending
and `candidates[i]` already exceeds the remaining target, then every later candidate also exceeds it,
so you leave the loop rather than skipping one entry. Sorting is what turns a `continue` into a
`break`, and a `break` cuts the whole tail of the loop at that node.

**Test feasibility before recursing, not at the base case.** The naive shape recurses on every
candidate and discovers the overshoot when `remaining` goes negative one level down. That wastes one
call per dead branch, and those calls multiply at every level. Moving the test above the call removes
the entire subtree instead of its root.

**Keep constraint sets so the legality test is $O(1)$.** In N-Queens the question at row `r`, column
`c` is "does any earlier queen attack this square". Scanning the board is $O(n)$ per square. Instead
keep three sets: the used columns, the used values of `r - c`, and the used values of `r + c`. Two
squares share a descending diagonal exactly when `r - c` is equal, and an ascending diagonal exactly
when `r + c` is equal, so three set lookups answer the question in $O(1)$.

**Worked example.** Combination Sum on `[2, 3, 6, 7]` with target 29, counted both ways. The version
without pruning finds the overshoot at the base case. The version with pruning sorts and breaks out of
the loop as soon as a candidate is larger than what is left.

```python
def combination_sum_no_pruning(candidates, target):
    out, path = [], []
    calls = 0
    def dfs(start, remaining):
        nonlocal calls
        calls += 1
        if remaining == 0:
            out.append(path[:])
            return
        if remaining < 0:                     ## overshoot found only at the base case
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            dfs(i, remaining - candidates[i])
            path.pop()
    dfs(0, target)
    return out, calls

def combination_sum_pruned(candidates, target):
    candidates = sorted(candidates)           ## sorting is what licenses the break
    out, path = [], []
    calls = 0
    def dfs(start, remaining):
        nonlocal calls
        calls += 1
        if remaining == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                         ## every later candidate is larger: stop
            path.append(candidates[i])
            dfs(i, remaining - candidates[i])
            path.pop()
    dfs(0, target)
    return out, calls

## tests

plain, plain_calls = combination_sum_no_pruning([2, 3, 6, 7], 29)
fast, fast_calls = combination_sum_pruned([2, 3, 6, 7], 29)
assert sorted(map(sorted, plain)) == sorted(map(sorted, fast))
assert fast_calls < plain_calls
print("answers:", len(plain), "calls without pruning:", plain_calls, "calls with pruning:", fast_calls)
```

```
answers: 34 calls without pruning: 592 calls with pruning: 350
```

Both versions return the same 34 combinations. The unpruned version makes 592 recursive calls and the
pruned version makes 350, a reduction of about 41 percent from two lines. The saving grows with the
target and with the number of candidates, because each avoided call is the root of a subtree that is
never built. Say the two numbers in the interview: "sorting plus an early break takes this from 592
calls to 350 on this input" is a stronger statement than "I added pruning".

## The problems

### P1. Subsets — return every subset of a list of distinct integers

**Which template.** Template 2, the loop with a start index.
**The trick.** Record at every node rather than only at the leaves, because in this problem every
partial path is itself a complete answer. Copy the path with `path[:]` when you record it; appending
`path` itself stores a reference to a list that the recursion is about to mutate, so all your answers
end up identical and empty.

```python
def subsets(nums):
    out, path = [], []
    def dfs(start):
        out.append(path[:])                   ## every node on the tree is a subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)                        ## i + 1: never reuse nums[i]
            path.pop()
    dfs(0)
    return out

## tests

assert subsets([1, 2, 3]) == [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
assert len(subsets([1, 2, 3, 4, 5])) == 32
assert subsets([]) == [[]]
print(subsets([1, 2, 3]))
```

```
[[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

**Complexity.** $O(n \cdot 2^n)$ time and $O(n)$ extra space beyond the output.

### P2. Subsets II — return every distinct subset when the input may contain repeated values

**Which template.** Template 4: template 2 plus the duplicate skip.
**The trick.** Sort so equal values are adjacent, then skip a value at a level if the same value was
already the choice at that level. The guard is `i > start`. It removes sideways repeats and keeps the
downwards path that builds `[2, 2]` from two equal values.

```python
def subsets_with_dup(nums):
    nums = sorted(nums)                       ## duplicates must sit next to each other
    out, path = [], []
    def dfs(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                      ## skip a repeat at THIS depth only
            path.append(nums[i])
            dfs(i + 1)
            path.pop()
    dfs(0)
    return out

## tests

assert subsets_with_dup([1, 2, 2]) == [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
assert len(subsets_with_dup([4, 4, 4, 1, 4])) == 10
assert subsets_with_dup([0]) == [[], [0]]
print(subsets_with_dup([1, 2, 2]))
```

```
[[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
```

**Complexity.** $O(n \cdot 2^n)$ time worst case, $O(n)$ extra space. Sorting adds $O(n \log n)$.

### P3. Permutations — return every ordering of a list of distinct integers

**Which template.** Template 3, the `used` array.
**The trick.** There is no start index, because the choice is "which element comes next" and any unused
element qualifies. The undo has two parts and both matter: `path.pop()` and `used[i] = False`. Leaving
out the second one silently returns only the first permutation, because every element stays marked.

```python
def permute(nums):
    out, path = [], []
    used = [False] * len(nums)
    def dfs():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i in range(len(nums)):            ## every index, every level
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            dfs()
            path.pop()
            used[i] = False                   ## undo BOTH halves of the choice
    dfs()
    return out

## tests

assert permute([1, 2, 3]) == [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
assert permute([0, 1]) == [[0, 1], [1, 0]]
assert len(permute([1, 2, 3, 4, 5])) == 120
print(permute([1, 2, 3]))
```

```
[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```

**Complexity.** $O(n \cdot n!)$ time, $O(n)$ extra space.

### P4. Permutations II — return every distinct ordering when the input may contain repeated values

**Which template.** Template 3 plus a duplicate skip, but the skip line is **different** from template 4.
**The trick.** After sorting, skip `nums[i]` when it equals `nums[i - 1]` and the twin at `i - 1` is
**not currently used**. If the twin is unused, the branch that starts with the twin has already been
explored at this level, so this branch is a repeat. If the twin **is** used, it sits above you in the
path and this is the legitimate second copy. The condition `not used[i - 1]` is doing the same job that
`i > start` does in the subset form: it distinguishes sideways from downwards.

```python
def permute_unique(nums):
    nums = sorted(nums)
    out, path = [], []
    used = [False] * len(nums)
    def dfs():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue                      ## the earlier twin is free, so this branch repeats
            used[i] = True
            path.append(nums[i])
            dfs()
            path.pop()
            used[i] = False
    dfs()
    return out

## tests

assert permute_unique([1, 1, 2]) == [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
assert len(permute_unique([1, 2, 3])) == 6
assert len(permute_unique([2, 2, 1, 1])) == 6
print(permute_unique([1, 1, 2]))
```

```
[[1, 1, 2], [1, 2, 1], [2, 1, 1]]
```

**Complexity.** $O(n \cdot n!)$ time worst case, $O(n)$ extra space.

### P5. Combination Sum — all combinations of the candidates that sum to `target`, each number reusable

**Which template.** Template 2, but the recursive call passes `i` and not `i + 1`.
**The trick.** Reuse is allowed, so a candidate stays available after you choose it, and `dfs(i, ...)`
expresses that in one character. The start index still exists, and it is what stops `[2, 3]` and
`[3, 2]` from both appearing: you may repeat a candidate but you may never go back to an earlier one.
Sorting plus `break` prunes the tail of each loop.

```python
def combination_sum(candidates, target):
    candidates = sorted(candidates)           ## sorted so the break below is valid
    out, path = [], []
    def dfs(start, remaining):
        if remaining == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                         ## prune: all later candidates are larger
            path.append(candidates[i])
            dfs(i, remaining - candidates[i]) ## i, NOT i + 1: reuse is allowed
            path.pop()
    dfs(0, target)
    return out

## tests

assert combination_sum([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]
assert combination_sum([2, 3, 5], 8) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
assert combination_sum([2], 1) == []
print(combination_sum([2, 3, 6, 7], 7))
```

```
[[2, 2, 3], [7]]
```

**Complexity.** $O(n^{T/m})$ time in the worst case, where `T` is the target and `m` the smallest
candidate, and $O(T/m)$ recursion depth.

### P6. Combination Sum II — all distinct combinations summing to `target`, each input number used once

**Which template.** Template 2 with `i + 1`, plus the template 4 duplicate skip.
**The trick.** Two changes from P5, and they are independent. `i + 1` makes each **index** usable once.
The `i > start` skip makes each **value** usable once per level. You need both: without `i + 1` a
repeated value is reused forever, and without the skip two equal values at different indices generate
the same combination twice.

```python
def combination_sum2(candidates, target):
    candidates = sorted(candidates)
    out, path = [], []
    def dfs(start, remaining):
        if remaining == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                         ## prune the tail of the sorted list
            if i > start and candidates[i] == candidates[i - 1]:
                continue                      ## same value at the same depth
            path.append(candidates[i])
            dfs(i + 1, remaining - candidates[i])   ## i + 1: each index once
            path.pop()
    dfs(0, target)
    return out

## tests

assert combination_sum2([10, 1, 2, 7, 6, 1, 5], 8) == [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
assert combination_sum2([2, 5, 2, 1, 2], 5) == [[1, 2, 2], [5]]
assert combination_sum2([1], 5) == []
print(combination_sum2([10, 1, 2, 7, 6, 1, 5], 8))
```

```
[[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
```

**Complexity.** $O(n \cdot 2^n)$ time worst case, $O(n)$ recursion depth.

### P7. Combinations — every combination of `k` numbers chosen from 1 to `n`

**Which template.** Template 2 with a length target instead of a sum target.
**The trick.** Prune on the count. If you still need `need = k - len(path)` numbers, then any start
value above `n - need + 1` cannot work, because there are not enough numbers left above it. Capping the
loop at `n - need + 2` in `range` terms removes every doomed branch and is the cheapest possible
feasibility test.

```python
def combine(n, k):
    out, path = [], []
    def dfs(start):
        if len(path) == k:
            out.append(path[:])
            return
        need = k - len(path)                  ## how many numbers are still missing
        for i in range(start, n - need + 2):  ## prune: leave room for the rest
            path.append(i)
            dfs(i + 1)
            path.pop()
    dfs(1)
    return out

## tests

assert combine(4, 2) == [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
assert combine(1, 1) == [[1]]
assert len(combine(5, 3)) == 10
print(combine(4, 2))
```

```
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
```

**Complexity.** $O(k \cdot \binom{n}{k})$ time, $O(k)$ extra space.

### P8. Letter Combinations of a Phone Number — every string you can type from a digit string

**Which template.** Template 2 in its fixed-depth form: one choice per digit, always advance.
**The trick.** The depth is known in advance and equals `len(digits)`, so there is no start index and no
`used` array. The choices at level `k` are simply the letters on digit `k`. Return the empty list for
the empty input, not `[""]`; the graders test that case and the natural recursion gives the wrong one.

```python
def letter_combinations(digits):
    if not digits:
        return []
    keypad = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
              "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    out, path = [], []
    def dfs(index):
        if index == len(digits):              ## one letter chosen per digit
            out.append("".join(path))
            return
        for ch in keypad[digits[index]]:      ## the choices are this digit's letters
            path.append(ch)
            dfs(index + 1)                    ## always advance: depth is fixed
            path.pop()
    dfs(0)
    return out

## tests

assert letter_combinations("23") == ["ad","ae","af","bd","be","bf","cd","ce","cf"]
assert letter_combinations("") == []
assert letter_combinations("9") == ["w", "x", "y", "z"]
print(letter_combinations("23"))
```

```
['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
```

**Complexity.** $O(4^n \cdot n)$ time for `n` digits, $O(n)$ recursion depth.

### P9. Palindrome Partitioning — every way to cut a string into pieces that are all palindromes

**Which template.** Template 2 where the choice is a cut point, and the start index is the position of
the next cut.
**The trick.** The state is "how much of the string is already consumed", so `start` is both the loop
base and the base-case test. Check the palindrome property **before** recursing, which prunes the whole
subtree of a bad cut. Checking it at the base case instead would build every partition and then reject
most of them.

```python
def partition(s):
    out, path = [], []
    def is_palindrome(lo, hi):
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo, hi = lo + 1, hi - 1
        return True
    def dfs(start):
        if start == len(s):                   ## the whole string is consumed
            out.append(path[:])
            return
        for end in range(start, len(s)):      ## the choice is WHERE TO CUT
            if not is_palindrome(start, end):
                continue                      ## prune before recursing, not after
            path.append(s[start:end + 1])
            dfs(end + 1)
            path.pop()
    dfs(0)
    return out

## tests

assert partition("aab") == [["a", "a", "b"], ["aa", "b"]]
assert partition("a") == [["a"]]
assert len(partition("aaa")) == 4
print(partition("aab"))
```

```
[['a', 'a', 'b'], ['aa', 'b']]
```

**Complexity.** $O(n \cdot 2^n)$ time, $O(n)$ recursion depth. A precomputed palindrome table makes the
inner test $O(1)$ at a cost of $O(n^2)$ space.

### P10. Word Search — does a word appear in a grid along a path of adjacent cells, no cell reused

**Which template.** Grid DFS with mark and unmark. The choice is which of the four neighbours to step
to.
**The trick.** The visited set is the board itself. Overwrite the cell with a sentinel before you
recurse and restore the original character after, so the path cannot cross itself and no separate set
is needed. Restoring is not optional: the outer double loop starts a fresh search from every cell, and
a board left dirty makes later searches fail.

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])
    def dfs(r, c, k):
        if k == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[k]:
            return False
        board[r][c] = "#"                     ## mark visited IN PLACE
        found = (dfs(r + 1, c, k + 1) or dfs(r - 1, c, k + 1) or
                 dfs(r, c + 1, k + 1) or dfs(r, c - 1, k + 1))
        board[r][c] = word[k]                 ## unmark on the way out
        return found
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False

## tests

grid = [["A","B","C","E"], ["S","F","C","S"], ["A","D","E","E"]]
assert exist(grid, "ABCCED") is True
assert exist(grid, "SEE") is True
assert exist(grid, "ABCB") is False
assert grid == [["A","B","C","E"], ["S","F","C","S"], ["A","D","E","E"]]
print(exist(grid, "ABCCED"), exist(grid, "SEE"), exist(grid, "ABCB"))
```

```
True True False
```

**Complexity.** $O(R \cdot C \cdot 3^L)$ time for a word of length `L`, because after the first step
only three directions are new, and $O(L)$ recursion depth.

### P11. Generate Parentheses — every well-formed string of `n` pairs of brackets

**Which template.** Template 1 in spirit: two choices at each level, but each is guarded by a
feasibility test.
**The trick.** Do not generate all $2^{2n}$ strings and filter. Keep the counts `opened` and `closed`
and allow only legal moves: an open bracket while `opened < n`, and a close bracket while
`closed < opened`. Those two guards make every leaf of the tree a valid answer, so there is no
validation step at all. This problem also appears in the stack chapter from the other angle, where you
are given a string and must decide whether it is balanced; here you build only the balanced ones, and
the counter pair is the same invariant seen from the producing side.

```python
def generate_parenthesis(n):
    out, path = [], []
    def dfs(opened, closed):
        if len(path) == 2 * n:
            out.append("".join(path))
            return
        if opened < n:                        ## an open bracket is always legal here
            path.append("(")
            dfs(opened + 1, closed)
            path.pop()
        if closed < opened:                   ## close only what is already open
            path.append(")")
            dfs(opened, closed + 1)
            path.pop()
    dfs(0, 0)
    return out

## tests

assert generate_parenthesis(1) == ["()"]
assert generate_parenthesis(3) == ["((()))","(()())","(())()","()(())","()()()"]
assert len(generate_parenthesis(4)) == 14
print(generate_parenthesis(3))
```

```
['((()))', '(()())', '(())()', '()(())', '()()()']
```

**Complexity.** The output has the `n`-th Catalan number of strings, so time is
$O(4^n / \sqrt{n})$ and recursion depth is $O(n)$.

### P12. N-Queens — place `n` queens on an `n` by `n` board so that none attacks another

**Which template.** Row-by-row DFS with three constraint sets. The choice at level `r` is the column
for row `r`.
**The trick.** One queen per row is built into the recursion, so only columns and diagonals need
checking. Two squares are on the same descending diagonal exactly when `r - c` is equal, and on the
same ascending diagonal exactly when `r + c` is equal, because moving one step down and one step right
keeps `r - c` fixed while moving down and left keeps `r + c` fixed. Store those two integers in sets
alongside the used columns and the attack test is three hash lookups, $O(1)$ instead of $O(n)$.

```python
def solve_n_queens(n):
    out = []
    cols, diag, anti = set(), set(), set()    ## diag = r - c, anti = r + c
    queen_col = [0] * n
    def dfs(r):
        if r == n:
            out.append(["." * c + "Q" + "." * (n - c - 1) for c in queen_col])
            return
        for c in range(n):
            if c in cols or (r - c) in diag or (r + c) in anti:
                continue                      ## O(1) attack test
            cols.add(c); diag.add(r - c); anti.add(r + c)
            queen_col[r] = c
            dfs(r + 1)
            cols.remove(c); diag.remove(r - c); anti.remove(r + c)
    dfs(0)
    return out

## tests

assert solve_n_queens(4) == [[".Q..", "...Q", "Q...", "..Q."],
                             ["..Q.", "Q...", "...Q", ".Q.."]]
assert len(solve_n_queens(1)) == 1
assert len(solve_n_queens(2)) == 0
assert len(solve_n_queens(8)) == 92
print(len(solve_n_queens(8)), solve_n_queens(4)[0])
```

```
92 ['.Q..', '...Q', 'Q...', '..Q.']
```

**Complexity.** $O(n!)$ time in the worst case, $O(n)$ space for the sets and the row assignment.

### P13. Sudoku Solver — fill every blank cell of a 9 by 9 grid so the rules hold

**Which template.** Constraint-set DFS over a precomputed list of blanks, returning a boolean.
**The trick.** Two ideas. First, collect the blank cells once into a list so the recursion advances by
an index rather than searching the grid for the next hole at every level. Second, the function returns
`True` as soon as one solution is found, and on success you do **not** undo — the board must keep the
answer. That makes this the one problem on the page where the undo is conditional, so write
`if dfs(k + 1): return True` above the undo lines and not below them.

```python
def solve_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    blanks = []
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == ".":
                blanks.append((r, c))
            else:
                rows[r].add(v); cols[c].add(v); boxes[(r // 3) * 3 + c // 3].add(v)
    def dfs(k):
        if k == len(blanks):
            return True                       ## every blank is filled: stop at once
        r, c = blanks[k]
        b = (r // 3) * 3 + c // 3
        for v in "123456789":
            if v in rows[r] or v in cols[c] or v in boxes[b]:
                continue
            rows[r].add(v); cols[c].add(v); boxes[b].add(v); board[r][c] = v
            if dfs(k + 1):
                return True                   ## one solution is enough: do not undo
            rows[r].remove(v); cols[c].remove(v); boxes[b].remove(v); board[r][c] = "."
        return False
    dfs(0)
    return board

## tests

puzzle = [list(row) for row in [
    "53..7....", "6..195...", ".98....6.", "8...6...3", "4..8.3..1",
    "7...2...6", ".6....28.", "...419..5", "....8..79"]]
solved = solve_sudoku(puzzle)
assert "".join(solved[0]) == "534678912"
assert all(set(row) == set("123456789") for row in solved)
assert all(set(col) == set("123456789") for col in zip(*solved))
print("".join(solved[0]))
```

```
534678912
```

**Complexity.** Exponential in the number of blanks in the worst case, but the constraint sets keep it
fast in practice. Space is $O(1)$, because the board is a fixed 81 cells.

### P14. Restore IP Addresses — every valid IP address you can form by inserting three dots into a digit string

**Which template.** Template 2 with a bounded piece length: the choice is how many digits this octet
takes.
**The trick.** The depth is fixed at four, so the base case is two conditions and not one: you have four
parts **and** you have consumed the whole string. Both prunes use `break` rather than `continue`,
because the loop runs over increasing lengths: once a piece has a leading zero and length above one,
every longer piece does too, and once a piece exceeds 255, every longer piece does too.

```python
def restore_ip_addresses(s):
    out, path = [], []
    def dfs(start):
        if len(path) == 4:
            if start == len(s):               ## four parts AND nothing left over
                out.append(".".join(path))
            return
        for length in (1, 2, 3):
            end = start + length
            if end > len(s):
                break
            piece = s[start:end]
            if piece[0] == "0" and length > 1:
                break                         ## no leading zeros, and none longer either
            if int(piece) > 255:
                break                         ## longer pieces only get bigger
            path.append(piece)
            dfs(end)
            path.pop()
    dfs(0)
    return out

## tests

assert restore_ip_addresses("25525511135") == ["255.255.11.135", "255.255.111.35"]
assert restore_ip_addresses("0000") == ["0.0.0.0"]
assert restore_ip_addresses("101023") == ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
assert restore_ip_addresses("1") == []
print(restore_ip_addresses("25525511135"))
```

```
['255.255.11.135', '255.255.111.35']
```

**Complexity.** $O(1)$ in the strict sense, because at most $3^4$ splits are ever examined. Space is
$O(1)$ beyond the output.

### P15. Word Break II — every sentence you can form by cutting a string into dictionary words

**Which template.** Template 2 by cut point, with memoisation on the start index.
**The trick.** Plain backtracking is exponential on inputs like `"aaaaaaaaab"` with dictionary
`["a", "aa", "aaa"]`, because the same suffix is re-solved along every path that reaches it. The fix is
to make the recursion return the list of finished tails for a start index and cache it, so each suffix
is solved once. Note the base case returns `[""]` and not `[]`: one empty tail means "there is exactly
one way to finish from here", while an empty list would mean "there is no way", and that single choice
decides whether the function returns anything at all.

```python
def word_break(s, word_dict):
    words = set(word_dict)
    memo = {}                                 ## start index -> list of finished tails
    def dfs(start):
        if start == len(s):
            return [""]                       ## one empty tail, not zero tails
        if start in memo:
            return memo[start]
        results = []
        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if piece not in words:
                continue
            for tail in dfs(end):
                results.append(piece if tail == "" else piece + " " + tail)
        memo[start] = results
        return results
    return dfs(0)

## tests

assert word_break("catsanddog", ["cat","cats","and","sand","dog"]) == \
    ["cat sand dog", "cats and dog"]
assert word_break("pineapplepenapple", ["apple","pen","applepen","pine","pineapple"]) == \
    ["pine apple pen apple", "pine applepen apple", "pineapple pen apple"]
assert word_break("catsandog", ["cats","dog","sand","and","cat"]) == []
print(word_break("catsanddog", ["cat", "cats", "and", "sand", "dog"]))
```

```
['cat sand dog', 'cats and dog']
```

**Complexity.** $O(n^2)$ subproblem work plus the size of the output, which can itself be exponential.
Space is $O(n^2)$ for the memo plus the output.

### P16. Implement Trie — support `insert`, `search` and `startsWith` over a set of words

**Which template.** Template 5, written out in full.
**The trick.** The only real content of the class is the `is_word` flag, and the difference between the
two queries lives entirely in it. `search` must arrive at a node **and** find `is_word` set;
`startsWith` only has to arrive. Without the flag you cannot tell a stored word from a prefix of one,
which is the whole point of the structure.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def _walk(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.is_word   ## the flag, not just arrival

    def startsWith(self, prefix):
        return self._walk(prefix) is not None      ## arrival is enough

## tests

t = Trie()
t.insert("apple")
assert t.search("apple") is True
assert t.search("app") is False
assert t.startsWith("app") is True
t.insert("app")
assert t.search("app") is True
assert t.startsWith("z") is False
print(t.search("apple"), t.search("app"), t.startsWith("app"))
```

```
True True True
```

**Complexity.** $O(L)$ per operation for a word of length `L`, and $O(\text{total characters})$ space.

### P17. Design Add and Search Words — the same structure, but a query may contain the wildcard `.`

**Which template.** Template 5 for the storage, plus backtracking for the query.
**The trick.** Search stops being a walk and becomes a DFS the moment a wildcard is allowed. On a normal
character there is one child to follow, so the recursion is linear. On a `.` every child is a legal
next step, so you loop over `node.children.values()` and return `True` at the first success. This is
the smallest problem that shows why a trie and backtracking belong in the same chapter.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def search(self, word):
        def dfs(node, k):
            if k == len(word):
                return node.is_word
            ch = word[k]
            if ch == ".":
                for child in node.children.values():
                    if dfs(child, k + 1):     ## a dot branches over every child
                        return True
                return False
            if ch not in node.children:
                return False
            return dfs(node.children[ch], k + 1)
        return dfs(self.root, 0)

## tests

d = WordDictionary()
for w in ["bad", "dad", "mad"]:
    d.addWord(w)
assert d.search("pad") is False
assert d.search("bad") is True
assert d.search(".ad") is True
assert d.search("b..") is True
assert d.search("b....") is False
print(d.search("pad"), d.search(".ad"), d.search("b.."))
```

```
False True True
```

**Complexity.** $O(L)$ per search with no wildcards, and $O(26^L)$ in the worst case when the query is
all wildcards. Space is $O(\text{total characters})$.

### P18. Word Search II — find every word from a list that appears in the grid; the payoff problem

**Which template.** Grid DFS as in P10, but driven by a trie over the word list rather than by one
word.
**The trick.** Running P10 once per word costs `len(words)` full searches, and most of them fail late.
Instead insert every word into a trie and walk the grid and the trie together: one DFS carries a trie
node, and a step is only taken when the neighbouring letter is a child of that node. All words sharing
a prefix are then searched at the same time. Two details make it fast and correct. Store the whole word
at its terminal node and **pop** it when found, so each word is reported once and no result set is
needed. And when a node has no children left after the recursion, delete it from its parent, which
removes exhausted branches from the trie and stops later cells from re-exploring dead prefixes.

```python
def find_words(board, words):
    root = {}
    for w in words:                           ## build a trie of plain dicts
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["*"] = w                         ## "*" holds the whole word
    rows, cols = len(board), len(board[0])
    out = []
    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node:
            return
        nxt = node[ch]
        word = nxt.pop("*", None)             ## pop: report each word only once
        if word is not None:
            out.append(word)
        board[r][c] = "#"
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, nxt)
        board[r][c] = ch
        if not nxt:
            node.pop(ch)                      ## prune the dead branch out of the trie
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return out

## tests

b = [["o","a","a","n"], ["e","t","a","e"], ["i","h","k","r"], ["i","f","l","v"]]
assert sorted(find_words(b, ["oath","pea","eat","rain"])) == ["eat", "oath"]
assert b[0] == ["o", "a", "a", "n"]
assert find_words([["a","b"], ["c","d"]], ["abcd"]) == []
assert sorted(find_words([["a","b"], ["c","d"]], ["ab", "ac", "cd"])) == ["ab", "ac", "cd"]
print(sorted(find_words(b, ["oath", "pea", "eat", "rain"])))
```

```
['eat', 'oath']
```

**Complexity.** $O(R \cdot C \cdot 3^{L})$ time for the longest word of length `L`, plus
$O(\text{total characters})$ to build the trie. The per-word search factor is gone entirely.

### P19. Maximum Length of a Concatenated String with Unique Characters — longest concatenation of a subset of the strings with no repeated letter

**Which template.** Template 2, with the state compressed into a bitmask instead of a set.
**The trick.** Represent each word as a 26-bit mask, one bit per letter. Two words can be combined
exactly when `used & mask` is zero, and combining them is `used | mask`. That turns the expensive
"do these strings share a letter" test into one machine instruction. Drop any word that repeats a
letter inside itself during the preprocessing, because such a word can never be part of any answer.

```python
def max_length(arr):
    masks = []
    for word in arr:
        mask = 0
        for ch in word:
            bit = 1 << (ord(ch) - 97)
            if mask & bit:
                mask = 0                      ## the word repeats a letter: unusable
                break
            mask |= bit
        if mask:
            masks.append(mask)
    best = 0
    def dfs(i, used):
        nonlocal best
        best = max(best, bin(used).count("1"))
        for j in range(i, len(masks)):
            if used & masks[j]:
                continue                      ## an overlap, so this branch is dead
            dfs(j + 1, used | masks[j])
    dfs(0, 0)
    return best

## tests

assert max_length(["un", "iq", "ue"]) == 4
assert max_length(["cha","r","act","ers"]) == 6
assert max_length(["abcdefghijklmnopqrstuvwxyz"]) == 26
assert max_length(["aa", "bb"]) == 0
print(max_length(["un","iq","ue"]), max_length(["cha","r","act","ers"]))
```

```
4 6
```

**Complexity.** $O(2^n)$ time for `n` usable words, $O(n)$ recursion depth. The mask makes each node
$O(1)$ rather than $O(26)$.

## Tricks and tips

**Say the three questions out loud before you type.** What is the state, what are the choices at this
state, and when do I stop and record. For Combination Sum the state is a start index and a remaining
target, the choices are the candidates from `start` onwards, and you record when the remaining target
reaches zero. Three sentences, and the function is already written. Interviewers ask backtracking
questions partly to hear this decomposition, so do it aloud rather than silently.

**Copy the path when you record it.** `out.append(path)` stores a reference to the one list the
recursion mutates, so every entry in `out` points at the same object and the final answer is a list of
identical empty lists. Write `path[:]` or `list(path)`. This is the most common backtracking bug and it
is invisible until you print the result.

**`i` versus `i + 1` is the reuse switch.** Passing `i` to the recursive call keeps the current
candidate available, which is what "numbers may be reused" means. Passing `i + 1` consumes it. Nothing
else in the function changes between Combination Sum and Combination Sum II except that character and
the duplicate skip, so learn the pair as one fact.

**Sorting buys you three things at once.** It makes equal values adjacent, which is what the duplicate
skip needs. It makes the candidates ascending, which turns a `continue` into a `break`. And it makes
the output order deterministic, which makes your tests stable. If a problem mentions duplicates or a
target sum, sort first and say why.

**Prefer an explicit `used` array over `x in path`.** Membership in a list is $O(n)$ and it also
breaks when the input contains repeated values, because you cannot tell one copy from another. An array
indexed by position is $O(1)$ and distinguishes the copies, which is exactly what the Permutations II
skip condition relies on.

**Undo exactly what you did, immediately after the call.** Write the choice and its undo as a pair
before you write anything between them: `append` and `pop`, `add` and `remove`, set the cell and
restore the cell. The one exception is a search that stops at the first solution, such as Sudoku, where
you return `True` up the stack without undoing, because the answer must survive in the state.

**Marking a grid in place is legitimate and expected.** Overwriting a cell with a sentinel is the
standard visited set for grid DFS, and it costs no extra space. Restore the original value on the way
out. If the interviewer objects to mutating the input, offer a separate `visited` set of coordinates as
the alternative and note the space cost.

**Reach for a trie the moment the problem holds a set of words.** The signal is any question about
prefixes, or a search that would otherwise be repeated once per word. A trie turns "search the grid for
each of the words" into "search the grid once, guided by all the words", which is the entire idea
behind Word Search II.

**Memoise when subproblems repeat and only the answer count matters.** Backtracking that returns
answers for a suffix, as in Word Break II, can cache by start index. Backtracking that builds a path
with side conditions on the whole prefix usually cannot, because the state is the path itself.

## The bugs that cost the round

**Appending the mutable path instead of a copy.** Covered above, and worth repeating because it is the
single most frequent error. Every recorded answer must be `path[:]`.

**Choosing the wrong choice shape.** A subset problem written with a `used` array returns every
permutation of every subset, so `[1,2]` and `[2,1]` both appear. A permutation problem written with a
start index returns only the subsets, so most orderings are missing. Both look like duplicate-handling
bugs and neither is. Decide the shape from the diagnostic question before you write the loop.

**`i > 0` instead of `i > start` in the duplicate skip.** The wrong guard silently deletes correct
answers rather than duplicates: with `[2, 2]` you never build `[2, 2]` at all. Remember that the skip
must fire sideways and never downwards.

**Forgetting half of the undo in permutations.** `path.pop()` without `used[i] = False` leaves every
element marked as consumed, so the recursion produces exactly one permutation and returns. The two
lines are one action.

**Leaving the grid dirty in Word Search.** If the sentinel is not restored, the outer loop's later
starting cells search a corrupted board. The first test case usually still passes, which is what makes
this expensive.

**Missing the second half of a compound base case.** Restore IP Addresses needs four parts **and** the
whole string consumed. Checking only the part count produces addresses that ignore the tail of the
input.

**Testing feasibility at the base case instead of before the call.** It is not wrong, only slow, and on
a large input slow is wrong. Move every constraint test above the recursive call.

**Recursion depth.** Python's default limit is 1000. A backtracking solution whose depth is the length
of the input is fine for interview sizes, but say that you know the limit exists if the input can be
large.

## Done when

- Given a new problem statement, you can name the choice shape — which-comes-next or in-or-out — and
  say whether you need a `used` array or a start index, before writing any code.
- You can write the subset, permutation and combination-sum templates from a blank file, and explain
  why `dfs(i)` and `dfs(i + 1)` differ by exactly the reuse rule.
- You can explain why the duplicate skip uses `i > start` and not `i > 0`, using `[2, 2]` as the
  example, and give the matching `not used[i - 1]` form for permutations.
- You can state two prunings for any problem you are given, and quantify one of them the way the
  Combination Sum example does, in recursive calls rather than in adjectives.
