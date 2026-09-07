# Matrices, maths and bit tricks

This chapter collects the three families that do not belong to a bigger pattern but appear constantly:
in-place matrix manipulation, integer maths, and bit manipulation. What unites them is not a shared
technique. It is that each problem is solved by one specific trick that you either know or do not
know. There is very little to derive here and a great deal to memorise, which is the opposite of the
rest of the book, so the honest advice is to practise each one until the trick is automatic rather
than to reason it out in the room. You will not derive "transpose then reverse each row" under
pressure, and you do not need to: you need to have written it five times.

That makes the chapter cheap to study and high in value per hour. The problems are short, most fit in
fifteen lines, and the tricks recur. Three facts carry most of the weight. First, `n & (n - 1)` clears
the lowest set bit of `n`. Second, XOR cancels equal values, so `x ^ x = 0`. Third, when an
interviewer asks for $O(1)$ extra space on a matrix, they are asking you to store your working state
inside the matrix itself. Learn those three and roughly half the problems below become short.

## Recognising it from the phrasing

| The interviewer says | They mean | The trick |
|---|---|---|
| "rotate the matrix in place" | rotate by 90 degrees | transpose, then reverse each row |
| "print / fill the matrix in spiral order" | spiral walk | four boundaries shrinking inward |
| "set the whole row and column to zero, in place" | Set Matrix Zeroes | use row 0 and column 0 as the flags |
| "count the set bits" | population count | `n &= n - 1` in a loop |
| "every number appears twice except one" | Single Number | XOR the whole array |
| "check whether it is a power of two" | one set bit only | `n > 0 and n & (n - 1) == 0` |
| "add / multiply without using the operator" | bit arithmetic | XOR is the sum, AND shifted is the carry |
| "work with the digits of the number" | digit extraction | repeated `% 10` and `// 10` |

For every matrix problem, ask one question first: does the interviewer want $O(1)$ extra space? That
constraint is the entire problem. Without it you would allocate a second matrix, copy into it, and be
done in four obvious lines, and nobody would ask the question. With it you must find somewhere inside
the existing matrix to keep your working state, which is what forces the transpose-then-reverse for
rotation and the first-row-and-column marker for zeroing. For every bit problem, ask yourself what
`n & (n - 1)` does and say the answer out loud: it clears the lowest set bit. Subtracting one flips
that lowest set bit to zero and turns every zero below it into a one; the AND therefore keeps every
higher bit, clears the lowest set bit, and wipes the ones below. That single fact solves counting set
bits, checking powers of two, counting bits for a whole range, and several others.

## The templates

**Template 1 — rotate a square matrix by 90 degrees clockwise, in place.** Transpose across the main
diagonal, then reverse each row. The answer is the mutated matrix.

```python
def rotate_90_clockwise(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):                       ## upper triangle only, or you undo the swap
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()                                   ## mirror each row left to right
    return matrix

## tests

assert rotate_90_clockwise([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
assert rotate_90_clockwise([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
assert rotate_90_clockwise([[5]]) == [[5]]
print(rotate_90_clockwise([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
```

```
[[7, 4, 1], [8, 5, 2], [9, 6, 3]]
```

**Template 2 — the four-boundary spiral walk.** Use for any problem that reads or writes a matrix in
spiral order. Keep `top`, `bottom`, `left` and `right`, do four passes, and move the boundary inward
after each pass. The answer is the list you build, or the matrix you fill.

```python
def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    out = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            out.append(matrix[top][j])                  ## left to right along the top
        top += 1
        for i in range(top, bottom + 1):
            out.append(matrix[i][right])                ## top to bottom along the right
        right -= 1
        if top <= bottom:                               ## guard: the band may be one row tall
            for j in range(right, left - 1, -1):
                out.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:                               ## guard: the band may be one column wide
            for i in range(bottom, top - 1, -1):
                out.append(matrix[i][left])
            left += 1
    return out

## tests

assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral_order([[1, 2, 3, 4]]) == [1, 2, 3, 4]
assert spiral_order([[1], [2], [3]]) == [1, 2, 3]
assert spiral_order([]) == []
print(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
```

```
[1, 2, 3, 6, 9, 8, 7, 4, 5]
```

**Template 3 — clear the lowest set bit with `n & (n - 1)`.** Use whenever the number of set bits
matters. The answer is the number of loop iterations.

```python
def count_set_bits(n):
    count = 0
    while n:
        n &= n - 1                                      ## clears exactly the LOWEST set bit
        count += 1                                      ## so the loop runs once per set bit
    return count

def clearing_trace(n):
    steps = [format(n, "08b")]
    while n:
        n &= n - 1
        steps.append(format(n, "08b"))
    return steps

## tests

assert count_set_bits(0) == 0
assert count_set_bits(1) == 1
assert count_set_bits(11) == 3
assert count_set_bits(255) == 8
assert clearing_trace(12) == ["00001100", "00001000", "00000000"]
print(count_set_bits(11), clearing_trace(12))
```

```
3 ['00001100', '00001000', '00000000']
```

**Template 4 — XOR to cancel pairs.** XOR has three properties that together do all the work:
`x ^ x = 0`, `x ^ 0 = x`, and it is commutative and associative, so you may reorder the array freely.
Therefore XORing everything cancels every pair no matter where the pairs sit, and only the lonely
value survives. The answer is the accumulator.

```python
def single_number(nums):
    result = 0
    for x in nums:
        result ^= x                                     ## pairs cancel, order does not matter
    return result

## tests

assert 5 ^ 5 == 0                                       ## x ^ x = 0
assert 5 ^ 0 == 5                                       ## x ^ 0 = x
assert (3 ^ 7) ^ 2 == 3 ^ (7 ^ 2)                       ## associative
assert 3 ^ 7 == 7 ^ 3                                   ## commutative
assert single_number([2, 2, 1]) == 1
assert single_number([4, 1, 2, 1, 2]) == 4
assert single_number([7]) == 7
print(single_number([4, 1, 2, 1, 2]), 5 ^ 5, 5 ^ 0)
```

```
4 0 5
```

Templates 1 and 2 share the same idea, which is that a matrix operation becomes easy once you find the
right decomposition: rotation is two simple reflections, and a spiral is four straight walks. Templates
3 and 4 share the idea that a bitwise operator can carry an accumulator that no ordinary counter
could.

## Using the matrix itself as storage

This is the highest-value single technique in the chapter, because it generalises: whenever the space
constraint blocks an auxiliary array, look for space inside the input that you no longer need. Set
Matrix Zeroes is the clean example. You are given a matrix, and every cell that holds a zero must
cause its whole row and its whole column to become zero. The obvious solution keeps a set of rows to
clear and a set of columns to clear, which is $O(m + n)$ extra space. The interviewer will then ask
for $O(1)$, and the answer is to keep those two sets inside the first row and the first column of the
matrix itself.

The mechanism has two phases. In the marking phase you scan every cell from column 1 onwards, and when
you find a zero at `(i, j)` you write a zero into `matrix[i][0]` and into `matrix[0][j]`. Those two
cells are now flags meaning "clear this row" and "clear this column". In the writing phase you scan
again and set a cell to zero if either of its two flags is zero. The complication is that the first
row and the first column are doing two jobs at once: they are flags, and they are also real data that
may need clearing. The first row is safe, because `matrix[0][0]` is a legitimate flag for it. The
first column is not, because `matrix[0][0]` cannot mean two different things, so you need exactly one
extra scalar, `first_col_has_zero`, computed before any marking happens. That one boolean is the whole
subtlety, and it is what the interviewer is checking.

The second subtlety is order. You must write from the bottom-right corner backwards, and never touch
column 0 inside the inner loop, because the flags in row 0 and column 0 must survive until every cell
that depends on them has been written.

**Worked example.** Take

```
1 1 1
1 0 1
1 1 1
```

First, `first_col_has_zero` is false, because column 0 is `1, 1, 1`. Marking: the only zero is at
`(1, 1)`, so write a zero into `matrix[1][0]` and into `matrix[0][1]`. The matrix now reads

```
1 0 1
0 0 1
1 1 1
```

Writing, over columns 1 and 2 only: cell `(1, 1)` has row flag `matrix[1][0] = 0`, so it becomes 0.
Cell `(1, 2)` has the same row flag, so it becomes 0. Cell `(2, 1)` has column flag
`matrix[0][1] = 0`, so it becomes 0. Cell `(2, 2)` has neither flag set, so it stays 1. Cell `(0, 1)`
is already 0 and must stay 0, because column 1 genuinely contains a zero. Finally,
`first_col_has_zero` is false, so column 0 is left alone below row 1, and `matrix[1][0]` stays 0
because row 1 genuinely needed clearing. The result is

```
1 0 1
0 0 0
1 0 1
```

```python
def set_zeroes(matrix):
    if not matrix or not matrix[0]:
        return matrix
    rows, cols = len(matrix), len(matrix[0])
    first_col_has_zero = any(matrix[i][0] == 0 for i in range(rows))   ## the one extra scalar
    for i in range(rows):                               ## PHASE 1: mark
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0                        ## this row must go
                matrix[0][j] = 0                        ## this column must go
    for i in range(rows - 1, -1, -1):                   ## PHASE 2: write, bottom-up
        for j in range(cols - 1, 0, -1):                ## right to left, never touching column 0
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
        if first_col_has_zero:
            matrix[i][0] = 0                            ## column 0 last, using the saved flag
    return matrix

## tests

assert set_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
assert set_zeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]) == \
       [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
assert set_zeroes([[1, 0], [1, 1]]) == [[0, 0], [1, 0]]
assert set_zeroes([[0]]) == [[0]]
print(set_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
```

```
[[1, 0, 1], [0, 0, 0], [1, 0, 1]]
```

## The problems

### P1. Rotate Image — rotate an `n` by `n` matrix by 90 degrees clockwise, in place

**Which template.** Template 1: transpose, then reverse each row.
**The trick.** A rotation is two reflections. Transposing reflects across the main diagonal, and
reversing each row reflects across the vertical centre line; doing both in that order is exactly a
90-degree clockwise turn. The one detail that matters is the inner loop bound `range(i + 1, n)`: if it
starts at 0 you swap every pair twice and the matrix comes back unchanged. For anticlockwise, transpose
and then reverse the columns instead, which in code means reversing the list of rows.

```python
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):                       ## strictly above the diagonal
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix

## tests

assert rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
assert rotate([[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]) == \
       [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]
assert rotate([[1]]) == [[1]]
print(rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
```

```
[[7, 4, 1], [8, 5, 2], [9, 6, 3]]
```

**Complexity.** $O(n^2)$ time, $O(1)$ extra space.

### P2. Spiral Matrix — return every element of an `m` by `n` matrix in spiral order

**Which template.** Template 2, the four boundaries.
**The trick.** Do not think in directions and turns; think in four boundaries that shrink. Each pass
consumes one full edge and then moves its boundary inward. The two `if` guards before the bottom and
left passes are the entire difficulty: when the remaining band is a single row, the top pass has
already consumed it, and without the guard the bottom pass reads it again backwards. Test on a
one-row and a one-column input, because those are the cases the guards exist for.

```python
def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    out = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            out.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            out.append(matrix[i][right])
        right -= 1
        if top <= bottom:                               ## the guard that saves single-row inputs
            for j in range(right, left - 1, -1):
                out.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:                               ## the guard that saves single-column inputs
            for i in range(bottom, top - 1, -1):
                out.append(matrix[i][left])
            left += 1
    return out

## tests

assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == \
       [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
assert spiral_order([[7], [9], [6]]) == [7, 9, 6]
assert spiral_order([[]]) == []
print(spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
```

```
[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

**Complexity.** $O(mn)$ time, $O(1)$ extra space beyond the output.

### P3. Spiral Matrix II — fill an `n` by `n` matrix with 1 to `n` squared in spiral order

**Which template.** Template 2 again, writing instead of reading.
**The trick.** It is P2 with the assignment reversed. Say that out loud and then write the same
skeleton with a running `value` counter. Interviewers pair these two deliberately to see whether you
recognise a walk you already know when the direction of data flow changes.

```python
def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    value = 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = value
            value += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = value
            value += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = value
                value += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = value
                value += 1
            left += 1
    return matrix

## tests

assert generate_matrix(3) == [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
assert generate_matrix(1) == [[1]]
assert generate_matrix(2) == [[1, 2], [4, 3]]
print(generate_matrix(3))
```

```
[[1, 2, 3], [8, 9, 4], [7, 6, 5]]
```

**Complexity.** $O(n^2)$ time, $O(1)$ extra space beyond the output.

### P4. Set Matrix Zeroes — if a cell is 0, set its entire row and column to 0, in place

**Which template.** The matrix-as-storage technique above.
**The trick.** Store the row flags in column 0 and the column flags in row 0, and keep one extra
boolean for column 0 because `matrix[0][0]` can only serve one of the two. Then write bottom-up and
right-to-left so the flags survive until they have all been used. Compute the boolean before marking
begins, not after.

```python
def set_zeroes(matrix):
    rows, cols = len(matrix), len(matrix[0])
    first_col_has_zero = any(matrix[i][0] == 0 for i in range(rows))
    for i in range(rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, 0, -1):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
        if first_col_has_zero:
            matrix[i][0] = 0
    return matrix

## tests

assert set_zeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
assert set_zeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]) == \
       [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
assert set_zeroes([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
print(set_zeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]))
```

```
[[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]
```

**Complexity.** $O(mn)$ time, $O(1)$ extra space.

### P5. Valid Sudoku — is a partly filled 9 by 9 board valid

**Which template.** None. One pass with three arrays of sets.
**The trick.** All three constraints can be checked in the same pass, because each filled cell belongs
to exactly one row, one column and one box. The only line worth memorising is the box index,
`(i // 3) * 3 + j // 3`, which maps a cell to one of nine boxes. You do not have to solve the puzzle,
so do not start backtracking; the question is only about duplicates.

```python
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            v = board[i][j]
            if v == ".":
                continue
            b = (i // 3) * 3 + j // 3                   ## the box index, 0..8
            if v in rows[i] or v in cols[j] or v in boxes[b]:
                return False
            rows[i].add(v)
            cols[j].add(v)
            boxes[b].add(v)
    return True

## tests

good = [["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]]
bad = [r[:] for r in good]
bad[0][0] = "8"                                          ## clashes with the 8 in the same box
assert is_valid_sudoku(good) is True
assert is_valid_sudoku(bad) is False
print(is_valid_sudoku(good), is_valid_sudoku(bad))
```

```
True False
```

**Complexity.** $O(81)$ time, which is $O(1)$, and $O(1)$ space.

### P6. Game of Life — advance the board one generation, in place

**Which template.** In-place encoding, two bits per cell.
**The trick.** Every cell must be updated from the OLD state of its neighbours, so a naive in-place
update corrupts the cells that follow. The fix is to store both states in one integer: bit 0 holds the
old value and bit 1 holds the new one. Read neighbours with `board[r][c] & 1`, which always gives the
old value, and write with `board[i][j] |= 2`. When every cell is decided, a second pass shifts right
by one and the new state becomes the only state. The same two-bit idea works for any in-place
simultaneous update.

```python
def game_of_life(board):
    rows, cols = len(board), len(board[0])
    for i in range(rows):
        for j in range(cols):
            live = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    r, c = i + di, j + dj
                    if 0 <= r < rows and 0 <= c < cols:
                        live += board[r][c] & 1          ## bit 0 is always the OLD state
            if board[i][j] & 1:
                if live == 2 or live == 3:
                    board[i][j] |= 2                     ## bit 1 records the NEW state
            elif live == 3:
                board[i][j] |= 2
    for i in range(rows):
        for j in range(cols):
            board[i][j] >>= 1                            ## drop the old state, keep the new
    return board

## tests

assert game_of_life([[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]) == \
       [[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]
assert game_of_life([[1, 1], [1, 0]]) == [[1, 1], [1, 1]]
assert game_of_life([[0]]) == [[0]]
print(game_of_life([[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]))
```

```
[[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]]
```

**Complexity.** $O(mn)$ time, $O(1)$ extra space.

### P7. Diagonal Traverse — return the elements of a matrix in a zigzag diagonal order

**Which template.** None, but there is one observation that makes it easy.
**The trick.** Every cell on the same anti-diagonal has the same value of `i + j`. So bucket the cells
by `i + j`, then output bucket `d` forwards or reversed according to whether `d` is even. This costs
$O(mn)$ extra space and is far easier to get right in an interview than simulating the direction
changes with boundary checks at four corners. Say that you are trading space for correctness, and
offer the simulation as the $O(1)$-space alternative.

```python
def find_diagonal_order(matrix):
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    buckets = [[] for _ in range(rows + cols - 1)]
    for i in range(rows):
        for j in range(cols):
            buckets[i + j].append(matrix[i][j])          ## one bucket per anti-diagonal
    out = []
    for d, bucket in enumerate(buckets):
        if d % 2 == 0:
            out.extend(reversed(bucket))                 ## even diagonals run upward
        else:
            out.extend(bucket)                           ## odd diagonals run downward
    return out

## tests

assert find_diagonal_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 4, 7, 5, 3, 6, 8, 9]
assert find_diagonal_order([[1, 2], [3, 4]]) == [1, 2, 3, 4]
assert find_diagonal_order([[1, 2, 3]]) == [1, 2, 3]
assert find_diagonal_order([]) == []
print(find_diagonal_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
```

```
[1, 2, 4, 7, 5, 3, 6, 8, 9]
```

**Complexity.** $O(mn)$ time, $O(mn)$ space for the buckets.

### P8. Search a 2D Matrix — search a matrix whose rows are sorted and whose rows are in order

**Which template.** Binary search, from that chapter, over a virtual flat array.
**The trick.** The stated property means the matrix, read row by row, is one sorted array of length
`m * n`. So run an ordinary binary search over the index range `0` to `m * n - 1` and convert an index
to a cell with `matrix[mid // cols][mid % cols]`. That single line is the only matrix-aware part of
the solution, and the rest is the binary search template unchanged.

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows * cols - 1                       ## treat it as ONE sorted array
    while low <= high:
        mid = (low + high) // 2
        value = matrix[mid // cols][mid % cols]          ## the only matrix-specific line
        if value == target:
            return True
        if value < target:
            low = mid + 1
        else:
            high = mid - 1
    return False

## tests

assert search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3) is True
assert search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13) is False
assert search_matrix([[1]], 1) is True
assert search_matrix([[]], 1) is False
print(search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3),
      search_matrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13))
```

```
True False
```

**Complexity.** $O(\log(mn))$ time, $O(1)$ space.

### P9. Happy Number — repeatedly replace `n` by the sum of the squares of its digits; does it reach 1

**Which template.** Floyd cycle detection, the same technique as in the linked-list chapter.
**The trick.** The sequence of values is a linked list in disguise: each number has exactly one
successor, so the sequence must eventually either reach 1 or enter a cycle. Therefore run a slow and
a fast pointer over the transformation and stop when they meet. This uses $O(1)$ space where a `seen`
set uses $O(\log n)$, and naming the connection to the linked-list problem is worth as much as the
code. The digit extraction is the standard repeated `% 10` and `// 10`.

```python
def square_digit_sum(n):
    total = 0
    while n:
        d = n % 10                                       ## digits: repeated mod and divide
        total += d * d
        n //= 10
    return total

def is_happy(n):
    slow, fast = n, square_digit_sum(n)                  ## Floyd, exactly as on a linked list
    while fast != 1 and slow != fast:
        slow = square_digit_sum(slow)
        fast = square_digit_sum(square_digit_sum(fast))
    return fast == 1

## tests

assert is_happy(19) is True
assert is_happy(2) is False
assert is_happy(1) is True
assert is_happy(7) is True
print(is_happy(19), is_happy(2), square_digit_sum(19))
```

```
True False 82
```

**Complexity.** $O(\log n)$ time per step and a bounded number of steps, $O(1)$ space.

### P10. Plus One — add one to a number given as an array of digits

**Which template.** None. Carry propagation from the right.
**The trick.** Walk from the last digit. Any digit below 9 absorbs the increment and you return
immediately; a 9 becomes 0 and the carry continues. If the loop finishes, every digit was a 9, so the
number was all nines and the answer is a `1` followed by the zeros the loop already wrote. That last
line is the only case people forget, and `[9, 9, 9]` is the test that catches it.

```python
def plus_one(digits):
    out = digits[:]
    for i in range(len(out) - 1, -1, -1):
        if out[i] < 9:
            out[i] += 1
            return out                                   ## no carry: done
        out[i] = 0                                       ## carry continues left
    return [1] + out                                     ## all nines: the number gained a digit

## tests

assert plus_one([1, 2, 3]) == [1, 2, 4]
assert plus_one([4, 3, 2, 1]) == [4, 3, 2, 2]
assert plus_one([9]) == [1, 0]
assert plus_one([9, 9, 9]) == [1, 0, 0, 0]
print(plus_one([1, 2, 3]), plus_one([9, 9, 9]))
```

```
[1, 2, 4] [1, 0, 0, 0]
```

**Complexity.** $O(n)$ time, $O(n)$ space for the copy.

### P11. Pow(x, n) — compute `x` to the power `n` efficiently

**Which template.** Fast exponentiation, driven by the bits of `n`.
**The trick.** Write `n` in binary. Then `x` to the power `n` is the product of `x` to the power of
each set bit's place value, and those place values are `x`, `x` squared, `x` to the fourth, and so on,
each obtained from the last by one squaring. So walk the bits of `n` with `n & 1` and `n >>= 1`,
squaring `x` each step and multiplying into the result when the bit is set. Handle a negative `n`
first by inverting `x`, or the loop never terminates.

```python
def my_pow(x, n):
    if n < 0:
        x, n = 1 / x, -n                                 ## fold the negative exponent away first
    result = 1.0
    while n:
        if n & 1:                                        ## this bit of n is set
            result *= x
        x *= x                                           ## x now holds the next power of two
        n >>= 1
    return result

## tests

assert abs(my_pow(2.0, 10) - 1024.0) < 1e-9
assert abs(my_pow(2.1, 3) - 9.261) < 1e-9
assert abs(my_pow(2.0, -2) - 0.25) < 1e-9
assert abs(my_pow(5.0, 0) - 1.0) < 1e-9
print(my_pow(2.0, 10), my_pow(2.0, -2))
```

```
1024.0 0.25
```

**Complexity.** $O(\log n)$ time, $O(1)$ space.

### P12. Sqrt(x) — the integer square root, rounded down

**Which template.** Binary search on the answer, from the binary search chapter.
**The trick.** The predicate `mid * mid <= x` is monotone: true for every value below the answer and
false above it. That is exactly the shape binary search needs, so search the range and keep the last
value that satisfied it. Handle `x < 2` separately, and note that `mid * mid` cannot overflow in
Python, which is a point worth mentioning because in C or Java you would compare `mid <= x // mid`
instead.

```python
def my_sqrt(x):
    if x < 2:
        return x
    low, high, answer = 1, x // 2, 1
    while low <= high:
        mid = (low + high) // 2
        if mid * mid <= x:
            answer = mid                                 ## a candidate: keep it and try bigger
            low = mid + 1
        else:
            high = mid - 1
    return answer

## tests

assert my_sqrt(4) == 2
assert my_sqrt(8) == 2
assert my_sqrt(0) == 0
assert my_sqrt(1) == 1
assert my_sqrt(2147395600) == 46340
print(my_sqrt(8), my_sqrt(2147395600))
```

```
2 46340
```

**Complexity.** $O(\log x)$ time, $O(1)$ space.

### P13. Multiply Strings — multiply two non-negative integers given as strings

**Which template.** Long multiplication into a digit array.
**The trick.** The one fact that makes this manageable is the index rule: the product of `num1[i]` and
`num2[j]` lands in positions `i + j` and `i + j + 1` of a result array of length `m + n`, with the
units digit in the higher index. Once you have written that down, the code is a double loop that adds
into `product[i + j + 1]` and carries into `product[i + j]`. Strip the leading zeros at the end, and
handle the all-zero input before you start.

```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    product = [0] * (m + n)                              ## the answer needs at most m + n digits
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            low = i + j + 1                              ## the units place of this partial product
            total = mul + product[low]
            product[low] = total % 10
            product[i + j] += total // 10                ## carry into the next place up
    digits = "".join(str(d) for d in product).lstrip("0")
    return digits

## tests

assert multiply("2", "3") == "6"
assert multiply("123", "456") == "56088"
assert multiply("0", "999") == "0"
assert multiply("99", "99") == "9801"
print(multiply("123", "456"), multiply("99", "99"))
```

```
56088 9801
```

**Complexity.** $O(mn)$ time, $O(m + n)$ space.

### P14. Add Binary — add two binary numbers given as strings

**Which template.** Carry propagation from the right, the same skeleton as P10.
**The trick.** One loop condition handles the two different lengths and the final carry at once:
`while i >= 0 or j >= 0 or carry`. Digits are `total % 2` and the carry is `total // 2`, which is the
same code as decimal addition with the base changed. Build a list and reverse it at the end rather
than prepending to a string, because prepending is quadratic.

```python
def add_binary(a, b):
    i, j = len(a) - 1, len(b) - 1
    carry, out = 0, []
    while i >= 0 or j >= 0 or carry:
        total = carry
        if i >= 0:
            total += int(a[i])
            i -= 1
        if j >= 0:
            total += int(b[j])
            j -= 1
        out.append(str(total % 2))                       ## the digit
        carry = total // 2                               ## the carry
    return "".join(reversed(out))

## tests

assert add_binary("11", "1") == "100"
assert add_binary("1010", "1011") == "10101"
assert add_binary("0", "0") == "0"
assert add_binary("1", "111") == "1000"
print(add_binary("11", "1"), add_binary("1010", "1011"))
```

```
100 10101
```

**Complexity.** $O(\max(m, n))$ time, $O(\max(m, n))$ space.

### P15. Reverse Integer — reverse the digits of a signed 32-bit integer, returning 0 on overflow

**Which template.** Digit extraction with an overflow guard.
**The trick.** Python integers do not overflow, so the overflow is the whole exercise and you must
simulate it. Check before the multiply, not after: the step `result * 10 + digit` overflows exactly
when `result > (INT_MAX - digit) // 10`, so test that first and return 0. Handle the sign by taking
the absolute value up front, which avoids the different truncation rules for negative division.

```python
INT_MAX, INT_MIN = 2 ** 31 - 1, -(2 ** 31)

def reverse_integer(x):
    sign = -1 if x < 0 else 1
    n = abs(x)
    result = 0
    while n:
        digit = n % 10
        n //= 10
        if result > (INT_MAX - digit) // 10:             ## check BEFORE the multiply overflows
            return 0
        result = result * 10 + digit
    result *= sign
    return result if INT_MIN <= result <= INT_MAX else 0

## tests

assert reverse_integer(123) == 321
assert reverse_integer(-123) == -321
assert reverse_integer(120) == 21
assert reverse_integer(1534236469) == 0
assert reverse_integer(0) == 0
print(reverse_integer(123), reverse_integer(-123), reverse_integer(1534236469))
```

```
321 -321 0
```

**Complexity.** $O(\log x)$ time, $O(1)$ space.

### P16. Number of 1 Bits — count the set bits of a 32-bit unsigned integer

**Which template.** Template 3, `n &= n - 1`.
**The trick.** The loop runs once per set bit rather than 32 times, which matters when the interviewer
follows up with "and if the input is mostly zeros?". Be ready to explain `n & (n - 1)` in one
sentence, because that explanation is what the question is really testing.

```python
def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1                                       ## clear the lowest set bit
        count += 1
    return count

## tests

assert hamming_weight(0b00000000000000000000000000001011) == 3
assert hamming_weight(0b10000000000000000000000000000000) == 1
assert hamming_weight(0b11111111111111111111111111111101) == 31
assert hamming_weight(0) == 0
print(hamming_weight(11), hamming_weight(128))
```

```
3 1
```

**Complexity.** $O(k)$ time with `k` the number of set bits, $O(1)$ space.

### P17. Counting Bits — the set-bit count of every number from 0 to `n`

**Which template.** Dynamic programming over the bits, one line of recurrence.
**The trick.** Do not call the counter `n + 1` times. Note that `i >> 1` is `i` with its last bit
removed, so `bits[i] = bits[i >> 1] + (i & 1)`, and every subproblem is already solved because
`i >> 1 < i`. The Kernighan variant is just as good and worth mentioning:
`bits[i] = bits[i & (i - 1)] + 1`, because `i & (i - 1)` has exactly one fewer set bit than `i`. Both
are $O(n)$ overall.

```python
def count_bits(n):
    out = [0] * (n + 1)
    for i in range(1, n + 1):
        out[i] = out[i >> 1] + (i & 1)                   ## i has i>>1 's bits, plus its own last bit
    return out

def count_bits_kernighan(n):
    out = [0] * (n + 1)
    for i in range(1, n + 1):
        out[i] = out[i & (i - 1)] + 1                    ## one more bit than i with its lowest cleared
    return out

## tests

assert count_bits(2) == [0, 1, 1]
assert count_bits(5) == [0, 1, 1, 2, 1, 2]
assert count_bits(0) == [0]
assert count_bits(20) == count_bits_kernighan(20)
print(count_bits(5), count_bits_kernighan(5))
```

```
[0, 1, 1, 2, 1, 2] [0, 1, 1, 2, 1, 2]
```

**Complexity.** $O(n)$ time, $O(n)$ space for the output.

### P18. Reverse Bits — reverse the bits of a 32-bit unsigned integer

**Which template.** A fixed 32-step shift loop.
**The trick.** The loop must run exactly 32 times, not "while `n`". Stopping early when `n` becomes
zero drops the leading zeros of the input, which are trailing zeros of the answer, and the result is
too small by a power of two. Each step shifts the accumulator left to make room and pushes in `n & 1`.
If asked to call the function many times, mention memoising the four bytes in a lookup table.

```python
def reverse_bits(n):
    result = 0
    for _ in range(32):                                  ## exactly 32 iterations, always
        result = (result << 1) | (n & 1)                 ## take n's last bit, push it into result
        n >>= 1
    return result

## tests

assert reverse_bits(0b00000010100101000001111010011100) == 964176192
assert reverse_bits(0) == 0
assert reverse_bits(1) == 2 ** 31
assert reverse_bits(2 ** 32 - 1) == 2 ** 32 - 1
print(reverse_bits(0b00000010100101000001111010011100), reverse_bits(1))
```

```
964176192 2147483648
```

**Complexity.** $O(32)$ time, which is $O(1)$, and $O(1)$ space.

### P19. Missing Number — one number from 0 to `n` is missing from the array; find it

**Which template.** Template 4 by XOR, or the Gauss sum. Give both.
**The trick.** The XOR version pairs each index with each value: XOR together all the indices, all
the values, and `n` itself, and every present number cancels with its own index, leaving the missing
one. The sum version subtracts the array total from `n * (n + 1) / 2`. Both are $O(n)$ time and
$O(1)$ space; the XOR version is the one to lead with, because it cannot overflow in a fixed-width
language while the sum version can, and saying so is exactly the comparison the interviewer wants.

```python
def missing_number_xor(nums):
    result = len(nums)                                   ## seed with n, the index that has no partner
    for i, x in enumerate(nums):
        result ^= i ^ x                                  ## every present value cancels its index
    return result

def missing_number_sum(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)                  ## the Gauss formula minus what is there

## tests

for case, want in [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8), ([0], 1)]:
    assert missing_number_xor(case) == want
    assert missing_number_sum(case) == want
print(missing_number_xor([9, 6, 4, 2, 3, 5, 7, 0, 1]),
      missing_number_sum([9, 6, 4, 2, 3, 5, 7, 0, 1]))
```

```
8 8
```

**Complexity.** $O(n)$ time, $O(1)$ space, for both versions.

### P20. Sum of Two Integers — add two integers without using `+` or `-`

**Which template.** Bit arithmetic: XOR is the sum, AND shifted is the carry.
**The trick.** Adding two bits gives a sum bit and a carry bit. The sum bit is `a ^ b` and the carry
is `(a & b) << 1`, so repeat until the carry is zero. In Python the awkward part is that integers are
unbounded, so a negative result never terminates the loop; mask everything to 32 bits with
`0xFFFFFFFF` and convert a value above `INT_MAX` back to a negative Python integer at the end. Say
that this masking is a Python artefact and that the loop is the whole algorithm in C.

```python
MASK = 0xFFFFFFFF
INT_MAX = 0x7FFFFFFF

def get_sum(a, b):
    a, b = a & MASK, b & MASK
    while b:
        carry = (a & b) << 1                             ## bits where BOTH are 1 carry left
        a = (a ^ b) & MASK                               ## XOR is addition without carry
        b = carry & MASK
    return a if a <= INT_MAX else ~(a ^ MASK)            ## re-sign a 32-bit negative for Python

## tests

assert get_sum(1, 2) == 3
assert get_sum(2, 3) == 5
assert get_sum(-1, 1) == 0
assert get_sum(-2, -3) == -5
assert get_sum(0, 0) == 0
print(get_sum(2, 3), get_sum(-1, 1), get_sum(-2, -3))
```

```
5 0 -5
```

**Complexity.** $O(32)$ time, which is $O(1)$, and $O(1)$ space.

### P21. Single Number II — every element appears three times except one; find it

**Which template.** Bit counting, because plain XOR no longer cancels.
**The trick.** XOR cancels pairs, not triples, so template 4 fails here and you must say why. Instead
count each bit position independently across the whole array. Every triple contributes either 0 or 3
to a position, so the total in that position is a multiple of three plus the lonely number's bit.
Therefore the bit is set in the answer exactly when the count is not divisible by three. Re-sign the
result at the end, because Python integers have no fixed width and a set bit 31 would otherwise come
out as a large positive number.

```python
def single_number_ii(nums):
    result = 0
    for bit in range(32):
        total = 0
        for x in nums:
            total += (x >> bit) & 1                      ## count this bit across all numbers
        if total % 3:                                    ## the triples contribute 0 or 3
            result |= 1 << bit
    if result >= 2 ** 31:
        result -= 2 ** 32                                ## re-sign, because Python ints are unbounded
    return result

## tests

assert single_number_ii([2, 2, 3, 2]) == 3
assert single_number_ii([0, 1, 0, 1, 0, 1, 99]) == 99
assert single_number_ii([-2, -2, 1, -2]) == 1
assert single_number_ii([1, 1, 1, -4]) == -4
print(single_number_ii([0, 1, 0, 1, 0, 1, 99]), single_number_ii([1, 1, 1, -4]))
```

```
99 -4
```

**Complexity.** $O(32n)$ time, which is $O(n)$, and $O(1)$ space.

### P22. Encode and Decode Strings — serialise a list of strings into one string and back

**Which template.** A length prefix, not a delimiter.
**The trick.** Any separator character can also occur inside a string, so splitting on a separator is
wrong and the interviewer will produce the counter-example. Instead write the length, then a marker,
then the raw characters: `4#lint`. To decode, read up to the first `#` to get the length, then take
exactly that many characters without examining them at all. The `#` inside the payload is harmless
because you never search past the length field. Test with a string that contains `#` and a digit.

```python
def encode(strings):
    parts = []
    for s in strings:
        parts.append(str(len(s)) + "#" + s)              ## length, delimiter, then the raw bytes
    return "".join(parts)

def decode(data):
    out, i = [], 0
    while i < len(data):
        j = i
        while data[j] != "#":                            ## the FIRST '#' ends the length field
            j += 1
        length = int(data[i:j])
        out.append(data[j + 1:j + 1 + length])           ## take exactly that many characters
        i = j + 1 + length
    return out

## tests

assert decode(encode(["lint", "code", "love", "you"])) == ["lint", "code", "love", "you"]
assert decode(encode(["", ""])) == ["", ""]
assert decode(encode(["a#b", "3#c"])) == ["a#b", "3#c"]
assert decode(encode([])) == []
print(encode(["lint", "code"]), decode(encode(["a#b", "3#c"])))
```

```
4#lint4#code ['a#b', '3#c']
```

**Complexity.** $O(N)$ time in the total length, $O(N)$ space.

## Tricks and tips

**Say `n & (n - 1)` clears the lowest set bit, then use it three ways.** Counting bits is the obvious
one. A power of two has exactly one set bit, so `n > 0 and n & (n - 1) == 0` tests it in one line.
Counting bits for a whole range uses `bits[i] = bits[i & (i - 1)] + 1`. The companion trick is
`n & -n`, which isolates the lowest set bit rather than clearing it, and is what a Fenwick tree runs
on.

**XOR is addition without carry, and that is the whole story.** It is why XOR cancels pairs, why it
finds the missing number, and why it is the sum half of Sum of Two Integers with `(a & b) << 1` as the
carry half. When you see "everything appears twice except one", write the XOR before you finish
reading the question. When you see "three times", stop, because XOR does not cancel triples and you
need the bit-counting approach instead.

**Mask to 32 bits in Python and say why.** Python integers are unbounded and have no sign bit, so any
problem that says "32-bit signed integer" needs `& 0xFFFFFFFF` during the computation and a conversion
back at the end for values above `0x7FFFFFFF`. Mentioning this before the interviewer notices it reads
as fluency rather than as a bug you got away with.

**Decompose matrix operations into reflections and straight walks.** A 90-degree rotation is a
transpose followed by a row reversal. An anticlockwise rotation is a transpose followed by reversing
the order of the rows. A 180-degree rotation is reversing the rows and reversing each row. None of
these needs index arithmetic, and index arithmetic is where the mistakes live.

**When the space constraint blocks an array, look inside the input.** Set Matrix Zeroes puts its flags
in row 0 and column 0. Game of Life puts the new state in bit 1 of each cell. A sorted array of values
in the range 1 to n can encode a seen-set by negating entries. The question to ask is "what part of
the input have I already read and will never need again?".

**For digits, the pair is `% 10` and `// 10`.** Extract the last digit with the modulo, remove it with
the floor division, and loop while the number is non-zero. Build the reversed number with
`result = result * 10 + digit`. This appears in Reverse Integer, Happy Number, Palindrome Number and
Add Digits, and it is faster and cleaner than converting to a string.

**Practise these to speed, not to understanding.** Unlike the pattern chapters, the return here comes
from repetition. Write Rotate Image, Spiral Matrix and Set Matrix Zeroes from a blank file once a week
until they take three minutes each. That is a small investment for a family of problems that appears in
phone screens constantly.

## The bugs that cost the round

**Transposing the whole matrix instead of the upper triangle.** The inner loop must be
`range(i + 1, n)`. Starting it at 0 swaps every pair twice, the matrix returns to its original state,
and the bug is invisible on a symmetric test input.

**Missing the two guards in the spiral walk.** After the top and right passes, the remaining band may
be a single row or a single column. Without `if top <= bottom` before the bottom pass, and
`if left <= right` before the left pass, a single-row matrix is read forwards and then backwards. Test
`[[1, 2, 3, 4]]` and `[[1], [2], [3]]` every time.

**Reading the new state instead of the old one in Game of Life.** Any in-place simultaneous update
must read through a mask. `board[r][c] & 1` is the old value; plain `board[r][c]` is corrupted the
moment any neighbour has been decided.

**Losing the first column in Set Matrix Zeroes.** `matrix[0][0]` cannot be the flag for both the first
row and the first column. You need one extra boolean, computed before the marking phase, and the
writing loop must never touch column 0 until the very end of each row.

**Stopping the bit-reversal loop early.** `while n` instead of `for _ in range(32)` silently drops the
input's leading zeros. Test with the input `1`, whose reversal is `2 ** 31`.

**Checking for overflow after the multiply.** In Python the multiply cannot fail, so an after-the-fact
check works, but it is not what the question is testing and it does not transfer to a fixed-width
language. Check `result > (INT_MAX - digit) // 10` before the step.

**Splitting on a delimiter in Encode and Decode Strings.** Any character you pick can appear in the
payload. The length prefix is not one option among several; it is the answer.

## Done when

- You can write Rotate Image, Spiral Matrix and Set Matrix Zeroes from a blank file in under five
  minutes each, including the two spiral guards and the first-column boolean.
- You can say in one sentence what `n & (n - 1)` does and what `n & -n` does, and name two problems
  that each one solves.
- You can state the three XOR properties, use them to explain Single Number and Missing Number, and
  say precisely why the same argument fails for Single Number II.
- You can explain what masking with `0xFFFFFFFF` is for in Python and convert a masked value back to
  a signed integer without looking it up.
