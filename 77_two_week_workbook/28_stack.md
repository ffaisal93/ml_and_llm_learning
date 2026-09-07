# Stack and monotonic stack: every variation

A stack answers one question: what is the most recent unresolved thing. That single question covers
matching brackets, undo histories, and expression evaluation, because in each of them the item you
must deal with next is the one you met most recently and have not yet finished with. Push when
something opens, pop when it closes.

The monotonic stack is the variant people do not recognise, and it answers a different question
entirely: for each element, what is the next, or the previous, element greater or smaller than it. It
turns an $O(n^2)$ "look forward for each element" scan into $O(n)$, because each element is pushed
once and popped once, so the inner `while` loop is amortised constant. The stack holds the elements
that are still waiting for their answer, and it holds them in sorted order, which is why one
comparison is enough to decide who is finished.

The hard part is not the code. It is that "next greater" arrives in disguise: as temperatures, as
stock spans, as the width of a rectangle, as the digits you delete to make a number small.
Recognising the phrase under the costume is the whole skill.

## Recognising it from the phrasing

| The interviewer says | They mean | What the stack holds |
|---|---|---|
| "valid / balanced / matching" | plain stack | unclosed openers |
| "evaluate an expression, postfix" | stack of operands | operands not yet consumed |
| "next greater / warmer / higher" | monotonic **decreasing** stack | indices still waiting |
| "previous smaller" | monotonic **increasing** stack | candidate predecessors |
| "largest rectangle / maximal area" | monotonic stack with **index** and a sentinel | bar indices, heights increasing |
| "remove k digits to make the smallest number" | monotonic stack, greedy | the digits kept so far |
| "design a structure with min or max in O(1)" | two stacks, or pairs | value with its running minimum |
| "collision / fleet / asteroid" | stack of survivors | items not yet destroyed or absorbed |
| "nested / decode / calculator with parentheses" | stack of saved outer state | the enclosing context |

Before writing anything, ask one question: **does the answer for an element depend on a LATER element
that you have not yet seen, and once you find that later element is the earlier one finished
forever?** If both halves are yes, a monotonic stack is correct, because the stack then holds exactly
the elements still waiting for their answer, and each of them leaves the stack exactly once — at the
moment its answer arrives. If the answer for an element can change again after you have found it, the
stack is the wrong tool and you need a different structure. If the dependency runs backwards instead —
each element needs an *earlier* element — the same stack works, but you record the answer when you
push rather than when you pop, which is the only difference between templates 2 and 3 below.

## The templates

**Template 1 — plain stack for matching.** Use when the input has openers and closers, and every
closer must be paired with the most recent unmatched opener.

```python
def is_balanced(s):
    partner = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)                          ## 1. an opener is unresolved
        elif ch in partner:
            if not stack or stack[-1] != partner[ch]: ## 2. the top must be its match
                return False
            stack.pop()                               ## 3. resolved: remove it
    return not stack                                  ## 4. nothing may stay unresolved

## tests

assert is_balanced("([{}])") is True
assert is_balanced("([)]") is False
assert is_balanced("(") is False
assert is_balanced("") is True
print(is_balanced("([{}])"), is_balanced("([)]"))
```

```
True False
```

**Template 2 — monotonic decreasing stack, for NEXT GREATER.** Use when each element waits for a
later, bigger element. The answer is recorded on the **pop**, for the element being popped.

```python
def next_greater(nums):
    answer = [-1] * len(nums)
    stack = []                                        ## INDICES, their values decreasing
    for right in range(len(nums)):
        while stack and nums[stack[-1]] < nums[right]:
            waiting = stack.pop()                     ## this index finally has its answer
            answer[waiting] = nums[right]
        stack.append(right)                           ## right now waits for its own answer
    return answer

## tests

assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
assert next_greater([5, 4, 3]) == [-1, -1, -1]
assert next_greater([1, 2, 3]) == [2, 3, -1]
assert next_greater([]) == []
print(next_greater([2, 1, 2, 4, 3]))
```

```
[4, 2, 4, -1, -1]
```

**Template 3 — monotonic increasing stack, for PREVIOUS SMALLER.** Use when each element needs an
earlier, smaller element. The answer is recorded on the **push**, for the element being pushed, by
reading whatever survives on top.

```python
def previous_smaller(nums):
    answer = [-1] * len(nums)
    stack = []                                        ## INDICES, their values increasing
    for right in range(len(nums)):
        while stack and nums[stack[-1]] >= nums[right]:
            stack.pop()                               ## too big to ever be a previous smaller
        answer[right] = nums[stack[-1]] if stack else -1
        stack.append(right)
    return answer

## tests

assert previous_smaller([2, 1, 2, 4, 3]) == [-1, -1, 1, 2, 2]
assert previous_smaller([1, 2, 3]) == [-1, 1, 2]
assert previous_smaller([3, 2, 1]) == [-1, -1, -1]
print(previous_smaller([2, 1, 2, 4, 3]))
```

```
[-1, -1, 1, 2, 2]
```

Templates 2 and 3 differ in one character. Template 2 pops while `nums[stack[-1]] < nums[right]` and
so keeps values decreasing; template 3 pops while `nums[stack[-1]] >= nums[right]` and so keeps values
increasing. Flip the comparison and you flip greater to smaller. Move the record line from the pop to
the push and you flip next to previous. Those two independent switches give you all four of next
greater, next smaller, previous greater and previous smaller, and you should be able to say which
switch you are flipping out loud before you type.

**Template 4 — stack of pairs, for an $O(1)$ minimum.** Use when a data structure must report its
minimum or maximum as cheaply as it reports its top.

```python
class MinStack:
    def __init__(self):
        self.stack = []                               ## each entry is (value, min so far)

    def push(self, value):
        smallest = value if not self.stack else min(value, self.stack[-1][1])
        self.stack.append((value, smallest))

    def pop(self):
        return self.stack.pop()[0]

    def top(self):
        return self.stack[-1][0]

    def get_min(self):
        return self.stack[-1][1]                      ## O(1): the min travels with the value

## tests

st = MinStack()
for x in [5, 3, 7, 3]:
    st.push(x)
assert st.get_min() == 3
assert st.pop() == 3
assert st.get_min() == 3
assert st.pop() == 7
assert st.get_min() == 3
assert st.pop() == 3
assert st.get_min() == 5
print(st.top(), st.get_min())
```

```
5 5
```

## The sentinel, which is the difference between a clean solution and a buggy one

Every monotonic stack has the same loose end. When the loop over the input finishes, some elements are
still on the stack, because no later element ever beat them. Those elements still need their answer,
so the usual code adds a second drain loop after the main loop, which repeats the pop logic in a
slightly different form — no `right` value, a different width formula — and that duplicated,
subtly-different block is where the bugs live.

The sentinel removes it. Append one value to the input that is guaranteed to beat everything left on
the stack, and the main loop drains the stack itself. For a histogram use a height of `0`, because no
bar can be shorter, so every bar pops. For next-greater use `float("inf")`, because no value can be
larger. The post-loop code disappears and there is exactly one pop path to get right.

**Worked example.** Largest rectangle in histogram, `heights = [2, 1, 5, 6, 2, 3]`. Each bar is the
height of some rectangle, and that rectangle extends left until a strictly shorter bar and right until
a strictly shorter bar. So the width for a popped bar at index `i` is `right - left - 1`, where
`right` is the index that caused the pop — the next smaller — and `left` is the index now on top of
the stack — the previous smaller. Both boundaries are exclusive, which is why the formula subtracts
one.

Walk it with the appended sentinel `0`. Index 0 pushes `2`. Index 1 has height `1`, which pops `2`:
the stack is now empty so `left = -1`, and the width is `1 - (-1) - 1 = 1`, giving area `2`. Push `1`.
Indices 2 and 3 push `5` and `6`, both increasing. Index 4 has height `2`, which pops `6` with
`left = 2` and width `4 - 2 - 1 = 1`, area `6`; it then pops `5` with `left = 1` and width
`4 - 1 - 1 = 2`, area `10`. Push `2`, push `3`. Now the sentinel `0` at index 6 arrives and drains
everything: it pops `3` for area `3`, pops `2` with `left = 1` and width `6 - 1 - 1 = 4` for area `8`,
and pops `1` with `left = -1` and width `6 - (-1) - 1 = 6` for area `6`. The best is `10`, the
rectangle of height `5` spanning bars 2 and 3.

```python
def largest_rectangle(heights):
    stack = []                                        ## indices, heights increasing
    best = 0
    for right, h in enumerate(heights + [0]):         ## the 0 sentinel drains the stack
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            left = stack[-1] if stack else -1         ## previous smaller index
            width = right - left - 1                  ## strictly between the two smaller bars
            best = max(best, height * width)
        stack.append(right)
    return best

## tests

assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle([2, 4]) == 4
assert largest_rectangle([1, 1, 1, 1]) == 4
assert largest_rectangle([]) == 0
print(largest_rectangle([2, 1, 5, 6, 2, 3]))
```

```
10
```

## The problems

### P1. Valid Parentheses — decide whether a string of brackets is correctly opened and closed

**Which template.** Template 1, the plain matching stack.
**The trick.** Map each closer to its opener, not the other way round, so the test on a closer is one
dictionary lookup. Two conditions fail: a closer arriving when the stack is empty, and a leftover
opener at the end. Handle the empty stack with a sentinel character that can never match, so there is
no separate `if not stack` branch to forget.

```python
def is_valid(s):
    partner = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in partner:
            top = stack.pop() if stack else "#"       ## "#" can never match
            if top != partner[ch]:
                return False
        else:
            stack.append(ch)
    return not stack

## tests

assert is_valid("()") is True
assert is_valid("()[]{}") is True
assert is_valid("(]") is False
assert is_valid("([)]") is False
assert is_valid("{[]}") is True
assert is_valid("]") is False
print(is_valid("()[]{}"), is_valid("([)]"))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P2. Min Stack — a stack with push, pop, top and a minimum, all in $O(1)$

**Which template.** Template 4, here with two parallel stacks rather than pairs.
**The trick.** The minimum cannot be recomputed on demand, so it must be stored. Store, alongside each
value, the minimum of the whole stack up to and including that value. Popping then restores the
previous minimum for free, because it was recorded when that element was pushed. This is the general
move: when a query must be $O(1)$, precompute its answer at push time.

```python
class MinStack:
    def __init__(self):
        self.values = []
        self.mins = []                                ## mins[i] is the min of values[:i+1]

    def push(self, val):
        self.values.append(val)
        self.mins.append(val if not self.mins else min(val, self.mins[-1]))

    def pop(self):
        self.mins.pop()
        return self.values.pop()

    def top(self):
        return self.values[-1]

    def getMin(self):
        return self.mins[-1]

## tests

st = MinStack()
st.push(-2); st.push(0); st.push(-3)
assert st.getMin() == -3
assert st.pop() == -3
assert st.top() == 0
assert st.getMin() == -2
st.push(-5)
assert st.getMin() == -5
print(st.top(), st.getMin())
```

```
-5 -5
```

**Complexity.** $O(1)$ per operation, $O(n)$ space.

### P3. Evaluate Reverse Polish Notation — evaluate a postfix expression given as a token list

**Which template.** A plain stack of operands.
**The trick.** Postfix needs no parentheses and no precedence rules, because the order of the tokens
already encodes the tree. Push numbers; on an operator pop two operands and push the result. The
operand order is the trap: the **second** pop is the left operand. Python's `//` floors toward
negative infinity, but the problem truncates toward zero, so use `int(left / right)`.

```python
def eval_rpn(tokens):
    stack = []
    for token in tokens:
        if token in ("+", "-", "*", "/"):
            right = stack.pop()                       ## the SECOND operand pops first
            left = stack.pop()
            if token == "+":   stack.append(left + right)
            elif token == "-": stack.append(left - right)
            elif token == "*": stack.append(left * right)
            else:              stack.append(int(left / right))   ## truncate toward zero
        else:
            stack.append(int(token))
    return stack[-1]

## tests

assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
assert eval_rpn(["7", "-3", "/"]) == -2
print(eval_rpn(["2", "1", "+", "3", "*"]), eval_rpn(["7", "-3", "/"]))
```

```
9 -2
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P4. Generate Parentheses — list every well-formed string of `n` pairs of parentheses

**Which template.** Template 1 inverted: instead of checking a string with a stack, you build strings
that a stack would accept.
**The trick.** Two counters make every generated string valid by construction, so there is no filter
step. You may open while `opened < n`, and you may close only while `closed < opened`, which is the
rule that a matching stack enforces. The list acting as the path is itself a stack: append before the
recursive call, pop after it.

```python
def generate_parenthesis(n):
    out, stack = [], []
    def build(opened, closed):
        if len(stack) == 2 * n:
            out.append("".join(stack))
            return
        if opened < n:                                ## you may always open
            stack.append("(")
            build(opened + 1, closed)
            stack.pop()                               ## undo: the stack is the path
        if closed < opened:                           ## close only what is unresolved
            stack.append(")")
            build(opened, closed + 1)
            stack.pop()
    build(0, 0)
    return out

## tests

assert generate_parenthesis(1) == ["()"]
assert sorted(generate_parenthesis(2)) == ["(())", "()()"]
assert len(generate_parenthesis(3)) == 5
assert len(generate_parenthesis(4)) == 14
print(generate_parenthesis(3))
```

```
['((()))', '(()())', '(())()', '()(())', '()()()']
```

**Complexity.** $O(4^n / \sqrt{n})$ time, the Catalan number of results, and $O(n)$ stack depth.

### P5. Daily Temperatures — for each day, how many days until a warmer temperature

**Which template.** Template 2, the monotonic decreasing stack.
**The trick.** This is next-greater wearing a costume, and the only change is what you record. The
stack holds **indices**, not temperatures, so when the answer arrives you can subtract to get the
distance `right - waiting`. Store indices in every monotonic stack by default; you can always read the
value from the index, but you cannot recover an index from a value.

```python
def daily_temperatures(temperatures):
    answer = [0] * len(temperatures)
    stack = []                                        ## indices with decreasing temperatures
    for right in range(len(temperatures)):
        while stack and temperatures[stack[-1]] < temperatures[right]:
            waiting = stack.pop()
            answer[waiting] = right - waiting         ## record the DISTANCE, not the value
        stack.append(right)
    return answer

## tests

assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
assert daily_temperatures([30, 60, 90]) == [1, 1, 0]
assert daily_temperatures([50, 50, 50]) == [0, 0, 0]
print(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]))
```

```
[1, 1, 4, 2, 1, 1, 0, 0]
```

**Complexity.** $O(n)$ time — each index is pushed once and popped once — and $O(n)$ space.

### P6. Next Greater Element I — for each value of `nums1`, its next greater element inside `nums2`

**Which template.** Template 2, plus a hash map to redirect the answers.
**The trick.** `nums1` is a subset of `nums2` and the values are distinct, so solve the whole of
`nums2` once with a monotonic stack, storing `value -> next greater` in a map, then read the answers
off for `nums1`. Because the values are distinct here, the stack can hold values instead of indices.
Anything not in the map has no next greater, so `get(x, -1)` covers the default.

```python
def next_greater_element(nums1, nums2):
    greater = {}                                      ## value -> its next greater in nums2
    stack = []
    for x in nums2:
        while stack and stack[-1] < x:
            greater[stack.pop()] = x
        stack.append(x)
    return [greater.get(x, -1) for x in nums1]

## tests

assert next_greater_element([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
assert next_greater_element([2, 4], [1, 2, 3, 4]) == [3, -1]
assert next_greater_element([1], [1]) == [-1]
print(next_greater_element([4, 1, 2], [1, 3, 4, 2]))
```

```
[-1, 3, -1]
```

**Complexity.** $O(n + m)$ time, $O(n)$ space.

### P7. Next Greater Element II — next greater element in a circular array, wrapping past the end

**Which template.** Template 2, run over two laps.
**The trick.** Circularity means an element may find its answer by wrapping around, so scan indices
`0 .. 2n-1` and use `step % n` to read the array. The essential detail is that you **push only on the
first lap**: the second lap exists purely to answer elements still waiting, and pushing again would
leave duplicates on the stack that never resolve. Two laps are enough, because an element that is
unanswered after seeing every other element is the maximum, and its answer is `-1`.

```python
def next_greater_elements_circular(nums):
    n = len(nums)
    answer = [-1] * n
    stack = []                                        ## indices, decreasing values
    for step in range(2 * n):                         ## two laps around the array
        right = step % n
        while stack and nums[stack[-1]] < nums[right]:
            answer[stack.pop()] = nums[right]
        if step < n:                                  ## push only on the FIRST lap
            stack.append(right)
    return answer

## tests

assert next_greater_elements_circular([1, 2, 1]) == [2, -1, 2]
assert next_greater_elements_circular([1, 2, 3, 4, 3]) == [2, 3, 4, -1, 4]
assert next_greater_elements_circular([5, 4, 3, 2, 1]) == [-1, 5, 5, 5, 5]
print(next_greater_elements_circular([1, 2, 3, 4, 3]))
```

```
[2, 3, 4, -1, 4]
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P8. Online Stock Span — for each day's price, how many consecutive days back it was the highest

**Which template.** Template 2, but reporting a count rather than a value, and answering online.
**The trick.** Store `(price, span)` pairs. When a new price beats the top, that day can never be an
answer again, so absorb its span into the current one and discard it. The absorbed spans mean each
popped day carries its whole run with it, so the total work stays linear even though a single call can
pop many entries. This is the previous-greater problem stated as a length.

```python
class StockSpanner:
    def __init__(self):
        self.stack = []                               ## (price, span) with prices decreasing

    def next(self, price):
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]               ## absorb the span of every beaten day
        self.stack.append((price, span))
        return span

## tests

sp = StockSpanner()
assert [sp.next(p) for p in [100, 80, 60, 70, 60, 75, 85]] == [1, 1, 1, 2, 1, 4, 6]
sp2 = StockSpanner()
assert [sp2.next(p) for p in [1, 2, 3, 4]] == [1, 2, 3, 4]
sp3 = StockSpanner()
assert [sp3.next(p) for p in [5, 4, 3]] == [1, 1, 1]
sp4 = StockSpanner()
print([sp4.next(p) for p in [100, 80, 60, 70, 60, 75, 85]])
```

```
[1, 1, 1, 2, 1, 4, 6]
```

**Complexity.** $O(1)$ amortised per call, $O(n)$ space.

### P9. Car Fleet — cars driving to the same target; count how many fleets arrive

**Which template.** A stack of survivors, after sorting by position.
**The trick.** Sort the cars from nearest the target to furthest, then process them in that order. A
car joins the fleet in front of it exactly when its arrival time is less than or equal to that fleet's
arrival time, because it would catch up before the target and then be stuck at the slower speed. So
push a new arrival time only when it is strictly greater than the top; the answer is the stack size.
Comparing times, not speeds or distances, is what makes this simple.

```python
def car_fleet(target, position, speed):
    cars = sorted(zip(position, speed), reverse=True) ## nearest the target first
    stack = []                                        ## arrival times of the fleet leaders
    for pos, spd in cars:
        time = (target - pos) / spd
        if not stack or time > stack[-1]:             ## strictly slower: a NEW fleet
            stack.append(time)
        ## otherwise this car catches the fleet ahead and is absorbed
    return len(stack)

## tests

assert car_fleet(12, [10, 8, 0, 5, 3], [2, 4, 1, 1, 3]) == 3
assert car_fleet(10, [3], [3]) == 1
assert car_fleet(100, [0, 2, 4], [4, 2, 1]) == 1
assert car_fleet(10, [0, 4, 2], [2, 1, 3]) == 1
print(car_fleet(12, [10, 8, 0, 5, 3], [2, 4, 1, 1, 3]))
```

```
3
```

**Complexity.** $O(n \log n)$ time for the sort, $O(n)$ space.

### P10. Largest Rectangle in Histogram — the largest axis-aligned rectangle fitting under the bars

**Which template.** Monotonic increasing stack of indices, with the `0` sentinel.
**The trick.** Fix the rectangle by its height: every maximal rectangle has some bar as its exact
height, and it extends until a strictly shorter bar on each side. So the answer for each bar is
`height * (next_smaller - previous_smaller - 1)`, and one increasing stack supplies both bounds — the
index that pops you is your next smaller, and the index left underneath is your previous smaller. The
sentinel drains the stack so there is only one pop path.

```python
def largest_rectangle_area(heights):
    stack = []                                        ## indices, heights strictly increasing
    best = 0
    for right, h in enumerate(heights + [0]):         ## 0 sentinel drains the stack
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            left = stack[-1] if stack else -1
            best = max(best, height * (right - left - 1))
        stack.append(right)
    return best

## tests

assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle_area([2, 4]) == 4
assert largest_rectangle_area([5]) == 5
assert largest_rectangle_area([3, 3, 3]) == 9
assert largest_rectangle_area([]) == 0
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))
```

```
10
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P11. Maximal Rectangle — the largest all-ones rectangle in a binary matrix

**Which template.** P10 applied once per row.
**The trick.** Read the matrix as a stack of histograms. For each row, `heights[c]` is the number of
consecutive ones ending at that row in column `c`; a `1` extends the bar and a `0` resets it to zero.
The largest all-ones rectangle whose bottom edge is this row is exactly the largest rectangle in that
histogram, so run P10 on each row and take the best. Say "this is largest rectangle in histogram, per
row" before you write anything, because that sentence is the whole solution.

```python
def maximal_rectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    def largest(heights):
        stack, best = [], 0
        for right, h in enumerate(heights + [0]):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                left = stack[-1] if stack else -1
                best = max(best, height * (right - left - 1))
            stack.append(right)
        return best
    width = len(matrix[0])
    heights, best = [0] * width, 0
    for row in matrix:
        for c in range(width):
            heights[c] = heights[c] + 1 if row[c] == "1" else 0   ## a zero RESETS the bar
        best = max(best, largest(heights))
    return best

## tests

grid = [["1","0","1","0","0"], ["1","0","1","1","1"],
        ["1","1","1","1","1"], ["1","0","0","1","0"]]
assert maximal_rectangle(grid) == 6
assert maximal_rectangle([["0"]]) == 0
assert maximal_rectangle([["1"]]) == 1
assert maximal_rectangle([["1","1"], ["1","1"]]) == 4
print(maximal_rectangle(grid))
```

```
6
```

**Complexity.** $O(rows \times cols)$ time, $O(cols)$ space.

### P12. Trapping Rain Water — total water trapped between bars of an elevation map

**Which template.** Template 2, a decreasing stack, filling water in horizontal layers.
**The trick.** The stack version computes water layer by layer rather than column by column. When a
taller bar arrives it pops the bar below it, and that popped bar is the **floor** of a puddle whose
left wall is the new stack top and whose right wall is the incoming bar. The depth is
`min(left, right) - floor` and the width is `right - left - 1`. If the stack empties after a pop there
is no left wall, so the water escapes and you stop. The two-pointer solution to this problem is in the
two-pointers chapter; it uses $O(1)$ space and is the better answer to give, so know both and say why
you chose one.

```python
def trap(height):
    stack = []                                        ## indices, heights decreasing
    water = 0
    for right in range(len(height)):
        while stack and height[stack[-1]] < height[right]:
            floor = height[stack.pop()]               ## the bottom of a horizontal puddle
            if not stack:
                break                                 ## no left wall: the water escapes
            left = stack[-1]
            depth = min(height[left], height[right]) - floor
            water += depth * (right - left - 1)
        stack.append(right)
    return water

## tests

assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
assert trap([3, 2, 1]) == 0
assert trap([]) == 0
print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), trap([4, 2, 0, 3, 2, 5]))
```

```
6 9
```

**Complexity.** $O(n)$ time, $O(n)$ space — worse in space than the two-pointer version.

### P13. Remove K Digits — delete exactly `k` digits from a numeric string to leave the smallest number

**Which template.** Monotonic increasing stack, used greedily.
**The trick.** The leftmost digit dominates the value, so a digit is worth deleting exactly when a
smaller digit follows it. Scan left to right and, while you still have budget and the top of the stack
is bigger than the incoming digit, pop. That leaves the kept digits non-decreasing. Two loose ends
decide the round: if budget remains at the end the string is already non-decreasing, so cut from the
tail; and leading zeros must be stripped, with `"0"` returned if nothing survives.

```python
def remove_k_digits(num, k):
    stack = []                                        ## digits kept, non-decreasing
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()                               ## a bigger digit in front is waste
            k -= 1
        stack.append(digit)
    if k > 0:
        stack = stack[:-k]                            ## still budget left: cut the tail
    answer = "".join(stack).lstrip("0")
    return answer if answer else "0"

## tests

assert remove_k_digits("1432219", 3) == "1219"
assert remove_k_digits("10200", 1) == "200"
assert remove_k_digits("10", 2) == "0"
assert remove_k_digits("112", 1) == "11"
print(remove_k_digits("1432219", 3), remove_k_digits("10200", 1))
```

```
1219 200
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P14. Asteroid Collision — asteroids move left or right at equal speed; report the survivors

**Which template.** A stack of survivors.
**The trick.** A collision happens only when a left-moving asteroid meets a right-moving one already
on the stack, that is when `a < 0` and `stack[-1] > 0`. Everything else simply pushes. The inner loop
needs three outcomes, not two: the stack top explodes and the loop continues, both explode, or the
incoming one explodes. An `alive` flag is clearer under pressure than breaking out of a loop and
testing how it ended.

```python
def asteroid_collision(asteroids):
    stack = []                                        ## the survivors so far
    for a in asteroids:
        alive = True
        while alive and a < 0 and stack and stack[-1] > 0:
            if stack[-1] < -a:
                stack.pop()                           ## the right-mover explodes, keep going
            elif stack[-1] == -a:
                stack.pop()
                alive = False                         ## both explode
            else:
                alive = False                         ## the incoming one explodes
        if alive:
            stack.append(a)
    return stack

## tests

assert asteroid_collision([5, 10, -5]) == [5, 10]
assert asteroid_collision([8, -8]) == []
assert asteroid_collision([10, 2, -5]) == [10]
assert asteroid_collision([-2, -1, 1, 2]) == [-2, -1, 1, 2]
print(asteroid_collision([5, 10, -5]), asteroid_collision([10, 2, -5]))
```

```
[5, 10] [10]
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P15. Decode String — expand a string like `3[a2[c]]` into its repeated form

**Which template.** A stack of saved outer state, one entry per `[`.
**The trick.** Keep the current string and the current number in plain variables, and push the
*enclosing* state onto stacks when a bracket opens. On `]` you pop the multiplier and the outer
string, and combine as `outer + current * count`. Build the count digit by digit with
`number * 10 + int(ch)`, because counts can exceed nine. What you push is the context you are leaving,
not the context you are entering; getting that backwards is the usual failure.

```python
def decode_string(s):
    count_stack, text_stack = [], []
    current, number = "", 0
    for ch in s:
        if ch.isdigit():
            number = number * 10 + int(ch)            ## multi-digit counts exist
        elif ch == "[":
            count_stack.append(number)                ## save the OUTER state
            text_stack.append(current)
            current, number = "", 0
        elif ch == "]":
            current = text_stack.pop() + current * count_stack.pop()
        else:
            current += ch
    return current

## tests

assert decode_string("3[a]2[bc]") == "aaabcbc"
assert decode_string("3[a2[c]]") == "accaccacc"
assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"
assert decode_string("10[a]") == "a" * 10
print(decode_string("3[a2[c]]"), decode_string("2[abc]3[cd]ef"))
```

```
accaccacc abcabccdcdcdef
```

**Complexity.** $O(\text{length of the output})$ time and space.

### P16. Simplify Path — reduce a Unix absolute path to its canonical form

**Which template.** Template 1 in spirit: `..` closes the most recent directory.
**The trick.** Split on `/` and the parsing disappears. Empty pieces come from repeated slashes and a
trailing slash, and `.` means stay, so both are skipped. Only `..` pops, and it must not pop an empty
stack, because the root has no parent. Every other piece is a directory name and is pushed unchanged —
including `...`, which is a legal name and not a special token.

```python
def simplify_path(path):
    stack = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue                                  ## empty from "//", or "stay here"
        if part == "..":
            if stack:
                stack.pop()                           ## go up, but never above the root
        else:
            stack.append(part)
    return "/" + "/".join(stack)

## tests

assert simplify_path("/home/") == "/home"
assert simplify_path("/../") == "/"
assert simplify_path("/home//foo/") == "/home/foo"
assert simplify_path("/a/./b/../../c/") == "/c"
assert simplify_path("/...") == "/..."
print(simplify_path("/a/./b/../../c/"), simplify_path("/home//foo/"))
```

```
/c /home/foo
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P17. Basic Calculator — evaluate an expression with `+`, `-`, digits, spaces and parentheses

**Which template.** A stack of saved outer state, like P15, holding `(total, sign)` per `(`.
**The trick.** With no multiplication there is no precedence, so a running `total` and a running
`sign` are enough. On `(` push the pair `(total, sign)` and reset both, so the group is evaluated from
zero. On `)` settle the pending number, then fold the group back with
`outer_total + outer_sign * group`, which is how a minus in front of a bracket distributes over
everything inside it. Numbers must be settled before any `+`, `-` or `)`, and once more after the
loop, because the last number has no terminator.

```python
def calculate(s):
    stack = []                                        ## saved (total, sign) at each "("
    total, number, sign = 0, 0, 1
    for ch in s:
        if ch.isdigit():
            number = number * 10 + int(ch)
        elif ch in "+-":
            total += sign * number                    ## settle the pending number
            number, sign = 0, 1 if ch == "+" else -1
        elif ch == "(":
            stack.append((total, sign))               ## park the outer expression
            total, sign = 0, 1
        elif ch == ")":
            total += sign * number
            number = 0
            outer_total, outer_sign = stack.pop()
            total = outer_total + outer_sign * total  ## the sign distributes over the group
            sign = 1
    return total + sign * number

## tests

assert calculate("1 + 1") == 2
assert calculate(" 2-1 + 2 ") == 3
assert calculate("(1+(4+5+2)-3)+(6+8)") == 23
assert calculate("2-(5-6)") == 3
assert calculate("1-(2+3-(4+5))") == 5
print(calculate("(1+(4+5+2)-3)+(6+8)"), calculate("1-(2+3-(4+5))"))
```

```
23 5
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P18. Sum of Subarray Minimums — sum the minimum of every subarray, modulo $10^9 + 7$

**Which template.** One pass of a monotonic increasing stack, giving previous-smaller and
next-smaller together.
**The trick.** Stop thinking about subarrays and count contributions instead. Element `i` is the
minimum of exactly `(i - prev[i]) * (next[i] - i)` subarrays: any start after its previous smaller, and
any end before its next smaller. Equal values would be counted twice by both neighbours, so break the
tie by making one side strict and the other not — here previous is smaller-or-equal and next is
strictly smaller. That asymmetry is the entire difficulty of the problem.

```python
def sum_subarray_mins(arr):
    MOD = 10 ** 9 + 7
    n = len(arr)
    prev_smaller = [-1] * n                           ## smaller-or-equal on the left
    next_smaller = [n] * n                            ## strictly smaller on the right
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            next_smaller[stack.pop()] = i
        prev_smaller[i] = stack[-1] if stack else -1
        stack.append(i)
    total = 0
    for i in range(n):
        left = i - prev_smaller[i]                    ## choices of start
        right = next_smaller[i] - i                   ## choices of end
        total += arr[i] * left * right
    return total % MOD

## tests

assert sum_subarray_mins([3, 1, 2, 4]) == 17
assert sum_subarray_mins([11, 81, 94, 43, 3]) == 444
assert sum_subarray_mins([1]) == 1
assert sum_subarray_mins([2, 2]) == 6
print(sum_subarray_mins([3, 1, 2, 4]), sum_subarray_mins([11, 81, 94, 43, 3]))
```

```
17 444
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P19. Implement Queue using Stacks — build a FIFO queue from two LIFO stacks

**Which template.** Two plain stacks, one for input and one for output.
**The trick.** Pouring one stack into another reverses it, so the oldest element ends up on top of the
output stack. The amortised bound depends on one rule: pour **only when the output stack is empty**.
Then each element is moved at most twice in its lifetime, so the average cost per operation is $O(1)$
even though one `pop` can cost $O(n)$. Pouring on every call is correct but quadratic, and the
interviewer is asking about exactly this.

```python
class MyQueue:
    def __init__(self):
        self.inbox = []                               ## newest at the top
        self.outbox = []                              ## oldest at the top

    def _shift(self):
        if not self.outbox:                           ## only when outbox is EMPTY
            while self.inbox:
                self.outbox.append(self.inbox.pop())

    def push(self, x):
        self.inbox.append(x)

    def pop(self):
        self._shift()
        return self.outbox.pop()

    def peek(self):
        self._shift()
        return self.outbox[-1]

    def empty(self):
        return not self.inbox and not self.outbox

## tests

q = MyQueue()
q.push(1); q.push(2)
assert q.peek() == 1
assert q.pop() == 1
q.push(3)
assert q.pop() == 2
assert q.pop() == 3
assert q.empty() is True
print(q.empty())
```

```
True
```

**Complexity.** $O(1)$ amortised per operation, $O(n)$ space.

### P20. Longest Valid Parentheses — the length of the longest well-formed substring

**Which template.** Template 1 with indices, and a base marker at the bottom.
**The trick.** The stack holds indices, and its bottom entry is always the index just before the
current valid stretch. Seed it with `-1`. On `(` push the index. On `)` pop first: if the stack is now
empty, this `)` is unmatched and becomes the new base; otherwise the length of the valid run ending
here is `i - stack[-1]`, measured back to whatever base survives. Measuring from the surviving bottom,
rather than counting matched pairs, is what makes lengths across several groups come out right.

```python
def longest_valid_parentheses(s):
    stack = [-1]                                      ## base: index before the last break
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)                       ## unmatched ")": a new base
            else:
                best = max(best, i - stack[-1])       ## length back to the base
    return best

## tests

assert longest_valid_parentheses("(()") == 2
assert longest_valid_parentheses(")()())") == 4
assert longest_valid_parentheses("") == 0
assert longest_valid_parentheses("()(()") == 2
assert longest_valid_parentheses("()(())") == 6
print(longest_valid_parentheses(")()())"), longest_valid_parentheses("()(())"))
```

```
4 6
```

**Complexity.** $O(n)$ time, $O(n)$ space.

## Tricks and tips

**Push indices, not values.** A value can always be read back with `nums[i]`, but an index cannot be
recovered from a value. Daily Temperatures needs indices to compute a distance, Largest Rectangle
needs them to compute a width, and Next Greater Element II needs them to write into an answer array.
Only push raw values when the problem guarantees distinct values and you genuinely want a value-keyed
map, as in Next Greater Element I. Make index-pushing the default and you will never have to rewrite
the loop halfway through.

**Say which of the four next/previous questions you are answering, out loud, before you type.** There
are exactly four: next greater, next smaller, previous greater, previous smaller. Greater versus
smaller is the direction of the comparison in the `while`. Next versus previous is whether you record
the answer when an element is popped or when it is pushed. Fix those two decisions first and the code
is four lines that you already know.

**Decide what happens on ties.** `<` and `<=` in the pop condition give different stacks, and for most
problems either works. For counting problems they do not. In Sum of Subarray Minimums, a run of equal
values would have every subarray counted once per equal element unless exactly one side of the
comparison is strict. Whenever the problem counts subarrays rather than reporting a single answer,
write down which side is strict before writing the loop.

**The stack is a sorted list you never sort.** In a monotonic stack the values from bottom to top are
already ordered, which is why one comparison against the top decides everything. That also means the
element just below any entry is its previous smaller, or previous greater, for free. Largest Rectangle
uses both facts at once: the popping index is the next smaller and the surviving index below is the
previous smaller.

**For nested structures, push the context you are leaving.** Decode String and Basic Calculator look
different but are the same program: local state in plain variables, and one stack entry pushed per
opening bracket holding the enclosing state. On the closing bracket you pop and combine. If you find
yourself pushing the state you are about to enter, you have it backwards.

**A sentinel removes the post-loop drain.** Append `0` to a histogram, or `float("inf")` when you want
every element answered. One pop path instead of two is fewer lines and far fewer bugs.

**Every problem with a linear stack scan should be quoted as $O(n)$ with the reason attached**: each
element is pushed once and popped at most once, so the inner `while` is amortised constant. Say the
sentence; an interviewer who sees a nested loop will otherwise assume $O(n^2)$.

## The bugs that cost the round

**Popping an empty stack.** Every `stack.pop()` and every `stack[-1]` needs the stack to be non-empty.
In Valid Parentheses an early closer empties it, in Largest Rectangle the last pop often empties it,
and in Trapping Rain Water an empty stack means the water escapes rather than that the input ended.
Each of these needs a different response, so write the guard at the moment you write the pop.

**Confusing the two record points.** Next-greater records the answer for the element being **popped**;
previous-smaller records the answer for the element being **pushed**, by reading the survivor beneath.
Writing the answer for the wrong element gives a result that looks plausible and is wrong on the
second test.

**Off-by-one in the width.** It is `right - left - 1`, because both boundaries are exclusive: `left` is
the previous smaller and `right` is the next smaller, and the rectangle occupies only the bars strictly
between them. When the stack is empty after a pop, `left` is `-1`, not `0`; using `0` silently loses
one column on exactly the widest rectangles.

**Forgetting the leftovers.** Elements still on the stack when the loop ends have no answer yet. Either
add the sentinel or write the drain loop, but do not assume the loop finished the job.

**Pushing twice on the circular pass.** In Next Greater Element II, pushing during the second lap
leaves entries that can never resolve, and the answer array gets overwritten with wrong values. Guard
the push with `if step < n`.

**Operand order in postfix.** The first pop is the right operand. `left - right` and `right - left`
both run, and only one is correct.

**Integer division that floors.** `-7 // 2` is `-4` in Python but Reverse Polish Notation wants `-3`.
Use `int(left / right)`.

## Done when

- Given a problem statement you have not seen, you can say within 30 seconds whether it is a plain
  stack, a monotonic stack, or neither, by asking whether an element's answer depends on a later
  element it has not yet seen.
- You can write next-greater and previous-smaller from a blank file and state the two switches — the
  comparison direction, and pop-time versus push-time recording — that turn one into the other.
- You can solve Largest Rectangle in Histogram with the sentinel, explain why the width is
  `right - left - 1`, and then extend it to Maximal Rectangle in one further sentence.
- You can explain why a loop containing a `while` pop is still $O(n)$, using the push-once pop-once
  argument, without being prompted.
