# Intervals and greedy: every variation

An interval problem almost always starts with a sort, and almost all of the difficulty is in one
decision: do you sort by START or by END. Sorting by start is for merging and for detecting overlap,
because it puts the intervals in the order they open, so a single pass can keep one block open and
extend it. Sorting by end is for scheduling the maximum number of non-overlapping items, because the
interval that finishes earliest leaves the most room for everything after it. That one sentence is the
chapter. Everything else here is bookkeeping around those two orderings.

The greedy half needs an honest statement too. A greedy algorithm is correct only when a local choice
cannot rule out the global optimum, and the interview skill is being able to say WHY the choice is
safe in one sentence, rather than asserting that it works. The argument has a standard shape, called
an exchange argument: assume an optimal solution that does not make your greedy choice, then show you
can swap your choice into it without making the result worse. For "take the earliest finisher" the
swap is direct. An optimal schedule contains some first interval; replace it with the earliest
finisher, which ends no later, so it clashes with nothing that the original allowed. The optimal
solution stays the same size. Therefore the greedy answer is at least as large. Practise saying that
out loud, because interviewers ask for it.

## Recognising it from the phrasing

| The interviewer says | They mean | Sort by | The move |
|---|---|---|---|
| "merge overlapping intervals" | merge, template 1 | **start** | extend the last block, or open a new one |
| "insert one interval into a sorted list" | insert, three phases | already sorted | before, overlapping, after |
| "maximum number of non-overlapping ..." | activity selection, template 2 | **end** | greedily take whatever fits |
| "minimum removals to make them non-overlapping" | the complement of the above | **end** | `total - maximum non-overlapping` |
| "how many rooms / cars / resources at once" | sweep line, template 3 | events | a running counter, take its maximum |
| "does any pair overlap" | conflict detection | **start** | check each neighbour pair only |
| "the smallest interval covering each query" | offline sweep with a heap | **start** | sort the queries too |
| "always take the best available right now" | greedy, template 4 | depends | state the exchange argument |

Before writing a line of code, ask exactly what it means for two intervals to overlap, and write the
condition down: `a.start < b.end and b.start < a.end`. Most interval bugs are an off-by-one in that
condition, because whether touching endpoints count as overlapping depends on the problem, and you
must ask the interviewer. Two meetings `[1, 5]` and `[5, 9]` do not conflict, because one room empties
as the next fills. Two balloons `[1, 5]` and `[5, 9]` do share the point 5, so one arrow pops both.
The difference between these two problems is a single `<` against a `<=`, so settle it before you
code and say your answer out loud, because it also fixes the tie rule in the sweep line below.

## The templates

Templates 1 and 2 have deliberately identical skeletons: sort, then one pass carrying a single piece
of state. Only the sort key and what you do with the state change. Learn the skeleton once and the two
become one decision.

**Template 1 — merge overlapping intervals.** Use when the output is intervals, not a count. Sort by
start, keep the last block open, and extend it. The answer is the list you build.

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda p: p[0])   ## sort by START
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:                      ## touches or overlaps the last kept block
            merged[-1][1] = max(merged[-1][1], end)     ## extend, never append
        else:
            merged.append([start, end])                 ## a clean gap: start a new block
    return merged

## tests

assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]
assert merge_intervals([]) == []
print(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]))
```

```
[[1, 6], [8, 10], [15, 18]]
```

**Template 2 — the maximum number of non-overlapping intervals.** Use when the output is a count.
Sort by end and take anything that starts at or after the last end. The answer is `taken`.

```python
def max_non_overlapping(intervals):
    intervals = sorted(intervals, key=lambda p: p[1])   ## sort by END
    taken, last_end = 0, float("-inf")
    for start, end in intervals:
        if start >= last_end:                           ## no clash with the last one taken
            taken += 1
            last_end = end                              ## the earliest finisher leaves the most room
    return taken

## tests

assert max_non_overlapping([[1, 2], [2, 3], [3, 4], [1, 3]]) == 3
assert max_non_overlapping([[1, 2], [1, 2], [1, 2]]) == 1
assert max_non_overlapping([[1, 100], [2, 3], [4, 5]]) == 2
assert max_non_overlapping([]) == 0
print(max_non_overlapping([[1, 2], [2, 3], [3, 4], [1, 3]]))
```

```
3
```

**Template 3 — the sweep line.** Use for any "how many are active at once" question. The answer is
the largest value the running counter reaches.

```python
def max_concurrent(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))                       ## one resource is taken
        events.append((end, -1))                        ## one resource is released
    events.sort()                                       ## -1 sorts before +1 at the same coordinate
    running, best = 0, 0
    for _, delta in events:
        running += delta
        best = max(best, running)
    return best

## tests

assert max_concurrent([[0, 30], [5, 10], [15, 20]]) == 2
assert max_concurrent([[1, 2], [2, 3], [3, 4]]) == 1
assert max_concurrent([[1, 5], [2, 6], [3, 7]]) == 3
assert max_concurrent([]) == 0
print(max_concurrent([[0, 30], [5, 10], [15, 20]]), max_concurrent([[1, 5], [2, 6], [3, 7]]))
```

```
2 3
```

**Template 4 — the greedy reach.** Use when you walk left to right and only need the best position
reachable so far. The answer is whether the walk survives, or the value of `reach` at the end.

```python
def can_reach_end(nums):
    reach = 0                                           ## furthest index reachable so far
    for i in range(len(nums)):
        if i > reach:                                   ## a gap: index i is unreachable
            return False
        reach = max(reach, i + nums[i])                 ## the greedy step: always take the best reach
    return True

## tests

assert can_reach_end([2, 3, 1, 1, 4]) is True
assert can_reach_end([3, 2, 1, 0, 4]) is False
assert can_reach_end([0]) is True
assert can_reach_end([1, 0, 1]) is False
print(can_reach_end([2, 3, 1, 1, 4]), can_reach_end([3, 2, 1, 0, 4]))
```

```
True False
```

The sort key is the whole difference between templates 1 and 2, and it is the line people get wrong
under pressure. Merging asks "which intervals touch each other", which is a question about the order
things open, so sort by start. Scheduling asks "how many can I fit", which is a question about how
soon the room is free again, so sort by end. Sorting a merge problem by end produces blocks in the
wrong order and a wrong answer that looks plausible on the sample input.

## The sweep line

The sweep line is the highest-value trick in this chapter, because it answers the whole "how many at
once" family with one skeleton and no data structure beyond a sorted list. The idea is to stop
thinking about intervals and start thinking about the two moments that matter in each one. Convert
every interval `[start, end]` into two events: `(start, +1)`, meaning one more resource is in use from
here, and `(end, -1)`, meaning one is released here. Throw the intervals away. Sort all the events by
coordinate, then walk them left to right carrying a running counter. The counter is the number of
intervals covering the current point, so the maximum value the counter ever reaches is the maximum
number of simultaneous intervals, and therefore the number of resources you must buy.

The tie rule is the only subtle part. When an end event and a start event share a coordinate, the
order you process them in decides whether touching intervals count as overlapping. Process the end
first and `[1, 5]` with `[5, 9]` peaks at 1, which is what a meeting room needs. Process the start
first and it peaks at 2. In Python you get the correct rule for free by sorting plain tuples, because
`-1 < +1`, so `(5, -1)` sorts before `(5, 1)`. Say that out loud in the interview rather than letting
it look like an accident.

**Worked example.** Take five meetings: `[0, 30]`, `[5, 10]`, `[15, 20]`, `[10, 25]`, `[20, 35]`. The
ten events, sorted, are `(0, +1)`, `(5, +1)`, `(10, -1)`, `(10, +1)`, `(15, +1)`, `(20, -1)`,
`(20, +1)`, `(25, -1)`, `(30, -1)`, `(35, -1)`. Walking them, the counter goes 1, 2, 1, 2, 3, 2, 3, 2,
1, 0. The maximum is 3, so three rooms are needed. Note the two places where the counter dips and
climbs again at the same coordinate, at 10 and at 20. That dip is the tie rule doing its work: without
it the counter would read 3 at coordinate 10 and the answer would be 4.

```python
def min_meeting_rooms_sweep(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))                       ## a meeting begins
        events.append((end, -1))                        ## a meeting ends
    events.sort()                                       ## (10, -1) sorts before (10, 1)
    running, best = 0, 0
    for _, delta in events:
        running += delta                                ## the counter is the rooms in use now
        best = max(best, running)
    return best

## tests

assert min_meeting_rooms_sweep([[0, 30], [5, 10], [15, 20], [10, 25], [20, 35]]) == 3
assert min_meeting_rooms_sweep([[7, 10], [2, 4]]) == 1
assert min_meeting_rooms_sweep([[1, 5], [5, 9], [9, 12]]) == 1
assert min_meeting_rooms_sweep([]) == 0
print(min_meeting_rooms_sweep([[0, 30], [5, 10], [15, 20], [10, 25], [20, 35]]))
```

```
3
```

## The problems

### P1. Merge Intervals — collapse a list of intervals so that no two of the results overlap

**Which template.** Template 1: sort by start, extend the last block.
**The trick.** After sorting by start, an interval can only overlap the block currently open, never an
earlier one, because every earlier block ended before this block opened. So one pass suffices and you
never look back further than `merged[-1]`. Use `max` when extending, because a long block can fully
contain the interval you are absorbing.

```python
def merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda p: p[0])   ## sort by START
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:                            ## overlaps or touches the open block
            last[1] = max(last[1], end)                 ## max, because the last block may swallow this one
        else:
            merged.append([start, end])
    return merged

## tests

assert merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge([[1, 4], [4, 5]]) == [[1, 5]]
assert merge([[1, 4], [0, 4]]) == [[0, 4]]
assert merge([[1, 4], [2, 3]]) == [[1, 4]]
print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
```

```
[[1, 6], [8, 10], [15, 18]]
```

**Complexity.** $O(n \log n)$ time for the sort, $O(n)$ space for the output.

### P2. Insert Interval — insert one interval into a list that is already sorted and non-overlapping

**Which template.** Three phases, and no sort at all, because the input is already sorted.
**The trick.** Name the three phases before you write them: the intervals that end strictly before the
new one starts, the intervals that overlap it, and the intervals that start strictly after it ends.
Phase 1 and phase 3 are copied through untouched. Phase 2 is folded into one interval by taking the
minimum start and the maximum end. Writing the loop as three separate `while` loops instead of one
loop with branches is what makes this problem easy to get right under pressure.

```python
def insert(intervals, new_interval):
    out = []
    start, end = new_interval
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < start:            ## phase 1: strictly BEFORE the new one
        out.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= end:             ## phase 2: every OVERLAPPING interval
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])
        i += 1
    out.append([start, end])                            ## the single fused interval
    while i < n:                                        ## phase 3: strictly AFTER
        out.append(intervals[i])
        i += 1
    return out

## tests

assert insert([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
assert insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [[1, 2], [3, 10], [12, 16]]
assert insert([], [5, 7]) == [[5, 7]]
assert insert([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]
print(insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]))
```

```
[[1, 2], [3, 10], [12, 16]]
```

**Complexity.** $O(n)$ time, $O(n)$ space. This is the one interval problem that does not need a sort.

### P3. Non-overlapping Intervals — the minimum number of intervals to remove so that the rest do not overlap

**Which template.** Template 2, then subtract.
**The trick.** Do not think about what to remove. Removing the fewest is the same as keeping the most,
and keeping the most non-overlapping intervals is exactly activity selection, so the answer is
`len(intervals) - maximum kept`. Inverting the objective turns a hard-sounding question into the
template you already know.

```python
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda p: p[1])   ## sort by END
    kept, last_end = 0, float("-inf")
    for start, end in intervals:
        if start >= last_end:                           ## fits after everything kept so far
            kept += 1
            last_end = end
    return len(intervals) - kept                        ## removals = total - the maximum kept

## tests

assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
assert erase_overlap_intervals([]) == 0
print(erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]))
```

```
1
```

**Complexity.** $O(n \log n)$ time, $O(1)$ extra space.

### P4. Meeting Rooms — can one person attend every meeting

**Which template.** Sort by start and check neighbours only.
**The trick.** After sorting by start, if any pair of intervals overlaps then some *adjacent* pair
overlaps, so checking `n - 1` neighbour pairs is enough and you never need the quadratic all-pairs
loop. Note the strict `<`: meetings that touch at an endpoint are fine, because you leave one as the
other begins.

```python
def can_attend_meetings(intervals):
    intervals = sorted(intervals, key=lambda p: p[0])   ## sort by START
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:       ## strict <: touching is allowed
            return False
    return True

## tests

assert can_attend_meetings([[0, 30], [5, 10], [15, 20]]) is False
assert can_attend_meetings([[7, 10], [2, 4]]) is True
assert can_attend_meetings([[1, 5], [5, 9]]) is True
assert can_attend_meetings([]) is True
print(can_attend_meetings([[0, 30], [5, 10], [15, 20]]), can_attend_meetings([[7, 10], [2, 4]]))
```

```
False True
```

**Complexity.** $O(n \log n)$ time, $O(1)$ extra space.

### P5. Meeting Rooms II — the minimum number of rooms needed to hold every meeting

**Which template.** Both a heap and a sweep line, and you should offer both.
**The trick.** The heap version sorts by start and keeps a min-heap of the end times of the busy
rooms. For each meeting, if the room that frees up earliest is already free, reuse it; otherwise open
a new room. The heap size at the end is the answer. The sweep-line version is shorter and needs no
heap: build plus-one and minus-one events, sort, and take the maximum of the running counter. Say
which you prefer and why, because "I would use the sweep line, it is $O(n \log n)$ either way but with
no heap to maintain" is exactly the kind of comparison the interviewer is listening for.

```python
import heapq

def min_meeting_rooms_heap(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda p: p[0])   ## sort by START
    ends = []                                           ## a min-heap of end times of busy rooms
    for start, end in intervals:
        if ends and ends[0] <= start:                   ## the earliest room to free up is free now
            heapq.heapreplace(ends, end)                ## reuse it
        else:
            heapq.heappush(ends, end)                   ## no free room: open a new one
    return len(ends)

def min_meeting_rooms_sweep(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    running, best = 0, 0
    for _, delta in events:
        running += delta
        best = max(best, running)
    return best

## tests

for case, want in [([[0, 30], [5, 10], [15, 20]], 2), ([[7, 10], [2, 4]], 1),
                   ([[0, 30], [5, 10], [15, 20], [10, 25], [20, 35]], 3), ([[1, 5], [5, 9]], 1)]:
    assert min_meeting_rooms_heap(case) == want
    assert min_meeting_rooms_sweep(case) == want
print(min_meeting_rooms_heap([[0, 30], [5, 10], [15, 20], [10, 25], [20, 35]]),
      min_meeting_rooms_sweep([[0, 30], [5, 10], [15, 20], [10, 25], [20, 35]]))
```

```
3 3
```

**Complexity.** $O(n \log n)$ time for both, $O(n)$ space for both.

### P6. Minimum Number of Arrows to Burst Balloons — the fewest vertical arrows that pop every balloon

**Which template.** Template 2, activity selection with the sense reversed.
**The trick.** This is the same greedy as maximum non-overlapping intervals, but you count the groups
rather than the members. Sort by end and shoot at the right edge of the first balloon, because that is
the latest position that still pops it and therefore the position that can pop the most others. Any
balloon starting at or before that coordinate is already popped. The comparison is `start > last_shot`
with a strict `>`, because touching at an endpoint still counts as being hit.

```python
def find_min_arrow_shots(points):
    if not points:
        return 0
    points = sorted(points, key=lambda p: p[1])         ## sort by END
    arrows, last_shot = 1, points[0][1]                 ## shoot at the first balloon's right edge
    for start, end in points[1:]:
        if start > last_shot:                           ## strict >: touching still gets popped
            arrows += 1
            last_shot = end
    return arrows

## tests

assert find_min_arrow_shots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
assert find_min_arrow_shots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
assert find_min_arrow_shots([[1, 2], [2, 3], [3, 4], [4, 5]]) == 2
assert find_min_arrow_shots([]) == 0
print(find_min_arrow_shots([[10, 16], [2, 8], [1, 6], [7, 12]]))
```

```
2
```

**Complexity.** $O(n \log n)$ time, $O(1)$ extra space.

### P7. Interval List Intersections — the intersection of two lists that are each sorted and non-overlapping

**Which template.** Two pointers walking the two lists, no sort.
**The trick.** For any pair, the intersection is `[max(starts), min(ends)]`, and it is non-empty
exactly when that low value is at most that high value. Then advance the pointer whose interval ends
first, because that interval can never intersect anything later in the other list. Both pointers move
forward only, so the walk is linear.

```python
def interval_intersection(first, second):
    out = []
    i, j = 0, 0
    while i < len(first) and j < len(second):
        low = max(first[i][0], second[j][0])            ## latest start
        high = min(first[i][1], second[j][1])           ## earliest end
        if low <= high:                                 ## a non-empty overlap
            out.append([low, high])
        if first[i][1] < second[j][1]:                  ## drop whichever ends first
            i += 1
        else:
            j += 1
    return out

## tests

assert interval_intersection([[0, 2], [5, 10], [13, 23], [24, 25]],
                             [[1, 5], [8, 12], [15, 24], [25, 26]]) == \
       [[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
assert interval_intersection([[1, 3], [5, 9]], []) == []
assert interval_intersection([[1, 7]], [[3, 10]]) == [[3, 7]]
print(interval_intersection([[0, 2], [5, 10], [13, 23], [24, 25]],
                            [[1, 5], [8, 12], [15, 24], [25, 26]]))
```

```
[[1, 2], [5, 5], [8, 10], [15, 23], [24, 24], [25, 25]]
```

**Complexity.** $O(m + n)$ time, $O(m + n)$ space for the output.

### P8. Employee Free Time — the intervals in which every employee is free

**Which template.** Template 1 on the pooled intervals, then read the gaps.
**The trick.** Forget which employee owns which interval. A moment is free for everyone exactly when
it is covered by no interval at all, so flatten every schedule into one list, merge it by template 1,
and the answer is the gaps between the merged blocks. The whole problem is the realisation that the
employee identities are noise.

```python
def employee_free_time(schedule):
    busy = []
    for person in schedule:
        busy.extend(person)                             ## forget who owns what
    busy.sort(key=lambda p: p[0])                       ## sort by START
    free = []
    if not busy:
        return free
    open_end = busy[0][1]
    for start, end in busy[1:]:
        if start > open_end:                            ## a gap nobody is working in
            free.append([open_end, start])
            open_end = end
        else:
            open_end = max(open_end, end)               ## ordinary merge
    return free

## tests

assert employee_free_time([[[1, 2], [5, 6]], [[1, 3]], [[4, 10]]]) == [[3, 4]]
assert employee_free_time([[[1, 3], [6, 7]], [[2, 4]], [[2, 5], [9, 12]]]) == [[5, 6], [7, 9]]
assert employee_free_time([[[1, 10]]]) == []
assert employee_free_time([]) == []
print(employee_free_time([[[1, 3], [6, 7]], [[2, 4]], [[2, 5], [9, 12]]]))
```

```
[[5, 6], [7, 9]]
```

**Complexity.** $O(n \log n)$ time in the total number of intervals, $O(n)$ space.

### P9. Car Pooling — can one car with a fixed capacity serve every trip

**Which template.** Template 3, the sweep line, with weighted events.
**The trick.** This is Meeting Rooms II with passengers instead of ones. A trip contributes
`+people` at its pickup and `-people` at its drop-off, and the car fails if the running total ever
exceeds the capacity. The tie rule matters here in the same way: at the same kilometre, passengers get
off before new passengers get on, and sorting plain tuples gives that for free because negative
deltas sort first.

```python
def car_pooling(trips, capacity):
    events = []
    for people, start, end in trips:
        events.append((start, people))                  ## board
        events.append((end, -people))                   ## alight
    events.sort()                                       ## at equal km, negative deltas come first
    onboard = 0
    for _, delta in events:
        onboard += delta
        if onboard > capacity:
            return False
    return True

## tests

assert car_pooling([[2, 1, 5], [3, 3, 7]], 4) is False
assert car_pooling([[2, 1, 5], [3, 3, 7]], 5) is True
assert car_pooling([[2, 1, 5], [3, 5, 7]], 3) is True
assert car_pooling([], 1) is True
print(car_pooling([[2, 1, 5], [3, 3, 7]], 4), car_pooling([[2, 1, 5], [3, 3, 7]], 5))
```

```
False True
```

**Complexity.** $O(n \log n)$ time, $O(n)$ space. With a bounded route length you can use a fixed
difference array instead and get $O(n + L)$.

### P10. Minimum Interval to Include Each Query — for each query point, the length of the smallest interval containing it

**Which template.** An offline sweep: sort the intervals by start, sort the queries, and use a heap.
**The trick.** Process the queries in increasing order, not in the order given. Then as the query
point moves right you only ever add intervals, never remove them because of their start. Push every
interval whose start has been passed onto a min-heap keyed by length, then discard from the top of the
heap any interval whose end lies behind the query. The top is now the shortest live interval. Keep the
original positions so you can put the answers back in the caller's order.

```python
import heapq

def min_interval(intervals, queries):
    intervals = sorted(intervals, key=lambda p: p[0])   ## sort by START
    order = sorted(range(len(queries)), key=lambda i: queries[i])
    answer = [-1] * len(queries)
    heap = []                                           ## (size, end) of intervals already opened
    i = 0
    for qi in order:
        q = queries[qi]
        while i < len(intervals) and intervals[i][0] <= q:
            start, end = intervals[i]
            heapq.heappush(heap, (end - start + 1, end))
            i += 1
        while heap and heap[0][1] < q:                  ## the smallest one has expired
            heapq.heappop(heap)
        answer[qi] = heap[0][0] if heap else -1
    return answer

## tests

assert min_interval([[1, 4], [2, 4], [3, 6], [4, 4]], [2, 3, 4, 5]) == [3, 3, 1, 4]
assert min_interval([[2, 3], [2, 5], [1, 8], [20, 25]], [2, 19, 5, 22]) == [2, -1, 4, 6]
assert min_interval([[1, 2]], [3]) == [-1]
print(min_interval([[1, 4], [2, 4], [3, 6], [4, 4]], [2, 3, 4, 5]))
```

```
[3, 3, 1, 4]
```

**Complexity.** $O((n + q) \log (n + q))$ time, $O(n)$ space.

### P11. Maximum Subarray — the largest sum over all contiguous subarrays, by Kadane

**Which template.** None of the four. This is dynamic programming, and people call it greedy.
**The trick.** Be precise about this, because the interviewer may test it. The state is
`ending_here`, the best sum of a subarray that *ends* at the current index, and the recurrence is
`ending_here = max(x, ending_here + x)`: either start a new subarray at `x`, or extend the best one
ending just before. That is a state definition and a transition, so it is DP with $O(1)$ rolling
state, not greed. The chapter on dynamic programming derives the same recurrence. The reason people
call it greedy is the discard rule, "throw away a running sum that has gone negative", which looks
like a local choice, but it is only correct because the DP state justifies it.

```python
def max_subarray(nums):
    best = nums[0]
    ending_here = nums[0]                               ## best sum of a subarray ENDING at i
    for x in nums[1:]:
        ending_here = max(x, ending_here + x)           ## start fresh, or extend the previous best
        best = max(best, ending_here)
    return best

## tests

assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5, 4, -1, 7, 8]) == 23
assert max_subarray([-3, -1, -2]) == -1
print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), max_subarray([-3, -1, -2]))
```

```
6 -1
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P12. Jump Game — can you reach the last index, where each value is a maximum jump length

**Which template.** Template 4, the greedy reach.
**The trick.** You do not need to know which jumps to make, only how far you can get. Carry the
furthest reachable index; if the loop ever stands at an index beyond that reach, there is a hole and
the answer is false. The exchange argument is one line: any solution that reaches index `i` reaches
every index before `i` too, so tracking only the maximum reach loses nothing.

```python
def can_jump(nums):
    reach = 0                                           ## furthest index reachable so far
    for i in range(len(nums)):
        if i > reach:                                   ## a hole you cannot cross
            return False
        reach = max(reach, i + nums[i])
    return True

## tests

assert can_jump([2, 3, 1, 1, 4]) is True
assert can_jump([3, 2, 1, 0, 4]) is False
assert can_jump([0]) is True
assert can_jump([2, 0, 0]) is True
print(can_jump([2, 3, 1, 1, 4]), can_jump([3, 2, 1, 0, 4]))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P13. Jump Game II — the minimum number of jumps needed to reach the last index

**Which template.** Template 4 with levels, which is a breadth-first search written as a scan.
**The trick.** Think of the indices reachable in exactly `j` jumps as level `j` of a BFS. The scan
walks the current level and records the furthest index any of its members can reach. When the walk
hits `current_end`, the level is exhausted, so increment the jump count and set the next level's end
to `farthest`. Stop the loop one index early, at `len(nums) - 1`, or you count a jump that lands
beyond the finish.

```python
def jump(nums):
    jumps, current_end, farthest = 0, 0, 0
    for i in range(len(nums) - 1):                      ## stop before the last index
        farthest = max(farthest, i + nums[i])           ## best landing spot from this level
        if i == current_end:                            ## the level is exhausted
            jumps += 1
            current_end = farthest                      ## the next level ends here
    return jumps

## tests

assert jump([2, 3, 1, 1, 4]) == 2
assert jump([2, 3, 0, 1, 4]) == 2
assert jump([0]) == 0
assert jump([1, 1, 1, 1]) == 3
print(jump([2, 3, 1, 1, 4]), jump([1, 1, 1, 1]))
```

```
2 3
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P14. Gas Station — the index to start from so that you can drive the full circle

**Which template.** Template 4, greedy with a restart, plus a feasibility check.
**The trick.** Two facts do all the work, and you should state both. First, a full circuit is possible
if and only if the total gas is at least the total cost, because the tank at the end of a lap is that
difference regardless of where you start. Second, if the tank goes negative somewhere between `start`
and `i`, then no station between them works either: any later start has an empty tank at that point
instead of a non-negative one, so it fails no later than `i` does. Therefore you may jump `start` all
the way to `i + 1` and never revisit. Those two facts together make one pass correct, and the answer
is unique when it exists.

```python
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost):
        return -1                                       ## no start can work at all
    start, tank = 0, 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:                                    ## i is unreachable from start
            start = i + 1                               ## so is every station between: skip them all
            tank = 0
    return start

## tests

assert can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
assert can_complete_circuit([2, 3, 4], [3, 4, 3]) == -1
assert can_complete_circuit([5], [4]) == 0
assert can_complete_circuit([3, 1, 1], [1, 2, 2]) == 0
print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
```

```
3
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P15. Partition Labels — cut the string into the most pieces such that each letter appears in only one piece

**Which template.** Template 1, merging, on intervals you have to build first.
**The trick.** Each letter defines the interval from its first to its last occurrence, and a valid
piece is a merged block of those intervals. So record the last index of every letter in one pass, then
sweep left to right extending the open block to `last[ch]`. When the loop index equals the block's end,
nothing inside the block reaches further, so you may cut. Seeing the intervals hiding in a string
problem is the whole exercise.

```python
def partition_labels(s):
    last = {}
    for i, ch in enumerate(s):
        last[ch] = i                                    ## the interval of a letter is [first, last]
    out = []
    start, end = 0, 0
    for i, ch in enumerate(s):
        end = max(end, last[ch])                        ## extend the open block, exactly like merging
        if i == end:                                    ## nothing inside reaches further: close it
            out.append(end - start + 1)
            start = i + 1
    return out

## tests

assert partition_labels("ababcbacadefegdehijhklij") == [9, 7, 8]
assert partition_labels("eccbbbbdec") == [10]
assert partition_labels("abc") == [1, 1, 1]
assert partition_labels("") == []
print(partition_labels("ababcbacadefegdehijhklij"))
```

```
[9, 7, 8]
```

**Complexity.** $O(n)$ time, $O(1)$ space for the 26 last-index entries.

### P16. Hand of Straights — can the cards be split into groups of consecutive values of a fixed size

**Which template.** Greedy on the smallest remaining value.
**The trick.** The exchange argument is easy here, so give it. The smallest card left must be the
first card of some group, because nothing smaller exists to precede it. Therefore its group is forced:
it consumes that card and the next `group_size - 1` values. Because the group is forced, greedy cannot
go wrong. Consume all copies of the smallest card at once rather than one group at a time, or the
solution becomes quadratic on inputs with many duplicates.

```python
from collections import Counter

def is_n_straight_hand(hand, group_size):
    if len(hand) % group_size != 0:
        return False
    count = Counter(hand)
    for card in sorted(count):                          ## the smallest card left must open a group
        needed = count[card]
        if needed == 0:
            continue
        for step in range(group_size):                  ## consume card, card+1, ... card+k-1
            if count[card + step] < needed:
                return False
            count[card + step] -= needed
    return True

## tests

assert is_n_straight_hand([1, 2, 3, 6, 2, 3, 4, 7, 8], 3) is True
assert is_n_straight_hand([1, 2, 3, 4, 5], 4) is False
assert is_n_straight_hand([1, 1, 2, 2, 3, 3], 3) is True
assert is_n_straight_hand([8, 10, 12], 3) is False
print(is_n_straight_hand([1, 2, 3, 6, 2, 3, 4, 7, 8], 3), is_n_straight_hand([1, 2, 3, 4, 5], 4))
```

```
True False
```

**Complexity.** $O(n \log n + nk)$ time with `k` the group size, $O(n)$ space.

### P17. Valid Parenthesis String — is the string balanced when every `*` may be `(`, `)`, or empty

**Which template.** Greedy with a range, not a single counter.
**The trick.** You cannot decide what a `*` means when you meet it, so do not decide. Carry two
counters, `low` and `high`, the smallest and largest number of open brackets that any reading of the
prefix could produce. A `*` pushes them apart by one in each direction. If `high` ever goes negative
then even the most generous reading has too many closing brackets, so fail immediately. Clamp `low` at
zero, because a reading that would go negative is simply not a reading you take. The string is valid
when zero open brackets are achievable at the end, which is `low == 0`.

```python
def check_valid_string(s):
    low, high = 0, 0                                    ## the range of possible open-bracket counts
    for ch in s:
        if ch == "(":
            low, high = low + 1, high + 1
        elif ch == ")":
            low, high = low - 1, high - 1
        else:                                           ## '*' can be ')', nothing, or '('
            low, high = low - 1, high + 1
        if high < 0:                                    ## too many ')' even in the best case
            return False
        low = max(low, 0)                               ## never let the optimistic count go negative
    return low == 0                                     ## zero open brackets must be achievable

## tests

assert check_valid_string("()") is True
assert check_valid_string("(*)") is True
assert check_valid_string("(*))") is True
assert check_valid_string(")(") is False
assert check_valid_string("(((**") is False
print(check_valid_string("(*))"), check_valid_string(")("), check_valid_string("(((**"))
```

```
True False False
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P18. Task Scheduler — the least time to run all tasks with a cooling gap of `n` between equal tasks

**Which template.** No simulation. A closed-form formula, derived.
**The trick.** Derive the formula rather than remembering it. Let `max_count` be the highest task
frequency and `num_max` the number of tasks tied at that frequency. The busiest task must run
`max_count` times, and consecutive runs of it must be at least `n + 1` slots apart. That builds a
skeleton of `max_count - 1` blocks, each `n + 1` slots long, followed by one final slot for the last
run. Every other task tied at `max_count` must also appear in that final block, so add `num_max`
rather than 1. The skeleton therefore costs `(max_count - 1) * (n + 1) + num_max` slots, and every
other task fits into the gaps because there is room by construction. However, if there are so many
distinct tasks that the gaps overflow, no idling happens at all and the answer is simply the number of
tasks. Take the maximum of the two.

```python
from collections import Counter

def least_interval(tasks, n):
    count = Counter(tasks)
    max_count = max(count.values())
    num_max = sum(1 for v in count.values() if v == max_count)
    frame = (max_count - 1) * (n + 1) + num_max         ## the skeleton built from the busiest task
    return max(len(tasks), frame)                       ## with many distinct tasks there is no idling

## tests

assert least_interval(["A", "A", "A", "B", "B", "B"], 2) == 8
assert least_interval(["A", "A", "A", "B", "B", "B"], 0) == 6
assert least_interval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2) == 16
assert least_interval(["A"], 5) == 1
print(least_interval(["A", "A", "A", "B", "B", "B"], 2),
      least_interval(["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"], 2))
```

```
8 16
```

**Complexity.** $O(n)$ time, $O(1)$ space for the 26 counts.

### P19. Candy — give each child at least one sweet, and more than any lower-rated neighbour

**Which template.** Two greedy passes, one in each direction.
**The trick.** One pass cannot satisfy both neighbours at once, because a child's requirement depends
on the left and the right. So satisfy them separately: a left-to-right pass fixes every rising run,
and a right-to-left pass fixes every falling run. The second pass must use `max(give[i],
give[i + 1] + 1)` rather than plain assignment, because overwriting would destroy the constraint the
first pass established. That single `max` is the whole problem.

```python
def candy(ratings):
    n = len(ratings)
    if n == 0:
        return 0
    give = [1] * n                                      ## everyone gets at least one
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            give[i] = give[i - 1] + 1                   ## left-to-right fixes the rising runs
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            give[i] = max(give[i], give[i + 1] + 1)     ## max keeps the first pass intact
    return sum(give)

## tests

assert candy([1, 0, 2]) == 5
assert candy([1, 2, 2]) == 4
assert candy([1, 3, 2, 2, 1]) == 7
assert candy([]) == 0
print(candy([1, 0, 2]), candy([1, 3, 2, 2, 1]))
```

```
5 7
```

**Complexity.** $O(n)$ time, $O(n)$ space.

### P20. Boats to Save People — the fewest boats when each boat carries at most two people under a weight limit

**Which template.** Greedy with two pointers on the sorted weights.
**The trick.** The heaviest person leaves on this boat no matter what, because there is no better
partner for them later than the lightest person available now. So pair the heaviest with the lightest
if they fit, and send the heaviest alone if they do not. The exchange argument: in any optimal
solution, if the heaviest person shares a boat with someone other than the lightest, swap that partner
for the lightest; the boat still fits, and the displaced person is no harder to place. So the answer
does not get worse.

```python
def num_rescue_boats(people, limit):
    people = sorted(people)
    left, right, boats = 0, len(people) - 1, 0
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1                                   ## the lightest rides with the heaviest
        right -= 1                                      ## the heaviest always leaves on this boat
        boats += 1
    return boats

## tests

assert num_rescue_boats([1, 2], 3) == 1
assert num_rescue_boats([3, 2, 2, 1], 3) == 3
assert num_rescue_boats([3, 5, 3, 4], 5) == 4
assert num_rescue_boats([], 5) == 0
print(num_rescue_boats([3, 2, 2, 1], 3), num_rescue_boats([3, 5, 3, 4], 5))
```

```
3 4
```

**Complexity.** $O(n \log n)$ time for the sort, $O(1)$ extra space.

## Tricks and tips

**Write the sort key on the page before anything else.** In an interval problem the sort key is the
algorithm; the loop afterwards is short and nearly the same in every case. Say "merge, so sort by
start" or "count, so sort by end" out loud, then write it. If you find yourself unsure, ask what the
output is: a list of intervals means merging and therefore start, a number of items or groups means
scheduling and therefore end.

**Sort tuples and let Python break the ties.** In a sweep line, `events.sort()` on `(coordinate,
delta)` tuples puts `-1` before `+1` at the same coordinate, which is exactly the rule you want when
touching intervals do not overlap. If the problem says touching DOES overlap, as with the balloons,
sort by `(coordinate, -delta)` instead, or use a start-sorted approach. Either way, state the rule you
are choosing before you write the sort.

**Prefer the sweep line to a heap when you only need a count.** Meeting Rooms II and Car Pooling both
have a neat heap solution and a shorter sweep-line solution. Both are $O(n \log n)$, but the sweep
line has no data structure to maintain and no reuse condition to get wrong. Offer the heap version as
the alternative, because it generalises to problems that need to know WHICH room, not just how many.

**Turn "minimum removals" into "maximum kept".** The complement trick appears constantly:
non-overlapping intervals, arrows, and any "delete the fewest so that a property holds" question. It
is almost always easier to build the largest valid set greedily and subtract than to reason about what
to delete.

**Every greedy answer needs a one-sentence justification.** Not a proof, a sentence. "The earliest
finisher leaves the most room." "The heaviest person must leave now, so give them the best partner."
"The smallest card must start a group, so its group is forced." If you cannot produce that sentence,
your greedy choice is probably wrong and the problem is dynamic programming. That is a genuinely
useful signal in the room, not just a presentational nicety.

**Watch for problems that hide intervals.** Partition Labels is interval merging on the first-to-last
range of each letter. Employee Free Time is interval merging with the owners removed. Car Pooling is
Meeting Rooms II with weights. The pattern here is that the word "interval" rarely appears; you have
to notice that a start and an end are being described.

**Sort a copy when the caller might care.** `intervals.sort()` mutates the argument. In an interview
it usually does not matter, but saying "I am sorting in place, which mutates the input, tell me if
that is a problem" costs one sentence and reads well.

## The bugs that cost the round

**Sorting by the wrong key.** This is the single most common failure. A merge sorted by end produces
blocks in an order that looks right on a small example and is wrong in general. An activity selection
sorted by start is simply the wrong greedy: it will happily take a long interval that blocks several
short ones. If you remember one thing, remember start for merging, end for counting.

**`<` against `<=` on the overlap test.** Whether touching endpoints overlap depends on the problem,
and the same character decides three things at once: the neighbour test in Meeting Rooms, the extend
test in Merge Intervals, and the tie rule in the sweep line. Ask the interviewer, then keep the
answer consistent across all three.

**Assigning instead of taking a maximum when merging.** `merged[-1][1] = end` is wrong; it must be
`max(merged[-1][1], end)`. An interval fully contained in the open block would otherwise shrink it.
The sample input `[[1, 4], [2, 3]]` catches this and almost nothing else does.

**Overwriting the first pass in Candy.** The right-to-left pass must use `max`. Plain assignment
passes the sample and fails on `[1, 3, 2, 2, 1]`.

**Counting a jump past the finish.** Jump Game II loops to `len(nums) - 1`, not `len(nums)`. Looping
one step too far increments the jump counter once more when the last index is also the end of a level.

**Forgetting the empty input.** Zero intervals, one interval, and all intervals identical. Each is one
line at the top, and each appears in the tests above.

**Simulating Task Scheduler with a heap when the formula exists.** The simulation is correct but long,
and under time pressure it is where you run out of minutes. Derive the formula in three sentences
instead, then write four lines.

## Done when

- Given an interval problem you have not seen, you can say within 30 seconds whether it sorts by start
  or by end, and give the one-sentence reason.
- You can write the overlap condition `a.start < b.end and b.start < a.end` from memory and say
  whether touching counts in the problem in front of you.
- You can build a sweep line from intervals to events to counter, and state the tie rule and why
  sorting plain tuples gives it to you.
- You can state an exchange argument for the earliest-finisher greedy, for Gas Station and for Boats
  to Save People, in one sentence each, without writing code.
