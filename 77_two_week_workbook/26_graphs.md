# Graphs: every variation

A graph is a set of nodes and a set of edges between them. That much is easy. The hard part is that
most graph problems in an interview are not presented as graphs. They arrive as a grid of characters,
a list of course prerequisites, a set of division equations, a ladder of words, a pile of email
accounts to merge, or a lock with four dials. None of those statements contain the word "graph". The
work is to see the graph, name its parts, and choose a representation. Once the graph is built, the
algorithm is almost always one of five standard ones, and you already know all five.

So the load-bearing sentence for this whole chapter is: **name the nodes, name the edges, then pick
the algorithm**. Candidates fail at the first two steps, not the third. A person who can say "a node
is a cell, an edge joins two cells that share a side" has already solved Number of Islands; a person
who jumps straight to writing a queue has not, and will spend the round patching bounds checks.

The second organising idea is the choice rule, and it is short. Use BFS when you need the fewest edges
on an unweighted graph, because BFS visits nodes in order of distance from the source. Use DFS when you
need to explore a whole component, detect a cycle, or build an ordering. Use Dijkstra when edges carry
non-negative weights. Use topological sort when the graph is a DAG and the question is about ordering
or dependency. Use union-find when the question is about connectivity and the edges arrive over time.

## Recognising it from the phrasing

| The interviewer says | They mean | Algorithm | Answer is |
|---|---|---|---|
| "shortest path", unweighted | fewest edges | BFS from the source | the level count |
| "shortest path", weighted, non-negative | cheapest path | Dijkstra with a heap | `dist[target]` |
| "number of islands / regions / connected components" | count the components | DFS or BFS flood fill, or union-find | a counter, or `dsu.components` |
| "course schedule / build order / dependency" | a valid ordering | topological sort, Kahn or DFS | the order list |
| "can these be merged / are these the same group / accounts, equations" | dynamic connectivity | union-find | the roots, grouped |
| "detect a cycle", directed | a back edge exists | DFS with three colours, or Kahn's leftover count | a boolean |
| "detect a cycle", undirected | a repeated meeting | DFS tracking the parent, or union-find | a boolean |
| "rotting oranges / spread over time / minimum steps for all" | simultaneous spread | multi-source BFS | the last level index |
| "clone / copy a graph" | a structural copy | DFS with an old-node-to-new-node map | the copy of the entry node |

Before writing any code, ask two questions out loud: **what is a node, and what is an edge?** In a grid
the node is a cell `(r, c)` and the edges join it to its four neighbours. In course scheduling the node
is a course and the edge points from a prerequisite to the course that needs it. In the equations
problem the node is a variable and the edge carries a ratio, so the path product is the answer. In Open
the Lock the node is a four-digit string, which is not a physical thing at all, and the edge is one
turn of one dial. Saying this out loud is worth real points, because it is the step that separates the
people who recognise a graph from the people who memorised BFS. It also fixes the representation for
you: if the nodes are `0..n-1` use a list of lists, and if the nodes are labels use a dict.

## The templates

Six templates. The first builds the graph, and the other five are the algorithms. Templates 2 and 3
share a skeleton — a container, a visited set, a pop, a neighbour loop — and differ only in whether the
container is a queue or a stack. That single change is the whole difference between "shortest" and
"reachable", so learn the skeleton once.

**Template 1 — build an adjacency list from an edge list.** Use when the input is a list of pairs.

```python
def build_graph(n, edges, directed=False):
    adj = {i: [] for i in range(n)}              ## every node exists, even isolated ones
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)                     ## the ONE line that differs
    return adj

def build_graph_labelled(edges, directed=False):
    adj = {}                                     ## nodes are arbitrary labels, not 0..n-1
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])                    ## make sure v exists as a key
        if not directed:
            adj[v].append(u)
    return adj

## tests

assert build_graph(3, [(0, 1), (1, 2)]) == {0: [1], 1: [0, 2], 2: [1]}
assert build_graph(3, [(0, 1), (1, 2)], directed=True) == {0: [1], 1: [2], 2: []}
assert build_graph(2, []) == {0: [], 1: []}
assert build_graph_labelled([("a", "b")], directed=True) == {"a": ["b"], "b": []}
print(build_graph(3, [(0, 1), (1, 2)]))
```

```
{0: [1], 1: [0, 2], 2: [1]}
```

**Template 2 — BFS for the fewest edges.** Use when the graph is unweighted and the question asks for
a minimum number of steps. The answer is recorded as the level counter, which increases once per
snapshot of the queue.

```python
from collections import deque

def bfs_shortest_edges(adj, source, target):
    if source == target:
        return 0
    visited = {source}                           ## mark ON PUSH, never on pop
    queue = deque([source])
    steps = 0
    while queue:
        steps += 1
        for _ in range(len(queue)):              ## the level snapshot: one whole layer
            node = queue.popleft()
            for nxt in adj[node]:
                if nxt in visited:
                    continue
                if nxt == target:
                    return steps
                visited.add(nxt)
                queue.append(nxt)
    return -1

## tests

adj = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2, 4], 4: [3]}
assert bfs_shortest_edges(adj, 0, 4) == 3
assert bfs_shortest_edges(adj, 0, 3) == 2
assert bfs_shortest_edges(adj, 0, 0) == 0
assert bfs_shortest_edges({0: [], 1: []}, 0, 1) == -1
print(bfs_shortest_edges(adj, 0, 4))
```

```
3
```

**Template 3 — DFS, recursive and iterative.** Use when you must visit a whole component. The recursive
form is shorter, but Python's recursion limit is about 1000 frames and a path of 200000 cells will
crash it, so the iterative form is the one to write for large inputs. Note the difference in where you
mark: the recursive form marks on entry, and the iterative form marks on pop, because a node can be
pushed onto the stack several times before it is popped.

```python
def dfs_recursive(adj, source, visited=None):
    if visited is None:
        visited = set()
    visited.add(source)                          ## mark on ENTRY
    for nxt in adj[source]:
        if nxt not in visited:
            dfs_recursive(adj, nxt, visited)
    return visited

def dfs_iterative(adj, source):
    visited = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node in visited:                      ## a node can be pushed many times
            continue
        visited.add(node)                        ## mark on POP, not on push
        for nxt in adj[node]:
            if nxt not in visited:
                stack.append(nxt)
    return visited

## tests

adj = {0: [1], 1: [0, 2], 2: [1], 3: []}
assert dfs_recursive(adj, 0) == {0, 1, 2}
assert dfs_iterative(adj, 0) == {0, 1, 2}
assert dfs_iterative(adj, 3) == {3}
chain = {i: ([i + 1] if i + 1 < 100000 else []) for i in range(100000)}
assert len(dfs_iterative(chain, 0)) == 100000    ## recursion would blow the stack here
print(sorted(dfs_recursive(adj, 0)), len(dfs_iterative(chain, 0)))
```

```
[0, 1, 2] 100000
```

**Template 4 — topological sort by Kahn's algorithm.** Use when the graph is a DAG and the question is
about ordering. The answer is the `order` list. The same run also answers "is there a cycle": if the
order is shorter than `n`, the leftover nodes all have a remaining in-degree, so each of them waits on
another leftover node, and that is a cycle. One algorithm, two questions.

```python
from collections import deque

def topological_order(n, edges):
    adj = {i: [] for i in range(n)}
    indegree = [0] * n
    for u, v in edges:                           ## edge u -> v means u comes first
        adj[u].append(v)
        indegree[v] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in adj[node]:
            indegree[nxt] -= 1                   ## remove the edge, not the node
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) < n:                           ## leftover nodes sit on a cycle
        return []
    return order

## tests

assert topological_order(4, [(0, 1), (1, 2), (2, 3)]) == [0, 1, 2, 3]
assert topological_order(2, [(0, 1), (1, 0)]) == []
assert topological_order(3, []) == [0, 1, 2]
assert len(topological_order(4, [(0, 1), (0, 2), (1, 3), (2, 3)])) == 4
print(topological_order(4, [(0, 1), (0, 2), (1, 3), (2, 3)]), topological_order(2, [(0, 1), (1, 0)]))
```

```
[0, 1, 2, 3] []
```

**Template 5 — Dijkstra with a heap.** Use when edges have non-negative weights. The answer is the
`dist` array. The line `if d > dist[u]: continue` is the whole trick of the practical version. A
textbook Dijkstra decreases a key inside the heap, but Python's `heapq` has no decrease-key operation.
So instead you push a second, cheaper entry for the same node and leave the old one in place. When the
stale entry surfaces later, its recorded distance `d` is worse than the `dist[u]` you have already
settled, so you skip it. The heap grows to $O(E)$ entries rather than $O(V)$, which changes nothing
about the $O(E \log V)$ bound because $\log E$ and $\log V$ differ by a constant factor.

```python
import heapq

def dijkstra(n, weighted_edges, source):
    adj = {i: [] for i in range(n)}
    for u, v, w in weighted_edges:               ## directed; add the reverse for undirected
        adj[u].append((v, w))
    dist = [float("inf")] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:                          ## a stale copy of u: skip it
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))   ## push a new entry, never decrease a key
    return dist

## tests

edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
assert dijkstra(4, edges, 0) == [0, 3, 1, 4]
assert dijkstra(2, [], 0) == [0, float("inf")]
assert dijkstra(3, [(0, 1, 7), (1, 2, 2)], 0) == [0, 7, 9]
print(dijkstra(4, edges, 0))
```

```
[0, 3, 1, 4]
```

**Template 6 — union-find with path compression and union by size.** Use when the question is about
connectivity and the edges arrive one at a time. The answer is either `find(a) == find(b)`, or the
`components` counter, or the return value of `union` — `False` means the two ends were already joined,
which is how you detect a redundant edge. With both optimisations the amortised cost per operation is
$O(\alpha(n))$, the inverse Ackermann function, which is below 5 for any `n` you can store. That is
not constant, but it is near-constant, and saying it that way is more accurate than saying $O(1)$.

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   ## path compression, halving form
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False                         ## already joined: this edge is redundant
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra                      ## union by size: small tree under big tree
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True

## tests

dsu = DSU(5)
assert dsu.union(0, 1) is True
assert dsu.union(1, 2) is True
assert dsu.union(0, 2) is False
assert dsu.find(2) == dsu.find(0)
assert dsu.components == 3
print(dsu.components, dsu.find(0) == dsu.find(2), dsu.union(0, 2))
```

```
3 True False
```

## The grid as a graph

A grid is the single most common disguise, so treat it as the default case rather than a special one.
The node is the pair `(r, c)`. The edges join a cell to its four orthogonal neighbours, and you
generate them with a fixed direction list rather than four copied blocks of code:

```
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
```

Then every traversal has the same three-line inner loop: add the offsets, check the bounds, check the
cell is the kind you can enter. The bounds check `0 <= nr < rows and 0 <= nc < cols` must come first,
because Python's negative indexing means `grid[-1][c]` is a legal read of the last row and will
silently wrap your search around the board. Eight-way movement is the same loop with an eight-entry
direction list, so nothing else changes.

There are two ways to mark a cell visited, and you should state your choice aloud. A `visited` set of
`(r, c)` pairs costs $O(rc)$ extra space and leaves the input untouched. Mutating the grid — writing a
`0` over an island cell, or a `2` over a fresh orange — costs $O(1)$ extra space and is faster, but it
destroys the caller's data. Say "I am going to mutate the input as my visited marker, which is $O(1)$
space; tell me if the grid must be preserved and I will use a set instead". Doing it silently is the
version that costs you the round.

**Multi-source BFS** is the trick people miss. When the question is "how long until everything is
reached", the naive plan is to run one BFS from each source and take the minimum, which costs
$O(S \cdot rc)$. Instead, push **every** source into the queue before the first pop. The queue then
holds all distance-0 nodes, then all distance-1 nodes, and so on, exactly as it would for one source.
Because the queue is still non-decreasing in distance, the first time BFS reaches any cell, it has
reached it by the shortest route from the nearest source. One pass, $O(rc)$.

**Worked example — rotting oranges.** The grid is `[[2,1,1],[1,1,0],[0,1,1]]`, where 2 is rotten, 1 is
fresh and 0 is empty. There is one rotten orange, at `(0,0)`. Minute 1 rots `(0,1)` and `(1,0)`. Minute
2 rots `(0,2)` and `(1,1)`. Minute 3 rots `(2,1)`. Minute 4 rots `(2,2)`. No fresh orange is left, so
the answer is 4. The loop runs one level per minute, and it stops as soon as `fresh` reaches zero so
that a final empty level does not add a phantom minute. If any fresh orange is unreachable, `fresh`
stays positive and the answer is -1.

```python
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))             ## SEED EVERY SOURCE before the first pop
            elif grid[r][c] == 1:
                fresh += 1
    minutes = 0
    while queue and fresh > 0:
        minutes += 1
        for _ in range(len(queue)):              ## one whole minute per level
            r, c = queue.popleft()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2             ## mutate the grid as the visited mark
                    fresh -= 1
                    queue.append((nr, nc))
    return -1 if fresh > 0 else minutes

## tests

assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
assert oranges_rotting([[0, 2]]) == 0
assert oranges_rotting([[2, 2], [1, 1]]) == 1
print(oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]))
```

```
4
```

## The problems

### P1. Number of islands, by DFS — count the connected groups of "1" cells in a character grid

**Which template.** Template 3, the iterative DFS, run once per unvisited land cell.
**The trick.** The outer double loop is not a search; it is a hunt for a component that has not been
seen yet. Each time it finds one, the counter goes up by one and the inner DFS erases the whole
component so it is never counted again. That separation — outer loop counts, inner loop consumes — is
the shape of every connected-components problem.

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = set()
    islands = 0
    for r0 in range(rows):
        for c0 in range(cols):
            if grid[r0][c0] != "1" or (r0, c0) in seen:
                continue
            islands += 1                         ## one new component starts here
            stack = [(r0, c0)]
            seen.add((r0, c0))
            while stack:                         ## iterative DFS: no recursion limit
                r, c = stack.pop()
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1" \
                            and (nr, nc) not in seen:
                        seen.add((nr, nc))       ## mark on PUSH to avoid duplicates
                        stack.append((nr, nc))
    return islands

## tests

g1 = [["1", "1", "0"], ["1", "0", "0"], ["0", "0", "1"]]
assert num_islands(g1) == 2
assert num_islands([["1", "1"], ["1", "1"]]) == 1
assert num_islands([["0", "0"], ["0", "0"]]) == 0
assert num_islands([]) == 0
print(num_islands(g1))
```

```
2
```

**Complexity.** $O(rc)$ time and $O(rc)$ space for the visited set and the stack.

### P2. Number of islands, by union-find — the same count, with the components merged instead of walked

**Which template.** Template 6.
**The trick.** Give every cell the index `r * cols + c` so the grid becomes a flat array of `rows * cols`
nodes. Then union each land cell with its neighbour above and its neighbour to the left only. Those two
directions are enough, because by the time you reach a cell the cells below and to the right have not
been processed, and every adjacency is therefore visited exactly once. The count starts at
`rows * cols`, so subtract the water cells, which were never merged with anything. Compare this against
P1 out loud: DFS is simpler here and union-find is the answer they want when islands are added one at a
time and the count must be re-reported after each addition.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True

def num_islands_dsu(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    dsu = DSU(rows * cols)
    water = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "1":
                water += 1                       ## water cells are components too: subtract them
                continue
            for dr, dc in ((-1, 0), (0, -1)):    ## up and left only: every edge seen once
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                    dsu.union(r * cols + c, nr * cols + nc)
    return dsu.components - water

## tests

g1 = [["1", "1", "0"], ["1", "0", "0"], ["0", "0", "1"]]
assert num_islands_dsu(g1) == 2
assert num_islands_dsu([["1", "1"], ["1", "1"]]) == 1
assert num_islands_dsu([["0", "0"], ["0", "0"]]) == 0
assert num_islands_dsu([["1", "0", "1"]]) == 2
print(num_islands_dsu(g1), num_islands_dsu([["1", "0", "1"]]))
```

```
2 2
```

**Complexity.** $O(rc \cdot \alpha(rc))$ time, $O(rc)$ space.

### P3. Max area of island — the size of the largest connected group of 1 cells

**Which template.** Template 3, with a counter instead of a flag.
**The trick.** Sink each cell as you push it, not as you pop it. If you sink on pop, a cell reachable
from two different neighbours gets pushed twice, popped twice, and counted twice, and the area comes
out too large on any island that is not a straight line. Counting on pop and marking on push is the
pairing that makes each cell contribute exactly once.

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def max_area_of_island(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    best = 0
    for r0 in range(rows):
        for c0 in range(cols):
            if grid[r0][c0] != 1:
                continue
            area = 0
            stack = [(r0, c0)]
            grid[r0][c0] = 0                     ## sink it: the grid IS the visited set
            while stack:
                r, c = stack.pop()
                area += 1                        ## count on POP, exactly once per cell
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 0         ## sink on PUSH, not on pop
                        stack.append((nr, nc))
            best = max(best, area)
    return best

## tests

g = [[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]]
assert max_area_of_island(g) == 3
assert max_area_of_island([[0, 0], [0, 0]]) == 0
assert max_area_of_island([[1, 1], [1, 1]]) == 4
assert max_area_of_island([]) == 0
print(max_area_of_island([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]]))
```

```
3
```

**Complexity.** $O(rc)$ time, $O(rc)$ space for the stack in the worst case.

### P4. Flood fill — recolour the region connected to a starting pixel

**Which template.** Template 3 on a grid, with the colour test in place of a visited set.
**The trick.** The guard `if old_colour == new_colour: return image` is the entire problem. Without it,
recolouring a pixel to the colour it already has leaves the entry test true forever, so the same cells
are pushed again and again and the loop never ends. Interviewers include that case deliberately.

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def flood_fill(image, sr, sc, new_colour):
    old_colour = image[sr][sc]
    if old_colour == new_colour:
        return image                             ## without this guard the loop never ends
    rows, cols = len(image), len(image[0])
    stack = [(sr, sc)]
    image[sr][sc] = new_colour
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == old_colour:
                image[nr][nc] = new_colour       ## recolouring is the visited mark
                stack.append((nr, nc))
    return image

## tests

assert flood_fill([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2) == [[2, 2, 2], [2, 2, 0], [2, 0, 1]]
assert flood_fill([[0, 0], [0, 0]], 0, 0, 0) == [[0, 0], [0, 0]]
assert flood_fill([[1, 2], [3, 4]], 0, 0, 9) == [[9, 2], [3, 4]]
print(flood_fill([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2))
```

```
[[2, 2, 2], [2, 2, 0], [2, 0, 1]]
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P5. Surrounded regions — flip every "O" region that does not touch the border to "X"

**Which template.** Template 3, started from the border instead of from the interior.
**The trick.** Invert the question. Finding the surrounded regions directly means proving a negative
for each one, which is awkward. Finding the *un*surrounded regions is easy, because they are exactly
the ones reachable from an "O" on the border. So run one traversal seeded with every border "O", mark
everything it reaches as safe, and then sweep the grid once: safe cells become "O", and everything else
becomes "X". The border-first move works whenever the property is "escapes to the outside".

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def solve_surrounded(board):
    if not board:
        return board
    rows, cols = len(board), len(board[0])
    border = [(r, c) for r in range(rows) for c in (0, cols - 1)]
    border += [(r, c) for c in range(cols) for r in (0, rows - 1)]
    stack = [(r, c) for r, c in border if board[r][c] == "O"]
    for r, c in stack:
        board[r][c] = "S"                        ## S = safe, reachable from the border
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == "O":
                board[nr][nc] = "S"
                stack.append((nr, nc))
    for r in range(rows):
        for c in range(cols):
            board[r][c] = "O" if board[r][c] == "S" else "X"   ## every other O was surrounded
    return board

## tests

b = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]
assert solve_surrounded(b) == [["X", "X", "X", "X"], ["X", "X", "X", "X"],
                               ["X", "X", "X", "X"], ["X", "O", "X", "X"]]
assert solve_surrounded([["O"]]) == [["O"]]
assert solve_surrounded([["X", "X"], ["X", "X"]]) == [["X", "X"], ["X", "X"]]
print(solve_surrounded([["X", "O", "X"], ["X", "O", "X"], ["X", "X", "X"]]))
```

```
[['X', 'O', 'X'], ['X', 'O', 'X'], ['X', 'X', 'X']]
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P6. Rotting oranges — minutes until no fresh orange remains, or -1 if one is unreachable

**Which template.** Multi-source BFS, seeded with every rotten orange.
**The trick.** This version carries the minute inside the queue entry as `(r, c, t)` instead of using
the level snapshot. Both are correct, and the reason is the same in each: the queue is non-decreasing
in `t`, so the last value popped is the largest distance reached. Use whichever you can write without
hesitating. Keep a `fresh` counter rather than rescanning the grid at the end, because the -1 case is
what the problem is really testing.

```python
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))          ## carry the minute in the queue entry
            elif grid[r][c] == 1:
                fresh += 1
    last = 0
    while queue:
        r, c, t = queue.popleft()
        last = t                                 ## the queue is non-decreasing in t
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, t + 1))
    return -1 if fresh > 0 else last

## tests

assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
assert oranges_rotting([[0, 2]]) == 0
assert oranges_rotting([[0]]) == 0
print(oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]))
```

```
4
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P7. 01 matrix — for each cell, the distance to the nearest 0

**Which template.** Multi-source BFS, seeded with every zero.
**The trick.** Running a BFS from each 1 to find the nearest 0 costs $O((rc)^2)$. Reverse it: start
from all the zeros at once and let the distances grow outwards. The first arrival at a cell is
necessarily the shortest distance to the closest zero, because BFS from a multi-source seed expands in
rings of equal distance. Use `dist[nr][nc] == -1` as the visited test, so one array does two jobs.

```python
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def update_matrix(mat):
    rows, cols = len(mat), len(mat[0])
    dist = [[-1] * cols for _ in range(rows)]
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dist[r][c] = 0
                queue.append((r, c))             ## seed EVERY zero, not one at a time
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1    ## first arrival is the nearest zero
                queue.append((nr, nc))
    return dist

## tests

assert update_matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
assert update_matrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]) == [[0, 0, 0], [0, 1, 0], [1, 2, 1]]
assert update_matrix([[0]]) == [[0]]
assert update_matrix([[0, 1, 1, 1]]) == [[0, 1, 2, 3]]
print(update_matrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]))
```

```
[[0, 0, 0], [0, 1, 0], [1, 2, 1]]
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P8. Walls and gates — fill each empty room with the distance to its nearest gate

**Which template.** Multi-source BFS again, seeded with every gate.
**The trick.** This is P7 with different labels, and you should say so. Gates are the zeros, walls are
-1 and are never entered, and empty rooms hold `INF`. Because only `INF` marks an unvisited empty room,
the value itself is the visited test and no separate set is needed. Recognising the reuse is the whole
answer.

```python
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = 2147483647

def walls_and_gates(rooms):
    if not rooms:
        return rooms
    rows, cols = len(rooms), len(rooms[0])
    queue = deque((r, c) for r in range(rows) for c in range(cols) if rooms[r][c] == 0)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == INF:
                rooms[nr][nc] = rooms[r][c] + 1  ## only INF cells are unvisited empty rooms
                queue.append((nr, nc))
    return rooms

## tests

grid = [[INF, -1, 0, INF], [INF, INF, INF, -1], [INF, -1, INF, -1], [0, -1, INF, INF]]
assert walls_and_gates(grid) == [[3, -1, 0, 1], [2, 2, 1, -1], [1, -1, 2, -1], [0, -1, 3, 4]]
assert walls_and_gates([[0]]) == [[0]]
assert walls_and_gates([[INF]]) == [[INF]]
assert walls_and_gates([]) == []
print(walls_and_gates([[INF, -1, 0], [INF, INF, INF]]))
```

```
[[4, -1, 0], [3, 2, 1]]
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P9. Pacific Atlantic water flow — cells from which water can reach both oceans

**Which template.** Template 3, run twice, from the two ocean borders.
**The trick.** Reverse the flow. Water runs downhill, so asking "which cells drain to the Pacific"
means tracing many paths forward from many starts. Instead, start at the ocean and walk **uphill**:
the neighbour test becomes `heights[nr][nc] >= heights[r][c]`. One such traversal from the Pacific
border marks every cell that drains to the Pacific, another from the Atlantic border does the same, and
the answer is the intersection of the two sets. Two traversals total, not one per cell.

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def pacific_atlantic(heights):
    if not heights:
        return []
    rows, cols = len(heights), len(heights[0])

    def reachable(starts):
        seen = set(starts)
        stack = list(starts)
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen \
                        and heights[nr][nc] >= heights[r][c]:   ## UPHILL: the flow reversed
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return seen

    pacific = [(r, 0) for r in range(rows)] + [(0, c) for c in range(cols)]
    atlantic = [(r, cols - 1) for r in range(rows)] + [(rows - 1, c) for c in range(cols)]
    return sorted(reachable(pacific) & reachable(atlantic))

## tests

h = [[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]
assert pacific_atlantic(h) == [(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)]
assert pacific_atlantic([[1]]) == [(0, 0)]
assert pacific_atlantic([[1, 1], [1, 1]]) == [(0, 0), (0, 1), (1, 0), (1, 1)]
print(pacific_atlantic(h))
```

```
[(0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0)]
```

**Complexity.** $O(rc)$ time, $O(rc)$ space.

### P10. Shortest path in binary matrix — fewest cells on a clear 8-directional path from corner to corner

**Which template.** Template 2, BFS on a grid with an eight-entry direction list.
**The trick.** The only difference from a four-direction BFS is `DIRS8`. It is worth doing because it
shows that "the edges" are a modelling decision, not a property of the grid. Note the answer counts
cells, not steps, so the level counter starts at 1, and a one-by-one clear grid answers 1. Check both
corners before starting, because a blocked start makes the whole thing -1.

```python
from collections import deque

DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] != 0 or grid[n - 1][n - 1] != 0:
        return -1
    queue = deque([(0, 0)])
    grid[0][0] = 1                               ## 1 marks visited as well as blocked
    length = 1
    while queue:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            if r == n - 1 and c == n - 1:
                return length
            for dr, dc in DIRS8:                 ## eight neighbours, not four
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    queue.append((nr, nc))
        length += 1
    return -1

## tests

assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1
assert shortest_path_binary_matrix([[0]]) == 1
print(shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]))
```

```
4
```

**Complexity.** $O(n^2)$ time, $O(n^2)$ space.

### P11. Word ladder — fewest transformations from `begin_word` to `end_word`, one letter at a time

**Which template.** Template 2, BFS, where the graph is never built explicitly.
**The trick.** The edges are implicit. Comparing every pair of words to find the neighbours costs
$O(N^2 L)$. Instead, generate the neighbours of a word by trying all 26 letters in each of its `L`
positions and keeping the results that are in the word set, which costs $O(26L)$ per word. Whenever the
node set is huge but the neighbour rule is cheap, generate the neighbours rather than storing the
graph. Put the word set in a `set` for $O(1)$ membership, and mark words visited on push so the same
word is not queued by two different predecessors.

```python
from collections import deque

def ladder_length(begin_word, end_word, word_list):
    words = set(word_list)
    if end_word not in words:
        return 0
    queue = deque([begin_word])
    visited = {begin_word}
    steps = 1
    while queue:
        for _ in range(len(queue)):
            word = queue.popleft()
            if word == end_word:
                return steps
            for i in range(len(word)):
                for ch in "abcdefghijklmnopqrstuvwxyz":
                    nxt = word[:i] + ch + word[i + 1:]     ## generate neighbours, do not compare pairs
                    if nxt in words and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        steps += 1
    return 0

## tests

assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
assert ladder_length("a", "c", ["a", "b", "c"]) == 2
assert ladder_length("hot", "hot", ["hot"]) == 1
print(ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]))
```

```
5
```

**Complexity.** $O(N \cdot 26 L^2)$ time with `L` the word length, because slicing to build each
candidate costs $O(L)$, and $O(NL)$ space.

### P12. Open the lock — fewest single-dial turns from "0000" to `target`, avoiding the deadends

**Which template.** Template 2, BFS over a state space.
**The trick.** A node need not be a physical thing. Here the node is the string on the dials, the
graph has 10000 nodes, and each has eight neighbours, being one dial turned one step in either
direction. Deadends are simply nodes you refuse to enter, which is why they go in the same test as the
visited set. Once you say "a node is a state and an edge is a legal move", the code is template 2
unchanged.

```python
from collections import deque

def open_lock(deadends, target):
    dead = set(deadends)
    if "0000" in dead:
        return -1
    queue = deque(["0000"])
    visited = {"0000"}
    steps = 0
    while queue:
        for _ in range(len(queue)):
            state = queue.popleft()
            if state == target:
                return steps
            for i in range(4):
                digit = int(state[i])
                for move in (1, -1):
                    nxt = state[:i] + str((digit + move) % 10) + state[i + 1:]
                    if nxt not in dead and nxt not in visited:   ## a node is a STATE, not a place
                        visited.add(nxt)
                        queue.append(nxt)
        steps += 1
    return -1

## tests

assert open_lock(["0201", "0101", "0102", "1212", "2002"], "0202") == 6
assert open_lock(["8888"], "0009") == 1
assert open_lock(["0000"], "8888") == -1
assert open_lock([], "0000") == 0
print(open_lock(["0201", "0101", "0102", "1212", "2002"], "0202"))
```

```
6
```

**Complexity.** $O(10^4 \cdot 8)$ time, $O(10^4)$ space, both constant in the input size.

### P13. Clone graph — return a deep copy of a connected undirected graph given one node

**Which template.** Template 3, with a dict from old node to new node in place of the visited set.
**The trick.** One dict does both jobs. `copies[old]` being present means "already visited", and its
value is the copy to link to. Create the copy of a neighbour at the moment you first see it, before you
push it, so that the edge from the current node can point at a real object immediately. If you create
the copies first and wire the edges in a second pass you also get a correct answer, but the one-pass
version is shorter and the cycles take care of themselves.

```python
class Node:
    def __init__(self, val, neighbours=None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else []

def clone_graph(node):
    if node is None:
        return None
    copies = {node: Node(node.val)}              ## old node -> new node, the visited set too
    stack = [node]
    while stack:
        old = stack.pop()
        for nb in old.neighbours:
            if nb not in copies:
                copies[nb] = Node(nb.val)        ## create the copy BEFORE recursing into it
                stack.append(nb)
            copies[old].neighbours.append(copies[nb])
    return copies[node]

## tests

a, b, c = Node(1), Node(2), Node(3)
a.neighbours, b.neighbours, c.neighbours = [b, c], [a, c], [a, b]
copy_a = clone_graph(a)
assert copy_a is not a and copy_a.val == 1
assert sorted(n.val for n in copy_a.neighbours) == [2, 3]
assert all(n is not b and n is not c for n in copy_a.neighbours)
assert copy_a.neighbours[0].neighbours[0] is copy_a   ## the cycle back to a is preserved
assert clone_graph(None) is None
print(copy_a.val, sorted(n.val for n in copy_a.neighbours))
```

```
1 [2, 3]
```

**Complexity.** $O(V + E)$ time, $O(V)$ space.

### P14. Course schedule — can every course be finished given the prerequisite pairs

**Which template.** Template 4, Kahn's algorithm, used only for its leftover count.
**The trick.** Get the edge direction right. The pair `[course, prereq]` means the course needs the
prerequisite, so the edge runs `prereq -> course`, because the prerequisite is what becomes available
first. Reversing it produces a graph that is still acyclic exactly when the original is, so the
boolean answer survives the mistake and the ordering in P15 does not. Fix the direction now, while it
costs nothing.

```python
from collections import deque

def can_finish(num_courses, prerequisites):
    adj = {i: [] for i in range(num_courses)}
    indegree = [0] * num_courses
    for course, prereq in prerequisites:         ## the pair reads "course needs prereq"
        adj[prereq].append(course)               ## so the EDGE goes prereq -> course
        indegree[course] += 1
    queue = deque(i for i in range(num_courses) if indegree[i] == 0)
    taken = 0
    while queue:
        node = queue.popleft()
        taken += 1
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return taken == num_courses                  ## leftover courses form a cycle

## tests

assert can_finish(2, [[1, 0]]) is True
assert can_finish(2, [[1, 0], [0, 1]]) is False
assert can_finish(3, []) is True
assert can_finish(4, [[1, 0], [2, 1], [3, 2], [1, 3]]) is False
print(can_finish(2, [[1, 0]]), can_finish(2, [[1, 0], [0, 1]]))
```

```
True False
```

**Complexity.** $O(V + E)$ time, $O(V + E)$ space.

### P15. Course schedule II — return a valid order in which to take the courses, or an empty list

**Which template.** Template 4 again, this time keeping the order.
**The trick.** The code is P14 with one line added: append the node when you pop it. That is worth
saying out loud, because it shows that cycle detection and ordering are the same computation read two
ways. A node leaves the queue only when its in-degree has fallen to zero, which means every
prerequisite has already been appended, so the order is valid by construction. Any node still holding
a positive in-degree at the end is on a cycle, and the answer is the empty list.

```python
from collections import deque

def find_order(num_courses, prerequisites):
    adj = {i: [] for i in range(num_courses)}
    indegree = [0] * num_courses
    for course, prereq in prerequisites:
        adj[prereq].append(course)
        indegree[course] += 1
    queue = deque(i for i in range(num_courses) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)                       ## the ONLY change from P13: keep the node
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == num_courses else []

## tests

assert find_order(2, [[1, 0]]) == [0, 1]
assert find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) == [0, 1, 2, 3]
assert find_order(2, [[1, 0], [0, 1]]) == []
assert find_order(1, []) == [0]
print(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]), find_order(2, [[1, 0], [0, 1]]))
```

```
[0, 1, 2, 3] []
```

**Complexity.** $O(V + E)$ time, $O(V + E)$ space.

### P16. Alien dictionary — recover the letter order from a list of words sorted in an unknown alphabet

**Which template.** Template 4 on edges you must infer first.
**The trick.** Two things, and the second is the one people miss. First, each adjacent pair of words
gives you exactly **one** edge: scan them together and stop at the first position where they differ,
because everything after that position tells you nothing. Second, the invalid-prefix case. If the two
words agree on every position up to the length of the shorter one, and the first word is the longer of
the two, then the input is impossible in any alphabet, because a prefix always sorts before the word
that extends it. Return the empty string there. A `for ... else` handles it cleanly: the `else` runs
only when the loop finished without breaking, which is exactly the all-equal case.

```python
from collections import deque

def alien_order(words):
    adj = {ch: [] for word in words for ch in word}
    indegree = {ch: 0 for ch in adj}
    seen_edges = set()
    for first, second in zip(words, words[1:]):
        for a, b in zip(first, second):
            if a != b:
                if (a, b) not in seen_edges:
                    seen_edges.add((a, b))       ## the FIRST difference gives the one edge
                    adj[a].append(b)
                    indegree[b] += 1
                break
        else:
            if len(first) > len(second):
                return ""                        ## "abc" before "ab" is impossible
    queue = deque(ch for ch in indegree if indegree[ch] == 0)
    order = []
    while queue:
        ch = queue.popleft()
        order.append(ch)
        for nxt in adj[ch]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return "".join(order) if len(order) == len(indegree) else ""

## tests

assert set(alien_order(["wrt", "wrf", "er", "ett", "rftt"])) == set("wertf")
assert alien_order(["wrt", "wrf", "er", "ett", "rftt"]) == "wertf"
assert alien_order(["z", "x", "z"]) == ""
assert alien_order(["abc", "ab"]) == ""          ## the invalid-prefix case
assert alien_order(["z", "x"]) == "zx"
print(alien_order(["wrt", "wrf", "er", "ett", "rftt"]), repr(alien_order(["abc", "ab"])))
```

```
wertf ''
```

**Complexity.** $O(C)$ time where `C` is the total number of characters, and $O(1)$ space for a fixed
alphabet.

### P17. Detect a cycle in a directed graph — is there any directed cycle

**Which template.** DFS with three colours, written iteratively.
**The trick.** Two marks are not enough. A plain visited set cannot tell "this node is on the path I
am currently walking" from "this node was fully explored earlier by another branch", and only the first
of those is a cycle. So use three colours: white is untouched, grey is on the current path, and black
is finished. An edge into a grey node is a back edge and proves a cycle. An edge into a black node is
a cross edge into a region already known to be cycle-free, and it proves nothing — which is why a
diamond, where two paths rejoin, is not a cycle. Keeping an iterator per stack frame is what lets the
iterative form know when a node is finished and can turn black.

```python
WHITE, GREY, BLACK = 0, 1, 2

def has_cycle_directed(n, edges):
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
    colour = [WHITE] * n
    for start in range(n):
        if colour[start] != WHITE:
            continue
        stack = [(start, iter(adj[start]))]
        colour[start] = GREY                     ## GREY = on the current path
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                colour[node] = BLACK             ## finished: it can never close a cycle again
                stack.pop()
            elif colour[nxt] == GREY:
                return True                      ## a back edge into the current path
            elif colour[nxt] == WHITE:
                colour[nxt] = GREY
                stack.append((nxt, iter(adj[nxt])))
    return False

## tests

assert has_cycle_directed(3, [(0, 1), (1, 2)]) is False
assert has_cycle_directed(3, [(0, 1), (1, 2), (2, 0)]) is True
assert has_cycle_directed(4, [(0, 1), (0, 2), (1, 3), (2, 3)]) is False   ## a diamond is not a cycle
assert has_cycle_directed(1, [(0, 0)]) is True
print(has_cycle_directed(3, [(0, 1), (1, 2), (2, 0)]),
      has_cycle_directed(4, [(0, 1), (0, 2), (1, 3), (2, 3)]))
```

```
True False
```

**Complexity.** $O(V + E)$ time, $O(V)$ space.

### P18. Number of connected components in an undirected graph — how many separate pieces

**Which template.** Template 6, in its simplest possible use.
**The trick.** Start the counter at `n`, with every node its own component, and decrement once per
successful union. A union fails when the two ends already share a root, which means the edge is inside
a component and changes nothing. DFS from every unvisited node also works and costs the same, so pick
union-find here only when the edges arrive over time and the count must be reported as they do.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True

def count_components(n, edges):
    dsu = DSU(n)
    for u, v in edges:
        dsu.union(u, v)                          ## each successful union merges two components
    return dsu.components

## tests

assert count_components(5, [(0, 1), (1, 2), (3, 4)]) == 2
assert count_components(5, [(0, 1), (1, 2), (2, 3), (3, 4)]) == 1
assert count_components(3, []) == 3
assert count_components(1, []) == 1
print(count_components(5, [(0, 1), (1, 2), (3, 4)]))
```

```
2
```

**Complexity.** $O(E \cdot \alpha(n))$ time, $O(n)$ space.

### P19. Graph valid tree — do `n` nodes and these edges form a tree

**Which template.** Template 6, with a counting check in front of it.
**The trick.** A tree is a connected acyclic graph, and on `n` nodes it has exactly `n - 1` edges. That
count is the cheap half of the test, so check it first and return early. Given `n - 1` edges, connected
and acyclic are the same condition, so you only have to verify one of them: run the unions, and if any
union fails there is a cycle. If none fails, `n - 1` successful merges have reduced `n` components to
one, so the graph is connected as well. The edge count is what lets one check cover both properties.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True

def valid_tree(n, edges):
    if len(edges) != n - 1:
        return False                             ## a tree on n nodes has exactly n-1 edges
    dsu = DSU(n)
    for u, v in edges:
        if not dsu.union(u, v):
            return False                         ## this edge closed a cycle
    return True                                  ## n-1 edges and no cycle implies connected

## tests

assert valid_tree(5, [(0, 1), (0, 2), (0, 3), (1, 4)]) is True
assert valid_tree(5, [(0, 1), (1, 2), (2, 3), (1, 3), (1, 4)]) is False
assert valid_tree(4, [(0, 1), (2, 3)]) is False
assert valid_tree(1, []) is True
print(valid_tree(5, [(0, 1), (0, 2), (0, 3), (1, 4)]), valid_tree(4, [(0, 1), (2, 3)]))
```

```
True False
```

**Complexity.** $O(n \cdot \alpha(n))$ time, $O(n)$ space.

### P20. Redundant connection — find the one edge whose removal leaves a tree

**Which template.** Template 6, returning at the first failed union.
**The trick.** The input is a tree plus exactly one extra edge. Process the edges in the given order
and the first edge whose two ends are already connected is the extra one, because every earlier edge
was needed to connect something new. The problem asks for the last such edge in the input order, and
scanning forwards and returning at the first failure gives exactly that, since there is only one.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True

def find_redundant_connection(edges):
    dsu = DSU(len(edges) + 1)                    ## nodes are 1..n, so allocate n+1 slots
    for u, v in edges:
        if not dsu.union(u, v):
            return [u, v]                        ## both ends already joined: this edge is the extra
    return []

## tests

assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
assert find_redundant_connection([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]) == [1, 4]
assert find_redundant_connection([[1, 2], [2, 3], [1, 3]]) == [1, 3]
print(find_redundant_connection([[1, 2], [1, 3], [2, 3]]),
      find_redundant_connection([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]))
```

```
[2, 3] [1, 4]
```

**Complexity.** $O(n \cdot \alpha(n))$ time, $O(n)$ space.

### P21. Accounts merge — group accounts that share at least one email address

**Which template.** Template 6, with a dict mapping each email to the account that first claimed it.
**The trick.** Union the accounts, not the emails. Each account is one node, indexed by its position in
the input. Walk the emails of each account: if an email has been seen before, union this account with
the one that owns it; otherwise record this account as the owner. Every shared email therefore merges
two account nodes. At the end, group the emails by the root of their owner and prepend the name, which
you can read from any account in the group because they all carry the same one. Trying to make each
email a node also works, but you then need a second map to recover the name, and the merge logic is
harder to state.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True

def accounts_merge(accounts):
    dsu = DSU(len(accounts))
    owner_of_email = {}
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in owner_of_email:
                dsu.union(i, owner_of_email[email])   ## a shared email merges two accounts
            else:
                owner_of_email[email] = i
    groups = {}
    for email, i in owner_of_email.items():
        groups.setdefault(dsu.find(i), set()).add(email)
    return sorted([accounts[root][0]] + sorted(emails) for root, emails in groups.items())

## tests

accounts = [["John", "a@x.com", "b@x.com"], ["John", "c@x.com"],
            ["John", "a@x.com", "c@x.com"], ["Mary", "m@x.com"]]
assert accounts_merge(accounts) == [["John", "a@x.com", "b@x.com", "c@x.com"], ["Mary", "m@x.com"]]
assert accounts_merge([["A", "p@x.com"]]) == [["A", "p@x.com"]]
assert len(accounts_merge([["A", "p@x.com"], ["B", "q@x.com"]])) == 2
print(accounts_merge(accounts))
```

```
[['John', 'a@x.com', 'b@x.com', 'c@x.com'], ['Mary', 'm@x.com']]
```

**Complexity.** $O(N L \log(N L))$ time, dominated by sorting the emails, with `N` accounts of `L`
emails, and $O(N L)$ space.

### P22. Evaluate division — answer queries `a / b` given a list of equations `a / b = value`

**Which template.** Template 3, on a weighted undirected graph, multiplying along the path.
**The trick.** The node is a variable and the edge weight is a ratio. An equation `a / b = 2.0` gives
an edge `a -> b` of weight 2.0 and the reverse edge `b -> a` of weight 0.5, because dividing runs
backwards. Then `a / c` is the product of the weights along any path from `a` to `c`, and any path
gives the same product when the input is consistent. Carry the running product in the stack entry.
Answer -1.0 when either variable is unknown or no path exists — and note that `a / a` is 1.0 only if
`a` appears somewhere in the equations.

```python
def calc_equation(equations, values, queries):
    adj = {}
    for (a, b), value in zip(equations, values):
        adj.setdefault(a, []).append((b, value))      ## a / b = value
        adj.setdefault(b, []).append((a, 1.0 / value))
    out = []
    for a, b in queries:
        if a not in adj or b not in adj:
            out.append(-1.0)
            continue
        stack, seen, answer = [(a, 1.0)], {a}, -1.0
        while stack:
            node, product = stack.pop()
            if node == b:
                answer = product                      ## the path product is the ratio
                break
            for nxt, weight in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, product * weight))
        out.append(answer)
    return out

## tests

eq, va = [["a", "b"], ["b", "c"]], [2.0, 3.0]
assert calc_equation(eq, va, [["a", "c"]]) == [6.0]
assert calc_equation(eq, va, [["c", "a"]]) == [1 / 6]
assert calc_equation(eq, va, [["a", "a"]]) == [1.0]
assert calc_equation(eq, va, [["x", "a"], ["a", "x"]]) == [-1.0, -1.0]
print(calc_equation(eq, va, [["a", "c"], ["b", "a"], ["a", "e"], ["x", "x"]]))
```

```
[6.0, 0.5, -1.0, -1.0]
```

**Complexity.** $O(Q(V + E))$ time for `Q` queries, $O(V + E)$ space.

### P23. Network delay time — how long until a signal from one node reaches every node

**Which template.** Template 5, Dijkstra with a heap.
**The trick.** The answer is the **maximum** of the shortest distances, not their sum and not the
distance to any one node, because the network is done only when the last node has heard the signal. If
any node is missing from `dist` at the end, it is unreachable and the answer is -1. This version uses
`if u in dist: continue` as the stale-entry skip, which is the same idea as `if d > dist[u]: continue`
written with a dict — a node's first pop is its final distance, so any later pop of the same node is
stale.

```python
import heapq

def network_delay_time(times, n, source):
    adj = {i: [] for i in range(1, n + 1)}       ## nodes are labelled 1..n
    for u, v, w in times:
        adj[u].append((v, w))
    dist = {}
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in dist:                            ## already finalised: a stale heap entry
            continue
        dist[u] = d
        for v, w in adj[u]:
            if v not in dist:
                heapq.heappush(heap, (d + w, v))
    if len(dist) < n:
        return -1                                ## some node is unreachable
    return max(dist.values())                    ## the signal ends when the LAST node hears it

## tests

assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
assert network_delay_time([[1, 2, 1]], 2, 1) == 1
assert network_delay_time([[1, 2, 1]], 2, 2) == -1
assert network_delay_time([], 1, 1) == 0
print(network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2))
```

```
2
```

**Complexity.** $O(E \log V)$ time, $O(V + E)$ space.

### P24. Cheapest flights within k stops — cheapest route from source to target using at most `k` stops

**Which template.** Bellman-Ford, run exactly `k + 1` times. Plain Dijkstra is wrong here.
**The trick.** Dijkstra is wrong because it settles each node at its cheapest distance and never
reconsiders it, but the cheapest route to a node may use too many hops while a dearer route to the same
node stays within budget and leads on to a better total. The node's state is not its distance alone; it
is the pair (distance, hops used), and Dijkstra only tracks the first half. Bellman-Ford is the right
shape because it counts rounds, and one round relaxes every edge once, so after `i` rounds `dist[v]`
holds the cheapest route using at most `i` edges. At most `k` stops means at most `k + 1` edges, so run
`k + 1` rounds. The line `previous = dist[:]` is what makes that true: without the copy, an edge
relaxed earlier in the same round could be used again later in that round, and a single round would
extend a path by two or more edges.

```python
def find_cheapest_price(n, flights, source, target, k):
    INF = float("inf")
    dist = [INF] * n
    dist[source] = 0
    for _ in range(k + 1):                       ## k stops means at most k+1 edges
        previous = dist[:]                       ## THE key line: relax from the previous round only
        for u, v, w in flights:
            if previous[u] + w < dist[v]:
                dist[v] = previous[u] + w
    return -1 if dist[target] == INF else dist[target]

## tests

flights = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
assert find_cheapest_price(3, flights, 0, 2, 1) == 200
assert find_cheapest_price(3, flights, 0, 2, 0) == 500
assert find_cheapest_price(3, [[0, 1, 2], [1, 2, 1], [0, 2, 10]], 0, 2, 1) == 3
assert find_cheapest_price(2, [], 0, 1, 5) == -1
print(find_cheapest_price(3, flights, 0, 2, 1), find_cheapest_price(3, flights, 0, 2, 0))
```

```
200 500
```

**Complexity.** $O(k \cdot E)$ time, $O(V)$ space.

### P25. Minimum spanning tree — cheapest set of edges connecting every node, by Kruskal and by Prim

**Which template.** Template 6 for Kruskal, and a heap for Prim. Both are here so you can pick one.
**The trick.** Kruskal sorts every edge by weight and keeps an edge only when its ends are in different
components, which is exactly what a failed union tells you. Prim grows one tree outwards, always taking
the cheapest edge that leaves the tree, and the heap supplies that edge. Kruskal is easier to write when
you already have the edge list and a DSU, so it is the default; Prim is better on a dense graph given
as an adjacency list, because it never sorts all $O(V^2)$ edges. Both must check at the end that they
covered every node, because a disconnected graph has no spanning tree at all.

```python
class DSU:
    def __init__(self, n):
        self.parent, self.size, self.components = list(range(n)), [1] * n, n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb], self.size[ra] = ra, self.size[ra] + self.size[rb]
        self.components -= 1
        return True
import heapq

def mst_kruskal(n, edges):
    dsu = DSU(n)
    total = 0
    for w, u, v in sorted(edges):                ## cheapest edge first
        if dsu.union(u, v):                      ## keep it only if it joins two components
            total += w
    return total if dsu.components == 1 else -1

def mst_prim(n, edges):
    adj = {i: [] for i in range(n)}
    for w, u, v in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    in_tree, total, heap = set(), 0, [(0, 0)]    ## start anywhere; the first edge costs 0
    while heap:
        w, u = heapq.heappop(heap)
        if u in in_tree:
            continue
        in_tree.add(u)
        total += w
        for w2, v in adj[u]:
            if v not in in_tree:
                heapq.heappush(heap, (w2, v))
    return total if len(in_tree) == n else -1

## tests

edges = [(1, 0, 1), (4, 0, 2), (3, 1, 2), (2, 1, 3), (5, 2, 3)]
assert mst_kruskal(4, edges) == 6
assert mst_prim(4, edges) == 6
assert mst_kruskal(3, [(1, 0, 1)]) == -1
assert mst_prim(3, [(1, 0, 1)]) == -1
assert mst_kruskal(1, []) == 0 and mst_prim(1, []) == 0
print(mst_kruskal(4, edges), mst_prim(4, edges))
```

```
6 6
```

**Complexity.** Kruskal is $O(E \log E)$ time, dominated by the sort. Prim is $O(E \log V)$ time. Both
use $O(V + E)$ space.

## Tricks and tips

**Mark visited on push, not on pop, when using a queue.** In BFS a node can be reached from several
nodes in the same level. If you mark it only when you pop it, it is queued several times, the queue
grows past $O(V)$, and on a dense graph the run degrades badly. In the iterative DFS the opposite
convention is normal — you mark on pop, and skip a node that is already marked — because there you want
the stack to hold the path frontier. Know which one you are writing and say it.

**Reverse the graph when the question points the wrong way.** Pacific Atlantic runs uphill from the
ocean instead of downhill from every cell. Surrounded Regions starts at the border instead of testing
each region. 01 Matrix starts at the zeros instead of at the ones. In each case one traversal from the
answer's side replaces one traversal per query. When a problem reads "for every X, find the nearest Y",
try starting from all the Y at once.

**Convert grid coordinates to integers when you need union-find.** The index `r * cols + c` is the
standard flattening, and `divmod(index, cols)` recovers the pair. DSU wants integer nodes, so this is
the bridge between the grid disguise and template 6.

**Get the edge direction right in dependency problems, and write down which way you chose.** "b
depends on a" is an edge `a -> b` in the graph you topologically sort, because `a` must be emitted
first. Half of all Course Schedule bugs are the reversed edge, and the boolean version hides the
mistake while the ordering version exposes it.

**Kahn's algorithm answers three questions with one run.** The order itself; whether a cycle exists,
from `len(order) < n`; and whether the order is unique, from whether the queue ever held more than one
node at a time. That last one is asked as a follow-up more often than people expect, and it costs one
extra line.

**Use Dijkstra only with non-negative weights.** A negative edge breaks the settled-node invariant,
because a node popped as final can later be improved through a negative edge. With negative weights use
Bellman-Ford, which is $O(VE)$ and also detects a negative cycle: run one extra round, and if anything
still improves, a negative cycle exists.

**When a constraint is not a distance, put it in the state.** Cheapest Flights adds a hop budget, so
the node becomes (city, hops). Problems with keys and doors add a key set, so the node becomes
(cell, keys) with the keys as a bitmask. This is the general escape hatch when a plain shortest-path
algorithm gives a wrong answer: enlarge the node until the algorithm's assumption holds again.

**For an undirected cycle check, track the parent, not just visited.** Every undirected edge appears
in both adjacency lists, so the edge you arrived by will always look like a cycle unless you skip the
one neighbour you came from. Union-find avoids the problem entirely and is easier to get right under
pressure.

## The bugs that cost the round

**Forgetting the bounds check, or writing it after the value check.** In Python `grid[-1][c]` is a
legal read of the last row, so a missing or late bounds check does not crash. It silently wraps the
search around the grid and returns an answer that is wrong on inputs where an island touches the top
and bottom edges. Write `0 <= nr < rows and 0 <= nc < cols` first, always, and let `and` short-circuit.

**Marking visited at the wrong moment in BFS.** Mark on push. If you mark on pop, a cell adjacent to
two cells in the same level is enqueued twice, and in Rotting Oranges or Word Ladder that turns a
linear pass into something much worse and can double-count a level.

**Reversing the edges in a dependency graph.** The pair `[a, b]` almost always means "a needs b", so
the edge is `b -> a`. Say the direction out loud before you write the loop.

**Using DFS for a shortest path.** DFS finds *a* path, not the shortest one, and no amount of
backtracking makes it BFS. If the question says "fewest" or "minimum steps" and the graph is
unweighted, it is BFS.

**Running one BFS per source.** The multi-source seed is the difference between $O(rc)$ and
$O(S \cdot rc)$, and the fix is one loop before the main loop.

**Hitting the recursion limit.** A recursive DFS over a 200000-cell grid or a long path graph raises
`RecursionError`. The iterative form is only three lines longer, so write that one when the input can
be large, and mention the limit rather than waiting to be asked.

**Forgetting isolated nodes.** Building the adjacency list from the edge list alone loses every node
with no edges, so the component count comes out too small and a topological sort silently drops them.
Initialise the dict with all `n` nodes first.

**Off-by-one in the hop budget.** "At most `k` stops" means at most `k + 1` flights. Read the sentence
twice and write the conversion down.

## Done when

- Given a problem statement you have not seen, you can say what a node is, what an edge is, and which
  of the five algorithms applies, in under 30 seconds and before writing any code.
- You can write BFS with a level count, iterative DFS, and Kahn's topological sort from a blank file
  in five minutes each, with the visited marking in the right place in all three.
- You can write union-find with path compression and union by size from memory, and use its three
  return values — the root, the component count, and the `False` from a redundant union.
- You can explain why plain Dijkstra fails on Cheapest Flights Within K Stops, and enlarge the node to
  fix it.
