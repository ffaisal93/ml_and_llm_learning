# NeetCode 150 — Graphs + Advanced Graphs — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips.

Reusable templates:

```python
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
```

---

## Graphs (13)

### 1. Number of Islands (LC 200)

**Intuition.** DFS each '1' cell, mark visited as '0' (or use a set).

```python
def numIslands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    def dfs(r, c):
        if not (0 <= r < R and 0 <= c < C) or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        for dr, dc in DIRS: dfs(r + dr, c + dc)
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count
```

**Complexity.** $O(RC)$.

**Hook.** "DFS sink each island."

---

### 2. Max Area of Island (LC 695)

**Intuition.** Same as above, but DFS returns the size of the connected component.

```python
def maxAreaOfIsland(grid):
    R, C = len(grid), len(grid[0])
    def dfs(r, c):
        if not (0 <= r < R and 0 <= c < C) or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + sum(dfs(r + dr, c + dc) for dr, dc in DIRS)
    return max((dfs(r, c) for r in range(R) for c in range(C)), default=0)
```

**Complexity.** $O(RC)$.

**Hook.** "DFS returns size."

---

### 3. Clone Graph (LC 133)

**Intuition.** DFS or BFS with old→clone hash map.

```python
def cloneGraph(node):
    if not node: return None
    old_to_new = {}
    def dfs(n):
        if n in old_to_new: return old_to_new[n]
        copy = Node(n.val)
        old_to_new[n] = copy
        for nei in n.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node)
```

**Complexity.** $O(V + E)$.

**Hook.** "Hash old→new + recurse."

---

### 4. Walls and Gates (LC 286)

**Problem.** Fill each empty cell with distance to nearest gate.

**Intuition.** Multi-source BFS from all gates simultaneously. First time you reach a cell is shortest.

```python
from collections import deque
def wallsAndGates(rooms):
    if not rooms: return
    R, C = len(rooms), len(rooms[0])
    q = deque()
    for r in range(R):
        for c in range(C):
            if rooms[r][c] == 0:
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and rooms[nr][nc] == 2147483647:
                rooms[nr][nc] = rooms[r][c] + 1
                q.append((nr, nc))
```

**Complexity.** $O(RC)$.

**Hook.** "Push all gates first; BFS one wave."

---

### 5. Rotting Oranges (LC 994)

**Intuition.** Multi-source BFS from all rotten oranges. Track minutes via levels. After BFS, check for unreachable fresh oranges.

```python
from collections import deque
def orangesRotting(grid):
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2: q.append((r, c, 0))
            elif grid[r][c] == 1: fresh += 1
    minutes = 0
    while q:
        r, c, t = q.popleft()
        minutes = t
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                q.append((nr, nc, t + 1))
    return minutes if fresh == 0 else -1
```

**Complexity.** $O(RC)$.

**Hook.** "Multi-source BFS, count fresh remaining."

---

### 6. Pacific Atlantic Water Flow (LC 417)

**Problem.** Cells from which water can flow to *both* oceans (top/left = Pacific; bottom/right = Atlantic).

**Intuition.** Reverse the flow — start from each ocean's edge cells, climb to higher cells. Cells reached by both are answers.

```python
def pacificAtlantic(heights):
    R, C = len(heights), len(heights[0])
    pac, atl = set(), set()
    def dfs(r, c, visited, prev):
        if (r, c) in visited: return
        if not (0 <= r < R and 0 <= c < C): return
        if heights[r][c] < prev: return
        visited.add((r, c))
        for dr, dc in DIRS: dfs(r + dr, c + dc, visited, heights[r][c])
    for c in range(C):
        dfs(0, c, pac, heights[0][c])
        dfs(R - 1, c, atl, heights[R - 1][c])
    for r in range(R):
        dfs(r, 0, pac, heights[r][0])
        dfs(r, C - 1, atl, heights[r][C - 1])
    return list(pac & atl)
```

**Complexity.** $O(RC)$.

**Hook.** "Reverse the flow — climb up from both oceans."

---

### 7. Surrounded Regions (LC 130)

**Intuition.** O's connected to the border can't be flipped. DFS from border O's, mark "safe." Then flip all unsafe O's.

```python
def solve(board):
    R, C = len(board), len(board[0])
    def dfs(r, c):
        if not (0 <= r < R and 0 <= c < C) or board[r][c] != 'O': return
        board[r][c] = 'S'
        for dr, dc in DIRS: dfs(r + dr, c + dc)
    for r in range(R):
        dfs(r, 0); dfs(r, C - 1)
    for c in range(C):
        dfs(0, c); dfs(R - 1, c)
    for r in range(R):
        for c in range(C):
            board[r][c] = 'O' if board[r][c] == 'S' else 'X'
```

**Complexity.** $O(RC)$.

**Hook.** "Mark border-connected O's, then flip the rest."

---

### 8. Course Schedule (LC 207)

**Problem.** Can finish all courses (i.e., is the prereq graph acyclic)?

**Intuition.** Topological sort via DFS with three-state coloring (unvisited / visiting / done) to detect cycles.

```python
def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[a].append(b)
    state = [0] * numCourses    # 0 unseen, 1 visiting, 2 done
    def dfs(u):
        if state[u] == 1: return False     # cycle
        if state[u] == 2: return True
        state[u] = 1
        for v in graph[u]:
            if not dfs(v): return False
        state[u] = 2
        return True
    return all(dfs(i) for i in range(numCourses))
```

**Complexity.** $O(V + E)$.

**Hook.** "3-color DFS; return False on gray (visiting)."

---

### 9. Course Schedule II (LC 210)

**Intuition.** Same as above, but also produce the order. Reverse postorder DFS gives topological order.

```python
def findOrder(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for a, b in prerequisites:
        graph[a].append(b)
    state = [0] * numCourses
    order = []
    def dfs(u):
        if state[u] == 1: return False
        if state[u] == 2: return True
        state[u] = 1
        for v in graph[u]:
            if not dfs(v): return False
        state[u] = 2
        order.append(u)
        return True
    for i in range(numCourses):
        if not dfs(i): return []
    return order   # already in correct order: a comes after its prereqs
```

**Complexity.** $O(V + E)$.

**Hook.** "Same DFS; append on done."

**Watch out.** Direction matters — here `(a, b)` means `b → a` (a depends on b). With `graph[a].append(b)`, postorder gives correct order.

---

### 10. Graph Valid Tree (LC 261)

**Problem.** Is the graph (n nodes, list of edges) a tree?

**Intuition.** Tree iff: (a) exactly n-1 edges; (b) connected; (c) no cycle. Union-Find: if union returns False (already connected), there's a cycle.

```python
def validTree(n, edges):
    if len(edges) != n - 1: return False
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        parent[ra] = rb
        return True
    for a, b in edges:
        if not union(a, b): return False
    return True
```

**Complexity.** $O(n \alpha(n)) \approx O(n)$.

**Hook.** "Edges = n-1, no union failure."

---

### 11. Number of Connected Components (LC 323)

**Intuition.** Union-Find; component count = n − successful unions.

```python
def countComponents(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    components = n
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            components -= 1
    return components
```

**Complexity.** $O(E \alpha(n))$.

**Hook.** "n minus successful unions."

---

### 12. Redundant Connection (LC 684)

**Problem.** Find the edge whose removal makes the graph a tree.

**Intuition.** Union-Find. The edge that fails to union (creates a cycle) is the redundant one.

```python
def findRedundantConnection(edges):
    n = len(edges)
    parent = list(range(n + 1))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb: return [a, b]
        parent[ra] = rb
```

**Complexity.** $O(n \alpha(n))$.

**Hook.** "First union failure = answer."

---

### 13. Word Ladder (LC 127)

**Problem.** Min transformations from `beginWord` to `endWord`, changing one letter at a time, each intermediate must be in word list.

**Intuition.** BFS on word graph. Trick: don't enumerate edges; for each position, try each letter substitution and check membership in set.

```python
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    words = set(wordList)
    if endWord not in words: return 0
    q = deque([(beginWord, 1)])
    visited = {beginWord}
    while q:
        word, d = q.popleft()
        if word == endWord: return d
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                w = word[:i] + c + word[i+1:]
                if w in words and w not in visited:
                    visited.add(w)
                    q.append((w, d + 1))
    return 0
```

**Complexity.** $O(N \cdot L^2 \cdot 26)$ worst case.

**Hook.** "BFS; substitute each position."

**Pro tip.** Bidirectional BFS halves the search; advanced optimization.

---

## Advanced Graphs (6)

### 14. Reconstruct Itinerary (LC 332)

**Problem.** Find Eulerian path through tickets (every edge used exactly once), lexicographically smallest.

**Intuition.** Hierholzer's algorithm. DFS exhausting destinations in sorted order; append to result on the way back; reverse at end.

```python
from collections import defaultdict
import heapq
def findItinerary(tickets):
    graph = defaultdict(list)
    for a, b in tickets:
        heapq.heappush(graph[a], b)
    out = []
    def dfs(u):
        while graph[u]:
            v = heapq.heappop(graph[u])
            dfs(v)
        out.append(u)
    dfs("JFK")
    return out[::-1]
```

**Complexity.** $O(E \log E)$.

**Hook.** "Hierholzer; append on return; reverse at end."

---

### 15. Min Cost to Connect All Points (LC 1584)

**Problem.** Min total Manhattan-distance to connect all points (MST).

**Intuition.** Prim's algorithm: heap of (distance, point); grow tree from arbitrary start.

```python
import heapq
def minCostConnectPoints(points):
    n = len(points)
    visited = [False] * n
    h = [(0, 0)]   # (cost, idx)
    total = 0
    count = 0
    while count < n:
        d, i = heapq.heappop(h)
        if visited[i]: continue
        visited[i] = True
        total += d
        count += 1
        x, y = points[i]
        for j in range(n):
            if not visited[j]:
                xj, yj = points[j]
                heapq.heappush(h, (abs(x - xj) + abs(y - yj), j))
    return total
```

**Complexity.** $O(n^2 \log n)$.

**Hook.** "Prim's; heap of (dist, idx)."

---

### 16. Network Delay Time (LC 743)

**Problem.** Time for signal from `k` to reach all nodes (or -1).

**Intuition.** Dijkstra single-source; answer is max of dist[].

```python
import heapq
from collections import defaultdict
def networkDelayTime(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    dist = {}
    h = [(0, k)]
    while h:
        d, u = heapq.heappop(h)
        if u in dist: continue
        dist[u] = d
        for v, w in graph[u]:
            if v not in dist:
                heapq.heappush(h, (d + w, v))
    return max(dist.values()) if len(dist) == n else -1
```

**Complexity.** $O((V + E) \log V)$.

**Hook.** "Dijkstra; answer = max(dist)."

---

### 17. Swim in Rising Water (LC 778)

**Problem.** Cell heights are water rise threshold. Min time to reach (n-1, n-1) from (0, 0). Move to a neighbor only when water level ≥ neighbor's height.

**Intuition.** Dijkstra where "distance" to a cell = max height seen on the path. Min-heap by max-along-path.

```python
import heapq
def swimInWater(grid):
    n = len(grid)
    visited = [[False] * n for _ in range(n)]
    h = [(grid[0][0], 0, 0)]   # (max-so-far, r, c)
    while h:
        t, r, c = heapq.heappop(h)
        if r == n - 1 and c == n - 1: return t
        if visited[r][c]: continue
        visited[r][c] = True
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                heapq.heappush(h, (max(t, grid[nr][nc]), nr, nc))
```

**Complexity.** $O(n^2 \log n)$.

**Hook.** "Dijkstra-like with max-along-path."

---

### 18. Alien Dictionary (LC 269)

**Problem.** Order of letters in an alien language given a list of words sorted in that language.

**Intuition.** Build precedence graph from adjacent word pairs (first differing chars). Topological sort.

```python
from collections import defaultdict, deque
def alienOrder(words):
    graph = {c: set() for w in words for c in w}
    indeg = {c: 0 for c in graph}
    for w1, w2 in zip(words, words[1:]):
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""    # invalid (longer prefix appears first)
        for a, b in zip(w1, w2):
            if a != b:
                if b not in graph[a]:
                    graph[a].add(b)
                    indeg[b] += 1
                break
    q = deque([c for c in indeg if indeg[c] == 0])
    out = []
    while q:
        c = q.popleft()
        out.append(c)
        for nei in graph[c]:
            indeg[nei] -= 1
            if indeg[nei] == 0: q.append(nei)
    return ''.join(out) if len(out) == len(graph) else ""
```

**Complexity.** $O(\sum |w|)$.

**Hook.** "Adjacent words → first differing char gives edge; Kahn's topo."

**Watch out.** Edge case: `["abc", "ab"]` is invalid — return "".

---

### 19. Cheapest Flights Within K Stops (LC 787)

**Problem.** Cheapest path from `src` to `dst` with at most `k` intermediate stops.

**Intuition.** Bellman-Ford-style: relax edges k+1 times, using a *snapshot* of distances each round so we don't use multiple edges from the current round.

```python
def findCheapestPrice(n, flights, src, dst, k):
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        new = dist[:]
        for u, v, w in flights:
            if dist[u] + w < new[v]:
                new[v] = dist[u] + w
        dist = new
    return -1 if dist[dst] == INF else dist[dst]
```

**Complexity.** $O(k \cdot E)$.

**Hook.** "Bellman-Ford with k+1 rounds and a copy each round."

**Watch out.** Using `dist` (not a snapshot) lets a single round use multiple edges → wrong answer.

---

## Summary table — Graphs + Advanced Graphs

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Number of Islands | DFS sink | $O(RC)$ | mark visited |
| 2 | Max Area Island | DFS returns size | $O(RC)$ | size sum |
| 3 | Clone Graph | DFS + map | $O(V+E)$ | old→new |
| 4 | Walls and Gates | multi-source BFS | $O(RC)$ | gates first |
| 5 | Rotting Oranges | multi-source BFS | $O(RC)$ | track fresh |
| 6 | Pacific Atlantic | reverse-flow DFS | $O(RC)$ | climb up from oceans |
| 7 | Surrounded Regions | mark border | $O(RC)$ | border-safe DFS |
| 8 | Course Schedule | topo / cycle | $O(V+E)$ | 3-color DFS |
| 9 | Course Schedule II | topo with order | $O(V+E)$ | append on done |
| 10 | Graph Valid Tree | union-find | $O(n)$ | n-1 edges, no cycle |
| 11 | Connected Components | union-find | $O(E)$ | n − successful unions |
| 12 | Redundant Connection | union-find | $O(n)$ | first cycle edge |
| 13 | Word Ladder | BFS w/ subs | $O(NL^2)$ | substitute each pos |
| 14 | Reconstruct Itinerary | Hierholzer | $O(E\log E)$ | append on return |
| 15 | Min Cost Connect | Prim MST | $O(n^2\log n)$ | (dist, idx) heap |
| 16 | Network Delay | Dijkstra | $O((V+E)\log V)$ | max(dist) |
| 17 | Swim in Rising Water | Dijkstra max-path | $O(n^2\log n)$ | max-along-path |
| 18 | Alien Dictionary | Topo from word pairs | $O(\sum|w|)$ | first diff char |
| 19 | Cheapest K Stops | Bellman-Ford k+1 | $O(kE)$ | snapshot per round |

## How to drill

1. Day 1: Read all 19.
2. Day 2-3: Re-implement from hooks alone.
3. Day 7: Repeat from scratch on a blank file.
4. Mock interview: shuffle and pick 3 random problems; aim to solve each in ≤ 25 min.
