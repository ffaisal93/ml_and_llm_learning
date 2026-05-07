# NeetCode 150 — Trees — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips.

Standard tree node:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right
```

---

## Trees (15)

### 1. Invert Binary Tree (LC 226)

**Intuition.** Swap children at every node, recursively.

```python
def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

**Complexity.** $O(n)$ time, $O(h)$ space.

**Hook.** "Swap left/right then recurse."

---

### 2. Maximum Depth of Binary Tree (LC 104)

```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

**Complexity.** $O(n)$ time, $O(h)$ space.

**Hook.** "1 + max(left, right)."

---

### 3. Diameter of Binary Tree (LC 543)

**Problem.** Length of longest path between any two nodes (count of edges).

**Intuition.** At each node, the longest path *through* it is `left_depth + right_depth`. Recurse returning depth, track global max.

```python
def diameterOfBinaryTree(root):
    diameter = 0
    def depth(node):
        nonlocal diameter
        if not node: return 0
        l = depth(node.left)
        r = depth(node.right)
        diameter = max(diameter, l + r)
        return 1 + max(l, r)
    depth(root)
    return diameter
```

**Complexity.** $O(n)$ time, $O(h)$ space.

**Hook.** "Update global at each node = left + right depths."

---

### 4. Balanced Binary Tree (LC 110)

**Intuition.** Sentinel pattern: return depth, or -1 if unbalanced.

```python
def isBalanced(root):
    def check(node):
        if not node: return 0
        l = check(node.left)
        if l == -1: return -1
        r = check(node.right)
        if r == -1 or abs(l - r) > 1: return -1
        return 1 + max(l, r)
    return check(root) != -1
```

**Complexity.** $O(n)$ time, $O(h)$ space.

**Hook.** "Return -1 to abort early."

---

### 5. Same Tree (LC 100)

```python
def isSameTree(p, q):
    if not p and not q: return True
    if not p or not q: return False
    if p.val != q.val: return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

**Complexity.** $O(\min(m, n))$.

**Hook.** "Both null OK; one null bad; values equal; recurse."

---

### 6. Subtree of Another Tree (LC 572)

**Intuition.** For each node in `root`, check if subtree starting there equals `subRoot`. Reuse Same Tree.

```python
def isSubtree(root, subRoot):
    if not subRoot: return True
    if not root: return False
    if isSameTree(root, subRoot): return True
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)
```

**Complexity.** $O(m \cdot n)$ worst case.

**Hook.** "Same-tree at every position."

---

### 7. Lowest Common Ancestor of BST (LC 235)

**Intuition.** Walk down. If both `p` and `q` are smaller than current, go left; if both larger, go right; otherwise this is LCA.

```python
def lowestCommonAncestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
```

**Complexity.** $O(h)$ time.

**Hook.** "First node where p and q split."

**Watch out.** This is the *BST* version — exploit the ordering. General-tree LCA is different (next problem family).

---

### 8. Binary Tree Level Order Traversal (LC 102)

**Intuition.** BFS with queue, processing `len(queue)` nodes per level.

```python
from collections import deque
def levelOrder(root):
    if not root: return []
    q = deque([root])
    out = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

**Complexity.** $O(n)$ time, $O(w)$ space (max width).

**Hook.** "BFS; len(q) per level."

---

### 9. Binary Tree Right Side View (LC 199)

**Intuition.** BFS, last node of each level.

```python
from collections import deque
def rightSideView(root):
    if not root: return []
    q = deque([root])
    out = []
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if i == len(q):  # last in level (after popleft, len(q) is now level size minus already-popped)
                out.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
    return out
```

Cleaner version:

```python
def rightSideView(root):
    if not root: return []
    q = deque([root])
    out = []
    while q:
        size = len(q)
        for i in range(size):
            node = q.popleft()
            if i == size - 1: out.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
    return out
```

**Complexity.** $O(n)$ time.

**Hook.** "Last node of each BFS level."

---

### 10. Count Good Nodes in Binary Tree (LC 1448)

**Problem.** Node is "good" if no node on path from root to it has a greater value. Count good nodes.

**Intuition.** DFS carrying running max along the path.

```python
def goodNodes(root):
    def dfs(node, mx):
        if not node: return 0
        good = 1 if node.val >= mx else 0
        mx = max(mx, node.val)
        return good + dfs(node.left, mx) + dfs(node.right, mx)
    return dfs(root, root.val)
```

**Complexity.** $O(n)$ time.

**Hook.** "Pass running max down."

---

### 11. Validate Binary Search Tree (LC 98)

**Intuition.** Recurse with (low, high) bounds. Each node must be strictly between them.

```python
def isValidBST(root):
    def valid(node, lo, hi):
        if not node: return True
        if not (lo < node.val < hi): return False
        return valid(node.left, lo, node.val) and valid(node.right, node.val, hi)
    return valid(root, float('-inf'), float('inf'))
```

**Complexity.** $O(n)$ time.

**Hook.** "Pass (lo, hi) bounds down."

**Watch out.** Comparing only with parent is wrong — a deep right-grandchild can violate the root's bound.

---

### 12. Kth Smallest in BST (LC 230)

**Intuition.** Inorder traversal yields sorted order; stop at the kth visit.

```python
def kthSmallest(root, k):
    stack = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        if k == 0: return curr.val
        curr = curr.right
```

**Complexity.** $O(h + k)$ time.

**Hook.** "Iterative inorder; decrement k."

---

### 13. Construct Tree from Preorder + Inorder (LC 105)

**Intuition.** First preorder element is root. Find it in inorder to split left/right subtrees. Recurse.

```python
def buildTree(preorder, inorder):
    idx = {v: i for i, v in enumerate(inorder)}
    self_pre = iter(preorder)
    def build(l, r):
        if l > r: return None
        val = next(self_pre)
        node = TreeNode(val)
        m = idx[val]
        node.left  = build(l, m - 1)
        node.right = build(m + 1, r)
        return node
    return build(0, len(inorder) - 1)
```

**Complexity.** $O(n)$ time and space.

**Hook.** "Preorder gives root; inorder splits subtrees."

**Watch out.** Build *left first* (preorder is root, left, right).

---

### 14. Binary Tree Maximum Path Sum (LC 124)

**Problem.** Max sum of any path (any-to-any, must go through nodes).

**Intuition.** At each node, the best path *through* it = node.val + max(left_gain, 0) + max(right_gain, 0). What it *contributes upward* is node.val + max(left_gain, right_gain, 0). Track global max.

```python
def maxPathSum(root):
    best = float('-inf')
    def gain(node):
        nonlocal best
        if not node: return 0
        l = max(gain(node.left), 0)
        r = max(gain(node.right), 0)
        best = max(best, node.val + l + r)
        return node.val + max(l, r)
    gain(root)
    return best
```

**Complexity.** $O(n)$ time.

**Hook.** "Through-path = val + l + r; upward = val + max(l, r). Clip negative gains to 0."

---

### 15. Serialize and Deserialize Binary Tree (LC 297)

**Intuition.** Preorder with explicit null markers.

```python
class Codec:
    def serialize(self, root):
        out = []
        def dfs(node):
            if not node:
                out.append('#')
                return
            out.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(out)

    def deserialize(self, data):
        vals = iter(data.split(','))
        def build():
            v = next(vals)
            if v == '#': return None
            node = TreeNode(int(v))
            node.left = build()
            node.right = build()
            return node
        return build()
```

**Complexity.** $O(n)$ both.

**Hook.** "Preorder + '#' for null."

---

## Summary table — Trees

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Invert | recurse + swap | $O(n)$ | swap children |
| 2 | Max Depth | recursive max | $O(n)$ | 1 + max(l, r) |
| 3 | Diameter | DFS + global | $O(n)$ | left + right depths |
| 4 | Balanced | -1 sentinel | $O(n)$ | abort on -1 |
| 5 | Same Tree | parallel recurse | $O(n)$ | both null OK |
| 6 | Subtree | sameTree at each pos | $O(mn)$ | reuse same-tree |
| 7 | LCA BST | walk down | $O(h)$ | first split |
| 8 | Level Order | BFS with size | $O(n)$ | len(q) per level |
| 9 | Right Side View | BFS last per level | $O(n)$ | last in level |
| 10 | Good Nodes | DFS + running max | $O(n)$ | pass max down |
| 11 | Validate BST | (lo, hi) bounds | $O(n)$ | bounds down |
| 12 | Kth Smallest BST | iterative inorder | $O(h+k)$ | decrement k |
| 13 | Build from Pre+In | split via inorder map | $O(n)$ | preorder root |
| 14 | Max Path Sum | through vs up | $O(n)$ | clip negative |
| 15 | Serialize | preorder + null | $O(n)$ | '#' marker |

## How to drill

1. Day 1: Read all 15.
2. Day 2: Write each from the hook only — blank file, no peeking.
3. Day 5: Repeat from scratch.
4. Day 10: Repeat. Now they're locked in for the interview.
