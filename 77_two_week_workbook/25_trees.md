# Trees: every variation

Almost every binary tree problem is one of two things. It is a traversal, which visits every node in
some order and collects something, or it is a recursion in which each node combines the answers that
come back from its two children. The code is short, usually six or eight lines, and the base case is a
null node. That is why the pattern looks easy and then is not.

The thing that makes tree problems hard is not the recursion. It is deciding **what each call
returns**. Take the diameter of a binary tree. The recursion returns the *height* of a subtree, and the
answer — the longest path — is recorded in a variable outside the recursion. A candidate who tries to
return the diameter itself gets stuck, because the diameter of a subtree is not something the parent
can extend into a longer path. The parent needs a height. The question asked for a diameter. They are
different quantities, and the whole solution turns on keeping them apart.

The second organising idea is to choose the traversal by what the node needs. Use **pre-order** when
the parent must pass information *down*, such as bounds, a running path, or a depth. Use **post-order**
when the parent needs answers *up* from the children, such as a height, a subtree sum, or whether a
subtree is balanced. Use **in-order** when the tree is a binary search tree, because in-order on a BST
yields the values in sorted order, and that single fact solves a whole family of problems.

## Recognising it from the phrasing

| The interviewer says | They mean | Traversal | What the call returns |
|---|---|---|---|
| "depth / height / balanced / diameter" | combine two child answers | post-order | one quantity up; the answer goes in a nonlocal |
| "validate a BST", "every node in a range" | push bounds down | pre-order | a boolean, with `low` and `high` as arguments |
| "kth smallest", "sorted output", "two-sum in a BST" | in-order on a BST is sorted | in-order | nothing; you count or collect as you visit |
| "level order / zigzag / right side view / minimum depth" | one level per outer iteration | BFS with a queue | nothing; you append one list per level |
| "path sum", "all root-to-leaf paths" | carry the running path down | pre-order | nothing; append and pop a shared list |
| "lowest common ancestor" | the first node that sees both sides | post-order | the found node, or `None` |
| "serialise and deserialise" | a traversal with explicit null markers | pre-order | a string, rebuilt from an iterator |
| "build the tree from its traversals" | recursion on index ranges | pre-order construction | the root of the subtree it built |
| "count nodes better than every ancestor" | push a running maximum down | pre-order | a count, summed from the children |

Before you write anything, answer three questions in this order. First, **what does one call return?**
Second, **what does it need from its parent**, which fixes the extra arguments. Third, **what is the
base case for a null node?** If you can state those three sentences out loud, the code writes itself.
If you cannot state them, you are not ready to type, and typing anyway is how the round is lost. The
null base case is where most bugs live, because the quantities differ by one between problems: the
height of a null node is $0$, the depth of a null node is often $-1$, and the sum of an empty subtree
is $0$ while its minimum is $+\infty$. Mixing the height convention with the depth convention shifts
every answer in the tree by exactly one, and the shift is invisible on a one-node test.

## The templates

Every code block on this page repeats the same short prelude so that it runs on its own: a `TreeNode`
class and a `build` helper that makes a tree from a level-order list with `None` for a gap. Blocks that
must show a tree also re-declare `to_list`, which is the inverse of `build`. Templates 1 and 2 share a
skeleton, and only the return value and the extra `best` line differ.

**Template 1 — recursive DFS that returns one value.** Use when the answer at a node is a plain
function of the answers at its children.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def max_depth(root):
    if root is None:                                 ## the base case fixes the whole convention
        return 0
    left = max_depth(root.left)                      ## ask the children first: post-order
    right = max_depth(root.right)
    return 1 + max(left, right)                      ## combine, then hand one number up

## tests

assert max_depth(build([3, 9, 20, None, None, 15, 7])) == 3
assert max_depth(build([])) == 0
assert max_depth(build([1, None, 2, None, 3])) == 3
print(max_depth(build([3, 9, 20, None, None, 15, 7])))
```

```
3
```

**Template 2 — DFS that returns one value and records another.** Use when the answer asked for is a
combination *at* the node that the parent cannot use. The skeleton is template 1 with two lines added:
a `best` variable declared outside, and one `best = max(...)` line inside. This is the diameter and
maximum-path-sum shape.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def diameter(root):
    best = 0                                         ## the ANSWER lives here, not in the return

    def height(node):
        nonlocal best
        if node is None:
            return 0
        left = height(node.left)
        right = height(node.right)
        best = max(best, left + right)               ## a path THROUGH node, in edges
        return 1 + max(left, right)                  ## what the PARENT can use

    height(root)
    return best

## tests

assert diameter(build([1, 2, 3, 4, 5])) == 3
assert diameter(build([1, 2])) == 1
assert diameter(build([])) == 0
print(diameter(build([1, 2, 3, 4, 5])))
```

```
3
```

**Template 3 — BFS, one level per outer iteration.** Use whenever the question names levels, or asks
for the first node that satisfies something, because BFS reaches shallow nodes first.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def level_order(root):
    if root is None:
        return []
    levels, queue = [], deque([root])
    while queue:
        size = len(queue)                            ## SNAPSHOT: how many nodes are on this level
        level = []
        for _ in range(size):                        ## consume exactly that many
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)              ## children join the NEXT level
            if node.right:
                queue.append(node.right)
        levels.append(level)
    return levels

## tests

assert level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
assert level_order(build([])) == []
assert level_order(build([1])) == [[1]]
print(level_order(build([3, 9, 20, None, None, 15, 7])))
```

```
[[3], [9, 20], [15, 7]]
```

The line `size = len(queue)` is the whole template. It takes a snapshot of the queue length *before*
any child is pushed, so the `for` loop consumes exactly the nodes of the current level and no more.
Without the snapshot, the loop would run over a queue that grows while you iterate, the levels would
run together, and you would get one flat list. Read the snapshot as the sentence "this level has
`size` nodes in it".

**Template 4 — iterative in-order with an explicit stack.** Use when the interviewer asks for it, and
they do ask. It proves you know what the recursion is actually doing, and it does not hit the recursion
limit on a degenerate tree that is really a linked list of ten thousand nodes.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def inorder_iterative(root):
    out, stack, node = [], [], root
    while stack or node:
        while node:                                  ## descend left, remembering the way back
            stack.append(node)
            node = node.left
        node = stack.pop()                           ## the leftmost unvisited node
        out.append(node.val)                         ## VISIT here, between left and right
        node = node.right                            ## then start again from the right child
    return out

## tests

assert inorder_iterative(build([4, 2, 6, 1, 3, 5, 7])) == [1, 2, 3, 4, 5, 6, 7]
assert inorder_iterative(build([])) == []
assert inorder_iterative(build([1, None, 2])) == [1, 2]
print(inorder_iterative(build([4, 2, 6, 1, 3, 5, 7])))
```

```
[1, 2, 3, 4, 5, 6, 7]
```

**Template 5 — BST search and insert.** Use whenever the tree is a search tree, because the ordering
turns a whole-tree scan into a single root-to-leaf walk.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def bst_search(root, target):
    node = root
    while node:
        if target == node.val:
            return node
        node = node.left if target < node.val else node.right   ## one comparison, one branch
    return None

def bst_insert(root, val):
    if root is None:
        return TreeNode(val)                         ## the empty spot is where it belongs
    if val < root.val:
        root.left = bst_insert(root.left, val)       ## reassign: the child may have been created
    elif val > root.val:
        root.right = bst_insert(root.right, val)
    return root                                      ## always return the (possibly new) subtree

## tests

root = build([4, 2, 6, 1, 3, 5, 7])
assert bst_search(root, 5).val == 5
assert bst_search(root, 8) is None
tree = None
for x in [4, 2, 6, 1, 3]:
    tree = bst_insert(tree, x)
assert [n for n in [tree.val, tree.left.val, tree.right.val]] == [4, 2, 6]
print(bst_search(root, 5).val, tree.val, tree.left.val, tree.left.left.val)
```

```
5 4 2 1
```

Search and insert both cost $O(h)$, where $h$ is the height. That is $O(\log n)$ on a balanced tree
and $O(n)$ on a tree built by inserting a sorted sequence, which degenerates into a chain. Say the
worst case out loud, because the interviewer is waiting for it.

## The highest-value trick: return one thing, record another

This one shape solves the diameter of a binary tree, the maximum path sum, the longest univalue path,
and the one-pass balanced check. Learning it converts four separate hard problems into one, and a
candidate who names the shape out loud has effectively answered all four.

The shape is this. The recursion returns the quantity the **parent** can use. The quantity the question
**asks for** is a combination at the current node that the parent cannot use, so it does not go in the
return value. It goes into a variable declared outside the recursion, and every call updates it.

Work the diameter case. The diameter is the number of edges on the longest path between any two nodes.
For a fixed node, the longest path that bends at that node has `left_height + right_height` edges,
because it goes down one side and up the other. However, that bent path cannot be extended upwards
through the parent: a path may pass through a node only once, so from the parent's point of view only
one of the two arms is usable. Therefore the parent needs `1 + max(left_height, right_height)`, which
is the height, and the bent path is recorded on the side.

So the two lines at each node are:

$$\text{best} \leftarrow \max(\text{best},\; h_L + h_R), \qquad
\text{return } 1 + \max(h_L, h_R)$$

**Worked example.** Take the tree `[1, 2, 3, 4, 5]`, so node 1 has children 2 and 3, and node 2 has
children 4 and 5. Node 4 and node 5 are leaves with height 1 and a bent path of 0 edges. Node 2 has
`h_L = 1` and `h_R = 1`, so its bent path is 2 edges — that is the path 4-2-5 — and it returns height
2. Node 3 is a leaf, height 1. Node 1 has `h_L = 2` and `h_R = 1`, so its bent path is 3 edges, the
path 4-2-1-3, and it returns height 3. The best value ever recorded is 3, which is the diameter. Notice
that node 1 never sees the number 2 that node 2 recorded; the recorded values never travel, only the
heights do.

The code below prints the pair at every node so you can read the two quantities side by side.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def diameter_with_trace(root):
    best = 0
    trace = []

    def height(node):
        nonlocal best
        if node is None:
            return 0                                 ## a null subtree has height 0
        left = height(node.left)
        right = height(node.right)
        bent = left + right                          ## RECORDED: path through node, in edges
        best = max(best, bent)
        trace.append((node.val, left + right, 1 + max(left, right)))
        return 1 + max(left, right)                  ## RETURNED: what the parent can extend

    height(root)
    return best, trace

## tests

answer, trace = diameter_with_trace(build([1, 2, 3, 4, 5]))
assert answer == 3
assert trace == [(4, 0, 1), (5, 0, 1), (2, 2, 2), (3, 0, 1), (1, 3, 3)]
assert diameter_with_trace(build([1, 2, 3, 4, None, None, 5, 6, None, None, 7]))[0] == 6
print(answer)
for val, bent, returned in trace:
    print("node", val, "recorded", bent, "returned", returned)
```

```
3
node 4 recorded 0 returned 1
node 5 recorded 0 returned 1
node 2 recorded 2 returned 2
node 3 recorded 0 returned 1
node 1 recorded 3 returned 3
```

The rule for spotting the shape is short. If the thing being asked about is a path or a combination
that **bends** at a node, then it is not extendable upward, so it must be recorded rather than
returned. The maximum path sum bends. The longest univalue path bends. The diameter bends. A height, a
subtree sum and a node count do not bend, so they are returned.

## The problems

### P1. Maximum depth of a binary tree — the number of nodes on the longest root-to-leaf path

**Which template.** Template 1. It is the shortest post-order recursion there is.
**The trick.** The base case sets the convention and everything else follows from it. A null subtree has depth 0, therefore a leaf has depth 1, therefore the answer counts nodes rather than edges. If the interviewer asks for edges instead, return `-1` for a null node and change nothing else. The iterative version below carries the depth down on the stack, which is the pre-order way of getting the same number.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def max_depth(root):
    if root is None:
        return 0                                     ## nodes, not edges: use -1 for edges
    return 1 + max(max_depth(root.left), max_depth(root.right))

def max_depth_iterative(root):
    best, stack = 0, [(root, 1)]                     ## push the depth DOWN with each node
    while stack:
        node, depth = stack.pop()
        if node is None:
            continue
        best = max(best, depth)
        stack.append((node.left, depth + 1))
        stack.append((node.right, depth + 1))
    return best

## tests

for values in ([3, 9, 20, None, None, 15, 7], [], [1], [1, None, 2, None, 3]):
    assert max_depth(build(values)) == max_depth_iterative(build(values))
assert max_depth(build([3, 9, 20, None, None, 15, 7])) == 3
assert max_depth(build([])) == 0
assert max_depth_iterative(build([1, None, 2, None, 3])) == 3
print(max_depth(build([3, 9, 20, None, None, 15, 7])), max_depth_iterative(build([1, None, 2, None, 3])))
```

```
3 3
```

**Complexity.** $O(n)$ time. Space is $O(h)$ for the call stack, which is $O(n)$ on a degenerate tree.

### P2. Minimum depth of a binary tree — the number of nodes on the shortest root-to-leaf path

**Which template.** Template 3, BFS, because BFS can stop at the first leaf it meets.
**The trick.** The trap is the definition of a leaf. A leaf is a node with **no** children. A node with one missing child is not a leaf, so you must not treat that missing side as a path of length 0. Writing `1 + min(left, right)` returns 1 for the tree `[1, 2]`, which is wrong; the answer is 2. Handle the one-child case explicitly by taking the depth of the side that exists. BFS avoids the trap completely and returns as soon as it sees the first true leaf.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def min_depth_bfs(root):
    if root is None:
        return 0
    queue, depth = deque([root]), 1
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left is None and node.right is None:
                return depth                         ## first true leaf: BFS makes it the shallowest
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        depth += 1
    return depth

def min_depth_recursive(root):
    if root is None:
        return 0
    if root.left is None:                            ## one child missing: NOT a leaf
        return 1 + min_depth_recursive(root.right)
    if root.right is None:
        return 1 + min_depth_recursive(root.left)
    return 1 + min(min_depth_recursive(root.left), min_depth_recursive(root.right))

## tests

assert min_depth_bfs(build([3, 9, 20, None, None, 15, 7])) == 2
assert min_depth_bfs(build([2, None, 3, None, 4, None, 5, None, 6])) == 5
assert min_depth_bfs(build([1, 2])) == 2            ## the trap: a naive min() answers 1
assert min_depth_bfs(build([])) == 0
for values in ([3, 9, 20, None, None, 15, 7], [1, 2], [], [1], [2, None, 3, None, 4]):
    assert min_depth_bfs(build(values)) == min_depth_recursive(build(values))
print(min_depth_bfs(build([1, 2])), min_depth_bfs(build([2, None, 3, None, 4, None, 5, None, 6])))
```

```
2 5
```

**Complexity.** BFS is $O(n)$ time worst case but stops early, and $O(w)$ space for the widest level. The recursion is $O(n)$ time and $O(h)$ space.

### P3. Same tree — decide whether two trees have the same shape and the same values

**Which template.** Template 1, but the recursion walks two trees at once.
**The trick.** Compare structure before values. Two nulls are equal; one null and one node are not; otherwise compare the values and then recurse pairwise, left with left and right with right. The order of the three checks matters, because testing `p.val == q.val` before the null checks raises an `AttributeError` on the first missing child.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def is_same_tree(p, q):
    if p is None and q is None:
        return True                                  ## both empty: equal
    if p is None or q is None:
        return False                                 ## exactly one empty: different shape
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

## tests

assert is_same_tree(build([1, 2, 3]), build([1, 2, 3])) is True
assert is_same_tree(build([1, 2]), build([1, None, 2])) is False
assert is_same_tree(build([1, 2, 1]), build([1, 1, 2])) is False
assert is_same_tree(build([]), build([])) is True
print(is_same_tree(build([1, 2, 3]), build([1, 2, 3])), is_same_tree(build([1, 2]), build([1, None, 2])))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P4. Symmetric tree — decide whether a tree is a mirror image of itself

**Which template.** Template 1 on a pair of nodes, exactly like P3 but with the sides crossed.
**The trick.** Do not compare a tree with a reversed copy of itself, and do not compare the left subtree with the right subtree using `is_same_tree` — that tests equality, not mirroring. Write a helper on two nodes that recurses `a.left` against `b.right` and `a.right` against `b.left`. The single crossed pair of arguments is the entire problem.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def is_symmetric(root):
    def mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a.val != b.val:
            return False
        return mirror(a.left, b.right) and mirror(a.right, b.left)   ## CROSSED
    return root is None or mirror(root.left, root.right)

## tests

assert is_symmetric(build([1, 2, 2, 3, 4, 4, 3])) is True
assert is_symmetric(build([1, 2, 2, None, 3, None, 3])) is False
assert is_symmetric(build([1, 2, 2])) is True
assert is_symmetric(build([])) is True
print(is_symmetric(build([1, 2, 2, 3, 4, 4, 3])), is_symmetric(build([1, 2, 2, None, 3, None, 3])))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P5. Invert a binary tree — swap the left and right child of every node

**Which template.** Template 1, post-order, returning the node itself.
**The trick.** Swap the two children at every node and recurse. The one thing to be careful about is the order of assignment: if you write `node.left = invert(node.right)` first, then `node.right = invert(node.left)` reads the value you have just overwritten and duplicates a subtree. Use a tuple swap, which evaluates the right-hand side completely before assigning.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
def to_list(root):                                   ## inverse of build, trailing Nones trimmed
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(None if node is None else node.val)
        if node:
            queue.append(node.left)
            queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out

def invert(root):
    if root is None:
        return None
    root.left, root.right = invert(root.right), invert(root.left)   ## tuple swap: no clobbering
    return root

## tests

assert to_list(invert(build([4, 2, 7, 1, 3, 6, 9]))) == [4, 7, 2, 9, 6, 3, 1]
assert to_list(invert(build([2, 1, 3]))) == [2, 3, 1]
assert to_list(invert(build([]))) == []
print(to_list(invert(build([4, 2, 7, 1, 3, 6, 9]))))
```

```
[4, 7, 2, 9, 6, 3, 1]
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P6. Balanced binary tree — decide whether every node has left and right heights differing by at most one

**Which template.** Template 2, in its sentinel form. The recursion returns a height, and a special value carries the failure up.
**The trick.** The naive solution calls a separate `height` function at every node, which is $O(n^2)$ on a chain. Fix it in one pass by returning the height normally and returning the sentinel `-1` the moment any subtree is unbalanced. A real height is never negative, so `-1` cannot be confused with an answer, and once it appears it propagates straight to the root. This is the same "return one thing" idea as the diameter, with a sentinel taking the place of the nonlocal variable.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def is_balanced(root):
    def height(node):
        if node is None:
            return 0
        left = height(node.left)
        if left == -1:
            return -1                                ## sentinel: a real height is never negative
        right = height(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1                                ## this node is the one that fails
        return 1 + max(left, right)
    return height(root) != -1

## tests

assert is_balanced(build([3, 9, 20, None, None, 15, 7])) is True
assert is_balanced(build([1, 2, 2, 3, 3, None, None, 4, 4])) is False
assert is_balanced(build([])) is True
assert is_balanced(build([1, 2, None, 3])) is False
print(is_balanced(build([3, 9, 20, None, None, 15, 7])), is_balanced(build([1, 2, 2, 3, 3, None, None, 4, 4])))
```

```
True False
```

**Complexity.** $O(n)$ time and $O(h)$ space, against $O(n^2)$ for the two-function version.

### P7. Diameter of a binary tree — the number of edges on the longest path between any two nodes

**Which template.** Template 2. This is the canonical "return one thing, record another".
**The trick.** The returned value is the height. The recorded value is `left + right`, the bent path through the current node. The second function below returns a pair `(height, best)` instead of using a nonlocal, to show that the nonlocal is a convenience and not the mechanism; the two answers agree on every tree. Also note that the path need not pass through the root, which is why a single top-level `height(left) + height(right)` is wrong.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def diameter(root):
    best = 0

    def height(node):
        nonlocal best
        if node is None:
            return 0
        left, right = height(node.left), height(node.right)
        best = max(best, left + right)               ## record the bent path
        return 1 + max(left, right)                  ## return what the parent can use

    height(root)
    return best

def diameter_pair(root):                             ## same thing without nonlocal
    def solve(node):
        if node is None:
            return 0, 0                              ## (height, best diameter in this subtree)
        lh, lb = solve(node.left)
        rh, rb = solve(node.right)
        return 1 + max(lh, rh), max(lb, rb, lh + rh)
    return solve(root)[1]

## tests

cases = [[1, 2, 3, 4, 5], [1, 2], [], [1], [1, 2, 3, 4, None, None, 5, 6, None, None, 7]]
for values in cases:
    assert diameter(build(values)) == diameter_pair(build(values))
assert diameter(build([1, 2, 3, 4, 5])) == 3
assert diameter(build([1, 2])) == 1
assert diameter(build([1, 2, 3, 4, None, None, 5, 6, None, None, 7])) == 6
print([diameter(build(v)) for v in cases])
```

```
[3, 1, 0, 0, 6]
```

**Complexity.** $O(n)$ time, $O(h)$ space. Each node is visited once, unlike the $O(n^2)$ version that recomputes heights.

### P8. Path sum — does some root-to-leaf path have a given total

**Which template.** Template 1, pre-order, with the remaining target pushed down.
**The trick.** Subtract as you descend instead of adding as you return. The question at a node becomes "does a path from here to a leaf sum to `remaining`", which is the same question with a smaller number, so the recursion is direct. The base case is the trap again: return `True` only at a real leaf, where both children are missing. Returning `remaining == 0` at a null node accepts a half-path, so the tree `[1, 2]` with target 1 would wrongly answer `True`.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def has_path_sum(root, target):
    if root is None:
        return False                                 ## an empty tree has no root-to-leaf path
    remaining = target - root.val
    if root.left is None and root.right is None:     ## a real leaf: decide here
        return remaining == 0
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)

## tests

tree = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
assert has_path_sum(tree, 22) is True
assert has_path_sum(tree, 26) is True
assert has_path_sum(build([1, 2]), 1) is False       ## the half-path trap
assert has_path_sum(build([]), 0) is False
print(has_path_sum(tree, 22), has_path_sum(build([1, 2]), 1))
```

```
True False
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P9. Path sum II — return every root-to-leaf path whose values sum to a given total

**Which template.** Template 1, pre-order, carrying a shared list down and backtracking on the way out.
**The trick.** Push the node onto `path`, recurse, then pop it. The pop is the backtracking step, and it must run on every exit from the node, including the leaf case. Copy the list with `list(path)` when you record a hit, because `path` keeps mutating afterwards; appending `path` itself gives you a result full of references to one list that ends up empty.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def path_sum_all(root, target):
    out, path = [], []

    def walk(node, remaining):
        if node is None:
            return
        path.append(node.val)                        ## enter
        remaining -= node.val
        if node.left is None and node.right is None and remaining == 0:
            out.append(list(path))                   ## COPY, because path keeps changing
        walk(node.left, remaining)
        walk(node.right, remaining)
        path.pop()                                   ## leave: undo exactly what enter did

    walk(root, target)
    return out

## tests

tree = build([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
assert path_sum_all(tree, 22) == [[5, 4, 11, 2], [5, 8, 4, 5]]
assert path_sum_all(build([1, 2, 3]), 3) == [[1, 2]]
assert path_sum_all(build([1, 2]), 0) == []
assert path_sum_all(build([]), 0) == []
print(path_sum_all(tree, 22))
```

```
[[5, 4, 11, 2], [5, 8, 4, 5]]
```

**Complexity.** $O(n h)$ time in the worst case, because each recorded path costs $O(h)$ to copy, and $O(h)$ space beyond the output.

### P10. Binary tree maximum path sum — the largest sum over any path between any two nodes

**Which template.** Template 2. The bent path is recorded, the straight arm is returned.
**The trick.** Two things make this harder than the diameter. First, values may be negative, so an arm that sums to less than zero should be dropped rather than used: clamp each child contribution with `max(0, child)`, which means "take this arm only if it helps". Second, the recorded quantity is `node.val + left + right`, using the clamped arms, while the returned quantity is `node.val + max(left, right)` — one arm only, because the parent extends a straight line through the node. Forgetting the clamp gives the wrong answer on any tree with a negative subtree.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def max_path_sum(root):
    best = float("-inf")

    def gain(node):
        nonlocal best
        if node is None:
            return 0
        left = max(0, gain(node.left))               ## clamp: a negative arm is worth skipping
        right = max(0, gain(node.right))
        best = max(best, node.val + left + right)    ## RECORD the bent path through node
        return node.val + max(left, right)           ## RETURN one arm only

    gain(root)
    return best

## tests

assert max_path_sum(build([1, 2, 3])) == 6
assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42
assert max_path_sum(build([-3])) == -3
assert max_path_sum(build([2, -1])) == 2             ## the clamp drops the -1 arm
print(max_path_sum(build([-10, 9, 20, None, None, 15, 7])), max_path_sum(build([2, -1])))
```

```
42 2
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P11. Binary tree level order traversal — the node values level by level, as a list of lists

**Which template.** Template 3, the plain BFS.
**The trick.** The snapshot `size = len(queue)` separates the levels. Everything else in this family — zigzag, right side view, averages, level maxima, the bottom-up variant — is this loop with one line changed at the point where the level is appended, so learn this one exactly and describe the others as edits to it.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def level_order(root):
    if root is None:
        return []
    levels, queue = [], deque([root])
    while queue:
        size = len(queue)                            ## nodes on the CURRENT level
        level = []
        for _ in range(size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        levels.append(level)
    return levels

## tests

assert level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
assert level_order(build([1, 2, 3, 4, None, None, 5])) == [[1], [2, 3], [4, 5]]
assert level_order(build([])) == []
print(level_order(build([1, 2, 3, 4, None, None, 5])))
```

```
[[1], [2, 3], [4, 5]]
```

**Complexity.** $O(n)$ time, $O(w)$ space, where $w$ is the widest level.

### P12. Zigzag level order traversal — level order, but alternate levels run right to left

**Which template.** Template 3 with one extra line.
**The trick.** Do not reverse the queue and do not push children in a different order on alternate levels; both break the level structure. Build each level left to right exactly as usual, then reverse the finished list when the level index is odd. The traversal is unchanged and only the presentation flips.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def zigzag_level_order(root):
    if root is None:
        return []
    levels, queue, left_to_right = [], deque([root]), True
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        levels.append(level if left_to_right else level[::-1])   ## flip the finished level
        left_to_right = not left_to_right
    return levels

## tests

assert zigzag_level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [20, 9], [15, 7]]
assert zigzag_level_order(build([1, 2, 3, 4, 5, 6, 7])) == [[1], [3, 2], [4, 5, 6, 7]]
assert zigzag_level_order(build([])) == []
print(zigzag_level_order(build([1, 2, 3, 4, 5, 6, 7])))
```

```
[[1], [3, 2], [4, 5, 6, 7]]
```

**Complexity.** $O(n)$ time, $O(w)$ space.

### P13. Binary tree right side view — the values you see looking at the tree from the right

**Which template.** Template 3, recording the last node of each level.
**The trick.** The right side view is not the right spine. If the right child is missing, the visible node on that level comes from the left subtree, so a walk down `node.right` gives the wrong answer. With BFS it is one line: the last node dequeued on a level is the rightmost one, so record it when the loop index reaches `size - 1`.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def right_side_view(root):
    if root is None:
        return []
    view, queue = [], deque([root])
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                view.append(node.val)                ## the LAST node of the level is visible
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return view

## tests

assert right_side_view(build([1, 2, 3, None, 5, None, 4])) == [1, 3, 4]
assert right_side_view(build([1, 2, 3, 4])) == [1, 3, 4]     ## visible node comes from the LEFT subtree
assert right_side_view(build([1, None, 3])) == [1, 3]
assert right_side_view(build([])) == []
print(right_side_view(build([1, 2, 3, None, 5, None, 4])), right_side_view(build([1, 2, 3, 4])))
```

```
[1, 3, 4] [1, 3, 4]
```

**Complexity.** $O(n)$ time, $O(w)$ space.

### P14. Average of levels in a binary tree — the mean of the values on each level

**Which template.** Template 3, summing instead of collecting.
**The trick.** Keep a running `total` for the level rather than a list, then divide by `size`. Use `size`, the snapshot taken at the top of the level, not `len(queue)` at the bottom — by then the queue holds the next level and the division is against the wrong count.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
from collections import deque

def average_of_levels(root):
    if root is None:
        return []
    out, queue = [], deque([root])
    while queue:
        size = len(queue)                            ## divide by THIS, not by len(queue) later
        total = 0
        for _ in range(size):
            node = queue.popleft()
            total += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        out.append(total / size)
    return out

## tests

assert average_of_levels(build([3, 9, 20, None, None, 15, 7])) == [3.0, 14.5, 11.0]
assert average_of_levels(build([1])) == [1.0]
assert average_of_levels(build([])) == []
print(average_of_levels(build([3, 9, 20, None, None, 15, 7])))
```

```
[3.0, 14.5, 11.0]
```

**Complexity.** $O(n)$ time, $O(w)$ space.

### P15. Lowest common ancestor of a binary tree — the deepest node that has both given nodes in its subtree

**Which template.** Template 1, post-order, returning a node instead of a number.
**The trick.** The recursion returns "one of the two targets, or the answer, or `None`". At a node, if the left call and the right call both return something, then the two targets are on opposite sides, so this node is the answer. If only one side returns something, pass that up unchanged. The single elegant point is that these two rules also handle the case where one target is an ancestor of the other: that target is returned by the base case before the recursion ever reaches the deeper one, and it travels up as the answer.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root                                  ## found a target, or ran out of tree
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root                                  ## targets on opposite sides: this is the LCA
    return left or right                             ## pass up whichever side found something

def find(node, val):
    if node is None or node.val == val:
        return node
    return find(node.left, val) or find(node.right, val)

## tests

tree = build([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
assert lowest_common_ancestor(tree, find(tree, 5), find(tree, 1)).val == 3
assert lowest_common_ancestor(tree, find(tree, 5), find(tree, 4)).val == 5   ## ancestor of the other
assert lowest_common_ancestor(tree, find(tree, 7), find(tree, 4)).val == 2
assert lowest_common_ancestor(tree, find(tree, 6), find(tree, 8)).val == 3
print(lowest_common_ancestor(tree, find(tree, 5), find(tree, 4)).val,
      lowest_common_ancestor(tree, find(tree, 7), find(tree, 4)).val)
```

```
5 2
```

**Complexity.** $O(n)$ time, $O(h)$ space. The whole tree may be searched, because there is no ordering to exploit.

### P16. Lowest common ancestor of a BST — the same question when the tree is a search tree

**Which template.** Template 5, a single root-to-leaf walk driven by the ordering.
**The trick.** You never need to search both sides. If both values are smaller than the current node, the answer is in the left subtree; if both are larger, it is in the right subtree; otherwise the two values split here, and the current node is the answer. "Split here" also covers the case where one value equals the current node. That gives an $O(h)$ loop with no recursion and no extra space, against $O(n)$ for P15.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def lca_bst(root, p_val, q_val):
    node = root
    low, high = min(p_val, q_val), max(p_val, q_val)
    while node:
        if high < node.val:
            node = node.left                         ## both targets are smaller
        elif low > node.val:
            node = node.right                        ## both targets are larger
        else:
            return node                              ## they split here: this is the LCA
    return None

## tests

tree = build([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
assert lca_bst(tree, 2, 8).val == 6
assert lca_bst(tree, 2, 4).val == 2                  ## one target is the ancestor
assert lca_bst(tree, 3, 5).val == 4
assert lca_bst(tree, 7, 9).val == 8
print(lca_bst(tree, 2, 8).val, lca_bst(tree, 2, 4).val, lca_bst(tree, 3, 5).val)
```

```
6 2 4
```

**Complexity.** $O(h)$ time, $O(1)$ space.

### P17. Validate a binary search tree — decide whether the whole tree satisfies the BST ordering

**Which template.** Template 1 with bounds pushed down, which is a pre-order use of the arguments.
**The trick.** Comparing each node only against its immediate parent is wrong, and it is worth saying why in one sentence: the BST property is about the whole subtree, not about one edge. In the tree `[5, 4, 6, None, None, 3, 7]` the node 3 is correctly less than its parent 6, but 3 sits in the right subtree of 5, where every value must exceed 5. A local check accepts it. The fix is to carry an open interval `(low, high)` down: going left tightens `high` to the current value, going right tightens `low`. Bounds are strict, because duplicates are not allowed.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def is_valid_bst(root):
    def check(node, low, high):
        if node is None:
            return True                              ## an empty subtree satisfies any bounds
        if not (low < node.val < high):
            return False
        return check(node.left, low, node.val) and check(node.right, node.val, high)
    return check(root, float("-inf"), float("inf"))

def is_valid_bst_local(root):                        ## the WRONG parent-only check, for contrast
    if root is None:
        return True
    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False
    return is_valid_bst_local(root.left) and is_valid_bst_local(root.right)

## tests

trap = build([5, 4, 6, None, None, 3, 7])
assert is_valid_bst(trap) is False
assert is_valid_bst_local(trap) is True              ## the local check is fooled by node 3
assert is_valid_bst(build([2, 1, 3])) is True
assert is_valid_bst(build([2, 2, 2])) is False       ## duplicates break strict ordering
assert is_valid_bst(build([])) is True
print(is_valid_bst(trap), is_valid_bst_local(trap), is_valid_bst(build([2, 1, 3])))
```

```
False True True
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P18. Kth smallest element in a BST — the k-th value in increasing order

**Which template.** Template 4, in-order, stopping as soon as the count reaches `k`.
**The trick.** In-order on a BST produces sorted order, so the k-th value visited is the answer. Do not build the whole sorted list and index into it; the iterative in-order lets you stop after `k` visits, which is $O(h + k)$ instead of $O(n)$. The counter must be incremented at the visit point, between the left subtree and the right subtree, and nowhere else.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def kth_smallest(root, k):
    stack, node, seen = [], root, 0
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        seen += 1                                    ## count at the VISIT point
        if seen == k:
            return node.val                          ## stop early: no need to finish the tree
        node = node.right
    return None

## tests

tree = build([5, 3, 6, 2, 4, None, None, 1])
assert kth_smallest(tree, 1) == 1
assert kth_smallest(tree, 3) == 3
assert kth_smallest(tree, 6) == 6
assert kth_smallest(tree, 7) is None
assert kth_smallest(build([2, 1, 3]), 2) == 2
print(kth_smallest(tree, 1), kth_smallest(tree, 3), kth_smallest(tree, 6))
```

```
1 3 6
```

**Complexity.** $O(h + k)$ time, $O(h)$ space.

### P19. BST iterator — a class with `next` and `has_next` returning the values in increasing order

**Which template.** Template 4, with the stack held as an attribute instead of a local variable.
**The trick.** This is the iterative in-order traversal cut in half. The constructor runs only the descend-left part and stops. Each `next` pops one node, runs the descend-left part on its right child, and returns the value. The stack therefore holds only the current path, so the space is $O(h)$ rather than $O(n)$, and `next` costs $O(1)$ on average because every node is pushed once and popped once across the whole traversal.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)                        ## the constructor is half of one loop iteration

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def has_next(self):
        return len(self.stack) > 0

    def next(self):
        node = self.stack.pop()                      ## the smallest unvisited node
        self._push_left(node.right)                  ## then set up its right subtree
        return node.val

## tests

it = BSTIterator(build([7, 3, 15, None, None, 9, 20]))
assert [it.next(), it.next(), it.next()] == [3, 7, 9]
assert it.has_next() is True
assert [it.next(), it.next()] == [15, 20]
assert it.has_next() is False
it2 = BSTIterator(build([4, 2, 6, 1, 3, 5, 7]))
out = []
while it2.has_next():
    out.append(it2.next())
assert out == [1, 2, 3, 4, 5, 6, 7]
print(out)
```

```
[1, 2, 3, 4, 5, 6, 7]
```

**Complexity.** Amortised $O(1)$ per `next`, $O(h)$ space.

### P20. Convert a sorted array to a height-balanced BST — build a BST of minimum height from sorted values

**Which template.** A construction recursion on index ranges. The returned value is the root of the subtree built.
**The trick.** The middle element of the range becomes the root, because that splits the remaining values into two halves of nearly equal size, and the halves become the two subtrees by the same rule. Pass index bounds rather than slicing the list; slicing copies and turns an $O(n)$ build into $O(n \log n)$. The base case is `low > high`, which returns `None` for an empty range.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
def to_list(root):                                   ## inverse of build, trailing Nones trimmed
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(None if node is None else node.val)
        if node:
            queue.append(node.left)
            queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out

def sorted_array_to_bst(nums):
    def build_range(low, high):
        if low > high:
            return None                              ## empty range
        mid = (low + high) // 2                      ## the middle value balances the two sides
        node = TreeNode(nums[mid])
        node.left = build_range(low, mid - 1)
        node.right = build_range(mid + 1, high)
        return node
    return build_range(0, len(nums) - 1)

def height(node):
    return 0 if node is None else 1 + max(height(node.left), height(node.right))

## tests

root = sorted_array_to_bst([-10, -3, 0, 5, 9])
assert to_list(root) == [0, -10, 5, None, -3, None, 9]
assert height(root) == 3
assert to_list(sorted_array_to_bst([])) == []
assert to_list(sorted_array_to_bst([1, 3])) == [1, None, 3]
assert height(sorted_array_to_bst(list(range(1023)))) == 10
print(to_list(root), height(sorted_array_to_bst(list(range(1023)))))
```

```
[0, -10, 5, None, -3, None, 9] 10
```

**Complexity.** $O(n)$ time, $O(\log n)$ space for the recursion, plus the tree itself.

### P21. Construct a binary tree from preorder and inorder — rebuild the tree from its two traversals

**Which template.** A construction recursion on index ranges, with a value-to-index map for the split point.
**The trick.** The first value of the preorder segment is the root. Find that value in the inorder list: everything to its left is the left subtree and everything to its right is the right subtree, and the count of those left values tells you how to split the preorder segment as well. Searching the inorder list each time costs $O(n)$ per node and makes the whole build $O(n^2)$; a dictionary from value to inorder index makes each split $O(1)$ and the build $O(n)$. Values must be distinct for the map to be well defined, which is worth confirming with the interviewer.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
def to_list(root):                                   ## inverse of build, trailing Nones trimmed
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(None if node is None else node.val)
        if node:
            queue.append(node.left)
            queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out

def build_tree(preorder, inorder):
    position = {val: i for i, val in enumerate(inorder)}   ## value -> inorder index, O(1) splits
    self_index = [0]                                       ## how far along preorder we are

    def build_range(low, high):
        if low > high:
            return None
        val = preorder[self_index[0]]                      ## preorder gives the root directly
        self_index[0] += 1
        node = TreeNode(val)
        mid = position[val]
        node.left = build_range(low, mid - 1)              ## left subtree consumes preorder FIRST
        node.right = build_range(mid + 1, high)
        return node

    return build_range(0, len(inorder) - 1)

## tests

root = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
assert to_list(root) == [3, 9, 20, None, None, 15, 7]
assert to_list(build_tree([-1], [-1])) == [-1]
assert to_list(build_tree([], [])) == []
assert to_list(build_tree([1, 2, 3], [3, 2, 1])) == [1, 2, None, 3]
print(to_list(root), to_list(build_tree([1, 2, 3], [3, 2, 1])))
```

```
[3, 9, 20, None, None, 15, 7] [1, 2, None, 3]
```

**Complexity.** $O(n)$ time and $O(n)$ space for the map, against $O(n^2)$ time without it.

### P22. Serialise and deserialise a binary tree — turn a tree into a string and back again

**Which template.** Pre-order with explicit null markers, in both directions.
**The trick.** A single traversal determines a tree only if the nulls are written down, so emit a marker such as `#` for every missing child. With the markers present, pre-order alone is enough, because the very next token after a node is always the root of its left subtree. Deserialise with an iterator over the tokens, so that the recursion consumes them in exactly the order they were produced and you never have to manage an index by hand.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None
def to_list(root):                                   ## inverse of build, trailing Nones trimmed
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(None if node is None else node.val)
        if node:
            queue.append(node.left)
            queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out

def serialize(root):
    out = []

    def walk(node):
        if node is None:
            out.append("#")                          ## the null marker is what makes this reversible
            return
        out.append(str(node.val))
        walk(node.left)
        walk(node.right)

    walk(root)
    return ",".join(out)

def deserialize(data):
    tokens = iter(data.split(","))                   ## an iterator consumes in traversal order

    def rebuild():
        token = next(tokens)
        if token == "#":
            return None
        node = TreeNode(int(token))
        node.left = rebuild()
        node.right = rebuild()
        return node

    return rebuild()

## tests

assert serialize(build([1, 2, 3, None, None, 4, 5])) == "1,2,#,#,3,4,#,#,5,#,#"
for values in ([1, 2, 3, None, None, 4, 5], [], [1], [1, None, 2, None, 3]):
    assert to_list(deserialize(serialize(build(values)))) == values
print(serialize(build([1, 2, 3, None, None, 4, 5])))
```

```
1,2,#,#,3,4,#,#,5,#,#
```

**Complexity.** $O(n)$ time and $O(n)$ space in each direction.

### P23. Count good nodes in a binary tree — count nodes with no larger value anywhere on the path from the root

**Which template.** Template 1 with a running maximum pushed down, which is a pre-order use of the arguments.
**The trick.** The condition looks at ancestors, so the information must travel downwards, not upwards. Carry the largest value seen on the path so far as an argument. A node is good when `node.val >= best_so_far`, and the value passed to the children is `max(best_so_far, node.val)`. The return value is just a count, summed from the two children. Start the recursion with the root value, or with negative infinity; both make the root good, which is correct.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def good_nodes(root):
    def count(node, best_so_far):
        if node is None:
            return 0
        is_good = 1 if node.val >= best_so_far else 0    ## >= because equal is still good
        best_so_far = max(best_so_far, node.val)         ## the value the CHILDREN will see
        return is_good + count(node.left, best_so_far) + count(node.right, best_so_far)
    return count(root, float("-inf"))

## tests

assert good_nodes(build([3, 1, 4, 3, None, 1, 5])) == 4
assert good_nodes(build([3, 3, None, 4, 2])) == 3
assert good_nodes(build([1])) == 1
assert good_nodes(build([])) == 0
print(good_nodes(build([3, 1, 4, 3, None, 1, 5])), good_nodes(build([3, 3, None, 4, 2])))
```

```
4 3
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P24. Subtree of another tree — decide whether one tree appears as a subtree of another

**Which template.** Template 1 wrapped around the P3 equality check.
**The trick.** Two recursions, and keeping them separate is the whole point. The outer one walks every node of the big tree and asks "does the subtree rooted here equal the target". The inner one is `is_same_tree` unchanged. Trying to write one recursion that does both produces a function that matches a partial shape and reports a false positive. Note also that matching must reach a leaf: a target that is a prefix of a branch is not a subtree.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def is_same_tree(p, q):
    if p is None and q is None:
        return True
    if p is None or q is None:
        return False
    return p.val == q.val and is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

def is_subtree(root, target):
    if target is None:
        return True
    if root is None:
        return False
    if is_same_tree(root, target):                   ## try a full match rooted HERE
        return True
    return is_subtree(root.left, target) or is_subtree(root.right, target)

## tests

assert is_subtree(build([3, 4, 5, 1, 2]), build([4, 1, 2])) is True
assert is_subtree(build([3, 4, 5, 1, 2, None, None, None, None, 0]), build([4, 1, 2])) is False
assert is_subtree(build([1, 1]), build([1])) is True
assert is_subtree(build([]), build([1])) is False
print(is_subtree(build([3, 4, 5, 1, 2]), build([4, 1, 2])),
      is_subtree(build([3, 4, 5, 1, 2, None, None, None, None, 0]), build([4, 1, 2])))
```

```
True False
```

**Complexity.** $O(n m)$ time in the worst case, where $n$ and $m$ are the two sizes, and $O(h)$ space.

### P25. Longest univalue path — the most edges on a path where every node has the same value

**Which template.** Template 2, the fourth member of the "return one thing, record another" family.
**The trick.** The returned quantity is the longest single arm of equal values starting at this node and going down. The recorded quantity is the two arms joined, because a bent path cannot be extended by the parent. The extra step compared with the diameter is that an arm counts only when the child value equals the node value; otherwise that arm contributes 0. Compute the child arms first and then zero them, rather than skipping the recursive call, because every node must still be visited as a potential centre.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def longest_univalue_path(root):
    best = 0

    def arm(node):
        nonlocal best
        if node is None:
            return 0
        left, right = arm(node.left), arm(node.right)     ## always recurse: every node is a centre
        left_arm = left + 1 if node.left and node.left.val == node.val else 0
        right_arm = right + 1 if node.right and node.right.val == node.val else 0
        best = max(best, left_arm + right_arm)            ## RECORD the bent path
        return max(left_arm, right_arm)                   ## RETURN one arm

    arm(root)
    return best

## tests

assert longest_univalue_path(build([5, 4, 5, 1, 1, None, 5])) == 2
assert longest_univalue_path(build([1, 4, 5, 4, 4, None, 5])) == 2
assert longest_univalue_path(build([1, 1, 1, 1, 1, None, 1])) == 4
assert longest_univalue_path(build([])) == 0
print(longest_univalue_path(build([5, 4, 5, 1, 1, None, 5])),
      longest_univalue_path(build([1, 1, 1, 1, 1, None, 1])))
```

```
2 4
```

**Complexity.** $O(n)$ time, $O(h)$ space.

### P26. Sum root to leaf numbers — read each root-to-leaf path as a decimal number and total them

**Which template.** Template 1 with the running number pushed down, and the totals summed on the way up.
**The trick.** Both directions are used in the same six lines, which makes this a good closing exercise. The running value `current * 10 + node.val` travels down as an argument, and the sum of the two child results travels up as the return value. Record the number only at a real leaf; returning `current` at a null node counts every one-child path twice, once through each missing side.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def build(values):                                   ## level-order list, None for a gap
    nodes = [None if v is None else TreeNode(v) for v in values]
    kids = nodes[:0:-1]                              ## children, in reverse so pop() is FIFO
    for node in nodes:
        if node:
            if kids: node.left = kids.pop()
            if kids: node.right = kids.pop()
    return nodes[0] if nodes else None

def sum_numbers(root):
    def walk(node, current):
        if node is None:
            return 0                                 ## contributes nothing, and is NOT a leaf
        current = current * 10 + node.val            ## the number so far travels DOWN
        if node.left is None and node.right is None:
            return current                           ## a real leaf closes the number
        return walk(node.left, current) + walk(node.right, current)
    return walk(root, 0)

## tests

assert sum_numbers(build([1, 2, 3])) == 25           ## 12 + 13
assert sum_numbers(build([4, 9, 0, 5, 1])) == 1026   ## 495 + 491 + 40
assert sum_numbers(build([1, 2])) == 12              ## one path only, counted once
assert sum_numbers(build([])) == 0
print(sum_numbers(build([1, 2, 3])), sum_numbers(build([4, 9, 0, 5, 1])), sum_numbers(build([1, 2])))
```

```
25 1026 12
```

**Complexity.** $O(n)$ time, $O(h)$ space.

## Tricks and tips

**Say the three sentences before you type.** What does one call return, what does it need from its
parent, and what happens at a null node. Those three sentences are the design, and the body of the
function is then mechanical. Saying them out loud also gives the interviewer something to correct
before you have written twenty lines in the wrong shape, which is the cheapest possible correction.

**A path that bends is recorded, a quantity that extends is returned.** Heights, subtree sums, node
counts and single arms all extend upwards, so they are return values. Diameters, maximum path sums and
longest univalue paths bend at a node, so the parent cannot use them and they belong in a nonlocal
variable. If you cannot decide which a quantity is, ask whether the parent could build a longer version
of it by adding itself on top. If the answer is no, it bends.

**A sentinel is a nonlocal variable in disguise.** The balanced check returns `-1` for "unbalanced"
because a real height is never negative. Any impossible value works: `None`, `float("-inf")`, a tuple.
Use a sentinel when the failure should stop the recursion immediately, and use a nonlocal when every
node must still be visited.

**BFS wins whenever the question mentions levels or the word "first".** Minimum depth, the right side
view, level averages, the deepest leaves and the first node at a given depth are all one loop with one
line changed. DFS can compute all of them, but it visits the whole tree, while BFS stops at the first
match.

**In-order on a BST is sorted, and that is the only fact you need for a large family.** The k-th
smallest, validation by checking the sequence is increasing, the BST iterator, the minimum absolute
difference between any two values, and converting a BST to a sorted list are all in-order traversals
with different bookkeeping at the visit point.

**Use the ordering to avoid searching both sides.** In a BST, search, insert, the lowest common
ancestor and range queries all become a single root-to-leaf walk that costs $O(h)$. Whenever the
statement says "binary search tree" and your solution still visits every node, you have missed the
point of the question.

**Pass index ranges, never slices.** In the sorted-array-to-BST and the preorder-plus-inorder builds,
slicing a list at each node copies it and turns an $O(n)$ construction into $O(n \log n)$ or $O(n^2)$.
Pass `low` and `high` and index the original list.

**A value-to-index dictionary turns a repeated search into a lookup.** In the preorder-plus-inorder
build, the linear search for the root inside the inorder segment is the only expensive step, and one
dictionary removes it.

**Offer the iterative version before you are asked.** Say that the recursion is $O(h)$ on the call
stack, that a degenerate tree makes that $O(n)$, and that Python's default recursion limit is about a
thousand frames. Then write the explicit stack if there is time.

## The bugs that cost the round

**Returning the wrong quantity.** Trying to return the diameter, or the maximum path sum, instead of
the height or the single arm. The recursion then has nothing useful to combine and you stall. Decide
the return value before anything else.

**Getting the null base case wrong by one.** A null node has height 0 and a leaf has height 1, so the
answer counts nodes. If the problem counts edges, the null case is `-1`. Mixing the two shifts every
answer in the tree by one, and a single-node test does not reveal it.

**Treating a node with one child as a leaf.** Minimum depth and path sum both break on this. A leaf has
**both** children missing. `1 + min(left, right)` answers 1 for the tree `[1, 2]`, and the correct
answer is 2.

**Validating a BST against the parent only.** The ordering constrains the whole subtree, not one edge.
Carry `low` and `high` down. P17 prints a tree where the local check says yes and the answer is no.

**Forgetting to clamp negative contributions.** In the maximum path sum, an arm worth less than zero
must be dropped with `max(0, child)`. Without the clamp, a negative subtree drags the answer down.

**Appending the shared path list instead of a copy.** In path sum II, `out.append(path)` stores a
reference, and by the end every stored path is the same empty list. Write `list(path)`.

**Losing the level boundaries in BFS.** The snapshot `size = len(queue)` must be taken before any child
is pushed. Without it every level runs into the next one.

**Overwriting a child before reading it.** In the invert, assign both children in one tuple assignment.
Two separate statements make the second one read a subtree that the first has already replaced.

## Done when

- Given a tree problem you have not seen, you can say what the recursion returns, what it carries down,
  and what the null base case is, in three sentences and before writing any code.
- You can write the diameter, the maximum path sum, the one-pass balanced check and the longest
  univalue path from a blank file in five minutes, and name the one shape all four share.
- You can write the level-order BFS from memory with the `size = len(queue)` snapshot, and convert it
  to zigzag, right side view and level averages by changing one line each.
- You can write the iterative in-order traversal from a blank file, and explain why a BST iterator is
  the same code split across a constructor and a `next` method.
