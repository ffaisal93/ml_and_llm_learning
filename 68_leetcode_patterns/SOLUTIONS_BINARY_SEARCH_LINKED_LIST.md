# NeetCode 150 — Binary Search + Linked List — Solutions

> Simplest-optimal Python solutions with intuition, memory hooks, and pressure tips.

---

## Binary Search (7)

### 1. Binary Search (LC 704)

**Problem.** Find target in sorted array, $O(\log n)$.

```python
def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target: return m
        if nums[m] < target:  l = m + 1
        else:                 r = m - 1
    return -1
```

**Complexity.** $O(\log n)$.

**Hook.** "Half each step. `l ≤ r` form."

**Watch out.** Pick one form (`l ≤ r` or `l < r`) and stick with it. Mixing causes off-by-ones.

---

### 2. Search a 2D Matrix (LC 74)

**Problem.** Sorted 2D matrix (each row sorted; first of each row > last of previous). Find target.

**Intuition.** Treat as flattened 1D. Map index $i$ to (row, col) = (i // n, i % n).

```python
def searchMatrix(matrix, target):
    if not matrix: return False
    m, n = len(matrix), len(matrix[0])
    l, r = 0, m * n - 1
    while l <= r:
        mid = (l + r) // 2
        v = matrix[mid // n][mid % n]
        if v == target: return True
        if v < target:  l = mid + 1
        else:           r = mid - 1
    return False
```

**Complexity.** $O(\log(mn))$.

**Hook.** "Flatten via (i//n, i%n)."

---

### 3. Koko Eating Bananas (LC 875)

**Problem.** Min eating speed `k` so Koko finishes piles within `h` hours.

**Intuition.** Binary search on the answer (speed). Feasibility = sum of `ceil(p / k)` ≤ h. Monotonic in k.

```python
def minEatingSpeed(piles, h):
    l, r = 1, max(piles)
    while l < r:
        m = (l + r) // 2
        hours = sum((p + m - 1) // m for p in piles)
        if hours <= h: r = m
        else:          l = m + 1
    return l
```

**Complexity.** $O(n \log(\max p))$.

**Hook.** "Binary search on speed; feasibility = ceil-sum."

---

### 4. Find Min in Rotated Sorted Array (LC 153)

**Problem.** Find min in array rotated some unknown times.

**Intuition.** Compare `mid` to `right`. If `nums[mid] > nums[right]`, min is in right half; else left half (including mid).

```python
def findMin(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        m = (l + r) // 2
        if nums[m] > nums[r]: l = m + 1
        else:                 r = m
    return nums[l]
```

**Complexity.** $O(\log n)$.

**Hook.** "Compare mid to right; pivot is in larger half."

**Watch out.** Compare to right, *not* left (avoids edge cases with not-rotated arrays).

---

### 5. Search in Rotated Sorted Array (LC 33)

**Problem.** Find target in rotated sorted array.

**Intuition.** At each step, one half is sorted. Decide which half target falls in.

```python
def search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] == target: return m
        if nums[l] <= nums[m]:                # left half sorted
            if nums[l] <= target < nums[m]: r = m - 1
            else:                           l = m + 1
        else:                                  # right half sorted
            if nums[m] < target <= nums[r]: l = m + 1
            else:                           r = m - 1
    return -1
```

**Complexity.** $O(\log n)$.

**Hook.** "One half is always sorted; check if target lies in it."

---

### 6. Time Based Key-Value Store (LC 981)

**Problem.** `set(key, val, ts)`, `get(key, ts)` returns latest value with timestamp ≤ ts.

**Intuition.** Per-key list of (ts, val), strictly increasing in ts (always set with monotonically increasing ts). Binary-search the list.

```python
class TimeMap:
    def __init__(self):
        self.store = {}   # key -> list of (ts, val)
    def set(self, key, val, ts):
        self.store.setdefault(key, []).append((ts, val))
    def get(self, key, ts):
        if key not in self.store: return ""
        arr = self.store[key]
        l, r = 0, len(arr) - 1
        ans = ""
        while l <= r:
            m = (l + r) // 2
            if arr[m][0] <= ts:
                ans = arr[m][1]
                l = m + 1
            else:
                r = m - 1
        return ans
```

**Complexity.** $O(1)$ set, $O(\log n)$ get.

**Hook.** "Per-key sorted list; bisect-right."

---

### 7. Median of Two Sorted Arrays (LC 4)

**Problem.** Median of two sorted arrays in $O(\log(\min(m, n)))$.

**Intuition.** Binary-search the partition position in the smaller array. Combined left half size = (m + n + 1) // 2. Verify left max ≤ right min on both sides.

```python
def findMedianSortedArrays(A, B):
    if len(A) > len(B): A, B = B, A
    m, n = len(A), len(B)
    half = (m + n + 1) // 2
    l, r = 0, m
    while l <= r:
        i = (l + r) // 2
        j = half - i
        Aleft  = A[i - 1] if i > 0 else float('-inf')
        Aright = A[i]     if i < m else float('inf')
        Bleft  = B[j - 1] if j > 0 else float('-inf')
        Bright = B[j]     if j < n else float('inf')
        if Aleft <= Bright and Bleft <= Aright:
            if (m + n) % 2:
                return max(Aleft, Bleft)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        if Aleft > Bright:
            r = i - 1
        else:
            l = i + 1
```

**Complexity.** $O(\log(\min(m, n)))$.

**Hook.** "Partition both; left-max ≤ right-min on both sides."

**Watch out.** Run BS on the smaller array (so `i` range is small). Use ±inf sentinels for boundary cases.

---

## Linked List (11)

Standard ListNode:

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next
```

### 8. Reverse Linked List (LC 206)

**Intuition.** Walk the list with three pointers (prev, curr, nxt). Flip each link.

```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Three pointers, flip in place."

---

### 9. Merge Two Sorted Lists (LC 21)

**Intuition.** Dummy node + tail pointer; pick smaller head each step.

```python
def mergeTwoLists(l1, l2):
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next
```

**Complexity.** $O(m + n)$.

**Hook.** "Dummy + tail."

---

### 10. Reorder List (LC 143)

**Problem.** L0 → L1 → ... → Ln becomes L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...

**Intuition.** Three steps: (1) find middle (fast/slow); (2) reverse second half; (3) merge two halves alternating.

```python
def reorderList(head):
    # 1. Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
    # 2. Reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        nxt = curr.next
        curr.next = prev
        prev, curr = curr, nxt
    # 3. Merge alternating
    first, second = head, prev
    while second:
        f1, s1 = first.next, second.next
        first.next = second
        second.next = f1
        first, second = f1, s1
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Middle → reverse → merge alternating."

---

### 11. Remove Nth from End (LC 19)

**Intuition.** Two pointers; advance fast by n+1 (so when fast is null, slow is at the node *before* the one to remove).

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Gap = n+1, advance both, slow.next = slow.next.next."

**Watch out.** Use a dummy so removing the head works uniformly.

---

### 12. Copy List with Random Pointer (LC 138)

**Intuition.** Hash map old→new. First pass creates new nodes; second pass wires next + random.

```python
def copyRandomList(head):
    if not head: return None
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.val)
        curr = curr.next
    curr = head
    while curr:
        old_to_new[curr].next   = old_to_new.get(curr.next)
        old_to_new[curr].random = old_to_new.get(curr.random)
        curr = curr.next
    return old_to_new[head]
```

**Complexity.** $O(n)$ time, $O(n)$ space.

**Hook.** "Map old→new; two passes."

---

### 13. Add Two Numbers (LC 2)

**Problem.** Two linked lists, digits in reverse order, return sum as linked list.

**Intuition.** Walk both with carry; build dummy-headed result.

```python
def addTwoNumbers(l1, l2):
    dummy = ListNode()
    tail = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        s = v1 + v2 + carry
        carry = s // 10
        tail.next = ListNode(s % 10)
        tail = tail.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```

**Complexity.** $O(\max(m, n))$.

**Hook.** "Carry-aware addition with dummy."

---

### 14. Linked List Cycle (LC 141)

**Intuition.** Floyd's tortoise and hare. If they meet, there's a cycle.

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast: return True
    return False
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Tortoise + hare."

---

### 15. Find the Duplicate Number (LC 287)

**Problem.** Array of n+1 integers in `[1, n]`, exactly one duplicate. Find without modifying, $O(1)$ extra space.

**Intuition.** Treat as linked list where `i → nums[i]`. Duplicate creates a cycle. Floyd's algorithm finds the entry of the cycle = the duplicate.

```python
def findDuplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = nums[0]
    while slow != fast:
        slow, fast = nums[slow], nums[fast]
    return slow
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Floyd cycle on the implicit i → nums[i] graph."

---

### 16. LRU Cache (LC 146)

**Intuition.** Doubly-linked list (recency order) + hash map (key → node). Get and put both $O(1)$.

```python
class Node:
    def __init__(self, k=0, v=0):
        self.k, self.v = k, v
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.map = {}
        self.head = Node(); self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, node):           # add right after head (MRU)
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key):
        if key not in self.map: return -1
        node = self.map[key]
        self._remove(node)
        self._add(node)
        return node.v

    def put(self, key, value):
        if key in self.map:
            self._remove(self.map[key])
        node = Node(key, value)
        self.map[key] = node
        self._add(node)
        if len(self.map) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.k]
```

**Complexity.** $O(1)$ get and put.

**Hook.** "DLL + hashmap. Head=MRU, tail=LRU."

**Watch out.** Always use sentinel head/tail dummies.

---

### 17. Merge K Sorted Lists (LC 23)

**Intuition.** Min-heap of (val, index, node). Pop smallest; push its `next`.

```python
import heapq
def mergeKLists(lists):
    h = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(h, (node.val, i, node))
    dummy = ListNode()
    tail = dummy
    while h:
        val, i, node = heapq.heappop(h)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(h, (node.next.val, i, node.next))
    return dummy.next
```

**Complexity.** $O(N \log k)$ where N = total nodes.

**Hook.** "Heap of heads."

**Watch out.** Tuple needs `i` as tiebreaker (nodes aren't comparable).

---

### 18. Reverse Nodes in k-Group (LC 25)

**Intuition.** Repeatedly find a group of k nodes; reverse just that segment; reattach. Use a `groupPrev` pointer.

```python
def reverseKGroup(head, k):
    dummy = ListNode(0, head)
    groupPrev = dummy
    while True:
        kth = groupPrev
        for _ in range(k):
            kth = kth.next
            if not kth: return dummy.next
        groupNext = kth.next
        # reverse [groupPrev.next ... kth]
        prev, curr = groupNext, groupPrev.next
        while curr != groupNext:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        # swap pointers
        tmp = groupPrev.next
        groupPrev.next = kth
        groupPrev = tmp
```

**Complexity.** $O(n)$ time, $O(1)$ space.

**Hook.** "Find kth, reverse segment, swap groupPrev."

**Watch out.** When fewer than k nodes remain, leave them as-is — `if not kth: return`.

---

## Summary table — Binary Search + Linked List

| # | Problem | Pattern | Time | Hook |
|---|---------|---------|------|------|
| 1 | Binary Search | classic BS | $O(\log n)$ | half each step |
| 2 | Search 2D Matrix | flatten 1D | $O(\log mn)$ | mid//n, mid%n |
| 3 | Koko Bananas | BS on answer | $O(n \log p)$ | feasibility = ceil sum |
| 4 | Min Rotated | mid vs right | $O(\log n)$ | mid > right ⇒ go right |
| 5 | Search Rotated | one half sorted | $O(\log n)$ | which half sorted? |
| 6 | TimeMap | per-key bisect | $O(\log n)$ | sorted ts list |
| 7 | Median Two Sorted | partition BS | $O(\log\min)$ | left-max ≤ right-min |
| 8 | Reverse List | 3 pointers | $O(n)$ | flip in place |
| 9 | Merge Two | dummy + tail | $O(n)$ | dummy |
| 10 | Reorder List | mid + rev + merge | $O(n)$ | 3 steps |
| 11 | Remove Nth | gap pointers | $O(n)$ | gap = n+1 |
| 12 | Copy w/ Random | hash old→new | $O(n)$ | 2 passes |
| 13 | Add Two Numbers | carry walk | $O(n)$ | carry until both null |
| 14 | Cycle | Floyd | $O(n)$ | tortoise + hare |
| 15 | Duplicate Number | Floyd implicit | $O(n)$ | i→nums[i] graph |
| 16 | LRU | DLL + hashmap | $O(1)$ | MRU at head |
| 17 | Merge K | heap of heads | $O(N\log k)$ | heap |
| 18 | Reverse k-Group | seg reverse | $O(n)$ | groupPrev pointer |
