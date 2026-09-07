# Linked lists: every variation

A linked-list problem is almost never about the algorithm. The algorithms are trivial: walk forward,
compare, join. The difficulty is pointer discipline. A node is reachable only through the pointer that
names it, so you get exactly one chance to hold a reference before you overwrite it. Write
`curr.next = prev` before you have saved `curr.next` and the rest of the list is gone, with no error
message and no way back.

Two habits solve most of these problems. The first is the **dummy head**: allocate one throwaway node
in front of the real list, so that inserting or deleting the first element is the same code as
inserting or deleting any other element. That single line removes the "what if the head changes" case
from every function you will write. The second is to **remember the next node before you rewrite a
pointer**. Every reversal loop is the same four lines in the same order — save, rewrite, advance, advance
— and the order is not negotiable.

Reversal is the primitive. Reverse Nodes in k-Group, Reorder List, Palindrome Linked List and Add Two
Numbers in forward order are all built from it. Therefore you must be able to write the three-pointer
reversal without thinking, correctly, on the first try. Practise that one until it is muscle memory and
half of this chapter becomes assembly rather than invention.

## Recognising it from the phrasing

| The interviewer says | They mean | The tool |
|---|---|---|
| "reverse the list", "reverse a part of it" | pointer reversal | the three-pointer walk |
| "delete the head", "insert before the first node", "the head may change" | head is a special case | a dummy head node |
| "find the middle", "does it have a cycle", "where does the cycle start" | one pass, no length | fast and slow pointers |
| "merge two sorted lists" | interleave by comparison | dummy head plus two pointers |
| "the kth node from the end" | one pass, no length | two pointers offset by k |
| "reorder", "interleave the halves", "is it a palindrome" | three phases | split, reverse, merge |
| "in groups of k", "every k nodes" | reverse a sublist | boundary pointers held on both sides |
| "copy a list with extra pointers" | clone with aliasing | interleave-and-split, or an old-to-new map |

Before you write a line, draw three consecutive nodes on paper and name every pointer you are going to
move. Then ask three questions in order. What happens when the list is empty? What happens when it has
exactly one node? What happens when the operation touches the head? Almost every linked-list bug is one
of those three cases and not the general case, because the general case is the one you were thinking
about while you wrote the loop. The dummy head answers the third question by construction, and the
first two are usually a single guard line at the top of the function. Two minutes of drawing is worth
more here than in any other pattern, because there is no array index to print and no way to see the
damage after it is done.

## The templates

Every block below defines `ListNode`, a `build` helper that makes a list from a Python list, and a
`to_list` helper that reads it back. They are repeated in every block so each one runs standalone.

**Template 1 — iterative reversal.** Use whenever direction must change. Three pointers: `prev`, `curr`
and the saved `next_node`. The answer is the final `prev`, which is the new head.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def reverse_list(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next        ## 1. SAVE before you destroy
        curr.next = prev             ## 2. rewrite
        prev = curr                  ## 3. advance prev
        curr = next_node             ## 4. advance curr
    return prev                      ## prev is the new head

## tests

assert to_list(reverse_list(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_list(build([1]))) == [1]
assert to_list(reverse_list(build([]))) == []
print(to_list(reverse_list(build([1, 2, 3, 4, 5]))))
```

```
[5, 4, 3, 2, 1]
```

**Template 2 — dummy head for insertion and deletion.** Use whenever the first node might be removed or
replaced. The answer is always `dummy.next`, never the original `head`.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def remove_all(head, target):
    dummy = ListNode(0, head)        ## one throwaway node in front of the real list
    prev = dummy
    while prev.next:
        if prev.next.val == target:
            prev.next = prev.next.next     ## unlink: prev does not move
        else:
            prev = prev.next               ## keep: prev moves
    return dummy.next                      ## the head may have changed

## tests

assert to_list(remove_all(build([1, 2, 6, 3, 4, 5, 6]), 6)) == [1, 2, 3, 4, 5]
assert to_list(remove_all(build([7, 7, 7]), 7)) == []
assert to_list(remove_all(build([]), 1)) == []
print(to_list(remove_all(build([1, 2, 6, 3, 4, 5, 6]), 6)))
```

```
[1, 2, 3, 4, 5]
```

**Template 3 — fast and slow pointers.** Use for the middle of the list and for cycle detection, both
in one pass and with $O(1)$ memory. See the two-pointers chapter for the same idea on arrays.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def middle_node(head):
    slow = fast = head
    while fast and fast.next:        ## fast moves two, slow moves one
        slow = slow.next
        fast = fast.next.next
    return slow                      ## for even length this is the SECOND middle

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:             ## identity, not equality
            return True
    return False

## tests

assert middle_node(build([1, 2, 3, 4, 5])).val == 3
assert middle_node(build([1, 2, 3, 4])).val == 3
looped = build([1, 2, 3])
looped.next.next.next = looped.next
assert has_cycle(looped) is True
assert has_cycle(build([1, 2, 3])) is False
print(middle_node(build([1, 2, 3, 4, 5])).val, has_cycle(looped))
```

```
3 True
```

**Template 4 — merge two sorted lists with a dummy head.** The skeleton is identical to template 2: a
dummy, a moving `tail`, and `dummy.next` returned at the end.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def merge_two(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:           ## <= keeps the merge stable
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b               ## one list is empty, attach the rest in O(1)
    return dummy.next

## tests

assert to_list(merge_two(build([1, 2, 4]), build([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
assert to_list(merge_two(build([]), build([0]))) == [0]
assert to_list(merge_two(build([]), build([]))) == []
print(to_list(merge_two(build([1, 2, 4]), build([1, 3, 4]))))
```

```
[1, 1, 2, 3, 4, 4]
```

The last line of the merge is worth naming. `tail.next = a or b` attaches the entire remaining tail in
one assignment, because the leftover list is already sorted and already linked. Copying it node by node
is not wrong, but it is longer code and one more loop to get wrong.

## Reversing a sublist

This is the highest-value trick on the page, because Reverse Nodes in k-Group, Reorder List, Palindrome
Linked List and Reverse Linked List II all reduce to it. Whole-list reversal is easy because both ends
are free. Sublist reversal is harder for one reason only: after you reverse the middle, the two nodes at
the boundary are pointing at the wrong things, and you cannot find them again once the middle is
reversed. Therefore you must hold them before you start.

You need exactly two boundary pointers. Call them `before`, the node immediately in front of the
sublist, and `after`, the node immediately behind it. Reverse the nodes strictly between them with the
ordinary three-pointer walk, seeding `prev` with `after` instead of with `None` so the tail of the
reversed piece is already reconnected. Then set `before.next` to the last node of the sublist, which is
now its first. Two reconnections, one of them free.

**Worked example.** Take `1 -> 2 -> 3 -> 4 -> 5 -> 6` and reverse in groups of `k = 2`. A dummy node
sits in front, so `before` starts as the dummy.

For the first group, `before` is the dummy and the group is `1 -> 2`. Walk `k` steps from `before` to
find `kth`, which is node 2, and `after` is `kth.next`, which is node 3. Now reverse from node 1 up to
but not including node 3, starting `prev` at node 3. Node 1 points at 3, then node 2 points at 1. The
piece is `2 -> 1 -> 3`. Set `before.next = kth`, so the dummy points at node 2. The list is now
`2 -> 1 -> 3 -> 4 -> 5 -> 6`. Finally set `before` to node 1, which is the tail of the group just
reversed and the node in front of the next group.

Repeat for `3 -> 4` and the list becomes `2 -> 1 -> 4 -> 3 -> 5 -> 6`, then for `5 -> 6` and it becomes
`2 -> 1 -> 4 -> 3 -> 6 -> 5`. The single line that people forget is the last one, saving the old
`before.next` as the *next* `before` before you overwrite it — after the reconnection that node is no
longer reachable from where you are standing.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def reverse_between(before, after):
    ## reverse the nodes strictly between `before` and `after`
    prev, curr = after, before.next
    first = before.next               ## it becomes the tail of the reversed piece
    while curr is not after:
        next_node = curr.next
        curr.next = prev
        prev, curr = curr, next_node
    before.next = prev                ## prev is the last node, now the first
    return first                      ## the new `before` for the next group

## tests

dummy = ListNode(0, build([1, 2, 3, 4, 5, 6]))
before = dummy
for _ in range(3):
    kth = before
    for _ in range(2):
        kth = kth.next
    before = reverse_between(before, kth.next)
assert to_list(dummy.next) == [2, 1, 4, 3, 6, 5]
print(to_list(dummy.next))
```

```
[2, 1, 4, 3, 6, 5]
```

## The problems

### P1. Reverse Linked List — return the list with all pointers reversed

**Which template.** Template 1, and the recursive form is the same walk written backwards.
**The trick.** The iterative version is four lines in a fixed order: save, rewrite, advance, advance.
The recursive version reverses the tail first and then makes the node after `head` point back at
`head`. The line `head.next.next = head` reads strangely, so say it in words: "the node that follows me
should now follow me in the other direction". Then `head.next = None`, or the last two nodes form a
two-cycle.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def reverse_iterative(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev, curr = curr, next_node
    return prev

def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)   ## the tail is reversed first
    head.next.next = head                     ## my successor now points back at me
    head.next = None                          ## or the last pair becomes a 2-cycle
    return new_head

## tests

assert to_list(reverse_iterative(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_recursive(build([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_recursive(build([]))) == []
print(to_list(reverse_iterative(build([1, 2, 3]))), to_list(reverse_recursive(build([1, 2, 3]))))
```

```
[3, 2, 1] [3, 2, 1]
```

**Complexity.** Both $O(n)$ time. Iterative is $O(1)$ space, recursive is $O(n)$ stack, which matters
on a list of a million nodes and is worth saying.

### P2. Merge Two Sorted Lists — splice two sorted lists into one sorted list

**Which template.** Template 4 exactly.
**The trick.** No node is created. You are only rewriting `next` pointers on nodes that already exist,
so the space is $O(1)$ apart from the dummy. Use `<=` rather than `<` so equal values keep their
original relative order, which is what "stable" means and what the follow-up question is about.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def merge_two_lists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next, list1 = list1, list1.next
        else:
            tail.next, list2 = list2, list2.next
        tail = tail.next
    tail.next = list1 or list2
    return dummy.next

## tests

assert to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))) == [1, 1, 2, 3, 4, 4]
assert to_list(merge_two_lists(build([]), build([]))) == []
assert to_list(merge_two_lists(build([5]), build([1, 2]))) == [1, 2, 5]
print(to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))))
```

```
[1, 1, 2, 3, 4, 4]
```

**Complexity.** $O(m + n)$ time, $O(1)$ space.

### P3. Linked List Cycle — does the list contain a loop, and where does it start

**Which template.** Template 3, Floyd's cycle detection.
**The trick.** If there is a cycle the fast pointer gains one position on the slow pointer per step, so
it must eventually land on it. For the entry point, the arithmetic is worth memorising: let the tail
before the loop have length `a` and let the meeting point be `b` steps into a loop of length `c`. Fast
has travelled twice as far as slow, so $a + b + mc = 2(a + b)$ for some integer m, which gives
$a = mc - b$. Therefore a pointer started at the head and a pointer started at the meeting point, both
moving one step at a time, meet exactly at the loop entry.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False

def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:                      ## phase 1: they meet inside the loop
            finder = head
            while finder is not slow:         ## phase 2: both move ONE step
                finder, slow = finder.next, slow.next
            return finder
    return None

## tests

looped = build([3, 2, 0, -4])
looped.next.next.next.next = looped.next      ## tail points at index 1
assert has_cycle(looped) is True
assert detect_cycle_start(looped).val == 2
assert has_cycle(build([1])) is False
assert detect_cycle_start(build([1, 2])) is None
print(has_cycle(looped), detect_cycle_start(looped).val)
```

```
True 2
```

**Complexity.** $O(n)$ time, $O(1)$ space. A `set` of visited nodes also works and is $O(n)$ space; say
you know it and prefer Floyd.

### P4. Reorder List — rearrange `L0 -> L1 -> ... -> Ln` into `L0 -> Ln -> L1 -> Ln-1 -> ...`

**Which template.** Three phases: template 3 to split, template 1 to reverse, then a zip merge.
**The trick.** Do not try to do it in one pass. Find the middle with fast and slow, cut the list there
by setting `slow.next = None`, reverse the second half, then interleave the two halves. Cutting is the
step people forget, and without it the reversed second half still points back into the first and the
merge builds a cycle.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def reorder_list(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next             ## slow lands on the END of the first half
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    second, slow.next = slow.next, None      ## CUT, or the merge builds a cycle
    prev = None
    while second:
        second.next, prev, second = prev, second, second.next
    first = head
    while prev:                              ## zip the two halves together
        first.next, prev.next, first, prev = prev, first.next, first.next, prev.next
    return head

## tests

assert to_list(reorder_list(build([1, 2, 3, 4]))) == [1, 4, 2, 3]
assert to_list(reorder_list(build([1, 2, 3, 4, 5]))) == [1, 5, 2, 4, 3]
assert to_list(reorder_list(build([1]))) == [1]
print(to_list(reorder_list(build([1, 2, 3, 4, 5]))))
```

```
[1, 5, 2, 4, 3]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P5. Remove Nth Node From End of List — delete the nth node counting from the tail, in one pass

**Which template.** Template 2 for the deletion plus two pointers offset by n.
**The trick.** Move `fast` n steps ahead, then advance both until `fast` falls off the end. The gap
between them is fixed, so `slow` lands exactly where you need it. Start both at the **dummy**, not at
the head: when n equals the length the node to delete is the head itself, and only the dummy gives you a
predecessor for it.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy                      ## BOTH start at the dummy
    for _ in range(n):
        fast = fast.next
    while fast.next:                         ## slow stops one BEFORE the target
        slow, fast = slow.next, fast.next
    slow.next = slow.next.next
    return dummy.next

## tests

assert to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
assert to_list(remove_nth_from_end(build([1]), 1)) == []
assert to_list(remove_nth_from_end(build([1, 2]), 2)) == [2]
print(to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)))
```

```
[1, 2, 3, 5]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P6. Copy List with Random Pointer — deep-copy a list where each node also has a random pointer

**Which template.** None of the four. Two accepted answers: a map from old node to new node, or the
interleave-and-split trick.
**The trick.** The problem is that a random pointer may target a node you have not created yet. The map
solves it by making every new node first and wiring the pointers in a second pass. The interleave trick
solves it with $O(1)$ extra space by putting each copy directly behind its original, so
`copy.random = original.random.next` reads the correct copy by construction. Then split the woven list
back into two. Restore the original list during the split, because the interviewer will check.

```python
class Node:
    def __init__(self, val, next=None, random=None):
        self.val, self.next, self.random = val, next, random

def copy_random_list(head):
    if not head:
        return None
    curr = head
    while curr:                              ## 1. weave: A -> A' -> B -> B' -> ...
        curr.next = Node(curr.val, curr.next)
        curr = curr.next.next
    curr = head
    while curr:                              ## 2. the copy of X sits at X.next
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    curr, new_head = head, head.next
    while curr:                              ## 3. unweave, restoring the original
        copy = curr.next
        curr.next = copy.next
        copy.next = copy.next.next if copy.next else None
        curr = curr.next
    return new_head

## tests

a, b, c = Node(7), Node(13), Node(11)
a.next, b.next = b, c
b.random, c.random = a, a
copied = copy_random_list(a)
assert [copied.val, copied.next.val, copied.next.next.val] == [7, 13, 11]
assert copied.next.random is copied and copied.next.random is not a
assert a.next is b and b.next is c        ## the original list is intact
assert copy_random_list(None) is None
print([copied.val, copied.next.val, copied.next.next.val], copied.next.random.val)
```

```
[7, 13, 11] 7
```

**Complexity.** $O(n)$ time. The weave version is $O(1)$ extra space; the map version is $O(n)$.

### P7. Add Two Numbers — two lists hold digits in reverse order; return their sum as a list

**Which template.** Template 2, building the answer behind a dummy head.
**The trick.** The digits are already least-significant-first, which is exactly the order addition
wants, so no reversal is needed. Run the loop while either list has nodes **or** the carry is non-zero:
`999 + 1` produces a fourth digit after both lists are exhausted, and a loop that stops when the lists
stop drops it.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def add_two_numbers(l1, l2):
    dummy = ListNode()
    tail, carry = dummy, 0
    while l1 or l2 or carry:                 ## "or carry" catches the final 1
        total = carry
        if l1:
            total, l1 = total + l1.val, l1.next
        if l2:
            total, l2 = total + l2.val, l2.next
        carry, digit = divmod(total, 10)
        tail.next = ListNode(digit)
        tail = tail.next
    return dummy.next

## tests

assert to_list(add_two_numbers(build([2, 4, 3]), build([5, 6, 4]))) == [7, 0, 8]
assert to_list(add_two_numbers(build([9, 9, 9]), build([1]))) == [0, 0, 0, 1]
assert to_list(add_two_numbers(build([0]), build([0]))) == [0]
print(to_list(add_two_numbers(build([9, 9, 9]), build([1]))))
```

```
[0, 0, 0, 1]
```

**Complexity.** $O(\max(m, n))$ time, $O(\max(m, n))$ space for the output.

### P8. Find the Duplicate Number — one repeated value in an array of `n+1` numbers from 1 to n

**Which template.** Template 3, on an array read as a linked list. This is the disguise worth knowing.
**The trick.** Read `nums` as a function: from index `i` you go to index `nums[i]`. Every value is
between 1 and n, so you can never step outside the array and the walk is an infinite sequence in a
finite set, which means it must cycle. Two indices point at the same successor exactly when they hold
the same value, so the duplicate is the entry point of that cycle, and Floyd finds it in $O(1)$ space
without modifying the array. Recognising that the array *is* a linked list is the entire problem.

```python
def find_duplicate(nums):
    slow = fast = nums[0]
    while True:                              ## phase 1: find a meeting point in the cycle
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    finder = nums[0]
    while finder != slow:                    ## phase 2: both move one step to the entry
        finder, slow = nums[finder], nums[slow]
    return finder

## tests

assert find_duplicate([1, 3, 4, 2, 2]) == 2
assert find_duplicate([3, 1, 3, 4, 2]) == 3
assert find_duplicate([2, 2, 2, 2, 2]) == 2
assert find_duplicate([1, 1]) == 1
print(find_duplicate([1, 3, 4, 2, 2]), find_duplicate([3, 1, 3, 4, 2]))
```

```
2 3
```

**Complexity.** $O(n)$ time, $O(1)$ space, and the input is not modified. Those three constraints
together are why the problem exists.

### P9. LRU Cache — `get` and `put` in $O(1)$, evicting the least recently used key when full

**Which template.** A doubly linked list for the order plus a hash map for the lookup. Write it out in
full; it is asked constantly.
**The trick.** Each structure supplies what the other lacks. The hash map finds a node in $O(1)$ but
knows nothing about order. The doubly linked list reorders in $O(1)$ but cannot search. Together they
give $O(1)$ for everything. Use two sentinel nodes, `head` and `tail`, so that no insertion or removal
ever needs a null check — that is the dummy-head habit applied at both ends. Keep the most recent next
to `head` and evict from next to `tail`.

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key, self.value = key, value
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.table = capacity, {}
        self.head, self.tail = Node(), Node()       ## sentinels at BOTH ends
        self.head.next, self.tail.prev = self.tail, self.head
    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev
    def _push_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node
    def get(self, key):
        if key not in self.table:
            return -1
        node = self.table[key]
        self._unlink(node)
        self._push_front(node)                      ## touching a key makes it most recent
        return node.value
    def put(self, key, value):
        if key in self.table:
            self._unlink(self.table[key])
        node = Node(key, value)
        self.table[key] = node
        self._push_front(node)
        if len(self.table) > self.capacity:
            oldest = self.tail.prev                 ## evict from the tail end
            self._unlink(oldest)
            del self.table[oldest.key]

## tests

cache = LRUCache(2)
cache.put(1, 1); cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)                                     ## evicts key 2, the least recently used
assert cache.get(2) == -1
cache.put(4, 4)                                     ## evicts key 1
assert cache.get(1) == -1
assert cache.get(3) == 3 and cache.get(4) == 4
print(cache.get(2), cache.get(3), cache.get(4))
```

```
-1 3 4
```

**Complexity.** $O(1)$ for `get` and `put`, $O(\text{capacity})$ space.

### P10. Merge K Sorted Lists — merge k sorted lists into one

**Which template.** Template 4 applied repeatedly, in a divide-and-conquer shape.
**The trick.** Merging one list at a time into an accumulator costs $O(Nk)$, because the accumulator is
re-walked every round. Pairing the lists instead halves their number each round, so there are
$\log k$ rounds and each round touches all N nodes once: $O(N \log k)$. That reasoning is the answer the
interviewer wants. A min-heap of the k heads gives the same bound and is in the heap chapter.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def merge_two(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next

def merge_k_lists(lists):
    if not lists:
        return None
    while len(lists) > 1:                    ## pair up, halving the count each round
        lists = [merge_two(lists[i], lists[i + 1] if i + 1 < len(lists) else None)
                 for i in range(0, len(lists), 2)]
    return lists[0]

## tests

assert to_list(merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])) == [1, 1, 2, 3, 4, 4, 5, 6]
assert to_list(merge_k_lists([])) == []
assert to_list(merge_k_lists([build([])])) == []
print(to_list(merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])))
```

```
[1, 1, 2, 3, 4, 4, 5, 6]
```

**Complexity.** $O(N \log k)$ time for N total nodes, $O(\log k)$ recursion-free stack here, so
effectively $O(1)$ extra space.

### P11. Reverse Nodes in k-Group — reverse every consecutive block of k nodes, leaving the remainder alone

**Which template.** The sublist reversal from the trick section, in a loop.
**The trick.** Before reversing a group you must know it is complete, so walk k steps from
`group_prev` and return the list unchanged if you fall off the end. Seed `prev` with `group_next`
rather than `None`, so the reversed block is already stitched to what follows. Then the two
reconnections: `group_prev.next = kth`, and the new `group_prev` is the node that *was*
`group_prev.next`, saved before the overwrite.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev
        for _ in range(k):                   ## is a full group of k available?
            kth = kth.next
            if not kth:
                return dummy.next
        group_next = kth.next
        prev, curr = group_next, group_prev.next   ## seed prev with what FOLLOWS the group
        while curr is not group_next:
            next_node = curr.next
            curr.next = prev
            prev, curr = curr, next_node
        new_group_prev = group_prev.next     ## save it BEFORE the overwrite
        group_prev.next = kth
        group_prev = new_group_prev

## tests

assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
assert to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6]), 2)) == [2, 1, 4, 3, 6, 5]
assert to_list(reverse_k_group(build([1]), 2)) == [1]
print(to_list(reverse_k_group(build([1, 2, 3, 4, 5, 6]), 2)))
```

```
[2, 1, 4, 3, 6, 5]
```

**Complexity.** $O(n)$ time — each node is visited a constant number of times — and $O(1)$ space.

### P12. Remove Duplicates from Sorted List, I and II — keep one copy of each value, or delete every value that repeats

**Which template.** Version I needs no dummy, because the head always survives. Version II needs
template 2, because a run of duplicates may start at the head.
**The trick.** The two versions differ in what the pointer means. In I, `curr` walks the survivors and
skips forward past equal values. In II, `prev` sits before a run and you compare `prev.next.val` with
`prev.next.next.val` to detect a run; if there is one, skip the whole run and do not move `prev`. The
question to ask the interviewer is which of the two they mean, because the names are almost identical.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def delete_duplicates(head):                 ## I: keep one copy of each value
    curr = head
    while curr and curr.next:
        if curr.next.val == curr.val:
            curr.next = curr.next.next       ## skip the copy, stay put
        else:
            curr = curr.next
    return head

def delete_all_duplicates(head):             ## II: delete every value that appears twice or more
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next:
        curr = prev.next
        while curr.next and curr.next.val == curr.val:
            curr = curr.next                 ## walk to the END of the run
        if curr is prev.next:
            prev = prev.next                 ## run of length 1: keep it
        else:
            prev.next = curr.next            ## drop the whole run, prev stays put
    return dummy.next

## tests

assert to_list(delete_duplicates(build([1, 1, 2, 3, 3]))) == [1, 2, 3]
assert to_list(delete_all_duplicates(build([1, 2, 3, 3, 4, 4, 5]))) == [1, 2, 5]
assert to_list(delete_all_duplicates(build([1, 1, 1, 2, 3]))) == [2, 3]
assert to_list(delete_all_duplicates(build([1, 1]))) == []
print(to_list(delete_duplicates(build([1, 1, 2, 3, 3]))),
      to_list(delete_all_duplicates(build([1, 2, 3, 3, 4, 4, 5]))))
```

```
[1, 2, 3] [1, 2, 5]
```

**Complexity.** $O(n)$ time, $O(1)$ space for both.

### P13. Odd Even Linked List — group the nodes at odd positions before the nodes at even positions

**Which template.** Two chains built at once, then joined. It is template 2 twice over.
**The trick.** Positions, not values. Keep an `odd` tail and an `even` tail, advance each by two, and
save `even_head` at the start because after the walk you can no longer find it. The loop condition is
`while even and even.next`: `even` must exist to advance `odd`, and `even.next` must exist to advance
`even`.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def odd_even_list(head):
    if not head or not head.next:
        return head
    odd, even = head, head.next
    even_head = even                         ## save it: you cannot find it later
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head                     ## join the odd chain to the even chain
    return head

## tests

assert to_list(odd_even_list(build([1, 2, 3, 4, 5]))) == [1, 3, 5, 2, 4]
assert to_list(odd_even_list(build([2, 1, 3, 5, 6, 4, 7]))) == [2, 3, 6, 7, 1, 5, 4]
assert to_list(odd_even_list(build([1, 2]))) == [1, 2]
print(to_list(odd_even_list(build([1, 2, 3, 4, 5]))))
```

```
[1, 3, 5, 2, 4]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P14. Intersection of Two Linked Lists — the node where two lists join, or nothing

**Which template.** Two pointers with the switch trick.
**The trick.** The lists have different lengths, so a naive parallel walk misaligns. Walk pointer `a`
through list A and then through list B, and walk pointer `b` through B and then through A. Both
therefore travel exactly $m + n$ steps, so they arrive at any shared suffix at the same moment and meet
at the first shared node. If there is no intersection both reach `None` at step $m + n$ and the loop
ends. Compare with `is`, never with `==`, because the question is about node identity and two distinct
nodes may hold the same value.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def get_intersection_node(head_a, head_b):
    if not head_a or not head_b:
        return None
    a, b = head_a, head_b
    while a is not b:                        ## identity, not value equality
        a = a.next if a else head_b          ## switch to the other list at the end
        b = b.next if b else head_a
    return a                                 ## the shared node, or None

## tests

shared = build([8, 4, 5])
first = build([4, 1]); first.next.next = shared
second = build([5, 6, 1]); second.next.next.next = shared
assert get_intersection_node(first, second) is shared
assert get_intersection_node(build([2, 6, 4]), build([1, 5])) is None
assert get_intersection_node(build([]), build([1])) is None
print(get_intersection_node(first, second).val)
```

```
8
```

**Complexity.** $O(m + n)$ time, $O(1)$ space.

### P15. Rotate List — move the list right by k places

**Which template.** Template 2, with the list closed into a ring and cut again.
**The trick.** Two facts. First, `k` may exceed the length, so reduce it with `k % length` or you will
walk the list many times for nothing, and `k % length == 0` means return the list unchanged. Second,
joining the tail to the head makes the rotation a single cut: the new tail is `length - k % length`
steps from the head, and the new head is the node after it. Closing the ring turns two-pointer
bookkeeping into one modulo and one cut.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def rotate_right(head, k):
    if not head or not head.next:
        return head
    length, tail = 1, head
    while tail.next:                         ## measure and hold the tail
        tail, length = tail.next, length + 1
    k %= length
    if k == 0:
        return head
    tail.next = head                         ## close the ring
    steps = length - k                       ## the new tail is `steps` from the head
    new_tail = head
    for _ in range(steps - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None                     ## cut the ring
    return new_head

## tests

assert to_list(rotate_right(build([1, 2, 3, 4, 5]), 2)) == [4, 5, 1, 2, 3]
assert to_list(rotate_right(build([0, 1, 2]), 4)) == [2, 0, 1]
assert to_list(rotate_right(build([1, 2, 3]), 3)) == [1, 2, 3]
assert to_list(rotate_right(build([]), 1)) == []
print(to_list(rotate_right(build([1, 2, 3, 4, 5]), 2)))
```

```
[4, 5, 1, 2, 3]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P16. Swap Nodes in Pairs — swap every two adjacent nodes

**Which template.** Template 2, and it is Reverse Nodes in k-Group with `k = 2`. Say that.
**The trick.** Three pointers change per swap: `prev.next`, `first.next` and `second.next`, and they
must be written in an order where nothing is lost. Write `prev.next = second` last if you are following
the general k-group shape, or use the explicit three-line form below. Swapping the values instead of the
nodes is a different problem and interviewers usually forbid it, so ask.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def swap_pairs(head):
    dummy = ListNode(0, head)
    prev = dummy
    while prev.next and prev.next.next:
        first, second = prev.next, prev.next.next
        first.next = second.next             ## 1. first now points past second
        second.next = first                  ## 2. second points back at first
        prev.next = second                   ## 3. the pair is reattached
        prev = first                         ## first is now the tail of the pair
    return dummy.next

## tests

assert to_list(swap_pairs(build([1, 2, 3, 4]))) == [2, 1, 4, 3]
assert to_list(swap_pairs(build([1, 2, 3]))) == [2, 1, 3]
assert to_list(swap_pairs(build([1]))) == [1]
assert to_list(swap_pairs(build([]))) == []
print(to_list(swap_pairs(build([1, 2, 3, 4]))))
```

```
[2, 1, 4, 3]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P17. Partition List — put every node below `x` before every node at or above `x`, keeping relative order

**Which template.** Template 2, twice: two dummy heads and two tails.
**The trick.** Build two separate chains as you walk once, then join them. Relative order is preserved
for free, because each chain appends in the order it meets nodes. The line people forget is
`after_tail.next = None` at the end: the last node of the "greater or equal" chain still points into
the original list, so without it you build a cycle.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

def partition(head, x):
    before = before_tail = ListNode()        ## chain of values < x
    after = after_tail = ListNode()          ## chain of values >= x
    curr = head
    while curr:
        if curr.val < x:
            before_tail.next, before_tail = curr, curr
        else:
            after_tail.next, after_tail = curr, curr
        curr = curr.next
    after_tail.next = None                   ## terminate, or you build a cycle
    before_tail.next = after.next            ## join the two chains
    return before.next

## tests

assert to_list(partition(build([1, 4, 3, 2, 5, 2]), 3)) == [1, 2, 2, 4, 3, 5]
assert to_list(partition(build([2, 1]), 2)) == [1, 2]
assert to_list(partition(build([]), 1)) == []
print(to_list(partition(build([1, 4, 3, 2, 5, 2]), 3)))
```

```
[1, 2, 2, 4, 3, 5]
```

**Complexity.** $O(n)$ time, $O(1)$ space.

### P18. Flatten a Multilevel Doubly Linked List — splice each child list in after its parent node

**Which template.** An explicit stack, because the structure is a tree walked as a list.
**The trick.** When a node has a child, push its `next` onto a stack, attach the child as the new
`next`, and set `child` to `None`. When you run off the end of a branch, pop the stack and continue.
That is a depth-first traversal written iteratively. Two details fail the hidden tests: you must fix the
`prev` pointer on every relink, because the list is doubly linked, and you must clear `child` or the
output still contains the old structure.

```python
class Node:
    def __init__(self, val, prev=None, next=None, child=None):
        self.val, self.prev, self.next, self.child = val, prev, next, child

def build_doubly(values):
    nodes = [Node(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next, nodes[i + 1].prev = nodes[i + 1], nodes[i]
    return nodes

def flatten(head):
    stack, curr = [], head
    while curr:
        if curr.child:
            if curr.next:
                stack.append(curr.next)      ## come back to it later
            curr.next, curr.child.prev = curr.child, curr
            curr.child = None                ## clear it, or the structure survives
        elif not curr.next and stack:
            nxt = stack.pop()
            curr.next, nxt.prev = nxt, curr
        curr = curr.next
    return head

def to_list(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

## tests

top = build_doubly([1, 2, 3, 4, 5, 6])
mid = build_doubly([7, 8, 9, 10])
low = build_doubly([11, 12])
top[2].child, mid[1].child = mid[0], low[0]
flat = flatten(top[0])
assert to_list(flat) == [1, 2, 3, 7, 8, 11, 12, 9, 10, 4, 5, 6]
assert all(n.child is None for n in top + mid + low)
assert flatten(None) is None
print(to_list(flat))
```

```
[1, 2, 3, 7, 8, 11, 12, 9, 10, 4, 5, 6]
```

**Complexity.** $O(n)$ time, $O(d)$ space for nesting depth d.

## Tricks and tips

**Use a dummy head by default and remove it only if you are sure.** It costs one line and one node, and
it deletes the entire class of "the head changed" bugs. Return `dummy.next`, never `head`, in any
function that has a dummy. The habit generalises: in a doubly linked list use sentinels at both ends,
which is what makes the LRU cache have no null checks anywhere.

**Save before you overwrite, always in the same order.** The reversal loop is save, rewrite, advance,
advance. Python lets you write it as one tuple assignment, `curr.next, prev, curr = prev, curr,
curr.next`, and that is fine because the right-hand side is evaluated first. However, under pressure the
four explicit lines are safer, because the tuple form hides exactly the ordering that people get wrong.

**Draw three nodes.** Not the whole list, three nodes: the one before, the one you are changing, and the
one after. Every pointer operation in this chapter is local to a window of three, and a window of three
fits on the corner of a whiteboard.

**Fast and slow gives you the middle for free, and which middle depends on where fast starts.** Starting
`fast = head` puts `slow` on the second middle of an even-length list. Starting `fast = head.next` puts
it on the first middle, which is what you want when you intend to cut the list into two halves, because
then the first half is the shorter or equal one. Reorder List uses the second form for exactly that
reason.

**Compare nodes with `is`, values with `==`.** Cycle detection and intersection are both about identity.
Using `==` on nodes compares object identity by default in Python, so it happens to work, but it reads
as a value comparison and it will be wrong the moment someone defines `__eq__`. Say `is` and mean it.

**Split, reverse, merge is a single move.** Reorder List, Palindrome Linked List and "is the second half
the reverse of the first" are all the same three phases. Learn them as one unit and each of those
problems is a two-minute write.

**Restore what you broke.** Palindrome Linked List and Copy List with Random Pointer both destroy the
input as an intermediate step. Good candidates restore it before returning and say so. It is a small
thing that reads as care.

**When the array is a function, it is a linked list.** Find the Duplicate Number is Floyd's algorithm on
`i -> nums[i]`. Any time indices map to indices and the range is closed, the same disguise is available.

## The bugs that cost the round

**Losing the rest of the list.** `curr.next = prev` without saving `curr.next` first drops everything
after `curr`, silently. There is no exception and no wrong value, just a short list. This is the bug,
and the fix is the fixed four-line order.

**Returning `head` when you built a dummy.** If the first node was deleted or replaced, `head` now
points at a node that is no longer in the list, or at nothing. Return `dummy.next`.

**Forgetting to cut.** In Reorder List, in Partition List and in any split, the tail of a piece still
points into the original list. Set it to `None` explicitly. A missing cut produces an infinite loop in
the test harness rather than a wrong answer, so it looks like a hang and costs you the remaining time.

**The wrong loop guard on fast and slow.** It is `while fast and fast.next`. Checking only `fast` throws
`AttributeError` on `fast.next.next` for an even-length list; checking only `fast.next` throws on the
empty list.

**Off-by-one in "nth from the end".** Start both pointers at the dummy and advance `fast` exactly n
times, then advance both while `fast.next` exists. Starting at `head` breaks when n equals the length,
which is the very case the test suite contains.

**Dropping the final carry in Add Two Numbers.** The loop condition must include `or carry`.

**Recursion depth.** The recursive reversal is elegant and it raises `RecursionError` at about 1000
nodes in CPython. Mention the limit and offer the iterative version.

## Done when

- You can write the three-pointer reversal, correctly and without hesitation, from a blank file in under
  thirty seconds.
- You can say why a dummy head is needed for Remove Nth Node From End and not needed for Remove
  Duplicates from Sorted List I, in one sentence each.
- You can reverse a sublist given only the node before it and the node after it, and explain both
  reconnections and why the new `before` must be saved first.
- You can write the LRU cache with sentinel nodes at both ends, and explain which of the two structures
  supplies $O(1)$ lookup and which supplies $O(1)$ reordering.
