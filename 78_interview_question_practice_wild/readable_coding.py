"""Readable supplementary exercises with explicitly simplified contracts."""
from collections import Counter


def encode_bpe(text, merges):
    """Character-level reference: texts are separate sequences; no special tokens."""
    tokens = list(text)
    for left, right in merges:
        updated = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i:i + 2] == [left, right]:
                updated.append(left + right)
                i += 2
            else:
                updated.append(tokens[i])
                i += 1
        tokens = updated
    return tokens


def train_bpe(texts, num_merges):
    """Count adjacent pairs, break frequency ties lexically, merge non-overlapping."""
    if num_merges < 0:
        raise ValueError("num_merges must be nonnegative")
    sequences = [list(text) for text in texts]
    merges = []
    for _ in range(num_merges):
        counts = Counter()
        for tokens in sequences:
            counts.update(zip(tokens, tokens[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        merges.append(pair)
        updated_sequences = []
        for tokens in sequences:
            updated = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tuple(tokens[i:i + 2]) == pair:
                    updated.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            updated_sequences.append(updated)
        sequences = updated_sequences
    return merges


def solve_crossword(board, dictionary):
    """Return one legal copied grid; slots are maximal runs of length >= 2."""
    if not board:
        return []
    rows, cols = len(board), len(board[0])
    if any(len(row) != cols for row in board):
        raise ValueError("board must be rectangular")
    grid = [list(row) for row in board]
    slots = []
    for row in range(rows):
        col = 0
        while col < cols:
            start = col
            while col < cols and grid[row][col] != "#":
                col += 1
            if col - start >= 2:
                slots.append([(row, c) for c in range(start, col)])
            col += 1
    for col in range(cols):
        row = 0
        while row < rows:
            start = row
            while row < rows and grid[row][col] != "#":
                row += 1
            if row - start >= 2:
                slots.append([(r, col) for r in range(start, row)])
            row += 1
    covered = {cell for slot in slots for cell in slot}
    if any(grid[r][c] == "." and (r, c) not in covered
           for r in range(rows) for c in range(cols)):
        return None
    words = {}
    for word in set(dictionary):
        words.setdefault(len(word), []).append(word)
    for candidates in words.values():
        candidates.sort()

    def search():
        chosen = None
        for slot in slots:
            candidates = [
                word for word in words.get(len(slot), [])
                if all(grid[r][c] in (".", char)
                       for (r, c), char in zip(slot, word))
            ]
            if not candidates:
                return None
            if any(grid[r][c] == "." for r, c in slot):
                if chosen is None or len(candidates) < len(chosen[1]):
                    chosen = (slot, candidates)
        if chosen is None:
            return [row[:] for row in grid]
        slot, candidates = chosen
        for word in candidates:
            changed = []
            for (r, c), char in zip(slot, word):
                if grid[r][c] == ".":
                    changed.append((r, c))
                    grid[r][c] = char
            answer = search()
            if answer is not None:
                return answer
            for r, c in changed:
                grid[r][c] = "."
        return None

    return search()
