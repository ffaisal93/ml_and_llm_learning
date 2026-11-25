import heapq, random, time


class ByteTokenizer:
    def __init__(self, token_alphabet):
        self.token_alphabet = token_alphabet
        self.token_to_id = {token: idx for idx, token in enumerate(token_alphabet)}
        self.pair_rank = {}
        for rank in range(256, len(token_alphabet)):
            token = token_alphabet[rank]
            for cut in range(1, len(token)):
                a = self.token_to_id.get(token[:cut])
                b = self.token_to_id.get(token[cut:])
                if a is not None and b is not None:
                    self.pair_rank[a, b] = rank

    def slow_tokenize(self, text):
        token_ids = list(text)
        for cur_token_id in range(256, len(self.token_alphabet)):
            cur_token_bytes = self.token_alphabet[cur_token_id]
            pairs_to_merge = set()
            for split_point in range(1, len(cur_token_bytes)):
                left_id = self.token_to_id.get(cur_token_bytes[:split_point])
                right_id = self.token_to_id.get(cur_token_bytes[split_point:])
                if left_id is not None and right_id is not None:
                    pairs_to_merge.add((left_id, right_id))
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if (
                    i < len(token_ids) - 1
                    and (token_ids[i], token_ids[i + 1]) in pairs_to_merge
                ):
                    new_token_ids.append(cur_token_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = new_token_ids
        return token_ids

    def tokenize(self, text):
        n = len(text)
        if not n:
            return []
        # Nodes retain original leftmost byte positions. Removed nodes are tombstoned.
        ids = list(text)
        previous = list(range(-1, n - 1))
        following = list(range(1, n + 1))
        alive = bytearray([1]) * n
        versions = [0] * n
        heap = []

        def offer(left):
            if left < 0 or left >= n or not alive[left]:
                return
            right = following[left]
            if right == n:
                return
            rank = self.pair_rank.get((ids[left], ids[right]))
            if rank is not None:
                heapq.heappush(
                    heap, (rank, left, right, versions[left], versions[right])
                )

        for left in range(n - 1):
            offer(left)
        while heap:
            rank, left, right, lv, rv = heapq.heappop(heap)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or versions[left] != lv
                or versions[right] != rv
            ):
                continue
            ids[left] = rank
            versions[left] += 1
            alive[right] = 0
            after = following[right]
            following[left] = after
            if after < n:
                previous[after] = left
            offer(previous[left])
            offer(left)
        result = []
        node = 0
        while node < n:
            result.append(ids[node])
            node = following[node]
        return result

    def estimate_token_count(self, text, sample_size, rng):
        n = len(text)
        if sample_size < 1:
            raise ValueError("positive sample budget required")
        if sample_size >= n:
            return len(self.tokenize(text))
        # Stratified random contiguous windows reduce slow spatial distribution drift.
        # Use longer windows at larger budgets to reduce boundary bias.
        target_window = 256 if sample_size <= 10000 else 2048
        windows = max(1, sample_size // target_window)
        base, extra = divmod(sample_size, windows)
        tokens = bytes_used = 0
        for j in range(windows):
            length = base + (j < extra)
            lo = j * n // windows
            hi = (j + 1) * n // windows
            start = (
                rng.randrange(lo, hi - length + 1)
                if hi - lo >= length
                else rng.randrange(0, n - length + 1)
            )
            tokens += len(self.tokenize(text[start : start + length]))
            bytes_used += length
        return int(round(tokens * n / bytes_used))


if __name__ == "__main__":
    base = [bytes([i]) for i in range(256)]
    t = ByteTokenizer(base + [b"ab", b"bc", b"abc"])
    assert t.tokenize(b"abc") == [258] and t.tokenize(b"abcab") == [258, 256]
    assert ByteTokenizer(base + [b"bc", b"ab"]).tokenize(b"abc") == [97, 256]
    assert ByteTokenizer(base + [b"aa", b"aaa", b"aaaa"]).tokenize(b"aaaaaaa") == [
        258,
        257,
    ]
    rng = random.Random(7)
    # Randomized equivalence to the unchanged authoritative reference.
    for _ in range(100):
        alphabet = base + sorted(
            {
                bytes(rng.choice(b"abc") for _ in range(rng.randint(2, 6)))
                for j in range(30)
            },
            key=lambda b: (len(b), b),
        )
        tok = ByteTokenizer(alphabet)
        for _ in range(10):
            text = bytes(rng.choice(b"abc") for j in range(rng.randrange(40)))
            assert tok.tokenize(text) == tok.slow_tokenize(text)
    t = ByteTokenizer(base + [b"ab"])
    assert t.estimate_token_count(b"ab" * 600, 1200, random.Random(0)) == 600
    assert (
        abs(t.estimate_token_count(b"ab" * 5000, 1000, random.Random(0)) - 5000) / 5000
        < 0.02
    )
    # Instrument sampled-byte budget and reproducibility, no whole-text scan.
    count = [0]
    original = t.tokenize

    def counted(b):
        count[0] += len(b)
        return original(b)

    t.tokenize = counted
    first = t.estimate_token_count(b"ab" * 5000, 1000, random.Random(3))
    assert count[0] == 1000
    assert first == t.estimate_token_count(b"ab" * 5000, 1000, random.Random(3))
    t.tokenize = original
    text = b"ab " * 333333
    tick = time.perf_counter()
    result = t.tokenize(text)
    print(
        "W014: 1000 randomized reference matches; sampled-byte budget passed; million-byte seconds",
        time.perf_counter() - tick,
    )
