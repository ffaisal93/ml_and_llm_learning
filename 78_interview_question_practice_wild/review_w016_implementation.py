from dataclasses import dataclass


@dataclass
class PackedSample:
    input_ids: list
    loss_mask: list
    position_ids: list


def pack(examples, seq_len, sep, pad_id):
    if seq_len < 1:
        raise ValueError("positive context length required")
    bins = []
    order = sorted(examples, key=lambda pair: -(len(pair[0]) + len(pair[1]) + 1))
    for prompt, answer in order:
        length = len(prompt) + len(answer) + 1
        if length > seq_len:
            continue
        selected = next((b for b in bins if len(b.input_ids) + length <= seq_len), None)
        if selected is None:
            selected = PackedSample([], [], [])
            bins.append(selected)
        selected.input_ids.extend(prompt + answer + [sep])
        selected.loss_mask.extend([0] * len(prompt) + [1] * (len(answer) + 1))
        selected.position_ids.extend(range(length))
    for b in bins:
        missing = seq_len - len(b.input_ids)
        b.input_ids.extend([pad_id] * missing)
        b.loss_mask.extend([0] * missing)
        b.position_ids.extend([0] * missing)
    return bins


r = pack([([5, 6], [7]), ([8], [9, 10]), ([11], [12])], 8, 99, 0)
assert r[0] == PackedSample(
    [5, 6, 7, 99, 8, 9, 10, 99], [0, 0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]
)
assert r[1] == PackedSample(
    [11, 12, 99, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0, 0, 0]
)
assert pack([([1], [2]), ([3, 4], [5, 6, 7])], 6, 99, 0)[0].input_ids == [
    3,
    4,
    5,
    6,
    7,
    99,
]
assert pack([([1, 2], [3, 4, 5]), ([6], [7])], 4, 99, 0)[0].input_ids == [6, 7, 99, 0]
assert pack([([], [])], 1, 99, 0) == [PackedSample([99], [1], [0])]
assert pack([], 8, 99, 0) == [] and pack([([1], [2])], 1, 99, 0) == []
print(
    "W016: all examples, stable first-fit order, oversize and empty prompt/answer passed"
)
