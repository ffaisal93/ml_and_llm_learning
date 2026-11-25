"""W093 exact annotation-cleaning contracts."""

from collections import defaultdict
from copy import deepcopy


def dataset_stats(train):
    return dict(
        num_raters=len({r["rater_id"] for r in train}),
        num_prompts=len({r["prompt_id"] for r in train}),
        label_counts={
            label: sum(r["label"] == label for r in train) for label in (0, 1)
        },
        embedding_dim=len(train[0]["embedding"]) if train else 0,
    )


def majority_labels(train):
    counts = {}
    for row in train:
        counts.setdefault(row["prompt_id"], [0, 0])[row["label"]] += 1
    return {p: (None if c[0] == c[1] else int(c[1] > c[0])) for p, c in counts.items()}


def rater_agreement(train, majority):
    totals = defaultdict(lambda: [0, 0])
    for row in train:
        target = majority.get(row["prompt_id"])
        if target is not None:
            record = totals[row["rater_id"]]
            record[0] += int(row["label"] == target)
            record[1] += 1
    return {r: correct / total for r, (correct, total) in totals.items()}


def assign_pools(agreement, expert_min=0.90, decent_min=0.70):
    return {
        r: ("expert" if a >= expert_min else "decent" if a >= decent_min else "spammer")
        for r, a in agreement.items()
    }


def filter_reliable(train, pools):
    return [row for row in train if pools.get(row["rater_id"]) in ("expert", "decent")]


def weighted_consensus(train, agreement):
    prompts = {}
    for row in train:
        pid = row["prompt_id"]
        if pid not in prompts:
            prompts[pid] = [row, 0.0, 0.0]
        weight = max(0.0, 2 * agreement.get(row["rater_id"], 0.0) - 1)
        prompts[pid][1] += weight
        prompts[pid][2] += weight * row["label"]
    output = []
    for pid, (row, total, positive) in prompts.items():
        if total > 0:
            output.append(
                dict(
                    prompt_id=pid,
                    rater_id="consensus",
                    embedding=deepcopy(row["embedding"]),
                    label=int(positive / total > 0.5),
                )
            )
    return output


if __name__ == "__main__":
    rows = []
    for pid, votes in [
        ("p1", {"A": 1, "B": 1, "C": 0}),
        ("p2", {"A": 0, "B": 0, "C": 1}),
        ("p3", {"A": 1, "C": 0}),
    ]:
        rows.extend(
            dict(prompt_id=pid, rater_id=r, embedding=[1.0, 2.0], label=y)
            for r, y in votes.items()
        )
    saved = deepcopy(rows)
    majority = majority_labels(rows)
    agreement = rater_agreement(rows, majority)
    assert majority == dict(p1=1, p2=0, p3=None)
    assert agreement == dict(A=1.0, B=1.0, C=0.0)
    assert len(filter_reliable(rows, assign_pools(agreement))) == 5
    assert [r["label"] for r in weighted_consensus(rows, agreement)] == [1, 0, 1]
    assert rows == saved and dataset_stats([])["embedding_dim"] == 0
    assert weighted_consensus(rows, {}) == []
    print("W093 contract and non-mutation checks passed")
