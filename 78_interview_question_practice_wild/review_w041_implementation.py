import asyncio


# async_compare(a,b) is provided by the caller/environment. Do not inspect scores.
async def batch_async_compare(pairs):
    return list(await asyncio.gather(*(async_compare(a, b) for a, b in pairs)))


async def _winner_index(answers, indices):
    while len(indices) > 1:
        pairs = [(indices[j], indices[j + 1]) for j in range(0, len(indices) - 1, 2)]
        # Each left subtree precedes its right subtree in original input order.
        winners = await batch_async_compare(
            [(answers[a], answers[b]) for a, b in pairs]
        )
        indices = [
            a if answer == answers[a] else b for (a, b), answer in zip(pairs, winners)
        ] + indices[len(pairs) * 2 :]
    return indices[0]


async def find_best_answer(answers):
    if not answers:
        raise ValueError("no candidates")
    return answers[await _winner_index(answers, list(range(len(answers))))]


async def find_top_k_answers(answers, k):
    if k < 0:
        raise ValueError("negative k")
    remaining = list(range(len(answers)))
    result = []
    for _ in range(min(k, len(answers))):
        winner = await _winner_index(answers, remaining)
        result.append(answers[winner])
        remaining.remove(winner)
    return result


if __name__ == "__main__":
    # Test-only stub: public ranking code above never uses this fixture's scores.
    scores = {"A": 3, "B": 9, "C": 5, "D": 1, "E": 9}
    calls = 0
    active = 0
    peak = 0

    async def async_compare(a, b):
        global calls, active, peak
        calls += 1
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return a if scores[a] >= scores[b] else b

    async def check():
        global calls, peak
        assert await batch_async_compare([("A", "B"), ("C", "D"), ("E", "B")]) == [
            "B",
            "C",
            "E",
        ]
        assert peak == 3
        calls = 0
        assert await find_best_answer(["A", "B", "C", "D", "E"]) == "B" and calls == 4
        calls = 0
        assert (
            await find_top_k_answers(["A", "B", "C", "D", "E"], 3) == ["B", "E", "C"]
            and calls <= 12
        )
        assert await find_top_k_answers(["C", "C", "A"], 2) == ["C", "C"]
        assert await find_top_k_answers(["A", "B", "C", "D", "E"], 10) == [
            "B",
            "E",
            "C",
            "A",
            "D",
        ]
        before = calls
        assert await find_best_answer(["A"]) == "A" and calls == before
        assert (
            await batch_async_compare([]) == []
            and await find_top_k_answers([], 0) == []
        )

    asyncio.run(check())
    print(
        "W041: examples, stable ties/duplicate occurrences, call bounds and parallel calls passed"
    )
