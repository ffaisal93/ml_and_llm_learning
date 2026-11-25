import math
import numpy as np


def beam_search(lm_predict, start_token, beam_width, num_steps, vocab_size):
    if beam_width < 1 or num_steps < 0 or vocab_size < 1:
        raise ValueError("invalid search configuration")
    beams = [([start_token], 0.0)]
    for _ in range(num_steps):
        candidates = []
        for sequence, score in beams:
            probs = np.asarray(lm_predict(sequence.copy()), dtype=np.float64)
            if probs.shape != (vocab_size,):
                raise ValueError("callback vocabulary mismatch")
            for token, probability in enumerate(probs):
                logprob = (
                    -math.inf if probability == 0 else math.log(float(probability))
                )
                candidates.append(
                    (sequence + [token], float(np.float64(score) + np.float64(logprob)))
                )
        beams = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:beam_width]
    return beams


def beam_search_eos(
    lm_predict, start_token, beam_width, num_steps, vocab_size, eos_token
):
    if beam_width < 1 or num_steps < 0 or vocab_size < 1:
        raise ValueError("invalid search configuration")
    beams = [([start_token], 0.0)]
    for _ in range(num_steps):
        candidates = []
        for sequence, score in beams:
            if sequence[-1] == eos_token:
                candidates.append((sequence, score))
                continue
            probs = lm_predict(sequence.copy())
            for token, p in enumerate(probs):
                candidates.append(
                    (
                        sequence + [token],
                        score + (math.log(float(p)) if p else -math.inf),
                    )
                )
        beams = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:beam_width]
        if all(s[-1] == eos_token for s, _ in beams):
            break
    return beams


def dummy_lm(tokens):
    p = np.full(5, 0.1)
    p[(tokens[-1] + 1) % 5] += 0.6
    return p / p.sum()


r = beam_search(dummy_lm, 0, 2, 2, 5)
assert [s for s, _ in r] == [[0, 1, 2], [0, 0, 1]]
assert np.allclose([v for _, v in r], [-0.9039702474861145, -2.8498803965414274])
assert beam_search(dummy_lm, 3, 4, 0, 5) == [([3], 0.0)]
assert beam_search(dummy_lm, 0, 1, 3, 5)[0][0] == [0, 1, 2, 3]


def mutating(seq):
    seq.append(999)
    return np.array([1.0, 0.0])


r = beam_search(mutating, 0, 5, 2, 2)
assert [s for s, _ in r] == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
assert all(len(s) == 3 for s, _ in r) and all(v == -math.inf for _, v in r[1:])
assert beam_search_eos(lambda _: np.array([0.0, 1.0]), 0, 1, 5, 2, 1) == [([0, 1], 0.0)]
print(
    "W012: literal beams/scores, zero-mass candidates, fresh callback lists and EOS passed"
)
