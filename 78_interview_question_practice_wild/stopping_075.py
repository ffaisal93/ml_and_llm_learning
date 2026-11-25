"""W075 finite-action minimax. Run: python stopping_075.py < input.json"""

import json
import math
import sys


def stopping_distribution(probabilities):
    if not probabilities or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("Need N>=1 probabilities in [0,1]")
    survival = 1.0
    pmf = []
    for probability in probabilities[:-1]:
        pmf.append(survival * probability)
        survival *= 1 - probability
    pmf.append(survival)  # emitted STOP or forced stop: both count as T=N
    expectation = math.fsum((t + 1) * mass for t, mass in enumerate(pmf))
    return pmf, expectation


def minimax(decisions, adversaries, reward):
    if not decisions or not adversaries:
        raise ValueError("Need at least one decision and one adversary")
    distributions = [stopping_distribution(a["p"])[0] for a in adversaries]
    if len({len(p) for p in distributions}) != 1:
        raise ValueError("All adversaries must use the same N")
    best = None
    best_value = float("-inf")
    for action in decisions:
        value = min(
            math.fsum(
                mass * reward(action, t + 1, adversary) for t, mass in enumerate(pmf)
            )
            for adversary, pmf in zip(adversaries, distributions)
        )
        if value > best_value:  # ties choose the first listed action
            best, best_value = action, value
    return best, best_value


def solve(data):
    N = data["N"]
    p = data["p"]
    actions = data["decisions"]
    adversaries = data["adversaries"]
    if N != len(p) or any(len(a["p"]) != N for a in adversaries):
        raise ValueError("N and probability lengths differ")
    if any(
        len(a["rewards"]) != len(actions) or any(len(row) != N for row in a["rewards"])
        for a in adversaries
    ):
        raise ValueError("Reward tensor must have shape (M,A,N)")
    # Distinct JSON scalar action labels keep row lookup unambiguous.
    index = {action: i for i, action in enumerate(actions)}
    if len(index) != len(actions):
        raise ValueError("Action labels must be unique")
    chosen, value = minimax(
        actions, adversaries, lambda a, t, adv: adv["rewards"][index[a]][t - 1]
    )
    pmf, expected = stopping_distribution(p)
    return dict(
        pmf=pmf, expected_T=expected, decision=chosen, worst_expected_reward=value
    )


if __name__ == "__main__":
    print(json.dumps(solve(json.load(sys.stdin))))
