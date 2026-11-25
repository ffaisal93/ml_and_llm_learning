import math


def tokenize(s):
    return s.split()


def classify_and_score(text, threshold, next_token_probs, l_pos=" yes", l_neg=" no"):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if l_pos == l_neg:
        raise ValueError("labels must differ")
    tokens = tokenize("Text:\n" + text + "\n\nLabel:")
    probabilities = next_token_probs(tokens)  # Exactly one model query.
    pos = probabilities.get(l_pos, 0.0)
    neg = probabilities.get(l_neg, 0.0)
    total = pos + neg
    pos, neg = (pos / total, neg / total) if total else (0.5, 0.5)
    return {
        "p_pos": pos,
        "p_neg": neg,
        "prediction": l_pos if pos >= threshold else l_neg,
    }


def bce_loss(result, y):
    if y not in (0, 1):
        raise ValueError("binary target required")
    return -(
        y * math.log(max(result["p_pos"], 1e-12))
        + (1 - y) * math.log(max(result["p_neg"], 1e-12))
    )


def check():
    calls = []

    def stub(tokens):
        calls.append(tokens)
        return {" yes": 0.6, " no": 0.2, " maybe": 0.2}

    r = classify_and_score("a good film", 0.7, stub)
    assert calls == [["Text:", "a", "good", "film", "Label:"]]
    assert set(r) == {"p_pos", "p_neg", "prediction"}
    assert math.isclose(r["p_pos"], 0.75) and r["prediction"] == " yes"
    assert math.isclose(bce_loss(r, 1), -math.log(0.75))
    r = classify_and_score("x", 0.5, lambda _: {" yes": 0.1, " no": 0.3})
    assert r["prediction"] == " no" and math.isclose(bce_loss(r, 1), -math.log(0.25))
    r = classify_and_score("x", 0.5, lambda _: {"other": 1.0})
    assert r == {"p_pos": 0.5, "p_neg": 0.5, "prediction": " yes"}
    assert classify_and_score("x", 0.6, lambda _: {})["prediction"] == " no"
    assert classify_and_score("x", 0, lambda _: {" no": 1.0})["prediction"] == " yes"
    assert math.isfinite(bce_loss({"p_pos": 0, "p_neg": 1}, 1))
    print("W107: all behavior checks passed")


if __name__ == "__main__":
    check()
