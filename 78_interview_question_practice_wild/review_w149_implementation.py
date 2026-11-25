"""Plain-list post-LN transformer; no array or differentiation libraries."""

import math


def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(r, s)] for r, s in zip(a, b)]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def linear(x, w, b=None):
    y = matmul(x, transpose(w))
    return [
        [v + (b[j] if b is not None else 0.0) for j, v in enumerate(row)] for row in y
    ]


def linear_grad(x, w, g):
    return (
        matmul(g, w),
        matmul(transpose(g), x),
        [sum(row[j] for row in g) for j in range(len(g[0]))],
    )


def cols(x, start, end):
    return [row[start:end] for row in x]


def set_cols(x, start, part):
    for row, p in zip(x, part):
        row[start : start + len(p)] = p


def softmax_rows(z):
    out = []
    for row in z:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        total = sum(e)
        out.append([v / total for v in e])
    return out


def softmax_rows_grad(P, g):
    result = []
    for p, grad in zip(P, g):
        dot = sum(a * b for a, b in zip(p, grad))
        result.append([a * (b - dot) for a, b in zip(p, grad)])
    return result


def layernorm(x, gamma, beta, eps):
    hats = []
    invs = []
    output = []
    d = len(gamma)
    for row in x:
        mean = sum(row) / d
        inv = 1 / math.sqrt(sum((v - mean) ** 2 for v in row) / d + eps)
        hat = [(v - mean) * inv for v in row]
        hats.append(hat)
        invs.append(inv)
        output.append([gamma[j] * v + beta[j] for j, v in enumerate(hat)])
    return output, hats, invs


def layernorm_grad(g, xhat, inv, gamma):
    d = len(gamma)
    dx = []
    dg = [0.0] * d
    db = [0.0] * d
    for row, hat, s in zip(g, xhat, inv):
        u = [v * a for v, a in zip(row, gamma)]
        mean = sum(u) / d
        mean_hat = sum(v * h for v, h in zip(u, hat)) / d
        dx.append([s * (v - mean - h * mean_hat) for v, h in zip(u, hat)])
        for j in range(d):
            dg[j] += row[j] * hat[j]
            db[j] += row[j]
    return dx, dg, db


class ManualTransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, params, eps=1e-5):
        if embed_dim % num_heads or num_heads < 1 or eps <= 0:
            raise ValueError("invalid head dimensions or epsilon")
        self.D, self.H, self.F = embed_dim, num_heads, ff_hidden_dim
        self.p, self.eps = params, eps
        self.cache = None

    def forward(self, x):
        p = self.p
        hd = self.D // self.H
        factor = 1 / math.sqrt(hd)
        # Caller must not mutate x or parameters before backward.
        q, k, v = [linear(x, p[name]) for name in ("W_Q", "W_K", "W_V")]
        context = zeros(len(x), self.D)
        heads = []
        for h in range(self.H):
            start = h * hd
            end = start + hd
            qh, kh, vh = [cols(a, start, end) for a in (q, k, v)]
            prob = softmax_rows(scale(matmul(qh, transpose(kh)), factor))
            set_cols(context, start, matmul(prob, vh))
            heads.append((qh, kh, vh, prob))
        attn = linear(context, p["W_O"])
        n1, hat1, inv1 = layernorm(
            add(x, attn), p["norm1_gamma"], p["norm1_beta"], self.eps
        )
        z = linear(n1, p["ffn1_w"], p["ffn1_b"])
        a = [[max(0.0, v) for v in row] for row in z]
        f = linear(a, p["ffn2_w"], p["ffn2_b"])
        y, hat2, inv2 = layernorm(
            add(n1, f), p["norm2_gamma"], p["norm2_beta"], self.eps
        )
        self.cache = (x, context, heads, n1, z, a, hat1, inv1, hat2, inv2)
        return y

    def backward(self, grad_output):
        if self.cache is None:
            raise RuntimeError("forward required")
        x, context, heads, n1, z, a, hat1, inv1, hat2, inv2 = self.cache
        p = self.p
        grads = {}
        hd = self.D // self.H
        factor = 1 / math.sqrt(hd)
        g, grads["norm2_gamma"], grads["norm2_beta"] = layernorm_grad(
            grad_output, hat2, inv2, p["norm2_gamma"]
        )
        ga, grads["ffn2_w"], grads["ffn2_b"] = linear_grad(a, p["ffn2_w"], g)
        gz = [[u * (v > 0.0) for u, v in zip(row, zrow)] for row, zrow in zip(ga, z)]
        gn, grads["ffn1_w"], grads["ffn1_b"] = linear_grad(n1, p["ffn1_w"], gz)
        gn = add(gn, g)  # The second residual's direct n1 path.
        gr, grads["norm1_gamma"], grads["norm1_beta"] = layernorm_grad(
            gn, hat1, inv1, p["norm1_gamma"]
        )
        gc, grads["W_O"], _ = linear_grad(context, p["W_O"], gr)
        gq, gk, gv = [zeros(len(x), self.D) for _ in range(3)]
        for h, (qh, kh, vh, prob) in enumerate(heads):
            gh = cols(gc, h * hd, (h + 1) * hd)
            gp = matmul(gh, transpose(vh))
            gs = softmax_rows_grad(prob, gp)
            set_cols(gq, h * hd, scale(matmul(gs, kh), factor))
            set_cols(gk, h * hd, scale(matmul(transpose(gs), qh), factor))
            set_cols(gv, h * hd, matmul(transpose(prob), gh))
        dx = [row[:] for row in gr]  # First residual's direct x path.
        for name, gproj in [("W_Q", gq), ("W_K", gk), ("W_V", gv)]:
            part, grads[name], _ = linear_grad(x, p[name], gproj)
            dx = add(dx, part)
        assert set(grads) == set(p)
        return dx, grads


def check():
    import random

    assert softmax_rows_grad([[0.5, 0.5]], [[1.0, 0.0]]) == [[0.25, -0.25]]
    assert softmax_rows_grad([[0.5, 0.5]], [[1.0, 1.0]]) == [[0.0, 0.0]]
    rng = random.Random(7)
    D, H, F = 4, 2, 3
    matrix = lambda a, b: [[rng.uniform(-0.4, 0.4) for _ in range(b)] for _ in range(a)]
    params = {name: matrix(D, D) for name in ["W_Q", "W_K", "W_V", "W_O"]}
    params.update(
        ffn1_w=matrix(F, D),
        ffn1_b=[0.2, -0.3, 0.4],
        ffn2_w=matrix(D, F),
        ffn2_b=[0.1] * D,
    )
    for n in [1, 2]:
        params[f"norm{n}_gamma"] = [rng.uniform(0.7, 1.3) for _ in range(D)]
        params[f"norm{n}_beta"] = [rng.uniform(-0.2, 0.2) for _ in range(D)]
    x = matrix(3, D)
    upstream = matrix(3, D)
    block = ManualTransformerBlock(D, H, F, params)
    block.forward(x)
    dx, grads = block.backward(upstream)

    def objective():
        y = block.forward(x)
        return sum(v * g for row, gr in zip(y, upstream) for v, g in zip(row, gr))

    eps = 1e-6
    worst = 0.0
    count = 0
    for name, values, analytic in [("input", x, dx)] + [
        (n, params[n], grads[n]) for n in params
    ]:
        is_matrix = isinstance(values[0], list)
        rows = values if is_matrix else [values]
        grows = analytic if is_matrix else [analytic]
        for row, grad in zip(rows, grows):
            for j in range(len(row)):
                old = row[j]
                row[j] = old + eps
                plus = objective()
                row[j] = old - eps
                minus = objective()
                row[j] = old
                err = abs((plus - minus) / (2 * eps) - grad[j])
                worst = max(worst, err)
                count += 1
                assert err < 1e-6, (name, j, err)
    # Stated one-token identity example.
    identity = [[1.0, 0.0], [0.0, 1.0]]
    p = {
        n: [r[:] for r in identity]
        for n in ["W_Q", "W_K", "W_V", "W_O", "ffn1_w", "ffn2_w"]
    }
    p.update(
        ffn1_b=[0.0, 0.0],
        ffn2_b=[0.0, 0.0],
        norm1_gamma=[1.0, 1.0],
        norm2_gamma=[1.0, 1.0],
        norm1_beta=[0.0, 0.0],
        norm2_beta=[0.0, 0.0],
    )
    y = ManualTransformerBlock(2, 1, 2, p).forward([[1.0, 3.0]])
    assert abs(y[0][0] + 0.9999978) < 1e-7
    print(
        f"W149: {count} central differences; worst absolute error {worst:.3g}; identity output {y}"
    )


if __name__ == "__main__":
    check()
