# Computation uses only Python lists and the provided all_gather interface.
def relu_grad(x, y_grad):
    return [float(g) if value > 0 else 0.0 for value, g in zip(x, y_grad)]


def sharded_matmul_grad(sharded_W, v, out_grad, comm):
    counts = comm.all_gather(len(sharded_W))
    start = sum(counts[: comm.my_shard_index])
    stop = start + len(sharded_W)
    local_g = out_grad[start:stop]
    W_grad = [[g * value for value in v] for g in local_g]
    partial = [
        sum(g * row[j] for g, row in zip(local_g, sharded_W)) for j in range(len(v))
    ]
    gathered = comm.all_gather(partial)
    v_grad = [sum(part[j] for part in gathered) for j in range(len(v))]
    return W_grad, v_grad


def sharded_mlp_grad(sharded_A, sharded_B, activations, y_grad, comm):
    B_grad, h_grad = sharded_matmul_grad(sharded_B, activations["h"], y_grad, comm)
    pre_grad = relu_grad(activations["h_pre"], h_grad)
    A_grad, _ = sharded_matmul_grad(sharded_A, activations["x"], pre_grad, comm)
    return {"sharded_A": A_grad, "sharded_B": B_grad}


if __name__ == "__main__":
    # Separate fixture imports; no NumPy or autograd inside either backward.
    from review_w035_implementation import Communicator
    from concurrent.futures import ThreadPoolExecutor

    def verify(a_splits, b_splits, zero=False):
        A = [[1, 0], [1, -2] if zero else [0, -1], [1, -1]]
        B = [[1, 2, 3], [0, 1, 0]]
        x = [2, 1]
        p = len(a_splits)
        h_pre = [sum(v * w for v, w in zip(row, x)) for row in A]
        h = [max(0.0, v) for v in h_pre]
        activations = {"x": x, "h_pre": h_pre, "h": h}
        backend = Communicator(p)

        class Comm:
            def __init__(self, rank):
                self.num_shards = p
                self.my_shard_index = rank

            def all_gather(self, value):
                return backend.all_gather(self.my_shard_index, value)

        def worker(rank):
            a = sum(a_splits[:rank])
            b = sum(b_splits[:rank])
            return sharded_mlp_grad(
                A[a : a + a_splits[rank]],
                B[b : b + b_splits[rank]],
                activations,
                [1, 1],
                Comm(rank),
            )

        with ThreadPoolExecutor(max_workers=p) as pool:
            result = list(pool.map(worker, range(p)))
        assert [row for r in result for row in r["sharded_A"]] == [
            [2.0, 1.0],
            [0.0, 0.0],
            [6.0, 3.0],
        ]
        assert [row for r in result for row in r["sharded_B"]] == [
            [2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
        assert backend.calls == [4] * p

    verify([2, 1], [1, 1])
    verify([2, 0, 1], [1, 0, 1], True)
    verify([3], [2])
    print(
        "W036: uneven/empty/one-shard examples, ReLU at zero, four collectives per rank passed"
    )
