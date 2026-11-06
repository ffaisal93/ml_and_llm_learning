"""Independent examples, invariants and finite-difference gradient checks."""
import math
import unittest
import numpy as np
import reference_solutions as r


def numerical_gradient(loss, array, h=1e-5):
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + h
        plus = loss()
        array[index] = original - h
        minus = loss()
        array[index] = original
        result[index] = (plus - minus) / (2 * h)
    return result


class NumericalPracticeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(78)

    def test_probability_shift_invariance(self):
        x = np.array([[0., 1., -2.], [3., -1., 2.]])
        np.testing.assert_allclose(r.softmax(x), r.softmax(x + 10000), atol=1e-12)
        np.testing.assert_allclose(r.softmax(x).sum(axis=1), 1)
        np.testing.assert_allclose(r.log_softmax(x), np.log(r.softmax(x)))

    def test_smoothed_loss_independent_target_distribution(self):
        logits = np.array([[1000., 999., 998.], [3., -2., 1.]])
        target = np.full((2, 3), .2 / 3)
        target[np.arange(2), [0, 2]] += .8
        expected = -(target * r.log_softmax(logits)).sum(axis=1)[0]
        self.assertAlmostEqual(r.masked_cross_entropy(logits, [0, 2], [1, 0], .2), expected)
        self.assertEqual(r.masked_cross_entropy(logits, [0, 2], [0, 0], .2), 0)

    def test_uniform_loss(self):
        for epsilon in [0, .1, .9]:
            self.assertAlmostEqual(r.masked_cross_entropy(np.zeros((3, 4)), [0, 1, 2], [1, 1, 1], epsilon), math.log(4))

    def test_attention_known_example_and_mask(self):
        x = np.eye(2)
        v = np.array([[1., 2.], [3., 4.]])
        result = r.attention(x, x, x, v)
        np.testing.assert_allclose(result, [[1.6604769, 2.6604769], [2.3395231, 3.3395231]], atol=1e-7)
        masked = r.attention(x, x, x, v, [[True, False], [True, True]])
        np.testing.assert_allclose(masked[0], v[0])
        with self.assertRaises(ValueError):
            r.attention(x, x, x, v, np.zeros((2, 2), bool))

    def test_causal_future_invariance(self):
        q = self.rng.normal(size=(5, 3))
        k, v = self.rng.normal(size=(5, 3)), self.rng.normal(size=(5, 2))
        mask = np.tril(np.ones((5, 5), bool))
        original = r.attention_qkv(q, k, v, mask)
        k[3:] += 100
        v[3:] -= 100
        np.testing.assert_allclose(r.attention_qkv(q, k, v, mask)[:3], original[:3])

    def test_gqa_contiguous_head_mapping(self):
        q = np.zeros((1, 4, 2, 1))
        k = np.zeros((1, 2, 3, 1))
        v = np.array([[[[2.], [2.], [2.]], [[9.], [9.], [9.]]]])
        result = r.grouped_query_attention(q, k, v)
        np.testing.assert_allclose(result[0, :, 0, 0], [2, 2, 9, 9])
        with self.assertRaises(ValueError):
            r.grouped_query_attention(np.zeros((1, 3, 2, 1)), k, v)

    def test_cached_decode_matches_all_prefixes(self):
        q, k, v = [self.rng.normal(size=(7, 4)) for _ in range(3)]
        full = r.attention_qkv(q, k, v, np.tril(np.ones((7, 7), bool)))
        cache, outputs = None, []
        for i in range(7):
            out, cache = r.cached_attention_step(q[i:i+1], k[i:i+1], v[i:i+1], cache)
            outputs.append(out[0])
        np.testing.assert_allclose(outputs, full, atol=1e-12)

    def test_sampling_support_and_greedy(self):
        self.assertEqual(r.sample_top_k([1, 2, .5, 4], 1, self.rng), 3)
        samples = {r.sample_top_k([1, 2, .5, 4], 2, self.rng) for _ in range(300)}
        self.assertEqual(samples, {1, 3})
        ids, p = r.nucleus_candidates(np.log([.6, .25, .15]), .8)
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(p, [.6/.85, .25/.85])
        self.assertEqual(len(r.nucleus_candidates([2, 1, 0], 1)[0]), 3)

    def test_beam_stops_at_eos(self):
        def lp(tokens):
            return np.log([.1, .9])
        sequence, score = r.beam_search(lp, [7], eos=1, beam_width=2, max_new_tokens=5)
        self.assertEqual(sequence, [7, 1])
        self.assertAlmostEqual(score, math.log(.9))

    def test_entropy_lazy_contract_and_block_invariance(self):
        class IndexedOnly:
            def __len__(self): return 3
            def __iter__(self): raise AssertionError("iteration forbidden")
            def __getitem__(self, index):
                if not isinstance(index, int): raise AssertionError("slicing forbidden")
                return [1000., 999., -math.inf][index]
        for block in [1, 2, 10]:
            self.assertAlmostEqual(r.compute_entropy_blockwise(IndexedOnly(), block), .5822031088882179)
        self.assertEqual(r.compute_entropy_blockwise([-math.inf, 3], 1), 0)

    def test_entropy_validation(self):
        for logits, block in [([], 1), ([1], 0), ([math.nan], 1), ([math.inf], 1), ([-math.inf], 1)]:
            with self.assertRaises(ValueError): r.compute_entropy_blockwise(logits, block)

    def test_layernorm_all_gradients(self):
        x = self.rng.normal(size=(2, 3))
        gamma, beta = self.rng.normal(size=3), self.rng.normal(size=3)
        upstream = self.rng.normal(size=x.shape)
        _, cache = r.layernorm_forward(x, gamma, beta)
        analytical = r.layernorm_backward(upstream, cache)
        def loss(): return float(np.sum(r.layernorm_forward(x, gamma, beta)[0] * upstream))
        for array, gradient in zip([x, gamma, beta], analytical):
            np.testing.assert_allclose(gradient, numerical_gradient(loss, array), atol=1e-7, rtol=1e-6)

    def test_layernorm_offset_and_constant_row(self):
        x = np.array([[1., 2., 3., 4.], [2., 2., 2., 2.]])
        gamma, beta = np.ones(4), np.arange(4.)
        base, _ = r.layernorm_forward(x, gamma, beta)
        shifted, _ = r.layernorm_forward(x+1e8, gamma, beta)
        np.testing.assert_allclose(base, shifted, atol=1e-7)
        np.testing.assert_allclose(base[1], beta)

    def test_mlp_all_parameter_gradients(self):
        x = self.rng.uniform(.2, 1., (3, 2))
        y = self.rng.normal(size=(3, 1))
        shapes = [(2, 3), (3,), (3, 2), (2,), (2, 1), (1,)]
        params = [self.rng.uniform(.1, .5, shape) for shape in shapes]
        _, gradients = r.mlp_loss_and_gradients(x, y, params)
        def loss(): return r.mlp_loss_and_gradients(x, y, params)[0]
        for array, gradient in zip(params, gradients):
            np.testing.assert_allclose(gradient, numerical_gradient(loss, array), atol=1e-7, rtol=1e-6)

    def test_kmeans_centers_and_empty_clusters(self):
        x = np.array([[0.], [1.], [9.], [10.]])
        centers, labels = r.kmeans(x, 2, seed=0)
        np.testing.assert_allclose(np.sort(centers[:, 0]), [.5, 9.5])
        np.testing.assert_array_equal(labels, ((x[:, None, :] - centers[None, :, :])**2).sum(-1).argmin(1))
        centers, _ = r.kmeans(np.ones((4, 2)), 3)
        self.assertTrue(np.isfinite(centers).all())

    def test_knn_tie_convention(self):
        self.assertEqual(r.knn_predict([[0], [2]], [2, 1], [1], 2), 1)
        self.assertEqual(r.knn_predict([[0], [2], [3]], [0, 1, 1], [2.5], 3), 1)

    def test_calibration_equal_bin_weighting(self):
        confidence = np.array([.9, .95, 1, .5, .55])
        self.assertAlmostEqual(r.ace_calc(np.log(confidence), [1, 1, 1, 0, 1], 5), .0375)
        self.assertEqual(r.ace_calc([], [], 5), 0)
        self.assertEqual(r.ace_calc([0, 0], [1, 1], 15), 0)

    def test_focal_reduces_to_bce_and_stays_finite(self):
        z, y = np.array([-1000., 1000., 0]), np.array([0, 1, 1])
        expected = np.mean(np.logaddexp(0, z) - y*z)
        self.assertAlmostEqual(r.binary_focal_loss(z, y, gamma=0), expected)
        self.assertTrue(np.isfinite(r.binary_focal_loss(-z, y)))

    def test_contrastive_symmetry_and_scale_invariance(self):
        q, k = self.rng.normal(size=(4, 3)), self.rng.normal(size=(4, 3))
        self.assertAlmostEqual(r.info_nce_loss(q, k), r.info_nce_loss(k, q))
        self.assertAlmostEqual(r.info_nce_loss(q, k), r.info_nce_loss(q*3, k*7))
        self.assertAlmostEqual(r.info_nce_loss([[1, 0]], [[0, 1]]), 0)

    def test_batch_deadlines_backlog_and_mapping(self):
        calls = []
        def model(inputs):
            calls.append(inputs)
            return [x*10 for x in inputs]
        b = r.DynamicBatcher(model, 3, .05)
        for i in range(5): self.assertEqual(b.submit(i, 0), i)
        self.assertEqual(b.dispatch(.1), [(0, 0), (1, 10), (2, 20)])
        self.assertEqual(b.dispatch(.1), [(3, 30), (4, 40)])
        self.assertEqual(calls, [[0, 1, 2], [3, 4]])
        self.assertIsNone(b.next_deadline())
        b.submit(9, 1)
        self.assertEqual(b.dispatch(1.049), [])
        self.assertEqual(b.dispatch(1.05), [(5, 90)])

    def test_packing_example_and_oversize(self):
        examples = [([5, 6], [7]), ([8], [9, 10]), ([11], [12])]
        result = r.pack(examples, 8, 99, 0)
        self.assertEqual(result[0].input_ids, [5, 6, 7, 99, 8, 9, 10, 99])
        self.assertEqual(result[0].loss_mask, [0, 0, 1, 1, 0, 1, 1, 1])
        self.assertEqual(result[0].position_ids, [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(result[1].input_ids, [11, 12, 99, 0, 0, 0, 0, 0])
        self.assertEqual(r.pack([([1, 2], [3, 4])], 4, 99, 0), [])

    def test_worked_numbers(self):
        self.assertAlmostEqual(math.log(6)/5, .358351893845611, places=12)
        self.assertAlmostEqual(.009/(.009+.0495), .15384615384615385)
        self.assertEqual(2*1*32*4096*8*128*2, 512*1024**2)
        self.assertAlmostEqual(-math.log(1/(1+math.exp(-1))), .31326168751822286)
        np.testing.assert_allclose((np.array([0,0,1,1])-.5)/.5, [-1,-1,1,1])


if __name__ == "__main__":
    unittest.main()
