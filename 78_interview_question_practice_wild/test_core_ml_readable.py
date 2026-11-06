"""Check the numerical behavior and execute the F041–F050 article examples."""
from pathlib import Path
import ast
import contextlib
import io
import re
import unittest
import numpy as np
import core_ml_readable as ml
try:
    import torch
except ImportError:
    torch = None


def numerical_gradient(array, objective, step=1e-6):
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        try:
            array[index] = original + step
            plus = objective()
            array[index] = original - step
            minus = objective()
        finally:
            array[index] = original
        result[index] = (plus - minus) / (2 * step)
    return result


class CoreMLTests(unittest.TestCase):
    def test_softmax_normalization_shift_invariance_and_log_underflow(self):
        logits = np.array([[1000., 1001., 999.], [-1000., 0., 1.]])
        probabilities = ml.softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(probabilities, ml.softmax(logits + 5000))
        np.testing.assert_allclose(np.exp(ml.log_softmax(logits)), probabilities)
        self.assertEqual(ml.log_softmax([-1000., 0.])[0], -1000.)
        self.assertTrue(np.isfinite(ml.log_softmax(logits)).all())

    def test_layer_norm_gradients_for_inputs_scale_and_bias(self):
        random = np.random.default_rng(8)
        x = random.normal(size=(3, 4))
        gamma = random.normal(size=4)
        beta = random.normal(size=4)
        upstream = random.normal(size=x.shape)
        _, cache = ml.layer_norm_forward(x, gamma, beta)
        gradients = ml.layer_norm_backward(upstream, cache)
        def objective():
            output, _ = ml.layer_norm_forward(x, gamma, beta)
            return np.sum(output * upstream)
        for variable, analytical in zip([x, gamma, beta], gradients):
            np.testing.assert_allclose(analytical, numerical_gradient(variable, objective),
                                       rtol=1e-5, atol=1e-6)

    def test_layer_norm_constant_rows_and_row_independence(self):
        x = np.array([[4., 4., 4.], [1., 2., 3.]])
        output, cache = ml.layer_norm_forward(x, np.ones(3), np.array([1., 2., 3.]))
        np.testing.assert_allclose(output[0], [1., 2., 3.])
        modified = x.copy()
        modified[1] *= 10
        changed, _ = ml.layer_norm_forward(modified, np.ones(3), np.array([1., 2., 3.]))
        np.testing.assert_allclose(output[0], changed[0])
        dx, _, _ = ml.layer_norm_backward(np.ones_like(x), cache)
        np.testing.assert_allclose(dx, 0, atol=1e-12)

    def test_mlp_all_parameter_gradients_with_multi_output_average(self):
        random = np.random.default_rng(2)
        x = random.normal(size=(3, 2))
        targets = random.normal(size=(3, 2))
        parameters = ml.initialize_mlp(2, 3, 4, 2, seed=3)
        for name in ["W1", "W2", "W3"]:
            parameters[name] *= 0.1
        parameters["b1"][:] = 0.8
        parameters["b2"][:] = 0.8
        _, _, gradients = ml.mlp_loss_and_gradients(x, targets, parameters)
        def objective():
            return ml.mlp_loss_and_gradients(x, targets, parameters)[0]
        for name in parameters:
            np.testing.assert_allclose(gradients[name], numerical_gradient(parameters[name], objective),
                                       rtol=1e-5, atol=1e-6, err_msg=name)

    def test_mlp_relu_blocks_inactive_path_and_update_reduces_loss(self):
        parameters = ml.initialize_mlp(1, 1, 1, 1)
        for name in ["W1", "W2", "W3"]:
            parameters[name][:] = 1
        parameters["b1"][:] = -3
        loss, _, gradients = ml.mlp_loss_and_gradients([[1]], [[1]], parameters)
        np.testing.assert_array_equal(gradients["W1"], 0)
        np.testing.assert_array_equal(gradients["W2"], 0)
        for name in parameters:
            parameters[name] -= 0.01 * gradients[name]
        self.assertLess(ml.mlp_loss_and_gradients([[1]], [[1]], parameters)[0], loss)

    def test_kmeans_separates_points_and_returns_consistent_labels(self):
        points = np.array([[0.], [1.], [9.], [10.]])
        centers, labels = ml.k_means(points, 2, seed=0)
        np.testing.assert_allclose(np.sort(centers[:, 0]), [0.5, 9.5])
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        points = np.array([[0., 2.], [5., 1.], [2., 8.], [11., 4.], [2., 6.]])
        centers, labels = ml.k_means(points, 2, max_iterations=1, seed=2)
        expected = np.argmin(np.sum((points[:, None] - centers[None]) ** 2, axis=-1), axis=1)
        np.testing.assert_array_equal(labels, expected)

    def test_kmeans_empty_clusters_and_invalid_k(self):
        centers, labels = ml.k_means(np.ones((3, 2)), 3)
        self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_array_equal(labels, [0, 0, 0])
        with self.assertRaises(ValueError):
            ml.k_means([[1]], 2)

    def test_knn_predictions_distance_ties_and_class_ties(self):
        result = ml.knn_predict([[0], [1], [9], [10]], [0, 0, 1, 1], [[.2], [9.2]], 3)
        np.testing.assert_array_equal(result, [0, 1])
        # Equidistant k=1 picks the first training row.
        np.testing.assert_array_equal(ml.knn_predict([[-1], [1]], [9, 2], [[0]], 1), [9])
        # k=2 vote tie picks the smallest label.
        np.testing.assert_array_equal(ml.knn_predict([[-1], [1]], [9, 2], [[0]], 2), [2])
        with self.assertRaises(ValueError):
            ml.knn_predict([[0]], [1], [[2]], 0)

    def test_pairwise_distances_rectangular_shapes_and_agreement(self):
        x = np.array([[0, 0], [3, 4]])
        y = np.array([[0, 0], [0, 4], [3, 0]])
        expected = np.array([[0., 4., 3.], [5., 3., 4.]])
        np.testing.assert_allclose(ml.pairwise_euclidean(x, y), expected)
        np.testing.assert_allclose(ml.pairwise_euclidean_fast(x, y), expected)
        self.assertEqual(ml.pairwise_euclidean(np.empty((0, 2)), y).shape, (0, 3))

    def test_infonce_known_value_symmetry_scaling_and_single_pair(self):
        identity = np.eye(2)
        self.assertAlmostEqual(ml.symmetric_info_nce(identity, identity, 1), np.log1p(np.exp(-1)))
        queries = np.array([[1, 2], [-1, 3], [2, -.2]])
        keys = np.array([[2, 1], [.2, 3], [1, -.7]])
        loss = ml.symmetric_info_nce(queries, keys)
        self.assertAlmostEqual(loss, ml.symmetric_info_nce(keys, queries))
        self.assertAlmostEqual(loss, ml.symmetric_info_nce(queries * 7, keys * 3))
        self.assertAlmostEqual(ml.symmetric_info_nce([[1., 0]], [[0., 1]]), 0)
        with self.assertRaises(ValueError):
            ml.symmetric_info_nce([[0., 0]], [[1., 0]])
        with self.assertRaises(ValueError):
            ml.symmetric_info_nce(identity, identity, 0)

    @unittest.skipIf(torch is None, "PyTorch is optional")
    def test_torch_infonce_matches_numpy_and_backpropagates(self):
        queries = torch.tensor([[1., .2], [.1, 1.]], dtype=torch.float64, requires_grad=True)
        keys = torch.tensor([[.9, 0.], [0., 1.1]], dtype=torch.float64, requires_grad=True)
        loss = ml.symmetric_info_nce_torch(queries, keys, .5)
        expected = ml.symmetric_info_nce(queries.detach().numpy(), keys.detach().numpy(), .5)
        self.assertAlmostEqual(loss.item(), expected)
        loss.backward()
        for gradient in [queries.grad, keys.grad]:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_focal_loss_bce_limit_class_weights_extremes(self):
        self.assertAlmostEqual(ml.binary_focal_loss([0], [1], gamma=0), np.log(2))
        self.assertAlmostEqual(ml.binary_focal_loss([0], [1], gamma=2), np.log(2) / 4)
        self.assertAlmostEqual(ml.binary_focal_loss([0], [1], gamma=0, alpha=.25), np.log(2) * .25)
        self.assertAlmostEqual(ml.binary_focal_loss([0], [0], gamma=0, alpha=.25), np.log(2) * .75)
        self.assertAlmostEqual(ml.binary_focal_loss([1000, -1000], [1, 0]), 0)
        self.assertAlmostEqual(ml.binary_focal_loss([1000, -1000], [0, 1]), 1000)
        self.assertLess(ml.binary_focal_loss([5], [1]), ml.binary_focal_loss([0], [1]))

    def test_calibration_equal_bins_endpoints_and_empty_data(self):
        confidence = [.9, .95, 1., .5, .55]
        error, report = ml.calibration_error(np.log(confidence), [1, 1, 1, 0, 1], 5)
        self.assertAlmostEqual(error, .0375)
        self.assertEqual([row["count"] for row in report], [2, 3])
        self.assertNotAlmostEqual(error, sum(row["count"] * row["gap"] for row in report) / 5)
        error, report = ml.calibration_error([-np.inf, 0], [0, 1], 5)
        self.assertEqual(error, 0)
        self.assertEqual([row["bin"] for row in report], [0, 4])
        self.assertEqual(ml.calibration_error([], []), (0., []))

    def test_annotation_leave_one_out_and_tie_handling(self):
        annotations = []
        for item, label in [("i1", 1), ("i2", 0), ("i3", 1)]:
            annotations += [(item, r, label) for r in ["A", "B", "C"]]
            annotations.append((item, "D", 1 - label))
        _, report, cleaned = ml.clean_annotations(annotations)
        self.assertEqual(report["A"]["agreement"], 1)
        self.assertEqual(report["D"]["agreement"], 0)
        self.assertEqual(len(cleaned), 9)
        consensus, report, _ = ml.clean_annotations([("x", "A", 1), ("x", "B", 0)])
        self.assertIsNone(consensus["x"])
        self.assertEqual(report["A"]["agreement"], 0)
        # With a 2-to-1 split, the majority raters have tied peers and no score.
        _, report, _ = ml.clean_annotations([("x", "A", 1), ("x", "B", 1), ("x", "C", 0)])
        self.assertIsNone(report["A"]["agreement"])
        self.assertEqual(report["A"]["resolved_items"], 0)
        self.assertFalse(report["A"]["keep"])
        with self.assertRaises(ValueError):
            ml.clean_annotations([("x", "A", 1), ("x", "A", 0)])

    def test_every_article_example_executes(self):
        page = Path(__file__).parent / "05_core_ml_coding.md"
        namespace = {}
        for index, code in enumerate(re.findall(r"```python\n(.*?)```", page.read_text(), re.S)):
            ast.parse(code)
            if code.startswith("import torch") and torch is None:
                continue
            with self.subTest(block=index):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, f"F041-F050 example {index}", "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
