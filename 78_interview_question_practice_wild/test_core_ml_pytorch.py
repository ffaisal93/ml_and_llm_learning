"""Check the tensor implementations and their gradients against references."""
import unittest
import numpy as np
import core_ml_readable as reference
try:
    import torch
    from torch.nn import functional as F
    import core_ml_pytorch as examples
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional")
class PyTorchCoreMLTests(unittest.TestCase):
    def test_softmax_and_log_softmax_values_and_gradient(self):
        logits = torch.tensor([[1000., 1001., 999.], [-1000., 0., 1.]],
                              dtype=torch.float64, requires_grad=True)
        probabilities = examples.softmax_torch(logits)
        log_probabilities = examples.log_softmax_torch(logits)
        torch.testing.assert_close(probabilities, F.softmax(logits, dim=-1))
        torch.testing.assert_close(log_probabilities, F.log_softmax(logits, dim=-1))
        np.testing.assert_allclose(probabilities.detach().numpy(), reference.softmax(logits.detach().numpy()))
        loss = -log_probabilities[0, 1]
        loss.backward()
        expected_gradient = torch.zeros_like(logits)
        expected_gradient[0] = probabilities.detach()[0]
        expected_gradient[0, 1] -= 1
        torch.testing.assert_close(logits.grad, expected_gradient)
        extreme = examples.log_softmax_torch(torch.tensor([-1000., 0.]))
        self.assertTrue(torch.isfinite(extreme).all())
        self.assertEqual(extreme[0].item(), -1000.)
        ordinary = torch.tensor([[.2, -.7, 1.1]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(examples.softmax_torch, (ordinary,)))

    def test_layernorm_output_and_all_gradients_match_manual_backward(self):
        torch.manual_seed(4)
        inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        normalizer = examples.ReadableLayerNorm(4).double()
        with torch.no_grad():
            normalizer.gamma.copy_(torch.tensor([.5, 1., 2., -.7]))
            normalizer.beta.copy_(torch.tensor([.2, -.3, .4, 1.]))
        upstream = torch.randn_like(inputs)
        output = normalizer(inputs)
        expected_output, cache = reference.layer_norm_forward(
            inputs.detach().numpy(), normalizer.gamma.detach().numpy(), normalizer.beta.detach().numpy()
        )
        np.testing.assert_allclose(output.detach().numpy(), expected_output, atol=1e-12)
        expected_gradients = reference.layer_norm_backward(upstream.numpy(), cache)
        (output * upstream).sum().backward()
        for actual, expected in zip([inputs.grad, normalizer.gamma.grad, normalizer.beta.grad], expected_gradients):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-9, atol=1e-10)
        built_in = F.layer_norm(inputs, (4,), normalizer.gamma, normalizer.beta, eps=1e-5)
        torch.testing.assert_close(output, built_in)
        constant = torch.full((1, 4), 9., dtype=torch.float64)
        torch.testing.assert_close(normalizer(constant), normalizer.beta[None, :])

    def test_mlp_numpy_weights_and_gradients_match_with_transposed_storage(self):
        random = np.random.default_rng(4)
        inputs = random.normal(size=(3, 2))
        targets = random.normal(size=(3, 2))
        parameters = reference.initialize_mlp(2, 3, 4, 2, seed=5)
        parameters["b1"][:] = .8
        parameters["b2"][:] = .8
        expected_loss, expected_predictions, expected_gradients = reference.mlp_loss_and_gradients(
            inputs, targets, parameters
        )
        model = examples.TwoHiddenLayerMLP(2, 3, 4, 2).double()
        layers = [model.layer1, model.layer2, model.output]
        with torch.no_grad():
            for i, layer in enumerate(layers, 1):
                layer.weight.copy_(torch.from_numpy(parameters[f"W{i}"].T))
                layer.bias.copy_(torch.from_numpy(parameters[f"b{i}"]))
        predictions = model(torch.from_numpy(inputs))
        loss = F.mse_loss(predictions, torch.from_numpy(targets))
        np.testing.assert_allclose(predictions.detach().numpy(), expected_predictions, atol=1e-12)
        self.assertAlmostEqual(loss.item(), expected_loss)
        loss.backward()
        for i, layer in enumerate(layers, 1):
            np.testing.assert_allclose(layer.weight.grad.numpy().T, expected_gradients[f"W{i}"], atol=1e-12)
            np.testing.assert_allclose(layer.bias.grad.numpy(), expected_gradients[f"b{i}"], atol=1e-12)

    def test_mlp_scalar_example_and_training_loop(self):
        scalar_model = examples.TwoHiddenLayerMLP(1, 1, 1, 1)
        with torch.no_grad():
            for layer in [scalar_model.layer1, scalar_model.layer2, scalar_model.output]:
                layer.weight.fill_(1)
                layer.bias.zero_()
        loss = F.mse_loss(scalar_model(torch.tensor([[2.]])), torch.tensor([[1.]]))
        self.assertEqual(loss.item(), 1)
        loss.backward()
        self.assertEqual(scalar_model.layer1.weight.grad.item(), 4)
        torch.manual_seed(0)
        model = examples.TwoHiddenLayerMLP(2, 8, 8, 1)
        inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        targets = inputs.sum(dim=1, keepdim=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        initial = F.mse_loss(model(inputs), targets).item()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = F.mse_loss(model(inputs), targets).item()
        self.assertLess(final, initial)
        self.assertLess(final, 1e-3)

    def test_pairwise_distances_values_and_backward(self):
        x = torch.tensor([[1., 1.], [3., 4.]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0., 0.], [0., 4.]], dtype=torch.float64, requires_grad=True)
        distances = examples.pairwise_euclidean_torch(x, y)
        expected = reference.pairwise_euclidean(x.detach().numpy(), y.detach().numpy())
        np.testing.assert_allclose(distances.detach().numpy(), expected)
        distances.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())

    def test_focal_loss_values_gradients_and_bce_limit(self):
        logits = torch.tensor([.2, -1.3, 3.], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1., 0., 1.], dtype=torch.float64)
        for gamma in [0., 1.5, 2.]:
            for alpha in [None, .25]:
                actual = examples.binary_focal_loss_torch(logits, labels, gamma, alpha)
                expected = reference.binary_focal_loss(logits.detach().numpy(), labels.numpy(), gamma, alpha)
                self.assertAlmostEqual(actual.item(), expected)
        torch.testing.assert_close(examples.binary_focal_loss_torch(logits, labels, gamma=0),
                                   F.binary_cross_entropy_with_logits(logits, labels))
        self.assertTrue(torch.autograd.gradcheck(
            lambda z: examples.binary_focal_loss_torch(z, labels, gamma=1.5, alpha=.25), (logits,)
        ))
        extreme = torch.tensor([1000., -1000., -1000., 1000.], requires_grad=True)
        extreme_labels = torch.tensor([1., 0., 1., 0.])
        loss = examples.binary_focal_loss_torch(extreme, extreme_labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(extreme.grad).all())


if __name__ == "__main__":
    unittest.main()
