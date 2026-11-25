import random
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from review_w112_implementation import TwoLayerNet
from review_w149_implementation import ManualTransformerBlock
from torch_comparisons_112_149 import (
    TorchTwoLayerNet,
    TorchPostNormBlock,
    train_classifier,
)


class TorchComparisonTests(unittest.TestCase):
    def test_two_layer_all_gradients_and_training(self):
        torch.set_num_threads(1)
        reference = TwoLayerNet(2, 8, 2, seed=3)
        model = TorchTwoLayerNet(2, 8, 2)
        model.load_numpy(reference.params)
        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])
        value, grads = reference.loss_and_grads(x, y)
        features = torch.tensor(x)
        labels = torch.tensor(y)
        loss = F.cross_entropy(model(features), labels)
        loss.backward()
        self.assertAlmostEqual(loss.item(), value, places=12)
        for name, parameter in [
            ("W1", model.first.weight),
            ("b1", model.first.bias),
            ("W2", model.second.weight),
            ("b2", model.second.bias),
        ]:
            actual = parameter.grad.detach().numpy()
            np.testing.assert_allclose(
                actual.T if name.startswith("W") else actual,
                grads[name],
                atol=1e-12,
                rtol=1e-12,
            )
        history = train_classifier(model, features, labels, epochs=500)
        self.assertLess(history[-1], 0.02)
        self.assertEqual(model(features).argmax(-1).tolist(), y.tolist())

    def test_full_postnorm_block_forward_and_every_gradient(self):
        rng = random.Random(7)
        D, H, hidden = 4, 2, 3
        matrix = lambda rows, cols: [
            [rng.uniform(-0.4, 0.4) for _ in range(cols)] for _ in range(rows)
        ]
        params = {name: matrix(D, D) for name in ["W_Q", "W_K", "W_V", "W_O"]}
        params.update(
            ffn1_w=matrix(hidden, D),
            ffn1_b=[0.2, -0.3, 0.4],
            ffn2_w=matrix(D, hidden),
            ffn2_b=[0.1] * D,
        )
        for n in [1, 2]:
            params[f"norm{n}_gamma"] = [rng.uniform(0.7, 1.3) for _ in range(D)]
            params[f"norm{n}_beta"] = [rng.uniform(-0.2, 0.2) for _ in range(D)]
        x = matrix(3, D)
        upstream = matrix(3, D)
        manual = ManualTransformerBlock(D, H, hidden, params)
        expected = manual.forward(x)
        dx, grads = manual.backward(upstream)
        model = TorchPostNormBlock(D, H, params)
        tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        result = model(tensor)
        np.testing.assert_allclose(
            result.detach().numpy(), expected, atol=1e-12, rtol=1e-12
        )
        (result * torch.tensor(upstream, dtype=torch.float64)).sum().backward()
        np.testing.assert_allclose(tensor.grad.numpy(), dx, atol=1e-11, rtol=1e-11)
        for name, parameter in model.params.items():
            np.testing.assert_allclose(
                parameter.grad.numpy(),
                grads[name],
                atol=1e-11,
                rtol=1e-11,
                err_msg=name,
            )


if __name__ == "__main__":
    unittest.main()
