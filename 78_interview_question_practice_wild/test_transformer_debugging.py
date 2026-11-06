"""Behavioral checks for W006; no corpus download or training run required."""
import unittest
from pathlib import Path
import ast
import re
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from transformer_debugging import GPT, ParityClassifier


@unittest.skipIf(torch is None, "PyTorch is required for Transformer behavior checks")
class TransformerDebuggingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(n_tokens=16, hiddens=16, n_heads=4, layers=2)
        self.model.eval()
        self.tokens = torch.randint(0, 256, (2, 8))

    def test_changing_future_tokens_cannot_change_earlier_logits(self):
        changed = self.tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 17) % 256
        with torch.no_grad():
            original = self.model(self.tokens, mode=None)
            modified = self.model(changed, mode=None)
        torch.testing.assert_close(original[:, :4], modified[:, :4], rtol=0, atol=1e-6)

    def test_cache_matches_full_prefix_at_every_decode_step(self):
        with torch.no_grad():
            self.model(self.tokens[:, :3], mode="prefill")
            for i in range(3, self.tokens.size(1)):
                cached = self.model(self.tokens[:, i:i + 1], mode="decode")
                full = self.model(self.tokens[:, :i + 1], mode=None)
                torch.testing.assert_close(cached[:, -1], full[:, -1], rtol=1e-4, atol=1e-5)
                self.assertEqual(self.model.size_of_kv_cache, i + 1)

    def test_prefill_resets_previous_request(self):
        with torch.no_grad():
            self.model(self.tokens[:, :6], mode="prefill")
            other = self.tokens[:, :2]
            self.model(other, mode="prefill")
            cached = self.model(self.tokens[:, 2:3], mode="decode")
            full = self.model(self.tokens[:, :3], mode=None)
        torch.testing.assert_close(cached[:, -1], full[:, -1], rtol=1e-4, atol=1e-5)
        self.assertEqual(self.model.blocks[0].attn.k_cache.size(2), 3)

    def test_loss_uses_next_tokens_and_gradients_are_finite(self):
        self.model.train()
        expected_logits = self.model(self.tokens[:, :-1], mode=None)
        expected = torch.nn.functional.cross_entropy(
            expected_logits.reshape(-1, 256), self.tokens[:, 1:].reshape(-1)
        )
        loss = self.model.loss(self.tokens, None, mode=None)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_parity_labels_shapes_and_loss(self):
        classifier = ParityClassifier(self.model)
        tokens = torch.tensor([[ord(c) for c in s] for s in ["00001234", "00001235"]])
        labels = torch.tensor([0, 1])
        logits = classifier(tokens)
        self.assertEqual(logits.shape, (2, 2))
        expected = torch.nn.functional.cross_entropy(logits, labels)
        torch.testing.assert_close(classifier.loss(tokens), expected)
        classifier.loss(tokens).backward()
        self.assertTrue(torch.isfinite(classifier.head.weight.grad).all())

    def test_multi_token_decode_is_rejected(self):
        with torch.no_grad():
            self.model(self.tokens[:, :2], mode="prefill")
            with self.assertRaises(ValueError):
                self.model(self.tokens[:, 2:4], mode="decode")

    def test_uninitialized_decode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.tokens[:, :1], mode="decode")


class ArticleCodeTests(unittest.TestCase):
    def test_all_python_snippets_parse(self):
        path = Path(__file__).parent / "questions" / "W006.md"
        if not path.exists():
            self.skipTest("Run within the chapter directory")
        for code in re.findall(r"```python\n(.*?)```", path.read_text(), re.S):
            ast.parse(code)


if __name__ == "__main__":
    unittest.main()
