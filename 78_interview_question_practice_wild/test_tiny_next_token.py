from pathlib import Path
import ast
import re
import unittest
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from tiny_next_token import (SyntheticDataset,TinyNextTokenModel,next_token_loss,
                                 train_one_epoch,evaluate_loss,run_demo)
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch is required for W073 behavior tests')
class TinyNextTokenTests(unittest.TestCase):
    def test_architecture_shape_and_position_independence(self):
        torch.manual_seed(1)
        model=TinyNextTokenModel(8,16)
        self.assertEqual(sum(isinstance(m,nn.Linear) for m in model.modules()),2)
        a=torch.tensor([[1,2,3]])
        b=torch.tensor([[7,2,5]])
        self.assertEqual(model(a).shape,(1,3,8))
        torch.testing.assert_close(model(a)[:,1],model(b)[:,1])

    def test_shift_is_next_token_not_current_token(self):
        class PerfectSuccessor(nn.Module):
            def forward(self,tokens):
                self.seen=tokens.clone()
                return F.one_hot((tokens+1)%5,num_classes=5).float()*20
        model=PerfectSuccessor()
        sequences=torch.tensor([[2,3,4,0]])
        loss=next_token_loss(model,sequences)
        torch.testing.assert_close(model.seen,torch.tensor([[2,3,4]]))
        self.assertLess(loss.item(),1e-6)
        wrong=F.cross_entropy(model(model.seen).reshape(-1,5),sequences[:,:-1].reshape(-1))
        self.assertGreater(wrong.item(),10)

    def test_gradients_and_short_sequences(self):
        model=TinyNextTokenModel(8,16)
        batch=SyntheticDataset(4,5,8).sequences
        next_token_loss(model,batch).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaises(ValueError):next_token_loss(model,torch.tensor([[1]]))

    def test_evaluation_weights_last_batch_and_restores_mode(self):
        model=TinyNextTokenModel(8,16)
        dataset=SyntheticDataset(7,5,8)
        expected=next_token_loss(model,dataset.sequences).item()
        actual=evaluate_loss(model,DataLoader(dataset,batch_size=3))
        self.assertAlmostEqual(actual,expected,places=6)
        self.assertTrue(model.training)
        model.eval()
        evaluate_loss(model,DataLoader(dataset,batch_size=3))
        self.assertFalse(model.training)

    def test_reproducible_learnable_demo_decreases_loss_and_learns_rule(self):
        model,history=run_demo(seed=7,epochs=4)
        self.assertLess(history[-1],history[0])
        self.assertLess(history[-1],.01)
        with torch.no_grad():predicted=model(torch.arange(8)[None,:]).argmax(-1)
        torch.testing.assert_close(predicted,(torch.arange(8)[None,:]+1)%8)
        _,again=run_demo(seed=7,epochs=4)
        torch.testing.assert_close(torch.tensor(history),torch.tensor(again))

    def test_demo_dataset_contract(self):
        dataset=SyntheticDataset(9,4,5)
        self.assertEqual(dataset.sequences.dtype,torch.long)
        self.assertEqual(next(iter(DataLoader(dataset,batch_size=3))).shape,(3,4))
        torch.testing.assert_close(dataset.sequences[:,1:],(dataset.sequences[:,:-1]+1)%5)


if __name__=='__main__':unittest.main()
