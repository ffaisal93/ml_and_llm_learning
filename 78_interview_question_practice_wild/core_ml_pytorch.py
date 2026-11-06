"""Readable, differentiable PyTorch examples for the core ML coding page.

Inputs are floating tensors. Keep model parameters and inputs on the same
device and in the same dtype. No functions convert training tensors to NumPy.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def softmax_torch(logits, dim=-1):
    # Subtract one maximum per class row before exponentiating.
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def log_softmax_torch(logits, dim=-1):
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    log_total = shifted.exp().sum(dim=dim, keepdim=True).log()
    return shifted - log_total


class ReadableLayerNorm(nn.Module):
    def __init__(self, feature_count, epsilon=1e-5):
        super().__init__()
        if feature_count < 1 or epsilon <= 0:
            raise ValueError("need positive feature_count and epsilon")
        self.epsilon = epsilon
        # nn.Parameter registers these tensors so optimizers can find them.
        self.gamma = nn.Parameter(torch.ones(feature_count))
        self.beta = nn.Parameter(torch.zeros(feature_count))

    def forward(self, x):
        row_mean = x.mean(dim=-1, keepdim=True)
        centered = x - row_mean
        row_variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(row_variance + self.epsilon)
        return self.gamma * normalized + self.beta


class TwoHiddenLayerMLP(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, output_size)

    def forward(self, x):
        first_hidden = torch.relu(self.layer1(x))
        second_hidden = torch.relu(self.layer2(first_hidden))
        # Regression: leave the final output linear.
        return self.output(second_hidden)


def pairwise_euclidean_torch(x, y):
    # x: (N, D), y: (M, D), result: (N, M).
    return torch.cdist(x, y, p=2)


def binary_focal_loss_torch(logits, labels, gamma=2.0, alpha=None):
    if logits.shape != labels.shape or logits.numel() == 0:
        raise ValueError("logits and labels need matching nonempty shapes")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1, or None")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    one_minus_p_true = -torch.expm1(-bce)
    losses = one_minus_p_true.pow(gamma) * bce
    if alpha is not None:
        class_weights = labels * alpha + (1 - labels) * (1 - alpha)
        losses = class_weights * losses
    return losses.mean()
