"""Optional PyTorch study/verification models, separate from library-free submissions."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class TorchTwoLayerNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.first = nn.Linear(n_in, n_hidden, dtype=torch.float64)
        self.second = nn.Linear(n_hidden, n_out, dtype=torch.float64)
        nn.init.kaiming_normal_(self.first.weight, nonlinearity="relu")
        nn.init.normal_(self.second.weight, std=1 / math.sqrt(n_hidden))
        nn.init.zeros_(self.first.bias)
        nn.init.zeros_(self.second.bias)

    def forward(self, x):
        return self.second(F.relu(self.first(x)))

    @torch.no_grad()
    def load_numpy(self, params):
        # NumPy W is (input,output); nn.Linear weight is (output,input).
        self.first.weight.copy_(torch.as_tensor(params["W1"].T))
        self.first.bias.copy_(torch.as_tensor(params["b1"]))
        self.second.weight.copy_(torch.as_tensor(params["W2"].T))
        self.second.bias.copy_(torch.as_tensor(params["b2"]))


def train_classifier(model, x, labels, epochs=500, learning_rate=0.5):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), labels)  # logits (n,k), labels long (n,)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history


class TorchPostNormBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, params, eps=1e-5):
        super().__init__()
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("Head count must divide embed_dim")
        self.dimension = embed_dim
        self.heads = num_heads
        self.eps = eps
        # W149 already uses nn.Linear's output-by-input convention.
        self.params = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(value, dtype=torch.float64))
                for name, value in params.items()
            }
        )

    def forward(self, x):
        sequence, dimension = x.shape
        head_width = self.dimension // self.heads

        def split(weight):
            projected = F.linear(x, weight)
            return projected.reshape(sequence, self.heads, head_width).transpose(0, 1)

        p = self.params
        query = split(p["W_Q"])
        key = split(p["W_K"])
        value = split(p["W_V"])
        weights = (query @ key.transpose(-1, -2) / math.sqrt(head_width)).softmax(
            dim=-1
        )
        context = (weights @ value).transpose(0, 1).reshape(sequence, dimension)
        attention = F.linear(context, p["W_O"])
        norm1 = F.layer_norm(
            x + attention, (dimension,), p["norm1_gamma"], p["norm1_beta"], self.eps
        )
        hidden = F.relu(F.linear(norm1, p["ffn1_w"], p["ffn1_b"]))
        feedforward = F.linear(hidden, p["ffn2_w"], p["ffn2_b"])
        return F.layer_norm(
            norm1 + feedforward,
            (dimension,),
            p["norm2_gamma"],
            p["norm2_beta"],
            self.eps,
        )
