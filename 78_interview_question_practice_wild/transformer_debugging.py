"""W006: a small, explicit GPT for studying four correctness bugs.

Run without external data. Training uses mode=None. Cache decoding is an
inference exercise: prefill a prefix, then supply exactly one token per call.
Caches belong to this model instance; prefill resets them for a new request.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hiddens, n_heads):
        super().__init__()
        if hiddens % n_heads:
            raise ValueError("hidden size must be divisible by head count")
        self.n_heads = n_heads
        self.key = nn.Linear(hiddens, hiddens)
        self.query = nn.Linear(hiddens, hiddens)
        self.value = nn.Linear(hiddens, hiddens)
        self.out = nn.Linear(hiddens, hiddens)
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, mode="prefill"):
        b, t, c = x.shape
        h = self.n_heads
        d = c // h
        q = self.query(x).view(b, t, h, d).transpose(1, 2)
        k = self.key(x).view(b, t, h, d).transpose(1, 2)
        v = self.value(x).view(b, t, h, d).transpose(1, 2)
        # q, k, v: (batch, heads, tokens, head_dim)
        if mode == "prefill":
            self.k_cache, self.v_cache = k, v
        elif mode == "decode":
            if t != 1 or self.k_cache is None or self.v_cache is None:
                raise ValueError("decode needs a prefilled cache and one token")
            if self.k_cache.shape[:2] != (b, h):
                raise ValueError("batch changed; start a new prefill")
            k = torch.cat((self.k_cache, k), dim=2)
            v = torch.cat((self.v_cache, v), dim=2)
            self.k_cache, self.v_cache = k, v
        elif mode is not None:
            raise ValueError("mode must be None, 'prefill' or 'decode'")

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        if mode != "decode":
            # True means forbidden: only strictly future keys are blocked.
            future = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        # One-token decode sees only past tokens plus itself, so all keys
        # already in the cache are allowed. A top-left 1 x K triangle is wrong.
        weights = F.softmax(scores, dim=-1)
        y = weights @ v                           # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # (B, T, H*D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, hiddens, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(hiddens)
        self.ln2 = nn.LayerNorm(hiddens)
        self.attn = CausalSelfAttention(hiddens, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hiddens, 4 * hiddens),
            nn.GELU(),
            nn.Linear(4 * hiddens, hiddens),
        )

    def forward(self, x, mode="prefill"):
        x = x + self.attn(self.ln1(x), mode=mode)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_tokens=64, hiddens=64, n_heads=4,
                 n_classes=256, layers=4, vocab=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, hiddens)
        self.pos_emb = nn.Parameter(torch.empty(1, n_tokens, hiddens))
        nn.init.normal_(self.pos_emb)
        self.blocks = nn.ModuleList(Block(hiddens, n_heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(hiddens)
        self.final = nn.Linear(hiddens, n_classes, bias=False)
        self.size_of_kv_cache = 0

    def hidden_states(self, tokens, mode=None):
        if mode not in (None, "prefill", "decode"):
            raise ValueError("invalid mode")
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("tokens must have shape (batch, nonempty sequence)")
        t = tokens.size(1)
        start = self.size_of_kv_cache if mode == "decode" else 0
        if mode == "decode" and (t != 1 or start == 0):
            raise ValueError("decode needs a prefilled cache and one token")
        if start + t > self.pos_emb.size(1):
            raise ValueError("sequence exceeds positional embedding capacity")
        x = self.tok_emb(tokens) + self.pos_emb[:, start:start + t, :]
        for block in self.blocks:
            x = block(x, mode=mode)
        if mode in ("prefill", "decode"):
            self.size_of_kv_cache = start + t
        return self.ln_f(x)

    def forward(self, x, mode="prefill"):
        return self.final(self.hidden_states(x, mode))

    def loss(self, tokens, train_label=None, mode="prefill"):
        # The interface retains train_label, but LM targets come from tokens.
        if tokens.size(1) < 2:
            raise ValueError("next-token loss needs at least two tokens")
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = self.forward(inputs, mode=mode)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )


class ParityClassifier(nn.Module):
    """An intentionally overpowered Transformer for an 8-digit binary task."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.tok_emb.embedding_dim, 2)

    def forward(self, tokens):
        # The last position can see all eight digits under the causal mask.
        hidden = self.backbone.hidden_states(tokens, mode=None)
        return self.head(hidden[:, -1, :])

    def loss(self, tokens, train_label=None):
        labels = (tokens[:, -1] - ord("0")) % 2 if train_label is None else train_label
        return F.cross_entropy(self(tokens), labels.long())
