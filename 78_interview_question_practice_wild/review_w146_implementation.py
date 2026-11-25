import hashlib
import torch
from torch import nn
import torch.nn.functional as F


class OnlineTextClassifier(nn.Module):
    def __init__(self, buckets=4096, dim=32, classes=2):
        super().__init__()
        if buckets < 2 or dim < 1 or classes < 2:
            raise ValueError("invalid model dimensions")
        self.buckets = buckets
        self.classes = classes
        self.embedding = nn.Embedding(buckets, dim)
        nn.init.normal_(self.embedding.weight, std=0.05)
        self.head = nn.Linear(dim, classes)

    def token_ids(self, text):
        # Stable across processes: Python hash() is deliberately randomized.
        tokens = text.lower().split()
        return [
            1
            + int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big")
            % (self.buckets - 1)
            for t in tokens
        ] or [0]

    def forward(self, text):
        ids = torch.tensor(
            self.token_ids(text), dtype=torch.long, device=self.embedding.weight.device
        )
        token_vectors = self.embedding(ids)
        document_vector = token_vectors.mean(dim=0)
        return self.head(document_vector), token_vectors, document_vector

    @torch.no_grad()
    def predict(self, text):
        return self(text)[0].softmax(dim=-1)


def learn_stream(model, records, lr=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    correct = total = 0
    loss_sum = 0.0
    for text, label in records:  # No list(records), vocabulary scan or second pass.
        if not 0 <= label < model.classes:
            raise ValueError("label out of range")
        model.train()
        logits, _, _ = model(text)
        # Progressive validation predicts BEFORE updating on this record.
        correct += int(logits.argmax().item() == label)
        total += 1
        loss = F.cross_entropy(
            logits[None, :], torch.tensor([label], device=logits.device)
        )
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {
        "records": total,
        "progressive_accuracy": correct / total if total else None,
        "progressive_logloss": loss_sum / total if total else None,
    }


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = OnlineTextClassifier(buckets=128)
    before = model.embedding.weight.detach().clone()

    # Chosen synthetic stream. Each event is consumed once; repeating text is
    # a property of the generator, not a hidden replay of an exhausted iterator.
    def events():
        for _ in range(200):
            yield "sunny warm", 1
            yield "rainy cold", 0

    report = learn_stream(model, events(), lr=0.3)
    assert report["records"] == 400 and report["progressive_accuracy"] > 0.9
    assert model.predict("sunny warm").argmax().item() == 1
    assert model.predict("rainy cold").argmax().item() == 0
    assert not torch.equal(before, model.embedding.weight)
    assert model.predict("").shape == (2,) and torch.isfinite(model.predict("")).all()
    assert model.token_ids("warm warm")[0] == model.token_ids("warm warm")[1]
    print(
        "W146:", report, "; positive probability", model.predict("sunny warm").tolist()
    )
