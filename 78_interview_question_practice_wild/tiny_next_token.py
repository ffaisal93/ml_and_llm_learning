"""W073: feed-forward next-token prediction with no recurrence or attention.

SyntheticDataset here is a LEARNABLE DEMONSTRATION, because the exercise does
not include its dataset implementation. Reuse the supplied dataset unchanged
when filling an existing interview skeleton.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size, seed=0):
        if num_samples < 1 or seq_len < 2 or vocab_size < 2:
            raise ValueError("need samples > 0, sequence length >= 2 and vocabulary >= 2")
        random = torch.Generator().manual_seed(seed)
        starts = torch.randint(vocab_size, (num_samples, 1), generator=random)
        offsets = torch.arange(seq_len).unsqueeze(0)
        # Demonstration rule: after token k comes (k + 1) modulo vocab_size.
        self.sequences = (starts + offsets) % vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class TinyNextTokenModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)  # (B, L, H)
        hidden_states = torch.relu(self.hidden(embeddings))
        return self.output(hidden_states)       # (B, L, V), raw logits


def next_token_loss(model, sequences):
    if sequences.ndim != 2 or sequences.size(1) < 2:
        raise ValueError("sequences must have shape (batch, length >= 2)")
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    logits = model(inputs)
    vocabulary_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
    )


def train_one_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    total_targets = 0
    for sequences in loader:
        sequences = sequences.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, sequences)
        loss.backward()
        optimizer.step()

        count = sequences.size(0) * (sequences.size(1) - 1)
        total_loss += loss.item() * count
        total_targets += count
    if total_targets == 0:
        raise ValueError("loader contains no target tokens")
    return total_loss / total_targets


def evaluate_loss(model, loader, device="cpu"):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    with torch.no_grad():
        for sequences in loader:
            sequences = sequences.to(device)
            count = sequences.size(0) * (sequences.size(1) - 1)
            total_loss += next_token_loss(model, sequences).item() * count
            total_targets += count
    model.train(was_training)
    if total_targets == 0:
        raise ValueError("loader contains no target tokens")
    return total_loss / total_targets


def run_demo(seed=7, epochs=6):
    torch.manual_seed(seed)
    vocabulary_size = 8
    dataset = SyntheticDataset(num_samples=256, seq_len=9,
                               vocab_size=vocabulary_size, seed=seed)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    evaluation_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = TinyNextTokenModel(vocabulary_size, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    history = [evaluate_loss(model, evaluation_loader)]
    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer)
        history.append(evaluate_loss(model, evaluation_loader))
    return model, history


if __name__ == "__main__":
    model, history = run_demo()
    for epoch, loss in enumerate(history):
        print(f"Evaluation after {epoch} epochs: {loss:.6f}")
    with torch.no_grad():
        predicted_next = model(torch.arange(8).unsqueeze(0)).argmax(dim=-1)
    print("Predicted next token for 0..7:", predicted_next.squeeze(0).tolist())
    assert history[-1] < history[0]
