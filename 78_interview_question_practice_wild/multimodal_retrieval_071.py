"""W071: frozen backbone features, trainable image/text projections."""

import torch
from torch import nn
from torch.nn import functional as F


def load_backbones(device="cpu"):
    # Optional: downloads pretrained weights on first use; not needed by demo.
    from torchvision.models import resnet50, ResNet50_Weights
    from sentence_transformers import SentenceTransformer

    weights = ResNet50_Weights.IMAGENET1K_V2
    image_model = resnet50(weights=weights)
    image_model.fc = nn.Identity()
    image_model = image_model.to(device).eval().requires_grad_(False)
    text_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    text_model.eval().requires_grad_(False)
    return image_model, text_model, weights.transforms()


@torch.no_grad()
def extract_features(image_model, text_model, image_batch, product_texts, queries):
    # image_batch must already use the returned weights.transforms().
    image_model.eval()
    text_model.eval()
    image = image_model(image_batch)
    title = text_model.encode(product_texts, convert_to_tensor=True)
    query = text_model.encode(queries, convert_to_tensor=True)
    return image.detach(), title.detach(), query.detach()


class JointEmbedding(nn.Module):
    def __init__(self, image_dim=2048, text_dim=384, dimension=128):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, dimension)
        self.text_projection = nn.Linear(text_dim, dimension)

    def queries(self, features):
        return F.normalize(self.text_projection(features), dim=-1)

    def products(self, image, text):
        image = F.normalize(self.image_projection(image), dim=-1)
        text = F.normalize(self.text_projection(text), dim=-1)
        return F.normalize((image + text) / 2, dim=-1)


def contrastive_loss(query, product, product_ids, temperature=0.1):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = query @ product.T / temperature
    # Same product may appear twice: both columns are positives, not negatives.
    positive = product_ids[:, None].eq(product_ids[None, :])
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    return (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def train_epoch(model, batches, optimizer):
    model.train()
    total = 0.0
    count = 0
    for query, image, text, product_ids in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = contrastive_loss(
            model.queries(query), model.products(image, text), product_ids
        )
        loss.backward()
        optimizer.step()
        total += loss.item() * len(query)
        count += len(query)
    return total / count


@torch.no_grad()
def evaluate(model, query_features, images, texts, catalog_ids, positive_sets, k=5):
    if len(set(catalog_ids)) != len(catalog_ids) or not catalog_ids or k < 1:
        raise ValueError("Need unique catalog IDs and positive k")
    if len(positive_sets) != len(query_features) or any(not p for p in positive_sets):
        raise ValueError("Each query needs at least one positive ID")
    was_training = model.training
    try:
        model.eval()
        scores = model.queries(query_features) @ model.products(images, texts).T
        indices = scores.topk(min(k, len(catalog_ids)), dim=1).indices.cpu().tolist()
        top_ids = [[catalog_ids[j] for j in row] for row in indices]
        recall = sum(
            len(set(row) & set(pos)) / len(set(pos))
            for row, pos in zip(top_ids, positive_sets)
        ) / len(top_ids)
        return top_ids, recall
    finally:
        model.train(was_training)


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # Random, frozen feature fixtures simulate encoder outputs, not actual images.
    n = 8
    query = torch.randn(n, 384)
    image = torch.randn(n, 2048)
    text = query.clone()
    ids = torch.arange(n)
    model = JointEmbedding(dimension=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch = [(query, image, text, ids)]
    first = train_epoch(model, batch, optimizer)
    for _ in range(79):
        last = train_epoch(model, batch, optimizer)
    top, recall = evaluate(
        model, query, image, text, list(range(n)), [{i} for i in range(n)], k=1
    )
    assert last < first and recall == 1.0
    assert (
        contrastive_loss(torch.eye(2), torch.eye(2), torch.tensor([1, 1])).item() == 0
    )
    print("Fixture loss", first, "->", last, "Recall@1", recall)
