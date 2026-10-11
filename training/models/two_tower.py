"""Hybrid two-tower retrieval model.

"Two-tower": users and items are encoded by two separate networks into the same
D-dimensional space. A user never needs to be run together with an item, so every item
vector can be precomputed and stored in a vector index (Qdrant), and retrieval is a
nearest-neighbour search.

"Hybrid": each tower sees both
- an ID embedding  -> collaborative signal ("users like you liked this"), and
- side features    -> content signal (age/occupation/region, genres/release year),
  which is all a rarely-rated item or a light user has to go on.

Each tower concatenates those inputs, passes them through a small MLP and L2-normalises
the result, so the dot product of a user and an item vector is a cosine similarity.

All indices here are zero-based model indices (`user_idx`, `item_idx`), never raw MovieLens IDs.
"""

# models/two_tower.py — the model and the loss
#
# - Job: define what the model is. Nothing is trained in this file.
# - `Tower`: one side of the model. It looks up an ID embedding, joins it with the feature
#   columns, passes it through the MLP, and scales the result to length 1.
# - `TwoTower`: two towers, one for users and one for films, both producing 64-number
#   vectors in the same space. The score of a user and a film is the cosine of their vectors.
# - `info_nce`: the loss. Given a user, one liked film and a list of negatives, it rewards
#   the model for scoring the liked film highest.
# - `save_model` / `load_model`: write and read `model.pt` and `meta.json`.

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Tower(nn.Module):
    """One side of the model (the user tower and the item tower are two instances)."""

    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int,
                 feature_dim: int, hidden_dim: int, dropout: float,
                 num_hidden_layers: int = 1) -> None:
        """
        cardinalities: vocabulary size of each categorical column; column 0 is the ID.
        dense_dim:     number of plain 0/1 feature columns fed in without an embedding.
        embedding_dim: size of the ID embedding AND of the tower's output vector (D).
        feature_dim:   size of the embedding of every other categorical column.
        hidden_dim:    width of every hidden layer.
        num_hidden_layers: depth; 0 leaves a single linear projection of the inputs.
        """
        super().__init__()
        sizes = [embedding_dim] + [feature_dim] * (len(cardinalities) - 1)
        self.embeddings = nn.ModuleList(nn.Embedding(n, d) for n, d in zip(cardinalities, sizes))
        layers: list[nn.Module] = []
        width = sum(sizes) + dense_dim
        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(width, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = hidden_dim
        layers.append(nn.Linear(width, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, categorical: Tensor, dense: Tensor | None = None) -> Tensor:
        parts = [emb(categorical[..., i]) for i, emb in enumerate(self.embeddings)]
        if dense is not None:
            parts.append(dense)
        hidden = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(hidden, dim=-1)


class TwoTower(nn.Module):
    """Feature tables are stored as buffers, so encoding only needs zero-based indices
    and the saved state dict carries exactly the features the model was trained with."""

    def __init__(self, user_categorical: Tensor, user_dense: Tensor, item_categorical: Tensor,
                 item_dense: Tensor,
                 user_cardinalities: list[int], item_cardinalities: list[int],
                 embedding_dim: int = 64, feature_dim: int = 8, hidden_dim: int = 128,
                 dropout: float = 0.1, num_hidden_layers: int = 1) -> None:
        super().__init__()
        self.register_buffer("user_categorical", user_categorical)
        self.register_buffer("user_dense", user_dense)
        self.register_buffer("item_categorical", item_categorical)
        self.register_buffer("item_dense", item_dense)  # (n_items, n_genres + 1): genre flags + scaled release year
        self.user_tower = Tower(user_cardinalities, user_dense.shape[1], embedding_dim, feature_dim,
                                hidden_dim, dropout, num_hidden_layers)
        self.item_tower = Tower(item_cardinalities, item_dense.shape[1], embedding_dim, feature_dim,
                                hidden_dim, dropout, num_hidden_layers)

    def encode_users(self, user_idx: Tensor) -> Tensor:
        return self.user_tower(self.user_categorical[user_idx], self.user_dense[user_idx])

    def encode_items(self, item_idx: Tensor) -> Tensor:
        return self.item_tower(self.item_categorical[item_idx], self.item_dense[item_idx])


def info_nce(user_vectors: Tensor, item_vectors: Tensor, temperature: float,
             false_negative: Tensor | None = None, correction: Tensor | None = None) -> Tensor:
    """Listwise InfoNCE: a softmax classification "which item in this list is the positive?".

        loss = -log( exp(s_pos / t) / sum_j exp(s_j / t) ),   s = cosine(user, item)

    user_vectors: (B, D). item_vectors: (B, 1 + K, D), the positive at position 0 and the
    user's K sampled negatives after it (K may be 0 when in-batch negatives are used).

    false_negative: optional (B, B) bool. Passing it switches on in-batch negatives: the
        positive of every other row in the batch becomes an extra candidate for each user.
        false_negative[b, j] = True removes row j's positive from user b's list, because
        user b has interacted with that item (it is not a negative for them).
    correction: optional, same shape as the final logits, (B, 1 + K) or (B, 1 + K + B).
        Subtracted from the logits; used for the logQ popularity correction (see train.py).
    """
    logits = torch.einsum("bd,bkd->bk", user_vectors, item_vectors) / temperature
    if false_negative is not None:
        in_batch = user_vectors @ item_vectors[:, 0].T / temperature
        logits = torch.cat([logits, in_batch.masked_fill(false_negative, -math.inf)], dim=1)
    if correction is not None:
        logits = logits - correction
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def save_model(model: TwoTower, meta: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "meta.json").write_text(json.dumps(meta))


def load_model(artifact_dir: Path) -> tuple[TwoTower, dict]:
    """Rebuild the model from model.pt + meta.json, in eval mode on CPU."""
    weights_path, meta_path = artifact_dir / "model.pt", artifact_dir / "meta.json"
    if not (weights_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"Model artifact not found in {artifact_dir} (need model.pt and meta.json). "
            "Run `python -m training.train` first."
        )
    meta = json.loads(meta_path.read_text())
    n_users, n_items = len(meta["user_ids"]), len(meta["item_ids"])
    model = TwoTower(
        torch.zeros((n_users, len(meta["user_cardinalities"])), dtype=torch.long),
        torch.zeros((n_users, meta["user_dense_dim"])),
        torch.zeros((n_items, len(meta["item_cardinalities"])), dtype=torch.long),
        torch.zeros((n_items, meta["item_dense_dim"])),
        meta["user_cardinalities"], meta["item_cardinalities"], **meta["model_config"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval(), meta
