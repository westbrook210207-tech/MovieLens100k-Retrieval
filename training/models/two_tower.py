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
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Tower(nn.Module):
    """One side of the model (the user tower and the item tower are two instances)."""

    def __init__(self, n_ids: int, dense_dim: int, embedding_dim: int,
                 hidden_dims: Sequence[int], dropouts: Sequence[float]) -> None:
        """
        n_ids:         how many different values the ID column can take:
                       943 for user_id, 1682 for item_id.
        dense_dim:     number of plain 0/1 feature columns fed in without an embedding.
        embedding_dim: size of the ID embedding AND of the tower's output vector (D).
        hidden_dims:   width of each hidden layer, in order. Its length is the depth; an empty
                       list leaves a single linear projection of the inputs.
        dropouts:      dropout rate after each hidden layer, one value per entry of hidden_dims.
        """
        super().__init__()
        self.embedding = nn.Embedding(n_ids, embedding_dim) # ID is embedded into a 64 dims embedding, each user_id and item_id is embedded into 64 dims
        layers: list[nn.Module] = [] # We use list here so later we can make use of for loop, instead of typing nn.Linear -> nn.ReLU -> nn.Dropout for each layer
        width = embedding_dim + dense_dim # the input dim is ID dim + feature dim. Such as 64(ID) + 34(user's features) for user, 64(ID) + 20(item's features) for item 
        # strict: a missing or extra dropout value raises instead of silently dropping a layer.
        for hidden_dim, dropout in zip(hidden_dims, dropouts, strict=True): # each (hidden_dims,dropouts) pair is dims (aka units) and dropout rate for each layer, corresponding to the idx inside the list. For example hidden_dims[0] and dropouts[0] is first layer's dims and dropout rate
            # Since every layer has the same pattern nn.Linear -> nn.ReLU -> nn.Dropout
            layers += [
                nn.Linear(width, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width = hidden_dim
        layers.append(nn.Linear(width, embedding_dim)) # The output layer is linear activation function just like normally, so the output can later be used correctly in loss
        self.mlp = nn.Sequential(*layers) # Unpack the list into mlp layers

    def forward(self, idx: Tensor, dense: Tensor) -> Tensor:
        # Look up the 64 ID numbers, put the 34 (user) or 20 (film) feature numbers next to them,
        # giving 98 or 84 numbers, and pass that through the MLP.
        # hidden is the 64 numbers that come out: the embedding before normalisation.
        hidden = self.mlp(torch.cat([self.embedding(idx), dense], dim=-1))
        return F.normalize(hidden, dim=-1) # Normalize embedding length = 1 so later dot product is cosine similarity


class TwoTower(nn.Module): # The role of this class is to take in users and items, turn them into embeddings using the skeleton above
    """Feature tables are stored as buffers, so encoding only needs zero-based indices
    and the saved state dict carries exactly the features the model was trained with."""

    def __init__(self, user_dense: Tensor, item_dense: Tensor, embedding_dim: int = 64, 
                 hidden_dims: Sequence[int] = (128,), dropouts: Sequence[float] = (0.1,)) -> None:
        super().__init__()

        self.register_buffer("user_dense", user_dense) # This line basically means self.user_dense = user_dense, the 943 x 34 feature table of users
        self.register_buffer("item_dense", item_dense) # This line basically means self.item_dense = item_dense, the 1682 x 20 feature table of films

        # len(user_dense) = 943 users, user_dense.shape[1] = 34 feature columns.
        # Only these two sizes are passed, not the feature values.
        self.user_tower = Tower(len(user_dense), user_dense.shape[1], embedding_dim,
                                hidden_dims, dropouts)
        
        # len(item_dense) = 1682 films, item_dense.shape[1] = 20 feature columns.
        # Only these two sizes are passed, not the feature values.
        self.item_tower = Tower(len(item_dense), item_dense.shape[1], embedding_dim,
                                hidden_dims, dropouts)

    def encode_users(self, user_idx: Tensor) -> Tensor:
        # user_idx is passed twice: once as the ID, for the lookup in the ID table,
        # and once to pick those users' rows from the feature table.
        # Returns one normalised 64-number vector per user.
        return self.user_tower(user_idx, self.user_dense[user_idx]) # So basically, the user_idx is to look for the 64 dims EMBEDDED ID (NOT FINAL 64 DIM EMBEDDING), the self.user_dense[user_idx] is to look for the 34 features of that ID in user_dense
                                                                    # So if wanna find the FINAL embedding of 1 user, we pass their ID into encode_user

    def encode_items(self, item_idx: Tensor) -> Tensor: # Same idea for items
        return self.item_tower(item_idx, self.item_dense[item_idx])

# InfoNCE loss for listwise
def info_nce(user_vectors: Tensor, item_vectors: Tensor, temperature: float,
             false_negative: Tensor | None = None, correction: Tensor | None = None) -> Tensor:
    """Listwise InfoNCE: a softmax classification "which item in this list is the positive?".

        loss = -log( exp(s_pos / t) / sum_j exp(s_j / t) ),   s = cosine(user, item) since magnitude is normalized

    user_vectors: (B, D). item_vectors: (B, 1 + K, D), the positive at position 0 and the
    user's K sampled negatives after it (K may be 0 when in-batch negatives are used).

    false_negative: optional (B, B) bool. Passing it switches on in-batch negatives: the
        positive of every other row in the batch becomes an extra candidate for each user.
        false_negative[b, j] = True removes row j's positive from user b's list, because
        user b has interacted with that item (it is not a negative for them).
    correction: optional, same shape as the final logits, (B, 1 + K) or (B, 1 + K + B).
        Subtracted from the logits; used for the logQ popularity correction (see train.py).
    """
    # item_vectors is the list of 21 movies including 1 pos and 20 negs for 256 rows
    # By that, item_vectors contains 1 pos and 277 negs we need for a row
    # The first index of item_vectors contains the pos of 256 rows, which is also the 256 in-batch sampling
    logits = torch.einsum("bd,bkd->bk", user_vectors, item_vectors) / temperature # torch.einsum("bd,bkd->bk", user_vectors, item_vectors) is the code to 
                                                                                # take dot product of each user embedding and their 21 item embeddings for score
                                                                                # then have temperature to scale the score so that normalized embeddings don't create the loss floor
                                                                                # that's called logit
    if false_negative is not None: # If in-batch is turned on
        in_batch = torch.einsum("bd,jd->bj", user_vectors, item_vectors[:, 0]) / temperature # Compute logit of each user inside current batch with every movies inside in-batch sampling, see how we extract in-batch sampling by pull out the pos of all 256 rows
        logits = torch.cat([logits, in_batch.masked_fill(false_negative, -math.inf)], dim=1) # Glue the 21 logits with the 256 logits in-batch, 
                                                                                            # if the movie is T inside false_negative, its logit is set to -inf bc exp(-inf) = 0
                                                                                            # so T movies (rated movies but in in-batch sampling) have no effect
                                                                                            # So basically after this line, we have the logit of all pair
                                                                                            # If that pair is a false_negative pair (rated movies but in in-batch sampling) inside In-batch, it's set to -inf
    if correction is not None:
        logits = logits - correction
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)

# Save the model after train.py has finished training it
def save_model(model: TwoTower, meta: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "meta.json").write_text(json.dumps(meta))

# evaluate.py loads the trained model to score the validation set (the default) or the test set
# (--split test), and logs those scores to the MLflow run of the training,
# and index.py loads the model to turns all 943 users and all 1,682 films into vectors, and stores those in Qdrant.
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
    # Build an empty model, the skeleton, with the same shape as the trained one.
    # The two feature tables are filled with zeros for now. They only need the right size,
    # for example 943 x 34 for users and 1682 x 20 for films. The layer sizes come from
    # meta["model_config"], which train.py saved next to the model.
    model = TwoTower(
        torch.zeros((n_users, meta["user_dense_dim"])),
        torch.zeros((n_items, meta["item_dense_dim"])),
        **meta["model_config"],
    )
    # Fill the skeleton with the trained numbers from model.pt. This covers the weights and
    # also the two feature tables, so the zeros above are replaced.
    # Every number in the file is stored under the name of the part it belongs to, and it is
    # put into the part of the skeleton with the same name. So this only works if the skeleton
    # has the same names and sizes as the model that was saved.
    # That is why the model has to be retrained after a change to the model itself, such as a
    # layer size or a removed part: the old file no longer fits. A change to sampling, the loss
    # or the training loop does not break loading.
    #
    # map_location="cpu" means: put the numbers on the CPU, wherever they were saved from.
    # Training runs on MPS (see --device in train.py), but train.py calls model.cpu() just
    # before save_model, so model.pt is written from the CPU. Loading on the CPU means the file
    # works on any machine. evaluate.py and index.py only run the model once over all users and
    # films, and the CPU handles that quickly.
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval(), meta
