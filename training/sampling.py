"""Negative sampling for listwise training examples (user, pos, {neg_1..neg_K}).

The sampled negatives are weak negatives: uniform draws from items the user has no
*training* interaction with. Unobserved is not the same as disliked, so some of these are
false negatives (on this split about 0.1% of draws hit one of the user's held-out positives).

Items the user rated low are NOT used as negatives (that was tested and did not help
retrieval). They only count as seen, like every other rated item, so they are never drawn.

Validation/test items are deliberately NOT excluded from the weak negatives:
excluding them would use held-out labels to build training examples.

In-batch negatives (other users' positives) are not produced here: they come for free
from the batch itself in training/train.py, which reuses this sampler's `seen` matrix
to mask out the ones that are not really negatives.

All indices are zero-based model indices (`user_idx`, `item_idx`).
"""

# sampling.py — supply the negatives
#
# - Job: the data says which films a user liked, never which they would not like. This file
#   manufactures the "wrong answers" the loss needs. 
#   Note that this file only for the 20 uniform negatives (random unrated movies only), not including in-batch negatives
# - Positives: every training rating ≥ 4 becomes one training row. These are taken as they
#   are, not sampled.
# - Negatives: for each row, 20 random films the user has not rated in training. They are
#   redrawn every epoch.
# - Also keeps: a `seen` table of which user rated which film in training, which `train.py`
#   reuses.

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class NegativeSampler:
    def __init__(self, train: pd.DataFrame, n_users: int, n_items: int, num_negatives: int,
                 positive_threshold: int, seed: int = 0) -> None:
        """`train` needs zero-based `user_idx`, `item_idx` and `rating` columns."""
        self.n_items = n_items # 1682 movies, from item.csv which will later be plugged in by train.py
        self.num_negatives = num_negatives # the number 20 is also defined in train.py
        self.rng = np.random.default_rng(seed)

        positives = train[train["rating"] >= positive_threshold] # train here is the train.csv from prepare.py
        self.users = positives["user_idx"].to_numpy(copy=True) # turn df to numpy arr, copy makes the array an independent copy of the column 
        self.positives = positives["item_idx"].to_numpy(copy=True) # turn df to numpy arr, copy makes the array an independent copy of the column 

        self.seen = np.zeros((n_users, n_items), dtype=bool) # Create a 0 matrix shape (n_users, n_items)
        self.seen[train["user_idx"], train["item_idx"]] = True # Mark 1 for pairs that are in train.csv (aka rated movies of users)


    def sample(self) -> tuple[Tensor, Tensor, Tensor]:
        """Draw fresh negatives for every positive. Returns user_idx (N,), pos_idx (N,), neg_idx (N, K).

        Called once per epoch, so over training each positive is contrasted with many
        different negatives instead of the same K every time.
        """
        n = len(self.users)
        negatives = self.rng.integers(0, self.n_items, size=(n, self.num_negatives))
        while True:
            collisions = self.seen[self.users[:, None], negatives]
            if not collisions.any():
                break
            negatives[collisions] = self.rng.integers(0, self.n_items, size=collisions.sum())

        return torch.from_numpy(self.users), torch.from_numpy(self.positives), torch.from_numpy(negatives)
