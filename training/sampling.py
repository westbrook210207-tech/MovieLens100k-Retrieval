"""Negative sampling for listwise training examples (user, pos, {neg_1..neg_K}).

Two kinds of sampled negatives:
- weak: uniform draws from items the user has no *training* interaction with.
  Unobserved is not the same as disliked, so some of these are false negatives
  (on this split about 0.1% of draws hit one of the user's held-out positives).
- explicit: items the user rated <= hard_negative_max_rating in the training period.
  These are "hard" in the sense that the user chose to watch them, so they look like
  the user's taste; whether they help retrieval is an empirical question (see README).

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
                 hard_negative_ratio: float, positive_threshold: int,
                 hard_negative_max_rating: int = 2, seed: int = 0) -> None:
        """`train` needs zero-based `user_idx`, `item_idx` and `rating` columns."""
        if not 0 <= hard_negative_ratio <= 1:
            raise ValueError("hard_negative_ratio must be in [0, 1]")
        if hard_negative_max_rating >= positive_threshold:
            raise ValueError("hard_negative_max_rating must be below positive_threshold")
        self.n_items = n_items
        self.num_negatives = num_negatives
        self.num_hard = round(num_negatives * hard_negative_ratio)
        self.rng = np.random.default_rng(seed)

        positives = train[train["rating"] >= positive_threshold]
        self.users = positives["user_idx"].to_numpy(copy=True)
        self.positives = positives["item_idx"].to_numpy(copy=True)

        self.seen = np.zeros((n_users, n_items), dtype=bool)
        self.seen[train["user_idx"], train["item_idx"]] = True

        lows = train[train["rating"] <= hard_negative_max_rating]
        self.low_items = {u: g.to_numpy() for u, g in lows.groupby("user_idx")["item_idx"]}
        self.rows_by_user = positives.reset_index(drop=True).groupby("user_idx").indices

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

        if self.num_hard:
            for user, rows in self.rows_by_user.items():
                lows = self.low_items.get(user)
                if lows is None:
                    continue
                h = min(self.num_hard, len(lows))
                tiled = np.tile(lows, (len(rows), 1))
                negatives[rows, :h] = self.rng.permuted(tiled, axis=1)[:, :h]

        return torch.from_numpy(self.users), torch.from_numpy(self.positives), torch.from_numpy(negatives)
