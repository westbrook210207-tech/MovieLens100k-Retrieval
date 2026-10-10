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
            # Otherwise the same (user, item) pair could be both the positive and a negative.
            raise ValueError("hard_negative_max_rating must be below positive_threshold")
        self.n_items = n_items
        self.num_negatives = num_negatives  # K
        # How many of the K slots are filled with explicit negatives (when the user has that many).
        self.num_hard = round(num_negatives * hard_negative_ratio)
        # A seeded generator of our own: the draws are reproducible and independent of
        # torch's global seed (which drives weight init, dropout and batch order).
        self.rng = np.random.default_rng(seed)

        # One training example per positive interaction: row n is (users[n], positives[n]).
        positives = train[train["rating"] >= positive_threshold]
        # copy=True: pandas hands out read-only views, which torch.from_numpy warns about.
        self.users = positives["user_idx"].to_numpy(copy=True)      # (N,)
        self.positives = positives["item_idx"].to_numpy(copy=True)  # (N,)

        # seen[u, i] is True if user u interacted with item i in training (any rating).
        # 943 x 1682 booleans = 1.6 MB, so a dense matrix is the simplest fast lookup.
        self.seen = np.zeros((n_users, n_items), dtype=bool)
        self.seen[train["user_idx"], train["item_idx"]] = True

        lows = train[train["rating"] <= hard_negative_max_rating]
        # user_idx -> array of that user's low-rated items.
        self.low_items = {u: g.to_numpy() for u, g in lows.groupby("user_idx")["item_idx"]}
        # Row positions of each user's positives, to fill explicit negatives user by user.
        # reset_index makes the positions line up with self.users / self.positives.
        self.rows_by_user = positives.reset_index(drop=True).groupby("user_idx").indices

    def sample(self) -> tuple[Tensor, Tensor, Tensor]:
        """Draw fresh negatives for every positive. Returns user_idx (N,), pos_idx (N,), neg_idx (N, K).

        Called once per epoch, so over training each positive is contrasted with many
        different negatives instead of the same K every time.
        """
        n = len(self.users)
        # Start with K uniform draws per row over the whole catalogue (with replacement, so
        # a list can occasionally repeat an item: ~11% of lists at K=20, harmless).
        negatives = self.rng.integers(0, self.n_items, size=(n, self.num_negatives))
        # Rejection sampling: redraw any negative the user has already seen in training.
        # Users have seen 6% of the catalogue on average, so this converges in a few rounds.
        while True:
            # self.users[:, None] is (N, 1) and broadcasts against (N, K): entry [n, k]
            # looks up seen[user of row n, negatives[n, k]].
            collisions = self.seen[self.users[:, None], negatives]
            if not collisions.any():
                break
            negatives[collisions] = self.rng.integers(0, self.n_items, size=collisions.sum())

        if self.num_hard:
            # Overwrite the first slots of each row with the user's own low-rated items.
            for user, rows in self.rows_by_user.items():
                lows = self.low_items.get(user)
                if lows is None:
                    continue  # no explicit lows: this user keeps all-uniform negatives
                h = min(self.num_hard, len(lows))  # a user with few lows fills fewer slots
                # Shuffle each row independently and keep the first h -> no repeats within a list.
                tiled = np.tile(lows, (len(rows), 1))  # (rows of this user, n_lows)
                negatives[rows, :h] = self.rng.permuted(tiled, axis=1)[:, :h]

        return torch.from_numpy(self.users), torch.from_numpy(self.positives), torch.from_numpy(negatives)
