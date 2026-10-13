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
        n = self.positives.shape[0] # Number of positive ratings
        negatives = self.rng.integers(0, self.n_items, size=(n, self.num_negatives)) # create a matrix shape (n,20), note that this line only takes random 20 movies per row, it doesn't know which movie is a valid negative for which user
        while True:
            collisions = self.seen[self.users[:, None], negatives]  # Mark the bad random sampling, which are movies that user already rated
                                                                    # Explain the logic of this marking line
                                                                    # self.users[:, None] turns self.users from [0,0,0,...,1,1,...] which is a 1D arr (n,) into a 2D (n,1)
                                                                    # self.seen() check if that pair is in train.csv or not, mark True if yes False if no
                                                                    # Since self.seen() only check if pair is in train.csv or not, 
                                                                    # it doesn't cover pairs that exist but not un training (user X might still rated movie Y 
                                                                    # but because that pair is not in train.csv, 
                                                                    # it still can be randomly picked as a negative sample among 20)
            if not collisions.any(): # If nothing is marked, it's clean and good to break out of this checking loop
                break
            negatives[collisions] = self.rng.integers(0, self.n_items, size=collisions.sum())   # If there's any movie that user already rated, replace it with a new random movie
                                                                                                # Recheck to see if that new random movie is rated by that user or not?
                                                                                                # If still then loop until all movies among 20 movies is unrated by that user
                                                                                                # Explain the logic of this replacing line
                                                                                                # collisions.sum() counts the True cells. True counts as 1, so the collisions.sum() is basically how many rated movies in the bad sampling
                                                                                                # Randomly take collisions.sum() number of movies again, so if the previous sampling has 2 rated movies, randomly take 2 new movies to replace them

        return torch.from_numpy(self.users), torch.from_numpy(self.positives), torch.from_numpy(negatives) # Turn them into tensors
