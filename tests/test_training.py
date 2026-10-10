"""Offline pipeline tests on tiny synthetic data: no dataset, services or trained model needed."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from training.evaluate import rank_metrics
from training.models.two_tower import TwoTower, info_nce, load_model, save_model
from training.prepare import split
from training.sampling import NegativeSampler


def test_split_is_temporal_and_has_no_leakage():
    # User 1: positives at t=1,3,4,6 and non-positives at t=2,5,7. User 2: only 2 positives.
    ratings = pd.DataFrame({
        "user_id":   [1, 1, 1, 1, 1, 1, 1, 2, 2],
        "item_id":   [10, 11, 12, 13, 14, 15, 16, 10, 11],
        "rating":    [5, 1, 4, 4, 2, 5, 1, 5, 4],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 1, 2],
    })
    # Row order must not matter (u.data is shuffled): same result on the reversed frame.
    for frame in (ratings, ratings[::-1]):
        train, val, test = split(frame, positive_threshold=4)
        assert test["item_id"].tolist() == [15]   # user 1's latest positive
        assert val["item_id"].tolist() == [13]    # second-latest positive
        # Only what came before the validation positive; the lows at t=5 and t=7 are dropped.
        assert train[train["user_id"] == 1]["item_id"].tolist() == [10, 11, 12]
        assert train[train["user_id"] == 1]["timestamp"].max() < val["timestamp"].iloc[0]
        # A user with fewer than 3 positives stays entirely in training.
        assert train[train["user_id"] == 2]["item_id"].tolist() == [10, 11]


@pytest.fixture
def train_frame() -> pd.DataFrame:
    # 3 users x 30 items. Each user rates items 0-9: the first five high, the next five low.
    rows = [(u, i, 5 if i < 5 else (1 if i < 8 else 3)) for u in range(3) for i in range(10)]
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating"])


def test_weak_negatives_are_never_seen_items(train_frame):
    sampler = NegativeSampler(train_frame, 3, 30, num_negatives=8, hard_negative_ratio=0,
                              positive_threshold=4)
    users, positives, negatives = sampler.sample()
    assert negatives.shape == (15, 8) and len(users) == len(positives) == 15  # 3 users x 5 positives
    assert (negatives >= 10).all()  # items 0-9 are seen by every user


@pytest.mark.parametrize("max_rating, allowed", [(2, {5, 6, 7}), (3, {5, 6, 7, 8, 9})])
def test_explicit_negatives_are_the_users_low_rated_items(train_frame, max_rating, allowed):
    sampler = NegativeSampler(train_frame, 3, 30, num_negatives=4, hard_negative_ratio=0.5,
                              positive_threshold=4, hard_negative_max_rating=max_rating)
    _, _, negatives = sampler.sample()
    hard, weak = negatives[:, :2], negatives[:, 2:]
    assert set(hard.flatten().tolist()) <= allowed
    assert (hard[:, 0] != hard[:, 1]).all()  # no repeats within a list
    assert (weak >= 10).all()


def test_sampler_supports_zero_sampled_negatives_and_rejects_bad_settings(train_frame):
    sampler = NegativeSampler(train_frame, 3, 30, num_negatives=0, hard_negative_ratio=0,
                              positive_threshold=4)
    assert sampler.sample()[2].shape == (15, 0)  # in-batch-only training
    with pytest.raises(ValueError):
        NegativeSampler(train_frame, 3, 30, 4, 0.5, positive_threshold=4, hard_negative_max_rating=4)


def test_info_nce_values():
    user = torch.tensor([[1.0, 0.0]])
    # All four candidates identical -> the model cannot tell them apart -> loss = log(4).
    same = torch.tensor([[[1.0, 0.0]] * 4])
    assert info_nce(user, same, temperature=0.1).item() == pytest.approx(math.log(4))
    # Positive aligned with the user, negatives opposite -> loss close to 0.
    good = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])
    assert info_nce(user, good, temperature=0.1).item() < 1e-6


def test_info_nce_in_batch_mask_and_correction():
    # Two users with identical vectors and identical positives, no sampled negatives.
    users = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    items = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])  # (B=2, 1 + K=1, D=2)
    # Unmasked: each user sees [own positive, positive 0, positive 1], all equal -> log(3).
    nothing_masked = torch.zeros(2, 2, dtype=torch.bool)
    assert info_nce(users, items, 0.1, nothing_masked).item() == pytest.approx(math.log(3))
    # Fully masked (every in-batch candidate is a false negative): only the positive is left.
    everything_masked = torch.ones(2, 2, dtype=torch.bool)
    assert info_nce(users, items, 0.1, everything_masked).item() == pytest.approx(0.0)
    # Correction: mask only the diagonal, and discount the remaining in-batch negative by
    # log(3), as if it were sampled 3x as often -> loss = log(1 + 1/3).
    diagonal = torch.eye(2, dtype=torch.bool)
    correction = torch.tensor([[0.0, math.log(3), math.log(3)]] * 2)
    assert info_nce(users, items, 0.1, diagonal, correction).item() == pytest.approx(math.log(4 / 3))


def test_rank_metrics_on_known_ranks():
    scores = torch.tensor([
        [0.9, 0.1, 0.5, 0.3],   # target 0 is ranked 1st
        [0.9, 0.1, 0.5, 0.3],   # target 3 is ranked 3rd
        [0.5, 0.5, 0.5, 0.5],   # all tied: pessimistic rank = last
    ])
    metrics = rank_metrics(scores, torch.tensor([0, 3, 1]), ks=(2,))
    assert metrics["recall@2"] == pytest.approx(1 / 3)
    assert metrics["precision@2"] == pytest.approx(1 / 6)
    assert metrics["ndcg@2"] == pytest.approx(1 / 3)  # only the rank-1 hit counts: 1 / log2(2)


# {} is an artifact saved before num_hidden_layers existed: it must still load (default 1).
@pytest.mark.parametrize("depth", [{}, {"num_hidden_layers": 0}, {"num_hidden_layers": 2}])
def test_save_and_load_round_trip(tmp_path, depth):
    torch.manual_seed(0)
    model_config = {"embedding_dim": 4, "feature_dim": 2, "hidden_dim": 8, "dropout": 0.0, **depth}
    user_categorical = torch.arange(3)[:, None]
    item_categorical = torch.stack([torch.arange(5), torch.tensor([0, 1, 2, 1, 0])], dim=1)
    model = TwoTower(user_categorical, torch.rand(3, 6), item_categorical, torch.rand(5, 7),
                     user_cardinalities=[3], item_cardinalities=[5, 3], **model_config).eval()
    meta = {"model_config": model_config, "user_cardinalities": [3], "user_dense_dim": 6,
            "item_cardinalities": [5, 3], "user_ids": [1, 2, 3], "item_ids": [1, 2, 3, 4, 5],
            "genres": list("abcdefg")}
    save_model(model, meta, tmp_path)
    loaded, loaded_meta = load_model(tmp_path)
    assert loaded_meta == meta
    with torch.no_grad():
        vectors = loaded.encode_items(torch.arange(5))
        assert torch.equal(vectors, model.encode_items(torch.arange(5)))
        n_linear = sum(isinstance(layer, torch.nn.Linear) for layer in loaded.item_tower.mlp)
        assert n_linear == depth.get("num_hidden_layers", 1) + 1  # hidden layers + output layer
        assert np.allclose(vectors.norm(dim=-1), 1.0)  # unit length -> dot product is a cosine
        # Item lists keep their leading shape: (B, 1 + K) -> (B, 1 + K, D).
        assert loaded.encode_items(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3, 4)


def test_missing_artifact_fails_with_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.train"):
        load_model(tmp_path)
