"""Offline pipeline tests on tiny synthetic data: no dataset, services or trained model needed."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from training.evaluate import rank_metrics
from training.models.two_tower import Tower, TwoTower, info_nce, load_model, save_model
from training.prepare import split
from training.sampling import NegativeSampler
from training.train import parse_args


def test_split_is_temporal_and_has_no_leakage():
    ratings = pd.DataFrame({
        "user_id":   [1, 1, 1, 1, 1, 1, 1, 2, 2],
        "item_id":   [10, 11, 12, 13, 14, 15, 16, 10, 11],
        "rating":    [5, 1, 4, 4, 2, 5, 1, 5, 4],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 1, 2],
    })
    for frame in (ratings, ratings[::-1]):
        train, val, test = split(frame, positive_threshold=4)
        assert test["item_id"].tolist() == [15]
        assert val["item_id"].tolist() == [13]
        assert train[train["user_id"] == 1]["item_id"].tolist() == [10, 11, 12]
        assert train[train["user_id"] == 1]["timestamp"].max() < val["timestamp"].iloc[0]
        assert train[train["user_id"] == 2]["item_id"].tolist() == [10, 11]


@pytest.fixture
def train_frame() -> pd.DataFrame:
    rows = [(u, i, 5 if i < 5 else (1 if i < 8 else 3)) for u in range(3) for i in range(10)]
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "rating"])


def test_weak_negatives_are_never_seen_items(train_frame):
    sampler = NegativeSampler(train_frame, 3, 30, num_negatives=8, positive_threshold=4)
    users, positives, negatives = sampler.sample()
    assert negatives.shape == (15, 8) and len(users) == len(positives) == 15
    assert (negatives >= 10).all()


def test_sampler_supports_zero_sampled_negatives(train_frame):
    sampler = NegativeSampler(train_frame, 3, 30, num_negatives=0, positive_threshold=4)
    assert sampler.sample()[2].shape == (15, 0)


def test_info_nce_values():
    user = torch.tensor([[1.0, 0.0]])
    same = torch.tensor([[[1.0, 0.0]] * 4])
    assert info_nce(user, same, temperature=0.1).item() == pytest.approx(math.log(4))
    good = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]])
    assert info_nce(user, good, temperature=0.1).item() < 1e-6


def test_info_nce_in_batch_mask_and_correction():
    users = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    items = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    nothing_masked = torch.zeros(2, 2, dtype=torch.bool)
    assert info_nce(users, items, 0.1, nothing_masked).item() == pytest.approx(math.log(3))
    everything_masked = torch.ones(2, 2, dtype=torch.bool)
    assert info_nce(users, items, 0.1, everything_masked).item() == pytest.approx(0.0)
    diagonal = torch.eye(2, dtype=torch.bool)
    correction = torch.tensor([[0.0, math.log(3), math.log(3)]] * 2)
    assert info_nce(users, items, 0.1, diagonal, correction).item() == pytest.approx(math.log(4 / 3))


def test_rank_metrics_on_known_ranks():
    scores = torch.tensor([
        [0.9, 0.1, 0.5, 0.3],
        [0.9, 0.1, 0.5, 0.3],
        [0.5, 0.5, 0.5, 0.5],
    ])
    metrics = rank_metrics(scores, torch.tensor([0, 3, 1]), ks=(2,))
    assert metrics["recall@2"] == pytest.approx(1 / 3)
    assert metrics["precision@2"] == pytest.approx(1 / 6)
    assert metrics["ndcg@2"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("depth", [{"hidden_dims": [8], "dropouts": [0.0]},
                                   {"hidden_dims": [], "dropouts": []},
                                   {"hidden_dims": [8, 6], "dropouts": [0.0, 0.2]}])
def test_save_and_load_round_trip(tmp_path, depth):
    torch.manual_seed(0)
    model_config = {"embedding_dim": 4, **depth}
    model = TwoTower(torch.rand(3, 6), torch.rand(5, 7), **model_config).eval()
    meta = {"model_config": model_config, "user_dense_dim": 6,
            "item_dense_dim": 7, "user_ids": [1, 2, 3],
            "item_ids": [1, 2, 3, 4, 5], "genres": list("abcdefg")}
    save_model(model, meta, tmp_path)
    loaded, loaded_meta = load_model(tmp_path)
    assert loaded_meta == meta
    with torch.no_grad():
        vectors = loaded.encode_items(torch.arange(5))
        assert torch.equal(vectors, model.encode_items(torch.arange(5)))
        mlp = loaded.item_tower.mlp
        assert [layer.out_features for layer in mlp if isinstance(layer, torch.nn.Linear)] == depth["hidden_dims"] + [4]
        assert [layer.p for layer in mlp if isinstance(layer, torch.nn.Dropout)] == depth["dropouts"]
        assert np.allclose(vectors.norm(dim=-1), 1.0)
        assert loaded.encode_items(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3, 4)


def test_hidden_dims_and_dropouts_need_the_same_length():
    with pytest.raises(ValueError):
        Tower(3, 2, 4, hidden_dims=[8, 8], dropouts=[0.1])
    with pytest.raises(SystemExit):
        parse_args(["--hidden-dims", "8", "8", "--dropouts", "0.1"])


def test_missing_artifact_fails_with_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="training.train"):
        load_model(tmp_path)
