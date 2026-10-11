"""Guards against silent failures: settings, diverged models, bad raw data, stale index."""

import json
import math

import pandas as pd
import psycopg
import pytest
import torch
from pydantic import SecretStr

from api.main import warn_if_index_is_stale
from config import Settings, get_settings
from training.evaluate import evaluate_scores
from training.index import check_ids_match_artifact
from training.prepare import ZIP_REGIONS, build_features, validate


@pytest.mark.parametrize("password", ["plain", "has space", "quo'te", "back\\slash", "a=b c"])
def test_conninfo_survives_awkward_passwords(password):
    conninfo = Settings(pg_password=SecretStr(password)).pg_conninfo
    assert psycopg.conninfo.conninfo_to_dict(conninfo)["password"] == password


def test_unset_password_is_a_clear_error():
    with pytest.raises(RuntimeError, match="PG_PASSWORD"):
        Settings(pg_password=SecretStr("")).pg_conninfo


def test_password_is_not_in_settings_repr():
    assert "hunter2" not in repr(Settings(pg_password=SecretStr("hunter2")))


def test_diverged_model_is_rejected_not_scored_as_perfect():
    held_out = pd.DataFrame({"user_idx": [0], "item_idx": [1]})
    seen = pd.DataFrame({"user_idx": [0], "item_idx": [0]})
    with pytest.raises(ValueError, match="diverged"):
        evaluate_scores(torch.full((1, 3), math.nan), held_out, seen)


def _raw_frames():
    ratings = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 1], "rating": [5, 3], "timestamp": [1, 2]})
    users = pd.DataFrame({"user_id": [1, 2], "age": [24, 53], "gender": ["M", "F"],
                          "occupation": ["artist", "writer"], "zip_code": ["85711", "V3N4P"]})
    items = pd.DataFrame({"item_id": [1], "title": ["Film"], "release_date": ["01-Jan-1995"], "Drama": [1]})
    return ratings, users, items


def test_validate_rejects_unknown_gender():
    ratings, users, items = _raw_frames()
    validate(ratings, users, items)
    users.loc[0, "gender"] = "X"
    with pytest.raises(ValueError, match="gender"):
        validate(ratings, users, items)


def test_user_features_are_one_hot_and_zip_maps_to_region():
    _, users, items = _raw_frames()
    users.loc[len(users)] = [3, 30, "F", "artist", None]
    features, _, mappings = build_features(users, items, ["Drama"])
    assert features["is_male"].tolist() == [1, 0, 0]
    non_us = len(ZIP_REGIONS) - 1
    assert features["zip_region"].tolist() == [8, non_us, non_us]
    for prefix in ("age_bucket_", "occupation_", "zip_region_"):
        columns = [c for c in features.columns if c.startswith(prefix)]
        assert (features[columns].sum(axis=1) == 1).all(), prefix
    assert mappings["occupations"] == ["artist", "writer"]


def test_index_refuses_raw_data_that_differs_from_the_artifact():
    _, users, items = _raw_frames()
    meta = {"run_id": "run-a", "user_ids": [1, 2], "item_ids": [1]}
    check_ids_match_artifact(users, items, meta)
    with pytest.raises(ValueError, match="user IDs"):
        check_ids_match_artifact(users, items, {**meta, "user_ids": [1, 2, 3]})
    with pytest.raises(ValueError, match="item IDs"):
        check_ids_match_artifact(users, items, {**meta, "item_ids": [2]})


def test_stale_index_warning(tmp_path, monkeypatch, caplog):
    (tmp_path / "meta.json").write_text(json.dumps({"run_id": "newer-run"}))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        with caplog.at_level("WARNING", logger="api"):
            warn_if_index_is_stale("newer-run")
            assert "stale_index" not in caplog.text
            warn_if_index_is_stale("older-run")
            assert "stale_index" in caplog.text
    finally:
        get_settings.cache_clear()
