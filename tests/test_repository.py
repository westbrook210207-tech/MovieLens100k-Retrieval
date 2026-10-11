"""Integration tests for the real data-access code and the indexing writers.

Postgres: the compose database, inside a throwaway schema that is dropped afterwards
(skipped when the database is not reachable). Qdrant: the client's in-memory mode.
"""

import uuid

import pandas as pd
import psycopg
import pytest
import torch
from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from api.main import check_index
from api.repositories import Repository
from config import get_settings
from training.index import load_postgres, switch_aliases, write_collection

GENRES = ["Comedy", "Drama"]


def _conninfo() -> str | None:
    try:
        conninfo = make_conninfo(get_settings().pg_conninfo, connect_timeout=2)
        psycopg.connect(conninfo).close()
        return conninfo
    except Exception:
        return None


CONNINFO = _conninfo()
pytestmark = pytest.mark.skipif(CONNINFO is None, reason="Postgres from compose.yaml is not reachable")


def _frames():
    users = pd.DataFrame({"user_id": [1, 2], "age": [24, 53], "gender": ["M", "F"],
                          "occupation": ["artist", "writer"], "zip_code": ["01002", "V3N4P"]})
    items = pd.DataFrame({"item_id": [1, 2, 3], "title": ["Amélie", "Film 2", "Film 3"],
                          "release_year": pd.array([2001, None, 1995], dtype="Int64"),
                          "Comedy": [1, 0, 1], "Drama": [0, 0, 1]})
    ratings = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [1, 2, 3], "rating": [5, 1, 4],
                            "timestamp": [881250949, 881250950, 891717742]})
    return ratings, users, items


def _index(qdrant: QdrantClient, run_id: str, item_vectors: list[list[float]]) -> None:
    switch_aliases(qdrant, {
        "items": write_collection(qdrant, "items", [1, 2, 3], torch.tensor(item_vectors),
                                  [{"run_id": run_id}] * 3),
        "users": write_collection(qdrant, "users", [1, 2], torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                                  [{"run_id": run_id}] * 2),
    })


@pytest.fixture
def repository():
    schema = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(CONNINFO, autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    scoped = make_conninfo(CONNINFO, options=f"-c search_path={schema}")
    pool = ConnectionPool(scoped, min_size=1, max_size=2, open=True)
    qdrant = QdrantClient(":memory:")
    try:
        with psycopg.connect(scoped) as conn:
            load_postgres(conn, *_frames(), GENRES)
        _index(qdrant, "run-a", [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        yield Repository(pool, qdrant, "items", "users")
    finally:
        pool.close()
        with psycopg.connect(CONNINFO, autocommit=True) as admin:
            admin.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_ping_and_index_check(repository):
    repository.ping()
    assert repository.index_info("items") == (2, "run-a")
    assert repository.index_info("no_such_collection") is None
    assert check_index(repository) == "run-a"


def test_user_vector(repository):
    assert repository.user_vector(1) == pytest.approx([1.0, 0.0])
    assert repository.user_vector(999) is None


def test_max_user_id(repository):
    assert repository.max_user_id() == 2


def test_seen_items_and_metadata(repository):
    assert sorted(repository.seen_item_ids(1)) == [1, 2]
    assert repository.seen_item_ids(999) == []
    metadata = repository.items_metadata([1, 2, 404])
    assert metadata[1] == {"title": "Amélie", "release_year": 2001, "genres": ["Comedy"]}
    assert metadata[2] == {"title": "Film 2", "release_year": None, "genres": []}
    assert 404 not in metadata


def test_search_orders_by_similarity_and_excludes_ids(repository):
    hits = repository.search_items([1.0, 0.0], exclude_ids=[], k=3)
    assert [item_id for item_id, _ in hits] == [1, 2, 3]
    assert hits[0][1] == pytest.approx(1.0)
    assert [item_id for item_id, _ in repository.search_items([1.0, 0.0], exclude_ids=[1], k=2)] == [2, 3]
    assert repository.search_items([1.0, 0.0], exclude_ids=[1, 2, 3], k=5) == []


def test_reindex_switches_both_aliases_and_removes_old_collections(repository):
    _index(repository.qdrant, "run-b", [[0.0, 1.0], [0.9, 0.1], [1.0, 0.0]])
    assert check_index(repository) == "run-b"
    assert [item_id for item_id, _ in repository.search_items([1.0, 0.0], [], k=1)] == [3]
    assert len(repository.qdrant.get_collections().collections) == 2


def test_reload_replaces_rows_and_a_failed_reload_changes_nothing(repository):
    ratings, users, items = _frames()
    with repository.pool.connection() as conn:
        load_postgres(conn, ratings.iloc[:1], users, items, GENRES)
    assert repository.seen_item_ids(1) == [1]

    bad = ratings.assign(rating=[5, 9, 4])
    with repository.pool.connection() as conn, pytest.raises(psycopg.errors.CheckViolation):
        load_postgres(conn, bad, users, items, GENRES)
    assert repository.seen_item_ids(1) == [1]
