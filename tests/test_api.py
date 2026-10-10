"""API tests with a fake repository: no Postgres, Qdrant or trained model needed."""

import sys

import pytest
from fastapi.testclient import TestClient

from api.main import app, check_index
from api.routes import get_service
from api.service import RecommendationService

ITEMS = {i: {"title": f"Film {i}", "release_year": 1990 + i, "genres": ["Drama"]} for i in range(1, 6)}


class FakeRepository:
    item_collection, user_collection = "items", "users"

    def __init__(self) -> None:
        self.healthy = True
        self.user_vectors = {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.6, 0.8]}  # indexed users
        self.seen = {1: [1, 2]}  # user 1 has already rated items 1 and 2
        self.index = {"items": (2, "run-a"), "users": (2, "run-a")}

    def max_user_id(self) -> int:
        return 5  # users 4 and 5 are in range but have no vector

    def ping(self) -> None:
        if not self.healthy:
            raise RuntimeError("down")

    def index_info(self, collection: str):
        return self.index[collection]

    def user_vector(self, user_id: int) -> list[float] | None:
        return self.user_vectors.get(user_id)

    def seen_item_ids(self, user_id: int) -> list[int]:
        return self.seen.get(user_id, [])

    def search_items(self, vector: list[float], exclude_ids: list[int], k: int) -> list[tuple[int, float]]:
        candidates = [i for i in ITEMS if i not in exclude_ids]
        return [(item_id, 1.0 - 0.1 * rank) for rank, item_id in enumerate(candidates[:k])]

    def items_metadata(self, item_ids: list[int]) -> dict[int, dict]:
        return {i: ITEMS[i] for i in item_ids}


@pytest.fixture
def repository():
    return FakeRepository()


@pytest.fixture
def client(repository):
    app.dependency_overrides[get_service] = lambda: RecommendationService(repository, model_run_id="run-a")
    yield TestClient(app)  # not used as a context manager, so the real lifespan does not run
    app.dependency_overrides.clear()


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_run_id": "run-a"}


def test_health_reports_unavailable_dependency(client, repository):
    repository.healthy = False
    assert client.get("/health").status_code == 503


def test_recommendations_shape_and_seen_items_excluded(client):
    response = client.get("/users/1/recommendations", params={"k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert [item["item_id"] for item in body["items"]] == [3, 4]  # 1 and 2 were already rated
    assert set(body["items"][0]) == {"item_id", "title", "release_year", "genres", "score"}


def test_no_duplicate_recommendations(client):
    ids = [item["item_id"] for item in client.get("/users/2/recommendations").json()["items"]]
    assert len(ids) == len(set(ids)) == 5


def test_user_with_no_eligible_candidates_gets_empty_list(client, repository):
    repository.seen[3] = list(ITEMS)
    response = client.get("/users/3/recommendations")
    assert response.status_code == 200
    assert response.json()["items"] == []


def test_unknown_user_is_404(client):
    assert client.get("/users/4/recommendations").status_code == 404


@pytest.mark.parametrize("url", ["/users/1/recommendations?k=0", "/users/1/recommendations?k=101",
                                 "/users/0/recommendations", "/users/abc/recommendations",
                                 "/users/6/recommendations",  # one above the largest user ID
                                 f"/users/{2**31}/recommendations", f"/users/{10**30}/recommendations"])
def test_invalid_input_is_422(client, url):
    assert client.get(url).status_code == 422


def test_backend_failure_is_500_and_still_logged(repository, caplog):
    def broken(user_id):
        raise RuntimeError("postgres down")
    repository.seen_item_ids = broken
    app.dependency_overrides[get_service] = lambda: RecommendationService(repository)
    try:
        with caplog.at_level("INFO", logger="api"):
            response = TestClient(app, raise_server_exceptions=False).get("/users/1/recommendations")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert any('"status": 500' in record.getMessage() for record in caplog.records)


def test_missing_metadata_is_logged(client, repository, caplog):
    repository.items_metadata = lambda item_ids: {i: ITEMS[i] for i in item_ids if i != 3}
    with caplog.at_level("WARNING", logger="api"):
        items = client.get("/users/1/recommendations", params={"k": 2}).json()["items"]
    assert [item["item_id"] for item in items] == [4]  # item 3 has no Postgres row
    assert "not in Postgres" in caplog.text


def test_check_index_accepts_consistent_index_and_returns_run_id(repository):
    assert check_index(repository) == "run-a"


@pytest.mark.parametrize("users_info", [
    None,            # user vectors never indexed
    (2, "run-b"),    # user and item vectors from different model runs
    (4, "run-a"),    # different vector size
])
def test_startup_check_rejects_missing_or_inconsistent_index(repository, users_info):
    repository.index["users"] = users_info
    with pytest.raises(RuntimeError, match="training.index"):
        check_index(repository)


def test_api_does_not_import_torch_or_training():
    # Run in a clean interpreter: other test modules may import torch themselves.
    import subprocess
    code = "import sys, api.main; sys.exit(any(m in sys.modules for m in ('torch', 'training')))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
