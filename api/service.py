"""Recommendation orchestration: look up the user vector, retrieve candidates, attach metadata.

No model runs here. User and item vectors are computed offline by training/index.py.

Where this file sits:

    routes.py (HTTP) -> service.py (the recipe) -> repositories.py (Postgres / Qdrant calls)

The service decides WHAT happens and in which order. It contains no SQL, no Qdrant calls
and no HTTP. Because of that it can be tested with a fake repository and no running
database, which is exactly what tests/test_api.py does with FakeRepository.

All IDs in this file are raw MovieLens IDs.
"""

import logging

from api.schemas import RecommendedItem

logger = logging.getLogger("api")


class UserNotFoundError(Exception):
    pass


class RecommendationService:
    def __init__(self, repository, model_run_id: str | None = None) -> None:
        self.repository = repository
        self.model_run_id = model_run_id
        self.max_user_id: int = repository.max_user_id()

    def recommend(self, user_id: int, k: int) -> list[RecommendedItem]:
        """Top-k recommendations for one user, best first.

        user_id: raw MovieLens user ID. k: how many items (routes.py guarantees 1..100).
        Returns up to k RecommendedItem; fewer if the user has fewer than k unseen items.
        Raises UserNotFoundError when the user has no precomputed vector.
        """
        vector = self.repository.user_vector(user_id)
        if vector is None:
            raise UserNotFoundError(user_id)
        seen = self.repository.seen_item_ids(user_id)
        hits = self.repository.search_items(vector, exclude_ids=seen, k=k)
        metadata = self.repository.items_metadata([item_id for item_id, _ in hits])
        missing = [item_id for item_id, _ in hits if item_id not in metadata]
        if missing:
            logger.warning("items in Qdrant but not in Postgres: %s; re-run `python -m training.index`", missing)
        return [RecommendedItem(item_id=item_id, score=score, **metadata[item_id])
                for item_id, score in hits if item_id in metadata]
