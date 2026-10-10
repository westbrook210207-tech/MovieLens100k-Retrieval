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

# The Pydantic model for one recommended film (schemas.py). The service returns these so
# that routes.py can put them straight into the response.
from api.schemas import RecommendedItem

# The same "api" logger object that main.py configured (same name -> same logger).
logger = logging.getLogger("api")


# CONCEPT: domain exception
#   A custom exception type is made by subclassing Exception. `pass` means "empty body": the
#   class adds no behaviour, its NAME is the information.
#   The service raises it to say "I do not know this user" in business terms. routes.py
#   catches exactly this type and translates it into HTTP 404. Keeping the status code out of
#   this file means the same service would work unchanged behind a CLI or a batch job.
class UserNotFoundError(Exception):
    pass


class RecommendationService:
    # CONCEPT: dependency injection
    #   The service does not create its repository; it receives one. Whoever builds the
    #   service chooses which: main.py passes the real Repository, the tests pass a
    #   FakeRepository. `repository` has no type hint, so any object that has the four methods
    #   used in recommend() is accepted ("duck typing").
    #   `model_run_id: str | None = None` reads: a str or None, and None if the caller omits it.
    #   `-> None`: __init__ only stores attributes, it returns nothing.
    def __init__(self, repository, model_run_id: str | None = None) -> None:
        # Keep the repository on the instance so every method can reach it through self.
        self.repository = repository
        self.model_run_id = model_run_id  # MLflow run of the model whose vectors are being served
        # Upper bound for user_id, enforced in routes.py. Read once, at startup.
        # shortcut: not refreshed while the API runs; restart the API after a re-index that adds users.
        self.max_user_id: int = repository.max_user_id()

    def recommend(self, user_id: int, k: int) -> list[RecommendedItem]:
        """Top-k recommendations for one user, best first.

        user_id: raw MovieLens user ID. k: how many items (routes.py guarantees 1..100).
        Returns up to k RecommendedItem; fewer if the user has fewer than k unseen items.
        Raises UserNotFoundError when the user has no precomputed vector.
        """
        # Step 1 - who is this user, as a vector? A Qdrant lookup by ID that returns a
        # list[float], or None when there is no such user.
        vector = self.repository.user_vector(user_id)
        # None means this user was not part of the data when `training.index` last ran. With
        # no vector there is nothing to compare items against, so stop with a clear error
        # (routes.py turns it into 404) rather than return an empty or arbitrary list.
        # `is None` is the correct test for None; passing user_id stores it on the exception,
        # so it shows up in tracebacks.
        if vector is None:
            raise UserNotFoundError(user_id)
        # Step 2 - what has the user already rated? A list of item IDs from Postgres. It
        # covers every rating, including the ones held out of training for validation/test,
        # because a film the user has rated must never be recommended back to them.
        seen = self.repository.seen_item_ids(user_id)
        # Step 3 - candidate retrieval: the k unseen items whose vectors are most similar to
        # the user's. Returns a list of (item_id, score) tuples, highest score first.
        # That ORDER is the ranking the client receives, so nothing below may reorder it.
        hits = self.repository.search_items(vector, exclude_ids=seen, k=k)
        # Step 4 - Qdrant only returned IDs and scores. Fetch title, year and genres for all of
        # them from Postgres in ONE query.
        # [item_id for item_id, _ in hits] is a list comprehension. Each hit is a 2-tuple that
        # is unpacked into `item_id` and `_`; `_` is the conventional name for "a value I do
        # not use" (here the score).
        # Result: {item_id: {"title": ..., "release_year": ..., "genres": [...]}}
        metadata = self.repository.items_metadata([item_id for item_id, _ in hits])
        # Sanity check: every ID Qdrant returned should also exist in Postgres. `x not in d`
        # on a dict tests the KEYS. This collects the IDs that are in Qdrant only.
        missing = [item_id for item_id, _ in hits if item_id not in metadata]
        # An empty list is falsy, so this block runs only when something is missing.
        if missing:
            # Both stores are loaded by the same command (training.index), so a mismatch means
            # that run was interrupted or one store was changed by hand. We log it and carry
            # on: the client gets a slightly shorter list (fewer than k) instead of a 500.
            # Logging style: pass `missing` as an argument for %s rather than building an
            # f-string. logging fills it in itself, and only if the line is actually emitted.
            logger.warning("items in Qdrant but not in Postgres: %s; re-run `python -m training.index`", missing)
        # Build the response objects. The loop runs over `hits`, NOT over `metadata`, which
        # is what keeps the score order from step 3.
        #   if item_id in metadata  skips the missing ones reported above. Without it,
        #                           metadata[item_id] would raise KeyError and fail the
        #                           whole request.
        #   **metadata[item_id]     unpacks that dict into keyword arguments, so the call is
        #                           RecommendedItem(item_id=..., score=..., title=...,
        #                           release_year=..., genres=...).
        # RecommendedItem is a Pydantic model, so the field types are validated right here.
        return [RecommendedItem(item_id=item_id, score=score, **metadata[item_id])
                for item_id, score in hits if item_id in metadata]
