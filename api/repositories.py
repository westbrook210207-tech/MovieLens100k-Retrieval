"""Data access: Postgres (metadata, interactions) and Qdrant (precomputed user and item vectors).

Where this file sits:

    routes.py -> service.py -> repositories.py (here) -> Postgres / Qdrant

This is the only file in api/ that contains SQL or Qdrant client calls (the "repository
pattern"). Each method answers one question and returns plain Python values (lists, tuples,
dicts), so the service never handles a database cursor or a Qdrant object.

Qdrant vocabulary used below:
    collection  a named set of points, comparable to a table. There are two: item vectors
                and user vectors.
    point       one entry = id + vector + payload.
    id          here always the raw MovieLens ID (item_id or user_id). The same number is the
                primary key in Postgres, which is how the two stores are matched up.
    vector      the embedding: a list of floats produced by the trained model.
    payload     a small JSON object attached to a point (run_id; items also carry genres and
                release_year).
    alias       a second name that points at a collection and can be re-pointed atomically.

Every method is a plain blocking `def`. That is correct here because the endpoints in
routes.py are plain `def` too, and FastAPI runs those in worker threads (see
"CONCEPT: async / await" in main.py).
"""

from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition, SearchParams


class Repository:
    def __init__(self, pool: ConnectionPool, qdrant: QdrantClient, item_collection: str,
                 user_collection: str) -> None:
        self.pool = pool
        self.qdrant = qdrant
        self.item_collection = item_collection
        self.user_collection = user_collection

    def ping(self) -> None:
        """Raise if either store is unreachable or a vector collection is missing.
        Returns nothing on success. Called by GET /health in routes.py."""
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")
        for collection in (self.item_collection, self.user_collection):
            if not self.qdrant.collection_exists(collection):
                raise RuntimeError(f"Qdrant collection '{collection}' does not exist")

    def index_info(self, collection: str) -> tuple[int, str | None] | None:
        """(vector size, run_id of the model that produced the vectors), or None if the
        collection does not exist. Used once, at startup, by check_index() in main.py."""
        if not self.qdrant.collection_exists(collection):
            return None
        size = self.qdrant.get_collection(collection).config.params.vectors.size
        points, _ = self.qdrant.scroll(collection, limit=1, with_payload=["run_id"])
        return size, points[0].payload.get("run_id") if points else None

    def user_vector(self, user_id: int) -> list[float] | None:
        """The user's precomputed embedding, or None for a user that was never indexed."""
        points = self.qdrant.retrieve(self.user_collection, ids=[user_id], with_vectors=True,
                                      with_payload=False)
        return points[0].vector if points else None

    def max_user_id(self) -> int:
        """The largest raw user ID in Postgres; 0 when the table is empty."""
        with self.pool.connection() as conn:
            return conn.execute("SELECT COALESCE(max(user_id), 0) FROM users").fetchone()[0]

    def seen_item_ids(self, user_id: int) -> list[int]:
        """Raw item IDs of every film this user has rated; [] if there are none."""
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT item_id FROM interactions WHERE user_id = %s", (user_id,)).fetchall()
        return [row[0] for row in rows]

    def search_items(self, vector: list[float], exclude_ids: list[int], k: int) -> list[tuple[int, float]]:
        """The k highest-scoring items that are not in exclude_ids, as (item_id, score), best first."""
        query_filter = Filter(must_not=[HasIdCondition(has_id=exclude_ids)]) if exclude_ids else None
        result = self.qdrant.query_points(self.item_collection, query=vector, query_filter=query_filter,
                                          limit=k, with_payload=False,
                                          search_params=SearchParams(exact=True))
        return [(point.id, point.score) for point in result.points]

    def items_metadata(self, item_ids: list[int]) -> dict[int, dict]:
        """Title, release year and genres for the given item IDs, as {item_id: {...}}.
        IDs that are not in Postgres are absent from the result; [] gives {}."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT item_id, title, release_year, genres FROM items WHERE item_id = ANY(%s)",
                (item_ids,),
            ).fetchall()
        return {r[0]: {"title": r[1], "release_year": r[2], "genres": r[3]} for r in rows}
