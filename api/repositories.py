"""Data access: Postgres (metadata, interactions) and Qdrant (precomputed user and item vectors with payload).

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
        with self.pool.connection() as conn: # .pool.connection() automatically raise if there's a connection failure so we don't have to manually create an if/raise
            conn.execute("SELECT 1")
        for collection in (self.item_collection, self.user_collection):
            if not self.qdrant.collection_exists(collection): # .qdrant.collection_exists() automatically raise if there's a connection failure so we don't have to manually create an if/raise
                raise RuntimeError(f"Qdrant collection '{collection}' does not exist") # So only the missing collections need an if/raise

    def index_info(self, collection: str) -> tuple[int, str | None] | None:
        """(vector size, run_id of the model that produced the vectors), or None if the
        collection does not exist. Used once, at startup, by check_index() in main.py."""
        if not self.qdrant.collection_exists(collection):
            return None
        size = self.qdrant.get_collection(collection).config.params.vectors.size # return the vector size which is 64 by default
        points, _ = self.qdrant.scroll(collection, limit=1, with_payload=["run_id"]) # scroll returns 1 point and the next point of that point which means it returns a pair (current point, offset)
                                                                                    # But we don't need the offset in the returned pair, that's why we have store offset in _ and current point in points
                                                                                    # limit = 1 means 1 point at a time
                                                                                    # with_payload = ["run_id"] because we only need run_id for now, we don't need other info
                                                                                    # points = [Record(id=1, payload={"run_id": "a1b2..."}, vector=None)], _ = 2 (only number 2 bc 2 is the next offset of 1)
        return size, points[0].payload.get("run_id") if points else None # Returns size and Record(id=1, payload={"run_id": "a1b2..."}, vector=None)
                                                                        # If the collection is empty, points = [], and that is why the next line has if points else None
                                                                        # and if points = [] which means run_id = None, app still fails

    def user_vector(self, user_id: int) -> list[float] | None:
        """The user's precomputed embedding, or None for a user that was never indexed."""
        points = self.qdrant.retrieve(self.user_collection, ids=[user_id], with_vectors=True,
                                      with_payload=False) # with_vectors = True, with_payload = False means just give me vectors, don't give me payload
        return points[0].vector if points else None # points of user_vector contains only ID and vector (64 values)

    def max_user_id(self) -> int:
        """The largest raw user ID in Postgres; 0 when the table is empty."""
        with self.pool.connection() as conn:
            return conn.execute("SELECT COALESCE(max(user_id), 0) FROM users").fetchone()[0]

    def seen_item_ids(self, user_id: int) -> list[int]:
        """Raw item IDs of every film this user has rated; [] if there are none."""
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT item_id FROM interactions WHERE user_id = %s", (user_id,)).fetchall() # take all rated movies ID of user A
                                                                                                            # This returns a list of tuple [(43,), (67,), (99,),...] which each tuple is a rated movie ID
        return [row[0] for row in rows] # For each tuple inside that list, take the first element of the tuple. Which is the 43, 67, 99, ... movie IDs

    def search_items(self, vector: list[float], exclude_ids: list[int], k: int) -> list[tuple[int, float]]:
        """The k highest-scoring items that are not in exclude_ids, as (item_id, score), best first."""
        query_filter = Filter(must_not=[HasIdCondition(has_id=exclude_ids)]) if exclude_ids else None # exclude_ids are only rated movies of user A in this project (all rated movies depsite the rating)
        result = self.qdrant.query_points(self.item_collection, # Search in item_collection
                                          query=vector, # the user's 64 floats from user_vector(): "find films closest to THIS"
                                          query_filter=query_filter, # Apply query_filter above
                                          limit=k, # top_k
                                          with_payload=False,
                                          with_vectors=False,
                                          search_params=SearchParams(exact=True))  # compare against EVERY film, not an approximate shortcut
                                                                                    # If exact=False (the default) is HNSW. It walks a graph of neighbours and checks only some films, 
                                                                                    # so it's faster but may miss a few of the true top-k. 
        return [(point.id, point.score) for point in result.points] # Returns item_id, score in response body

    def items_metadata(self, item_ids: list[int]) -> dict[int, dict]:
        """Title, release year and genres for the given item IDs, as {item_id: {...}}.
        IDs that are not in Postgres are absent from the result; [] gives {}."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT item_id, title, release_year, genres FROM items WHERE item_id = ANY(%s)",
                (item_ids,),
            ).fetchall()
        return {r[0]: {"title": r[1], "release_year": r[2], "genres": r[3]} for r in rows} # Returns title, release_year and genres in response body
