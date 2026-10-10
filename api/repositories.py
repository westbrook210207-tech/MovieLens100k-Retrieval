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
# Building blocks for a Qdrant query: Filter (which points are allowed), HasIdCondition
# ("the point's ID is in this list"), SearchParams (how the search is executed).
from qdrant_client.models import Filter, HasIdCondition, SearchParams


class Repository:
    def __init__(self, pool: ConnectionPool, qdrant: QdrantClient, item_collection: str,
                 user_collection: str) -> None:
        # Nothing is opened here. The pool and the client are created, and later closed, by
        # lifespan() in main.py; this class only stores and uses them (dependency injection,
        # which also lets tests/test_repository.py pass an in-memory Qdrant client).
        #
        # item_collection and user_collection are Qdrant ALIASES, not real collection names.
        # On every run, training/index.py writes the new vectors into brand-new collections
        # (named like "items_<timestamp>") and then re-points both aliases at them in a
        # single atomic request. The API keeps asking for "items" and "users", so it sees
        # either the complete old index or the complete new one, never a missing or
        # half-written collection, and never users from one model with items from another.
        self.pool = pool
        self.qdrant = qdrant
        self.item_collection = item_collection
        self.user_collection = user_collection

    def ping(self) -> None:
        """Raise if either store is unreachable or a vector collection is missing.
        Returns nothing on success. Called by GET /health in routes.py."""
        # CONCEPT: `with` (context manager)
        #   `with X as name:` runs a setup step, then the indented block, then a cleanup
        #   step, and the cleanup runs even if the block raises. For pool.connection():
        #     setup    borrow a connection from the pool (waits up to the pool's timeout
        #              if all of them are in use)
        #     cleanup  commit if the block succeeded, roll back if it raised, then give the
        #              connection back to the pool
        #   So a connection cannot leak, and commit()/close() are never called by hand.
        with self.pool.connection() as conn:
            # The cheapest possible query: it reads no table, so success means exactly
            # "Postgres is reachable and answering". If it is not, execute() raises, and
            # that exception is the signal the caller wants.
            conn.execute("SELECT 1")
        # Loop over a 2-element tuple to run the same check on both collections.
        for collection in (self.item_collection, self.user_collection):
            # collection_exists is a network call. It raises if Qdrant is down, and returns
            # False if Qdrant is up but the collection is gone. The False case is turned into
            # an exception too, so the caller has a single rule: "ping raised -> 503".
            if not self.qdrant.collection_exists(collection):
                raise RuntimeError(f"Qdrant collection '{collection}' does not exist")

    # Reading the return type from the outside in:
    #   ... | None              the whole result may be None (the collection is missing)
    #   tuple[int, str | None]  otherwise a 2-tuple: an int, then a str-or-None
    def index_info(self, collection: str) -> tuple[int, str | None] | None:
        """(vector size, run_id of the model that produced the vectors), or None if the
        collection does not exist. Used once, at startup, by check_index() in main.py."""
        # Early return: nothing else below makes sense for a collection that is not there.
        if not self.qdrant.collection_exists(collection):
            return None
        # The vector size is a property of the collection, fixed when training/index.py
        # created it. get_collection returns a nested description of the collection; this
        # walks config -> params -> vectors -> size.
        size = self.qdrant.get_collection(collection).config.params.vectors.size
        # scroll means "list points" (no similarity involved).
        #   limit=1                  one point is enough, because the indexer stamps the same
        #                            run_id on every point of a collection
        #   with_payload=["run_id"]  fetch only that one payload field
        # scroll returns a pair (points, next_page_offset). `points, _ = ...` unpacks the pair
        # and discards the offset (`_` = "not used").
        points, _ = self.qdrant.scroll(collection, limit=1, with_payload=["run_id"])
        # Careful with precedence: `A if cond else B` binds tighter than the comma, so this is
        #     return (size, (points[0].payload.get("run_id") if points else None))
        # It ALWAYS returns a 2-tuple. Only the second element becomes None, when the
        # collection is empty (an empty list is falsy, and testing it first avoids an
        # IndexError on points[0]). .get("run_id") also gives None if the field is absent.
        return size, points[0].payload.get("run_id") if points else None

    def user_vector(self, user_id: int) -> list[float] | None:
        """The user's precomputed embedding, or None for a user that was never indexed."""
        # retrieve fetches points by their IDs. It is a key lookup, like
        # SELECT ... WHERE id = ..., and involves no similarity search.
        #   ids=[user_id]       the method takes a list of IDs; we ask for one
        #   with_vectors=True   vectors are large and NOT returned by default; ask for them
        #   with_payload=False  the payload is not needed here
        # IDs that do not exist are simply left out, so an unknown user gives [].
        points = self.qdrant.retrieve(self.user_collection, ids=[user_id], with_vectors=True,
                                      with_payload=False)
        # One point came back -> return its vector. Empty list (falsy) -> return None, which
        # service.py turns into UserNotFoundError.
        return points[0].vector if points else None

    def max_user_id(self) -> int:
        """The largest raw user ID in Postgres; 0 when the table is empty."""
        # MovieLens IDs run 1..N with no gaps, so this equals the number of users (943).
        # max() is used rather than count(*) so it stays correct if IDs ever have gaps.
        with self.pool.connection() as conn:
            return conn.execute("SELECT COALESCE(max(user_id), 0) FROM users").fetchone()[0]

    def seen_item_ids(self, user_id: int) -> list[int]:
        """Raw item IDs of every film this user has rated; [] if there are none."""
        # Borrow a connection for just this query (see CONCEPT: `with` in ping()).
        with self.pool.connection() as conn:
            # CONCEPT: parameterized SQL
            #   %s is a psycopg PLACEHOLDER, not Python string formatting. The SQL text and
            #   the values travel to Postgres separately, so a value is always treated as
            #   data and can never run as SQL. Never build a query with an f-string or `+`:
            #   that is how SQL injection happens.
            #   The values are given as a sequence, one per %s. (user_id,) is a 1-element
            #   tuple; the trailing comma is what makes it a tuple, since (user_id) is just
            #   user_id in parentheses.
            # execute() returns a cursor and .fetchall() reads all result rows into a list of
            # tuples, one tuple per row: [(242,), (302,), ...].
            # No extra index is needed for this query: the table's PRIMARY KEY
            # (user_id, item_id) in db/schema.sql already gives Postgres an index that starts
            # with user_id.
            rows = conn.execute("SELECT item_id FROM interactions WHERE user_id = %s", (user_id,)).fetchall()
        # The `with` block has ended, so the connection is already back in the pool.
        # Each row is a 1-column tuple and row[0] takes the value out:
        # [(242,), (302,)] -> [242, 302].
        return [row[0] for row in rows]

    def search_items(self, vector: list[float], exclude_ids: list[int], k: int) -> list[tuple[int, float]]:
        """The k highest-scoring items that are not in exclude_ids, as (item_id, score), best first."""
        # must_not + has_id filters out points by ID inside the vector search itself, so the
        # result is the top k of the remaining items (never "top k, then remove": that would
        # return fewer than k).
        #   Filter(must_not=[...])        keep a point only if it matches NONE of the conditions
        #   HasIdCondition(has_id=[...])  the condition "this point's ID is in the list"
        # `X if cond else Y` again: an empty exclude_ids is falsy, so a user with no ratings
        # gets None, which means "no filter, search every item".
        query_filter = Filter(must_not=[HasIdCondition(has_id=exclude_ids)]) if exclude_ids else None
        # exact=True: compare against every vector instead of using the approximate index.
        # With ~1.7k items that costs nothing and guarantees these really are the top k.
        # Background: a vector database normally answers through an approximate index (HNSW),
        # which is far faster on millions of vectors but may miss a few true neighbours.
        #   query=vector        the user's vector. Each item is scored by cosine similarity
        #                       to it (COSINE was chosen when training/index.py created the
        #                       collection); a higher score means a better match.
        #   query_filter        applied during the search, as explained above
        #   limit=k             return the k best
        #   with_payload=False  IDs and scores only; titles etc. come from Postgres
        result = self.qdrant.query_points(self.item_collection, query=vector, query_filter=query_filter,
                                          limit=k, with_payload=False,
                                          search_params=SearchParams(exact=True))
        # result.points is already sorted, highest score first. Each point is reduced to a
        # plain (item_id, score) tuple so that service.py does not depend on Qdrant's classes.
        return [(point.id, point.score) for point in result.points]

    def items_metadata(self, item_ids: list[int]) -> dict[int, dict]:
        """Title, release year and genres for the given item IDs, as {item_id: {...}}.
        IDs that are not in Postgres are absent from the result; [] gives {}."""
        with self.pool.connection() as conn:
            # One query for all IDs, instead of one query per item (the "N+1 queries"
            # mistake: k round trips to the database where 1 is enough).
            # `item_id = ANY(%s)` means "item_id equals any element of this array". psycopg
            # sends the Python list as ONE Postgres array parameter, so the SQL text is the
            # same for any number of IDs and stays parameterized. Joining the IDs into an
            # "IN (1, 2, 3)" string would bring the injection risk back.
            # (item_ids,) is a 1-element tuple again: one %s, one value, and that value is
            # the whole list.
            rows = conn.execute(
                "SELECT item_id, title, release_year, genres FROM items WHERE item_id = ANY(%s)",
                (item_ids,),
            ).fetchall()
        # Dict comprehension: {key: value for r in rows}. Each row r is the tuple
        # (item_id, title, release_year, genres), read by position in SELECT order.
        # genres is a Postgres text[] column, which psycopg hands back as a Python list of str.
        # The inner dict's keys are exactly RecommendedItem's field names; that is what lets
        # service.py write **metadata[item_id].
        # The result is keyed by ID and carries no ranking (SQL returns rows in no guaranteed
        # order without ORDER BY), which is why service.py takes the order from `hits`.
        return {r[0]: {"title": r[1], "release_year": r[2], "genres": r[3]} for r in rows}
