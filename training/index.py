"""Load metadata into Postgres and precomputed user + item vectors into Qdrant. Run: python -m training.index

Needs the compose services running and a trained artifact in settings.artifact_dir.
This is the only place the trained model is used for serving: the API reads these vectors.
Everything stored uses raw MovieLens IDs.

Safe to re-run while the API is serving:
- Postgres tables are rebuilt and reloaded in one transaction (readers see old or new, never half).
- Qdrant vectors go into fresh collections; the `items` / `users` aliases the API queries are
  then switched to them in a single atomic request, so user and item vectors always come
  from the same model.
"""

# index.py — hand the result over to serving
#
# - Job: the bridge from training to the API. This is the last time the model is used.
# - Postgres: loads users, films and all 100,000 ratings.
# - Qdrant: runs the model once to compute a vector for every user and every film, and
#   stores them.
# - Why it exists: after this, the API can recommend by looking up vectors, without PyTorch
#   or the model.

import re
import time

import pandas as pd
import psycopg
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation,
                                  Distance, PointStruct, VectorParams)

from config import PROJECT_ROOT, Settings, get_settings
from training.models.two_tower import TwoTower, load_model
from training.prepare import build_features, load_raw, validate

SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def load_postgres(conn: psycopg.Connection, ratings: pd.DataFrame, users: pd.DataFrame,
                  items: pd.DataFrame, genres: list[str]) -> None:
    """Rebuild the tables from db/schema.sql and load them, all in one transaction."""
    user_rows = users[["user_id", "age", "gender", "occupation", "zip_code"]].to_numpy(dtype=object).tolist()
    item_rows = []
    for row, flags in zip(items.itertuples(), items[genres].to_numpy()): # itertuples() lets you access column values using dot notation (e.g., row.column_name) rather than bracket indexing.
        # A film can have several genres; keep the names whose 0/1 flag is set.
        film_genres = []
        for genre, flag in zip(genres, flags):
            if flag:
                film_genres.append(genre)
        # One film (item 267, "unknown") has no release year; store NULL for it.
        if pd.isna(row.release_year):
            release_year = None
        else:
            release_year = int(row.release_year)
        item_rows.append((int(row.item_id), row.title, release_year, film_genres))
    rating_rows = ratings[["user_id", "item_id", "rating", "timestamp"]].to_numpy().tolist()
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text())
        cur.executemany(
            "INSERT INTO users (user_id, age, gender, occupation, zip_code) VALUES (%s, %s, %s, %s, %s)",
            user_rows)
        cur.executemany(
            "INSERT INTO items (item_id, title, release_year, genres) VALUES (%s, %s, %s, %s)", item_rows)
        cur.executemany(
            "INSERT INTO interactions (user_id, item_id, rating, rated_at) "
            "VALUES (%s, %s, %s, to_timestamp(%s))", rating_rows)


def write_collection(client: QdrantClient, alias: str, ids: list[int], vectors: torch.Tensor,
                     payloads: list[dict]) -> str:
    """Write one point per raw ID into a brand-new collection and return its name.
    vectors: (len(ids), D), row i belongs to ids[i]. Nothing reads the collection yet."""
    collection = f"{alias}_{time.time_ns()}"
    client.create_collection(collection, vectors_config=VectorParams(size=vectors.shape[1],
                                                                     distance=Distance.COSINE)) # Cosine similarity
    client.upsert(collection, points=[PointStruct(id=point_id, vector=vector, payload=payload)
                                      for point_id, vector, payload in zip(ids, vectors.tolist(), payloads)]) # Payload lets us store additional info along with vectors. 
                                                                                                                # For example, we can store "genre" of vectors. 
                                                                                                                # Then later tell Qdrant to return top_10 "Drama" recommendations for user A
                                                                                                                # Those recommendatins which are not "Drama" won't be recommended 
                                                                                                                # even though their score are higher
    return collection


# This function basically let us reuse the aliases "items" and "users": every time index.py reloads the embeddings
# into Qdrant as new collections, the aliases are switched to the new ones (in 1 request), then the old collections are deleted
# So the API always asks for "items" and "users" and never needs to change or restart
def switch_aliases(client: QdrantClient, new_collections: dict[str, str]) -> None:
    """Point each alias at its new collection in ONE request (atomic), then delete the old ones."""
    current = {a.alias_name: a.collection_name for a in client.get_aliases().aliases} # This asks Qdrant which aliases exist now and where they point: {"items": "items_1759...", "users": "users_1759..."}
    operations = [] # To-do list for Qdrant, NOT sent yet
    for alias, collection in new_collections.items(): # alias = "items", collection = "items_1760..." (then same for users)
        if alias in current: # Alias already exists
            operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))) # Add "unhook alias items from the OLD collection" to the to-do list
        elif client.collection_exists(alias): # No alias, but a REAL collection called "items" exists (old setup)
            client.delete_collection(alias) # Delete it right now, since an alias can't have the same name as a collection
        operations.append(CreateAliasOperation(
            create_alias=CreateAlias(collection_name=collection, alias_name=alias))) # Add "create alias items -> items_1760..." to the to-do list
    client.update_collection_aliases(change_aliases_operations=operations) # Send the whole to-do list in 1 request, so the API sees old pair or new pair, NEVER a mix

    keep = set(new_collections.values()) # The 2 new collections, {"items_1760...", "users_1760..."}
    for name in [c.name for c in client.get_collections().collections]: # Go through every collection in Qdrant
        if name not in keep and any(re.fullmatch(rf"{re.escape(alias)}_\d+", name) for alias in new_collections): # Delete name only if BOTH: 1) it's not one of the 2 new collections, 2) it's named items_<digits> or users_<digits>
                                                                                                                   # any(...) checks name against "items_<digits>" and against "users_<digits>", True if it matches at least one of them
                                                                                                                   # The pattern is "items_" + only digits, e.g. items_1759... matches, items_backup or my_items_1759 don't
                                                                                                                   # Only 1) would also delete unrelated collections (test_stuff), only 2) would also delete the NEW ones:
                                                                                                                   #                        1) not new?   2) items_/users_<digits>?   deleted?
                                                                                                                   #   items_1760... (new)      no              yes                    no
                                                                                                                   #   items_1759... (old)      yes             yes                    YES
                                                                                                                   #   test_stuff               yes             no                     no
                                                                                                                   # So basically: old items_<time> / users_<time> get deleted, any other collection is left alone
            client.delete_collection(name) 


@torch.no_grad()
def index_model(client: QdrantClient, settings: Settings, model: TwoTower, meta: dict,
                items: pd.DataFrame) -> None:
    """Offline retrieval step: encode every user and item once, so the API needs no model."""
    run = {"run_id": meta["run_id"]}
    item_vectors = model.encode_items(torch.arange(len(meta["item_ids"])))
    user_vectors = model.encode_users(torch.arange(len(meta["user_ids"])))

    items = items.set_index("item_id").loc[meta["item_ids"]]
    item_payloads = []
    for flags, year in zip(items[meta["genres"]].to_numpy(), items["release_year"]):
        # Keep the names of the genres whose 0/1 flag is set.
        film_genres = []
        for genre, flag in zip(meta["genres"], flags):
            if flag:
                film_genres.append(genre)
        # Item 267 has no release year; store null for it, as in Postgres.
        if pd.isna(year):
            release_year = None
        else:
            release_year = int(year)
        item_payloads.append({"genres": film_genres, "release_year": release_year, **run}) # 3 info is stored in each vector payload, which is genre, release_year and run_id (which is the model that produce this embedding)
                                                                                            # Later, we'll user run_id to check if user embedding and item embeddings come from the same model
                                                                                            # If not, raise Error
    switch_aliases(client, {
        settings.qdrant_item_collection: write_collection(
            client, settings.qdrant_item_collection, meta["item_ids"], item_vectors, item_payloads),
        settings.qdrant_user_collection: write_collection(
            client, settings.qdrant_user_collection, meta["user_ids"], user_vectors,
            [run] * len(meta["user_ids"])),
    })


def check_ids_match_artifact(users: pd.DataFrame, items: pd.DataFrame, meta: dict) -> None:
    """Raise ValueError if the raw data has other users or items than the model was trained on.

    Postgres is loaded from the raw files and Qdrant from the artifact, so a difference
    would leave rows without a vector (or vectors without a row). Both frames are sorted by
    raw ID (build_features), the same order as the artifact's lists."""
    for name, raw_ids, trained_ids in [("user", users["user_id"].tolist(), meta["user_ids"]),
                                       ("item", items["item_id"].tolist(), meta["item_ids"])]:
        if raw_ids != trained_ids:
            raise ValueError(
                f"The raw data has {len(raw_ids)} {name} IDs but the model artifact (run {meta['run_id']}) "
                f"was trained on {len(trained_ids)}, or the IDs differ. Nothing was written. Re-run "
                "`python -m training.prepare` and `python -m training.train`, or restore the raw data."
            )


def main() -> None:
    settings = get_settings()
    model, meta = load_model(settings.artifact_dir)
    ratings, users, items, genres = load_raw(settings.raw_data_dir)
    validate(ratings, users, items)
    users, items, _ = build_features(users, items, genres)
    check_ids_match_artifact(users, items, meta)

    with psycopg.connect(settings.pg_conninfo) as conn:
        load_postgres(conn, ratings, users, items, genres)
        counts = [conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("users", "items", "interactions")]
    client = QdrantClient(url=settings.qdrant_url)
    index_model(client, settings, model, meta, items)
    points = client.count(settings.qdrant_item_collection).count
    user_points = client.count(settings.qdrant_user_collection).count
    print(f"postgres users={counts[0]} items={counts[1]} interactions={counts[2]}; "
          f"qdrant '{settings.qdrant_item_collection}' points={points} "
          f"'{settings.qdrant_user_collection}' points={user_points} (run {meta['run_id']})")


if __name__ == "__main__":
    main()
