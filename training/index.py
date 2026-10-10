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
    # .tolist() converts numpy scalars to plain Python values that psycopg can adapt.
    user_rows = users[["user_id", "age", "gender", "occupation", "zip_code"]].to_numpy(dtype=object).tolist()
    item_rows = [
        (int(row.item_id), row.title, None if pd.isna(row.release_year) else int(row.release_year),
         [g for g, flag in zip(genres, flags) if flag])
        for row, flags in zip(items.itertuples(), items[genres].to_numpy())
    ]
    rating_rows = ratings[["user_id", "item_id", "rating", "timestamp"]].to_numpy().tolist()
    # conn.transaction() commits on success and rolls everything back (including the
    # DROP/CREATE: Postgres DDL is transactional) if any statement fails.
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text())  # several statements; allowed because there are no parameters
        cur.executemany(
            "INSERT INTO users (user_id, age, gender, occupation, zip_code) VALUES (%s, %s, %s, %s, %s)",
            user_rows)
        cur.executemany(
            "INSERT INTO items (item_id, title, release_year, genres) VALUES (%s, %s, %s, %s)", item_rows)
        # All interactions (including the held-out ones): serving must not recommend
        # anything the user has already rated.
        cur.executemany(
            "INSERT INTO interactions (user_id, item_id, rating, rated_at) "
            "VALUES (%s, %s, %s, to_timestamp(%s))", rating_rows)


def write_collection(client: QdrantClient, alias: str, ids: list[int], vectors: torch.Tensor,
                     payloads: list[dict]) -> str:
    """Write one point per raw ID into a brand-new collection and return its name.
    vectors: (len(ids), D), row i belongs to ids[i]. Nothing reads the collection yet."""
    collection = f"{alias}_{time.time_ns()}"
    client.create_collection(collection, vectors_config=VectorParams(size=vectors.shape[1],
                                                                     distance=Distance.COSINE))
    client.upsert(collection, points=[PointStruct(id=point_id, vector=vector, payload=payload)
                                      for point_id, vector, payload in zip(ids, vectors.tolist(), payloads)])
    return collection


def switch_aliases(client: QdrantClient, new_collections: dict[str, str]) -> None:
    """Point each alias at its new collection in ONE request (atomic), then delete the old ones."""
    current = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    operations = []
    for alias, collection in new_collections.items():
        if alias in current:
            operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
        elif client.collection_exists(alias):
            client.delete_collection(alias)  # a plain collection from before aliases were used
        operations.append(CreateAliasOperation(
            create_alias=CreateAlias(collection_name=collection, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=operations)

    # Clean up: collections the aliases used to point at, and leftovers of crashed runs.
    keep = set(new_collections.values())
    for name in [c.name for c in client.get_collections().collections]:
        if name not in keep and any(re.fullmatch(rf"{re.escape(alias)}_\d+", name) for alias in new_collections):
            client.delete_collection(name)


@torch.no_grad()
def index_model(client: QdrantClient, settings: Settings, model: TwoTower, meta: dict,
                items: pd.DataFrame) -> None:
    """Offline retrieval step: encode every user and item once, so the API needs no model."""
    # run_id on every point lets the API report which model it is serving.
    run = {"run_id": meta["run_id"]}
    item_vectors = model.encode_items(torch.arange(len(meta["item_ids"])))  # (n_items, D), row = item_idx
    user_vectors = model.encode_users(torch.arange(len(meta["user_ids"])))  # (n_users, D), row = user_idx

    items = items.set_index("item_id").loc[meta["item_ids"]]  # same order as the vectors
    item_payloads = [
        {"genres": [g for g, flag in zip(meta["genres"], flags) if flag],
         "release_year": None if pd.isna(year) else int(year), **run}
        for flags, year in zip(items[meta["genres"]].to_numpy(), items["release_year"])
    ]
    switch_aliases(client, {
        settings.qdrant_collection: write_collection(
            client, settings.qdrant_collection, meta["item_ids"], item_vectors, item_payloads),
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
    check_ids_match_artifact(users, items, meta)  # before any write to Postgres or Qdrant

    with psycopg.connect(settings.pg_conninfo) as conn:
        load_postgres(conn, ratings, users, items, genres)
        counts = [conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # fixed table names, not user input
                  for t in ("users", "items", "interactions")]
    client = QdrantClient(url=settings.qdrant_url)
    index_model(client, settings, model, meta, items)
    points = client.count(settings.qdrant_collection).count
    user_points = client.count(settings.qdrant_user_collection).count
    print(f"postgres users={counts[0]} items={counts[1]} interactions={counts[2]}; "
          f"qdrant '{settings.qdrant_collection}' points={points} "
          f"'{settings.qdrant_user_collection}' points={user_points} (run {meta['run_id']})")


if __name__ == "__main__":
    main()
