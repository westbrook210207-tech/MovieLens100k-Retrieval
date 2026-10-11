# Running order

How the project runs: which file produces the input of which, for the offline
`training/` pipeline and for the online `api/`.

There are two kinds of "input": data files a script reads from disk, and code a
script imports. Both are listed below.

---

## 1. Training pipeline

### Data flow

Each arrow is a file or database written by one step and read by the next.

```text
ml-100k/ (u.data, u.user, u.item, u.genre)
   │
   ▼
1. training/prepare.py ──► training/data/  (train.csv, val.csv, test.csv,
   │                                        users.csv, items.csv, mappings.json)
   ▼
2. training/train.py ────► training/artifacts/two_tower/  (model.pt, meta.json)
   │                       + one MLflow run
   ├──► 3. training/evaluate.py ──► metrics printed and added to that MLflow run
   ▼
4. training/index.py ────► Postgres (users, items, interactions)
   │                       Qdrant (item vectors, user vectors)
   ▼
5. api/  ────────────────► HTTP responses
```

### Steps

| Step | Run command | Reads | Writes |
|---|---|---|---|
| 1 | `python -m training.prepare` | the four raw files in `ml-100k/` | the five CSVs and `mappings.json` in `training/data/` |
| 2 | `python -m training.train` | `train.csv`, `val.csv`, `users.csv`, `items.csv`, `mappings.json` (never `test.csv`) | `model.pt`, `meta.json`, an MLflow run |
| 3 | `python -m training.evaluate --split val` or `--split test` | `model.pt`, `meta.json`, `train.csv`, `val.csv`, `mappings.json`, plus `test.csv` for `--split test` | metrics only |
| 4 | `python -m training.index` | `model.pt`, `meta.json`, the raw `ml-100k/` files again, `db/schema.sql` | Postgres tables and Qdrant vectors |
| 5 | `python -m uvicorn api.main:app` | Postgres and Qdrant; `meta.json` only to warn about a stale index | nothing; it only answers requests |

### Which code file feeds which

Helper files that a step imports rather than runs:

- **`config.py`**: imported by every step. It reads `.env` once and supplies paths,
  database addresses and the password.
- **`training/models/two_tower.py`**: the model, the loss, and save/load. Used by
  steps 2, 3 and 4.
- **`training/sampling.py`**: produces the negatives. Used by step 2 only.
- **`training/evaluate.py`**: also a helper. Step 2 imports its scoring functions to
  measure validation after every epoch.
- **`training/prepare.py`**: also a helper. Step 4 imports its loading, validation
  and feature functions to re-read the raw data.
- **`db/schema.sql`**: the table definitions, executed by step 4.
- **`compose.yaml`**: starts Postgres, Qdrant and MLflow. These must be running
  before steps 2, 3, 4 and 5.

### Reading order

Each file only depends on ones earlier in the list:

1. `config.py`
2. `training/prepare.py`
3. `training/models/two_tower.py`
4. `training/sampling.py`
5. `training/evaluate.py`
6. `training/train.py`
7. `db/schema.sql`, then `training/index.py`

---

## 2. API

The `api/` folder has two orders: which file imports which, and what runs when.

### Imports

An arrow means "imports from".

```text
main.py ──► repositories.py   (Repository)
        ──► routes.py         (router)
        ──► service.py        (RecommendationService)
        ──► config.py         (get_settings)

routes.py ──► service.py      (RecommendationService, UserNotFoundError)
          ──► schemas.py      (HealthResponse, RecommendationResponse)

service.py ──► schemas.py     (RecommendedItem)

repositories.py   imports nothing from the project (only psycopg_pool and qdrant_client)
schemas.py        imports nothing from the project (only pydantic)
```

- **`main.py` is the top**: it imports the others and nothing imports it.
- **`service.py` does not import `repositories.py`**: it receives a repository object
  from `main.py` at startup instead. That is why tests can hand it a fake one.
- **`api/__init__.py` is empty**: it only marks the folder as a package so `api.main`
  can be imported.

### Run order

**1. Import time** — `python -m uvicorn api.main:app` imports `api/main.py`, which
runs it top to bottom:

1. `repositories.py` is loaded (defines `Repository`).
2. `routes.py` is loaded, which first loads `schemas.py` and then `service.py`.
3. `config.py` is loaded (defines `Settings`; `.env` is not read yet).
4. Logging is configured.
5. `app = FastAPI(...)` is created, the routes are attached, and the logging
   middleware is registered.

No connection has been opened at this point.

**2. Startup** — uvicorn runs `lifespan()` in `main.py` up to its `yield`:

1. `get_settings()` reads and validates `.env`.
2. The Postgres pool and the Qdrant client are created, and the pool is opened.
3. A `Repository` is built from them.
4. `check_index()` asks Qdrant for both collections and stops startup if they are
   missing or inconsistent.
5. `warn_if_index_is_stale()` compares the served run ID with `meta.json`.
6. One `RecommendationService` is built (this runs one Postgres query for the
   largest user ID) and stored on `app.state`.
7. The `user_id` maximum is written into the API docs.

**3. Each request** — for example `GET /users/5/recommendations?k=10`:

1. `log_requests` in `main.py` starts a timer.
2. FastAPI matches the route in `routes.py` and validates `user_id` and `k`
   (422 if invalid).
3. `get_service` in `routes.py` fetches the service from `app.state`.
4. `recommendations()` in `routes.py` checks `user_id` against the maximum, then
   calls `service.recommend()`.
5. `recommend()` in `service.py` calls four methods in `repositories.py`, in this
   order:
   - `user_vector` (Qdrant) — a missing vector becomes a 404.
   - `seen_item_ids` (Postgres)
   - `search_items` (Qdrant)
   - `items_metadata` (Postgres)
6. `service.py` builds the list of `RecommendedItem` (from `schemas.py`), and
   `routes.py` wraps it in a `RecommendationResponse`.
7. `log_requests` writes one log line and the response goes back to the client.

`GET /health` is shorter: `routes.py` calls `repository.ping()` and returns either
`ok` with the run ID or a 503.

**4. Shutdown** — on Ctrl+C, `lifespan()` resumes after its `yield` and closes the
pool and the Qdrant client.

### Reading order

Following the imports from the bottom up: `schemas.py`, `repositories.py`,
`service.py`, `routes.py`, then `main.py`.
