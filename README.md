# MovieLens 100K recommender

Hybrid two-tower retrieval (ID embeddings + user/item features), trained offline with
listwise InfoNCE. After training, every user and item is encoded once and stored in Qdrant;
the FastAPI service only looks those vectors up (no PyTorch, no model loading online).

## Layout

| Path | Purpose |
|---|---|
| `ml-100k/` | Raw MovieLens 100K. Read-only, git-ignored ([download](https://grouplens.org/datasets/movielens/100k/)). |
| `training/` | Offline pipeline: `prepare.py` → `train.py` → `evaluate.py` → `index.py`. |
| `training/data/`, `training/artifacts/` | Generated split and exported model (git-ignored). |
| `api/` | Online serving: `routes.py` → `service.py` → `repositories.py`. |
| `db/schema.sql` | Postgres tables. Applied by `training.index`, which rebuilds them on every run. |
| `config.py` | All settings (`get_settings()`), read from environment / `.env`. |

## IDs

- **Raw MovieLens IDs** (1-based) in Postgres, Qdrant point IDs and the API.
- **Zero-based indices** (`user_idx`, `item_idx`) only inside the model. `meta.json` holds
  `user_ids` / `item_ids` lists whose positions are those indices.

## Setup

```sh
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # then set PG_PASSWORD
docker compose up -d postgres qdrant mlflow
```

| Service | Address | Runs in |
|---|---|---|
| Postgres | `localhost:5432` | Docker Compose |
| Qdrant | `http://localhost:6333` | Docker Compose |
| MLflow UI | `http://localhost:5001` | Docker Compose |
| API | `http://localhost:8000` | host (`uvicorn`) |

All three ports are bound to `127.0.0.1` only: Qdrant and MLflow have no authentication, so
they are not exposed to the network. `.env.example` documents every variable and which
service reads it.

The stack idles at roughly 0.5 GB of memory. MLflow runs one server worker with its
background job runner off (the defaults use over 2 GB), and Qdrant is limited to one thread
per pool; see the comments in `compose.yaml`.

## Pipeline

```sh
python -m training.prepare            # validate + split  -> training/data/
python -m training.train              # train, log to MLflow -> training/artifacts/two_tower/
                                      # uses the Apple GPU (MPS) when available; --device cpu to override
python -m training.evaluate --split val
python -m training.index              # Postgres rows + Qdrant user and item vectors
python -m uvicorn api.main:app --port 8000   # API runs on the host, not in Docker
```

Re-run `training.index` after every retrain; until then the API keeps serving the previous
model's vectors. If `meta.json` names a different run than the one indexed, the API logs a
`stale_index` warning at startup, and `/health` always reports the run being served.

`training.index` loads Postgres from the raw files and Qdrant from the model artifact. It
stops before writing anything if the raw user or item IDs differ from the ones the model was
trained on, so the two stores cannot end up describing different users or films.

`training.index` is safe to run while the API is serving:

- **Postgres:** the tables are dropped, recreated from `db/schema.sql` and reloaded in one
  transaction, so a schema change needs nothing more than a re-index. This is only valid while
  Postgres holds nothing but data derived from `ml-100k/`.
- **Qdrant:** vectors are written to new collections (`items_<timestamp>`, `users_<timestamp>`);
  the `items` and `users` aliases the API queries are then switched in one atomic request and
  the old collections deleted.

At startup the API refuses to run if Postgres or Qdrant is unreachable, or if the two aliases
are missing or come from different model runs.

## API

- `GET /health` → `{"status": "ok", "model_run_id": "<MLflow run being served>"}`, or 503 if
  Postgres/Qdrant is unreachable.
- `GET /users/{user_id}/recommendations?k=10` (`k` 1–100) → ranked items the user has not
  rated, with title, year, genres and cosine score. `user_id` must be between 1 and the largest user ID in Postgres (943), read at
  startup; anything else, or a bad `k`, → 422. A user in range with no vector → 404.
  A backend failure is a 500 within about 5 seconds, never a hang.

The API never trains and never runs the model. Per request it fetches the user's precomputed
vector from Qdrant, searches the item vectors excluding already-rated items, and reads titles
from Postgres. Only users present at indexing time can be served.

## Model artifact (`training/artifacts/two_tower/`)

- `model.pt`: state dict, including the user/item feature tables as buffers.
- `meta.json`: `run_id`, `split_hash`, `positive_threshold`, `model_config` (incl. `num_hidden_layers`), `sampling_config`,
  `user_cardinalities`, `user_dense_dim`, `item_cardinalities`, `user_ids`, `item_ids`, `genres`, `val_metrics`.

`mappings.json` (written by `training.prepare`) carries two hashes: `split_hash` changes when
the train/validation/test rows change, `feature_hash` when a user or item feature changes.
`training.evaluate` refuses to score a model on a split other than the one it was trained on.

Load with `training.models.two_tower.load_model`. Only `training.index` and
`training.evaluate` read it; the API does not.

## Evaluation protocol

Per-user temporal leave-one-out over positives (rating ≥ 4): ordering a user's ratings by
timestamp (ties broken by item ID), the latest positive is test, the second-latest is
validation, and everything before the validation positive is training. Low ratings made after
the validation positive are dropped, so training never contains anything later than
validation. The row order of `u.data` is shuffled and is not used. MovieLens timestamps are
rating times and many users rated in one sitting, so these numbers are noisy.
Users with fewer than 3 positives are
train-only. Each user is ranked against all items minus their other known interactions
(train for validation; train + validation for test); Recall@K, NDCG@K, Precision@K
(K = 10, 20, 200) are averaged over users. With one held-out item per user, Precision@K is at most 1/K.
Offline metrics on sampled history do not guarantee online performance.

## Negative sampling

Each training example is a list `(user, pos, {negatives})` scored with InfoNCE. Default
(`python -m training.train`): K=20 uniform items the user has not rated in training, plus the
other positives of the same batch (`--in-batch-negatives`, on by default). In-batch candidates
get two fixes, both from training data only:

- **False-negative mask:** a batch-mate's positive is dropped for a user who rated that item
  in training (about 1 in 3 in-batch candidates on this dataset).
- **logQ correction:** each logit is reduced by `log(K / n_items + (B - 1) * q_item)`, the
  expected number of times the item is drawn as a negative, so popular items are not
  pushed down just for appearing in many batches.

## Tower

Each tower is `inputs → [Linear → ReLU → Dropout] × --num-hidden-layers → Linear → L2-normalise`,
every hidden layer `--hidden-dim` wide. Default: two hidden layers of 256
(user 104 → 256 → 256 → 64, item 91 → 256 → 256 → 64).

## Model selection and hyperparameters

The checkpoint kept from a run, and early stopping, use **validation Recall@200**
(`SELECTION_METRIC` in `training/train.py`): this model is the retrieval stage, so what matters
is whether the user's next film is among the candidates handed on.

Defaults were set by a one-setting-at-a-time search on the temporal split (MPS, seeds 0/1/2,
100-epoch cap). A value replaced the current one only if it raised mean validation Recall@200
by at least 0.005, roughly the seed-to-seed spread. Best value per setting, against the value
in force at that stage:

| Setting | Values tried | Best (Recall@200) | In force (Recall@200) | Default now |
|---|---|---|---|---|
| `--lr` | 0.0003, 0.001, 0.003 | 0.003 (0.7948) | 0.001 (0.7820) | **0.003** |
| `--patience` | 5, 10, 20 | 10 (0.7969) | 5 (0.7948) | 5 |
| `--temperature` | 0.05, 0.07, 0.1, 0.15, 0.2 | 0.1 = 0.15 (0.7948) | 0.1 (0.7948) | 0.1 |
| `--batch-size` | 256, 512, 1024, 2048 | 256 (0.8001) | 1024 (0.7948) | **256** |
| `--dropout` | 0, 0.1, 0.2, 0.3 | 0.3 (0.8015) | 0.1 (0.8001) | 0.1 |
| `--weight-decay` | 0, 0.0001, 0.001, 0.01 | 0.01 (0.8043) | 0.0001 (0.8001) | 0.0001 |
| `--embedding-dim` | 32, 64, 128 | 64 (0.8001) | 64 (0.8001) | 64 |
| `--feature-dim` | 4, 8, 16 | 8 (0.8001) | 8 (0.8001) | 8 |
| `--hidden-dim` | 128, 256, 512 | 128 (0.8022) | 256 (0.8001) | 256 |
| `--num-hidden-layers` | 1, 2, 3 | 3 (0.8015) | 2 (0.8001) | 2 |

Negative sampling in the same search (learning rate 0.003, batch 1024, tower 256 → 256):

| Variant | Flags | Recall@200 | NDCG@10 |
|---|---|---|---|
| **Uniform K=20 + in-batch (default)** | none | 0.7948 ± 0.0053 | 0.0929 |
| Uniform K=50 + in-batch | `--num-negatives 50` | 0.7930 ± 0.0119 | 0.0934 |
| Uniform K=100 + in-batch | `--num-negatives 100` | 0.7941 ± 0.0076 | 0.0959 |
| In-batch only | `--num-negatives 0` | 0.7640 ± 0.0041 | 0.0903 |
| Uniform K=20 | `--no-in-batch-negatives` | 0.7987 ± 0.0028 | 0.0904 |
| Uniform K=100 | `--no-in-batch-negatives --num-negatives 100` | 0.7962 ± 0.0046 | 0.0931 |
| Uniform + in-batch + explicit ≤ 2 | `--hard-negative-ratio 0.2` | 0.7926 ± 0.0043 | 0.0945 |
| Uniform + explicit ≤ 2 | `--no-in-batch-negatives --hard-negative-ratio 0.2` | 0.7325 ± 0.0043 | 0.0692 |

Reading: the model is nearly flat in its hyperparameters; only the learning rate and batch
size moved Recall@200 by more than seed noise. Explicit low-rated negatives without in-batch
are clearly worst and in-batch alone is weak; the other sampling variants are tied.
Confirmation on fresh seeds 3/4/5: tuned defaults 0.7951 ± 0.0015 against 0.7895 ± 0.0039
before tuning (NDCG@10 0.0941 against 0.0952).

Not done: a joint search (settings were varied one at a time), learning-rate decay, and
combining the near-miss values (weight decay 0.01, dropout 0.3, patience 10).

## Exported model

Default settings, seed 0, trained on MPS. The test split has been scored three times in total
(for two earlier models as well); every choice was made on validation, but the test numbers
are not a fully untouched estimate.

| Split | Model | NDCG@10 | Recall@10 | Recall@20 | Recall@200 | NDCG@200 | Precision@200 |
|---|---|---|---|---|---|---|---|
| Validation | Two-tower | 0.0963 | 0.1900 | 0.2813 | 0.8015 | 0.2068 | 0.0040 |
| Validation | Popularity | 0.0385 | 0.0722 | 0.1253 | 0.5212 | 0.1164 | 0.0026 |
| Test | Two-tower | 0.0697 | 0.1338 | 0.2176 | 0.7325 | 0.1761 | 0.0037 |
| Test | Popularity | 0.0348 | 0.0679 | 0.1083 | 0.4639 | 0.1017 | 0.0023 |

## User features

Only the user ID is embedded. Everything else goes into the user tower as 0/1 columns:
`is_male` (F → 0, M → 1), a 7-bucket age one-hot, a 21-way occupation one-hot and an
11-way zip-region one-hot (first digit of a US zip code; anything else is "Non-US / unknown").

## Tests

```sh
python -m pytest
```

| File | Covers | Needs |
|---|---|---|
| `tests/test_api.py` | Endpoints, error codes, logging, startup index check | nothing |
| `tests/test_guards.py` | Settings, diverged-model guard, raw-data validation, raw-data/artifact ID check, stale-index warning | nothing |
| `tests/test_training.py` | Split, sampler, loss, metrics, model save/load | nothing |
| `tests/test_repository.py` | Real SQL and Qdrant calls, table reload, alias switch | the compose Postgres (skipped if down) |

`tests/test_repository.py` works inside a throwaway Postgres schema that it drops afterwards,
and uses Qdrant's in-memory mode, so it never touches the served data.
