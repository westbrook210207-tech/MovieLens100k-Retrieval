"""FastAPI application. Run: python -m uvicorn api.main:app --reload

Online retrieval only: the API reads vectors that training/index.py precomputed into
Qdrant. It does not import PyTorch or load the model. Startup connects to Postgres and
Qdrant once and refuses to start if either is unreachable or the vector index is
missing or inconsistent.

Where this file sits (one request, left to right):

    client -> log_requests (here) -> routes.py -> service.py -> repositories.py -> Postgres / Qdrant

This file is the wiring. It does three things:
  1. lifespan()      at startup: open the connections, check the index, build the service.
                     at shutdown: close the connections.
  2. app             the FastAPI object that uvicorn serves.
  3. log_requests()  write one log line for every HTTP request.
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from psycopg_pool import ConnectionPool, PoolTimeout
from qdrant_client import QdrantClient

from api.repositories import Repository
from api.routes import router
from api.service import RecommendationService
from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("api")

BACKEND_TIMEOUT_SECONDS = 5 # how long the API waits for one connection or one reply from Postgres or Qdrant before giving up.
                            # This is different from interval inside compose.yaml
                            # Interval is services/containers healthcheck
                            # BACKEND_TIMEOUT_SECONDS is connection/reply healthcheck

def check_index(repository: Repository) -> str:
    """Startup guard: refuse to serve unless the vector index is complete and consistent.

    User and item vectors must exist and come from the same model run (same size, same
    run_id). Returns that run_id.

    Why it matters: a recommendation score is the similarity between a USER vector and an
    ITEM vector. That number only means something if one trained model produced both.
    Vectors from two different models live in unrelated spaces; comparing them raises no
    error, it just returns confident-looking nonsense. So we check once, before serving.
    """
    items = repository.index_info(repository.item_collection)
    users = repository.index_info(repository.user_collection)
    if items is None or items != users or items[1] is None: # item[1] is None check for case where you have size so item != None but you don't have run_id
                                                            # we only need to check item is None and item[1] is None since item and user return exactly the same thing
                                                            # So item covers user
        raise RuntimeError(
            f"Qdrant vector index is missing or inconsistent: '{repository.item_collection}' "
            f"(size, run_id)={items}, '{repository.user_collection}' (size, run_id)={users}. "
            "Train a model, then run `python -m training.index`."
        )
    return items[1] # Returns run_id


def warn_if_index_is_stale(served_run_id: str) -> None:
    """The API serves whatever was last indexed. If a newer model has been exported but
    not indexed, say so (reads meta.json only; the model itself is never loaded here)."""
    meta_path = get_settings().artifact_dir / "meta.json"
    if not meta_path.exists():
        return
    exported_run_id = json.loads(meta_path.read_text()).get("run_id")
    if exported_run_id != served_run_id: # Check if the model that produces the collections and the model in meta is the same model
        logger.warning(json.dumps({ # here we only warning, not raise
            "event": "stale_index", "serving_run_id": served_run_id, "exported_run_id": exported_run_id,
            "hint": "run `python -m training.index` to serve the exported model",
        }))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    pool = ConnectionPool(settings.pg_conninfo, min_size=1, max_size=5,
                          timeout=BACKEND_TIMEOUT_SECONDS, open=False) # open = False means create pool but don't connect to PG yet
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=BACKEND_TIMEOUT_SECONDS)
    try:
        try:
            pool.open(wait=True, timeout=BACKEND_TIMEOUT_SECONDS) # Connect pool to PG here
        except PoolTimeout as error: # Raise if PG connection fails
            raise RuntimeError(
                f"Cannot connect to Postgres at {settings.pg_host}:{settings.pg_port} "
                f"(database '{settings.pg_db}'). Is `docker compose up -d` running, and do the "
                "PG_* values in .env match the ones the database was created with?"
            ) from error
        repository = Repository(pool, qdrant, settings.qdrant_item_collection, settings.qdrant_user_collection)
        try:
            run_id = check_index(repository)
        except RuntimeError: # raise RuntimeError in check_index() only stops check_index(), re-raise is needed to stop lifespan()
            raise
        except Exception as error: # Raise if Qdrant connection fails 
            raise RuntimeError(f"Cannot reach Qdrant at {settings.qdrant_url}. "
                               "Is `docker compose up -d` running?") from error
        warn_if_index_is_stale(run_id)
        app.state.service = RecommendationService(repository, model_run_id=run_id) # Build service object
        yield # Server runs here
    finally:
        pool.close()
        qdrant.close()


app = FastAPI(title="MovieLens recommender", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON log line per request, including requests whose handler raised.
    Paths only: no query strings, bodies or secrets."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        logger.info(json.dumps({
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }))
