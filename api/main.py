"""FastAPI application. Run: python -m uvicorn api.main:app

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

# Standard library: dict -> JSON text for log lines, and JSON text -> dict for meta.json.
import json
# Standard library: Python's built-in logging system.
import logging
# Standard library: used only to time how long a request took.
import time
# Turns an `async def` containing one `yield` into an async context manager (see lifespan).
from contextlib import asynccontextmanager

# FastAPI = the application class. Request = the object describing one incoming HTTP request.
from fastapi import FastAPI, Request
# ConnectionPool = a set of reusable Postgres connections. PoolTimeout = the error it raises
# when it cannot provide a connection in time.
from psycopg_pool import ConnectionPool, PoolTimeout
# The client object used to talk to the Qdrant vector database over HTTP.
from qdrant_client import QdrantClient

# Our own modules. Note the direction: main.py imports the other files, none of them import
# main.py. That one-way dependency is what keeps the layers separate.
from api.repositories import Repository
from api.routes import router
from api.service import RecommendationService
from config import get_settings

# CONCEPT: logging setup
#   basicConfig configures the ROOT logger once for the whole process: print every record of
#   level INFO or higher to the terminal.
#   format="%(message)s" means "print only the message, with no timestamp or level prefix".
#   The request log lines written below are already complete JSON strings, so printing them
#   bare keeps every output line valid JSON that a log tool can parse.
#   basicConfig does nothing if the root logger already has a handler, so calling it at import
#   time is safe.
logging.basicConfig(level=logging.INFO, format="%(message)s")
# getLogger("api") returns the logger named "api". Same name -> same object everywhere, so
# routes.py and service.py call getLogger("api") too and all three files share this one.
logger = logging.getLogger("api")

# The longest we wait on Postgres or Qdrant before giving up with an error. It is used three
# times in lifespan(): opening the pool, borrowing a pooled connection, and each Qdrant reply.
# UPPER_CASE is the Python convention for a module-level constant.
BACKEND_TIMEOUT_SECONDS = 5  # fail fast instead of hanging a request on a dead backend


def check_index(repository: Repository) -> str:
    """Startup guard: refuse to serve unless the vector index is complete and consistent.

    User and item vectors must exist and come from the same model run (same size, same
    run_id). Returns that run_id.

    Why it matters: a recommendation score is the similarity between a USER vector and an
    ITEM vector. That number only means something if one trained model produced both.
    Vectors from two different models live in unrelated spaces; comparing them raises no
    error, it just returns confident-looking nonsense. So we check once, before serving.
    """
    # index_info returns a tuple (vector_size, run_id) for one collection, for example
    # (64, "3f2a9c..."), or None if that collection does not exist.
    items = repository.index_info(repository.item_collection)
    users = repository.index_info(repository.user_collection)
    # Four ways the index can be unusable. `or` evaluates left to right and stops at the first
    # True condition ("short-circuit"); that order is what makes the last check safe.
    #   items is None     the item collection is missing (training.index never ran)
    #   users is None     the user collection is missing
    #   items != users    tuples compare element by element, so this is True when the vector
    #                     size differs OR the run_id differs: two different models
    #   items[1] is None  the collection exists but carries no run_id (it is empty, or was
    #                     written by older code). items[1] would crash if items were None,
    #                     but we only reach it when `items is None` was already False.
    if items is None or users is None or items != users or items[1] is None:
        # Raising here aborts startup (see lifespan). The message names both collections,
        # shows what was found, and ends with the command that fixes it: a good error tells
        # the reader what to do next.
        # Syntax: adjacent string literals are joined by Python into one string. The f prefix
        # is only needed on the pieces that contain {placeholders}.
        raise RuntimeError(
            f"Qdrant vector index is missing or inconsistent: '{repository.item_collection}' "
            f"(size, run_id)={items}, '{repository.user_collection}' (size, run_id)={users}. "
            "Train a model, then run `python -m training.index`."
        )
    # items == users at this point, so either tuple holds the shared run_id.
    # Index 1 is the second element of (size, run_id).
    return items[1]


def warn_if_index_is_stale(served_run_id: str) -> None:
    """The API serves whatever was last indexed. If a newer model has been exported but
    not indexed, say so (reads meta.json only; the model itself is never loaded here)."""
    # get_settings() returns the one shared Settings object (config.py).
    # artifact_dir is a pathlib.Path, and Path overloads the `/` operator to mean "join":
    # this builds training/artifacts/two_tower/meta.json without string concatenation.
    meta_path = get_settings().artifact_dir / "meta.json"
    # A missing meta.json is a normal situation (for example, the API runs on a machine that
    # never trained), so leave quietly. A bare `return` simply ends a `-> None` function.
    if not meta_path.exists():
        return
    # read_text() gives the file content as a str; json.loads() parses that into a dict.
    # .get("run_id") returns None when the key is absent, where ["run_id"] would raise KeyError.
    exported_run_id = json.loads(meta_path.read_text()).get("run_id")
    # Different IDs: training exported a newer model, but `training.index` was not re-run,
    # so Qdrant still holds the previous model's vectors.
    if exported_run_id != served_run_id:
        # A warning, not an error. The old index is still complete and consistent, so serving
        # it is correct, only outdated. Refusing to start would turn "slightly old
        # recommendations" into "no service at all".
        # json.dumps(dict) produces a one-line JSON string, the same shape as the request logs.
        logger.warning(json.dumps({
            "event": "stale_index", "serving_run_id": served_run_id, "exported_run_id": exported_run_id,
            "hint": "run `python -m training.index` to serve the exported model",
        }))


# CONCEPT: decorator
#   `@something` above a function means: lifespan = something(lifespan). The function is
#   passed to `something`, and the name now refers to whatever came back.
#
# CONCEPT: lifespan = startup and shutdown in one function
#   FastAPI needs code that runs ONCE before the first request (open connections) and ONCE
#   when the server stops (close them). It takes both as a single async context manager,
#   which it uses like this:
#
#       async with lifespan(app):     # runs the body up to `yield`      = startup
#           ...serve requests...      # the function stays paused at `yield`
#                                     # then runs the body after `yield` = shutdown
#
#   @asynccontextmanager builds that context manager from a function with exactly one `yield`.
#   `yield` is what makes the pause possible: unlike `return`, it suspends the function and
#   keeps its local variables (pool, qdrant) alive, so the shutdown half can still use them.
#
#   If the startup half raises, the app never starts: uvicorn prints the error and exits.
#   That is deliberate. A server that starts without its database only fails later, one
#   confusing request at a time.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # The one shared, validated Settings object (config.py): hosts, ports, collection names.
    settings = get_settings()

    # CONCEPT: connection pool
    #   Opening a Postgres connection is slow (network handshake + login), much slower than a
    #   small query. A pool opens a few connections once and lends one to each request, which
    #   hands it back when it is done.
    #     settings.pg_conninfo  the "host=... dbname=... user=... password=..." string. It
    #                           contains the password, so it is never logged or printed.
    #     min_size=1            keep at least 1 connection open and ready
    #     max_size=5            never more than 5; a 6th simultaneous query waits for a free one
    #     timeout=...           ...and waits at most this long before raising PoolTimeout
    #     open=False            only build the object; do not connect yet (reason just below)
    # open=False + open(wait=True): the default opens in the background, so a wrong
    # password or a stopped database would only surface later, as hanging requests.
    pool = ConnectionPool(settings.pg_conninfo, min_size=1, max_size=5,
                          timeout=BACKEND_TIMEOUT_SECONDS, open=False)
    # The Qdrant client. timeout = seconds to wait for each reply. Whether Qdrant is actually
    # reachable is found out at the first real call, which is check_index() below.
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=BACKEND_TIMEOUT_SECONDS)
    # Outer try/finally: from here on, whatever happens (an error during startup, or a normal
    # shutdown hours later), the `finally` at the bottom closes both clients.
    try:
        # Inner try/except: catch one specific failure and replace it with a clearer message.
        try:
            # Connect now. wait=True blocks until the min_size connection really exists, or
            # raises PoolTimeout after `timeout` seconds.
            # A blocking call inside `async def` is normally a mistake because it freezes the
            # event loop (see CONCEPT: async / await further down). Here it is fine: startup
            # runs before any request exists, so there is nothing to freeze.
            pool.open(wait=True, timeout=BACKEND_TIMEOUT_SECONDS)
        # `as error` binds the caught exception object to the name `error`.
        except PoolTimeout as error:
            # PoolTimeout alone says "no connection in time", not why. Raise a RuntimeError
            # that names the host, port and database and lists the two usual causes. The
            # password is left out on purpose: error text ends up in logs.
            # CONCEPT: exception chaining
            #   `raise New(...) from error` stores the original exception on the new one (as
            #   __cause__) and Python prints both tracebacks. You get the friendly message
            #   AND the low-level detail, instead of losing one of them.
            raise RuntimeError(
                f"Cannot connect to Postgres at {settings.pg_host}:{settings.pg_port} "
                f"(database '{settings.pg_db}'). Is `docker compose up -d` running, and do the "
                "PG_* values in .env match the ones the database was created with?"
            ) from error
        # The single object that talks to both stores (repositories.py). It receives the pool
        # and client built above plus the two collection names; it opens nothing itself.
        repository = Repository(pool, qdrant, settings.qdrant_collection, settings.qdrant_user_collection)
        try:
            # The first real Qdrant call. Verifies the index and returns the model's run_id.
            run_id = check_index(repository)
        # CONCEPT: order of except clauses
        #   Python tries them top to bottom and runs the first one that matches. RuntimeError
        #   is a subclass of Exception, so the specific clause has to come first; the other
        #   way round, `except Exception` would catch everything and this one would never run.
        # RuntimeError is what check_index raises itself, and its message already explains the
        # problem. A bare `raise` re-raises the same exception unchanged.
        except RuntimeError:
            raise
        # Anything else came out of the Qdrant client (connection refused, timeout, ...),
        # meaning the server could not be reached at all. Wrap it in a message that says so
        # and chain the original with `from error`, as above.
        except Exception as error:
            raise RuntimeError(f"Cannot reach Qdrant at {settings.qdrant_url}. "
                               "Is `docker compose up -d` running?") from error
        # Only logs a warning; it never stops startup.
        warn_if_index_is_stale(run_id)
        # CONCEPT: app.state
        #   A plain attribute holder on the app, for objects that must live as long as the
        #   server. ONE RecommendationService is created here; routes.py reads it back on each
        #   request through get_service() -> request.app.state.service. Nothing is rebuilt
        #   per request. run_id is kept so that /health can report which model is served.
        app.state.service = RecommendationService(repository, model_run_id=run_id)
        # The user_id upper bound is only known now, so write it into the generated OpenAPI
        # document by hand; Swagger UI (/docs) then shows "maximum: 943". app.openapi() caches
        # the document on the app, so this edit is what every later /openapi.json returns.
        # The check itself is in routes.py; this line is documentation only.
        for parameter in app.openapi()["paths"]["/users/{user_id}/recommendations"]["get"]["parameters"]:
            if parameter["name"] == "user_id":
                parameter["schema"]["maximum"] = app.state.service.max_user_id
        # Startup is finished. The function pauses on this line and FastAPI starts accepting
        # requests. It resumes below only when the server is shutting down (for example Ctrl+C).
        yield
    # `finally` runs on both paths: an exception during startup, or the normal resume after
    # `yield` at shutdown. Either way the connections are released instead of leaked.
    finally:
        # Close every Postgres connection held by the pool.
        pool.close()
        # Close the HTTP connections held by the Qdrant client.
        qdrant.close()


# The application object. `uvicorn api.main:app` means "import the module api.main and serve
# the variable named app". title is shown on the auto-generated docs page (/docs).
# lifespan=lifespan hands FastAPI the function above; FastAPI calls it, we never do.
app = FastAPI(title="MovieLens recommender", lifespan=lifespan)
# Attach the endpoints declared on `router` in routes.py: /health and
# /users/{user_id}/recommendations.
app.include_router(router)


# CONCEPT: middleware
#   A function wrapped around EVERY request, whichever endpoint it is for:
#
#       request  -> log_requests -> endpoint in routes.py
#       response <- log_requests <- endpoint in routes.py
#
#   It is the place for work that applies to all endpoints (logging, timing), so that no
#   endpoint has to repeat it. @app.middleware("http") registers the function below with the
#   app for all HTTP requests (decorator syntax again).
#
# CONCEPT: async / await
#   The server has one "event loop" thread that juggles many requests. An `async def`
#   function can pause at an `await` and hand the loop back, so other requests make progress
#   while this one waits. `await call_next(request)` means "run the rest of the app for this
#   request and wake me when the response is ready".
#   The rule that follows: never do slow blocking work directly inside `async def`, because
#   nothing else can run in the meantime. That is why the endpoints in routes.py are plain
#   `def`. FastAPI runs plain `def` endpoints in a worker thread, so their blocking Postgres
#   and Qdrant calls do not freeze the loop.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON log line per request, including requests whose handler raised.
    Paths only: no query strings, bodies or secrets."""
    # Start a stopwatch. perf_counter() is a clock made for measuring durations: it only
    # moves forward. time.time() is wall-clock time and can jump when the system clock is
    # corrected, which would give a wrong (even negative) duration.
    start = time.perf_counter()
    # Assume the worst before trying. If the endpoint raises an unexpected exception,
    # `call_next` raises too and the line `status = response.status_code` never runs. The
    # client then receives a 500 from FastAPI, so 500 is what the log line must say.
    status = 500
    try:
        # Hand the request to the next layer (routing, then the endpoint) and wait for the
        # response. Expected errors (404 unknown user, 422 invalid input, 503 unhealthy) have
        # already been turned into normal responses by FastAPI when they arrive here, so they
        # do NOT raise; only unexpected exceptions do.
        response = await call_next(request)
        # Reached only when a response came back: record its real status code.
        status = response.status_code
        # Pass the response on to the client unchanged.
        return response
    # `finally` runs in both cases: after `return response` has been decided, and when an
    # exception is passing through. That gives exactly one log line per request either way.
    # There is no `except`, so an exception is not swallowed: it continues upward after the
    # log line is written and FastAPI answers with the 500.
    finally:
        # logger.info(...) emits at INFO level; json.dumps turns the dict into one JSON line.
        logger.info(json.dumps({
            # GET, POST, ...
            "method": request.method,
            # .path leaves out the query string ("?k=5"). Query strings and bodies can carry
            # personal data or tokens, and logs are widely readable, so they are not logged.
            "path": request.url.path,
            "status": status,
            # Elapsed seconds * 1000 = milliseconds, rounded to 1 decimal place.
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }))
