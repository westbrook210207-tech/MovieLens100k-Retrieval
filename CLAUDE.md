# CLAUDE.md — Recommender System Project Instructions

## 1. Project goal

Build a maintainable recommender-system project that supports offline data preparation and model training, offline evaluation, and online recommendation serving through FastAPI.

Prioritize a correct, understandable end-to-end baseline before adding more models, abstractions, or infrastructure. Keep offline training and online serving responsibilities separate.

## 2. Working rules for the coding agent

- Inspect the existing repository, configuration, and data schema before making changes. Preserve working conventions and paths unless there is a clear reason to change them.
- Before substantial changes, state the intended approach and identify assumptions. For small, local changes, proceed directly.
- Prefer simple, readable Python over premature abstractions. Do not add a dependency, service, or framework without a concrete need; explain why it is needed.
- Keep variable names stable and descriptive, especially in notebooks and learning-oriented code.
- Add concise comments for non-obvious logic, unfamiliar syntax, important built-ins, tensor shapes, data transformations, and modeling decisions. Do not comment every obvious line.
- Use type hints and clear function boundaries. Validate external input at system boundaries.
- Add or update tests when changing behavior. Run the relevant tests or checks and report which commands actually ran and whether they passed.
- Never claim a command, test, training run, or evaluation succeeded unless it was actually run and its result was checked.
- After a meaningful implementation, summarize the data flow, key design decisions, and any assumptions or limitations.

## 3. Default architecture

Use these as the default architecture for this project. First check what already exists; do not recreate or reorganize working components without a reason.

- **Python + PyTorch:** data/model training and model evaluation.
- **PostgreSQL:** persistent user/item metadata and interaction records that the application needs to query. Keep relational metadata and interaction facts here when they need filtering, joins, or transactional updates.
- **Qdrant:** persistent vector storage and nearest-neighbor retrieval for item or user embeddings when the selected retrieval model produces vectors. Store the vector together with a stable item identifier and only the payload fields needed for filtering.
- **FastAPI:** HTTP/API boundary between the client application and recommendation services. Keep endpoints thin: validate inputs, call a service layer, and return a typed response. Do not put training loops or large model-building logic inside request handlers.
- **MLflow:** track training experiments, parameters, metrics, artifacts, and model versions where applicable. MLflow experiment tracking is not, by itself, production-service monitoring.
- **Docker Compose:** runs the backing services (PostgreSQL, Qdrant, MLflow) for local development. The API is not containerised: it runs on the host with `python -m uvicorn api.main:app`. Publish service ports on `127.0.0.1` only.

Use service names rather than `localhost` for communication between containers on the same Compose network. Use `localhost` only when a process running on the host machine needs to reach a published container port.

Keep the offline pipeline responsible for loading data, preparing examples, training/evaluating models, and generating/indexing embeddings. Keep the online API responsible for receiving requests, retrieving candidates, fetching metadata, and returning recommendations.

## 4. Repository and data layout

Use this as the intended layout for the project. Inspect the existing repository first and explain any proposed deviation; do not reorganize working files without a concrete reason.

```text
.
├── api/                      # FastAPI application and online recommendation serving
│   ├── main.py                # App, lifespan (connections, startup checks), request logging
│   ├── routes.py              # HTTP endpoints
│   ├── schemas.py             # Pydantic request/response models
│   ├── service.py             # Recommendation orchestration
│   └── repositories.py        # Postgres and Qdrant access
├── db/schema.sql              # Postgres tables; applied by training/index.py on every run
├── ml-100k/                   # Original MovieLens 100K dataset supplied by the user; keep unchanged
├── training/                   # Offline data preparation, training, and evaluation
│   ├── data/                   # Generated/prepared training data, if needed
│   ├── models/                 # Model definitions, if needed
│   ├── train.py                # Training entry point (or a clearly named equivalent)
│   ├── evaluate.py             # Offline evaluation entry point (or a clearly named equivalent)
│   ├── index.py                # Offline retrieval: load Postgres, precompute user/item vectors into Qdrant
│   └── artifacts/              # Exported trained model and required metadata for API loading
├── config.py                   # Shared Pydantic Settings definition and validation
├── .env                        # Local settings/secrets; must remain untracked
├── .env.example                # Required variable names with safe placeholder values
├── .gitignore
├── tests/
├── requirements.txt            # Or the dependency manifest already used by the project
└── compose.yaml                 # Or docker-compose.yml if already used
```

The `api/`, `ml-100k/`, and `training/` directories are the three main project areas. Create subfolders only when they serve a real purpose; do not add layers or boilerplate just to match the tree exactly.

Data and model artifact rules:

- Treat the existing `ml-100k/` directory (setting `RAW_DATA_DIR`) as the location of the user-provided raw MovieLens 100K dataset. Do not modify, overwrite, move, or delete the original dataset files. Store processed data separately under `training/data/` or another explicitly configured output path.
- Keep offline preparation, training, and evaluation in `training/`. After training, export the model and any required mappings, preprocessing objects, or metadata to a documented location under `training/artifacts/`.
- The API is online retrieval only: it serves the user and item vectors that `training/index.py` precomputed into Qdrant. It must not import PyTorch, load the model, retrain, or run the training pipeline. It must fail at startup with a clear error if Postgres or Qdrant is unreachable or the vector index is missing or inconsistent.
- Define data and artifact paths through `config.py` settings or another documented project-relative mechanism. Do not hard-code a personal absolute path.
- Use `config.py` with Pydantic Settings to validate and centralize settings needed by training and the API. Read secret and environment-specific values from environment variables / `.env`; keep defaults safe and non-sensitive. Avoid having independent modules parse `.env` in inconsistent ways.
- Do not commit secrets, private/user data, large raw datasets, generated model weights, or vector-database storage by default. Add appropriate paths to `.gitignore` and document how to obtain or generate excluded files.
- Validate input columns, dtypes, ID mappings, nulls, and duplicate behavior before training.
- Document whether user/item IDs are raw identifiers, zero-based model indices, or database IDs. Do not silently mix these representations.

## 5. Modeling approach, sampling, and evaluation

### Modeling approach

- Follow the modeling approach and training objective specified by the user for the current task. Do not assume or prescribe a particular model family, architecture, or training objective in these project-wide instructions.
- Inspect the existing code, dataset schema, task, and evaluation protocol before proposing model changes. Keep implementation aligned with the user's stated approach.
- If a key modeling decision has not been specified and would materially change the solution, explain the relevant options and trade-offs and recommend a direction rather than silently making a major change.
- Keep candidate generation/retrieval distinct from ranking when the system uses both stages. Make clear which component generates candidates and which component orders them.

### Negative sampling and training data

- Do not assume the current negative-sampling method is optimal. Inspect the dataset, data-collection process, existing implementation, task, and evaluation protocol; recommend improvements when there is a meaningful reason.
- Depending on the available data and objective, possible approaches may include uniform, popularity-aware, exposure-aware, in-batch, or hard/semi-hard negative sampling. Treat these as options to assess, not mandatory choices. Explain relevant trade-offs, including false negatives, popularity bias, leakage risk, and computational cost where applicable.
- Follow the user's explicit direction for the current task. Do not silently replace a user-selected sampling method or make consequential changes to the training formulation; explain the rationale and proposed change first when direction is not clear.
- Do not hard-code universal rating thresholds or one sampling strategy in these project-wide instructions. Treat thresholds and sampling settings as dataset- and experiment-specific decisions, and make them configurable when useful.
- Distinguish explicit negative feedback (for example, an item a user rated poorly) from an unobserved item. Unobserved items are not necessarily disliked; describe them as sampled or weak negatives when appropriate.
- For meaningful sampling changes, compare alternatives as controlled experiments using the same data split and evaluation protocol. Keep runs reproducible with recorded random seeds where randomness is involved.
- Prevent train/validation/test leakage. Do not use held-out labels to construct training examples or tune the sampler. Fit learned preprocessing only on the training partition, and respect event time when a temporal split is appropriate.

### Evaluation

- Use a documented and reproducible train/validation/test split; choose random or temporal splitting based on the intended product/use case.
- Evaluate the ranking task with relevant top-K metrics such as Recall@K, NDCG@K, and Precision@K. Define the candidate set, relevant-item threshold, treatment of already-seen items, and averaging method.
- Compare model variants on the same split and evaluation protocol. Do not select a model using the test set; use validation results for iteration and reserve test results for final assessment.
- Include sanity checks for empty histories, new users/items, duplicate recommendations, invalid IDs, and users with no eligible candidates.
- Do not claim sampled offline metrics guarantee online business performance.

## 6. MLflow and monitoring

- Log the experiment name, run parameters, data/split identifiers, random seed, model configuration, training/validation metrics, and relevant artifacts to MLflow.
- Record enough information to reproduce a run. Avoid logging secrets or personally identifiable user data.
- Keep offline model evaluation separate from online service observability.
- For the API, use structured logs and meaningful error handling. If production monitoring is added, track service health, latency, error rates, and suitable recommendation-quality or data-drift signals; do not describe MLflow alone as complete production monitoring.

## 7. Environment variables and secrets

- Treat `.env` as a local secret/configuration file. It may contain database passwords, API keys, access tokens, secret keys, and other sensitive values. Never commit it to Git or share its contents in chat, documentation, screenshots, or issue reports.
- Ensure `.env` is listed in `.gitignore` (including relevant local variants). If a real secret has already been committed, removing it from the file is not enough; follow the credential-rotation process and clean up repository history when appropriate.
- Keep a `.env.example` file for documenting required variable names. Use safe placeholders such as `PG_PASSWORD=replace_me`; never copy real credentials into it.
- Use the project-root `config.py` with Pydantic Settings to load, validate, and centralize application settings for the API and training code. Read credentials and environment-specific values from environment variables and the local `.env` file rather than hard-coding them in source code. Keep defaults safe and non-sensitive.
- Never hard-code, print, log, or return secret values. When debugging configuration, report variable names and whether they are set, not their sensitive values.
- Distinguish variables used to initialize a container (for example, a database's initial user/password/database) from variables the application itself reads. Document which service consumes each variable.
- Remember that services communicating inside the same Compose network usually use service names as hosts, while host-machine processes use published ports and `localhost` as appropriate.

## 8. FastAPI and data-access boundaries

- Use Pydantic request/response schemas for API contracts and type the response shape explicitly.
- Keep database access in a data-access/repository layer and recommendation orchestration in a service layer; endpoints should coordinate them rather than contain SQL, vector-indexing, or training code.
- Manage connection pools/clients through application lifespan or another explicit lifecycle mechanism. Avoid creating a new database connection for every query without a reason.
- Use parameterized SQL; never build queries by interpolating untrusted input.
- Return intentional HTTP errors (for example, a clear 404 for a genuinely missing user when appropriate, or a validation error for invalid input) rather than hiding failures behind generic success responses.
- Do not load or run a model in the API at all. Initialize reusable clients at startup and read the precomputed embeddings from Qdrant.
- Include a health endpoint and test both successful responses and important failure cases.

## 9. Adding dependencies and managing settings

- Before installing a new package, check whether the standard library or a dependency already in the project can do the job. Avoid adding packages without a concrete reason.
- When a new package is justified, explain what it adds, update the project's dependency manifest (and lock file if used), and run relevant checks.
- Keep two kinds of settings distinct:
  - **Environment/secrets settings:** credentials and environment-specific connection details, supplied through environment variables and local `.env` files.
  - **Experiment settings:** model and training choices worth changing or comparing, such as learning rate, embedding size, regularization, split settings, or a sampling method. Keep these in a clear configuration path or command-line arguments when useful.
- Avoid scattering tunable values as unexplained magic numbers, but do not build configuration switches for hypothetical needs. Add a setting when it supports a real experiment, reproducibility, or deployment requirement.

## 10. Definition of done

A change is done when:

1. The implementation matches the intended architecture and existing repository conventions.
2. Data assumptions, ID mappings, and important model decisions are explicit.
3. Relevant tests or checks have been run, with results reported accurately.
4. New configuration variables are documented and secret values remain untracked.
5. Any meaningful API, data schema, model artifact, or run-command change is documented.
6. The final response summarizes what changed, why, how it was checked, and any known limitation or next step.

## 11. Modeling decisions and remaining project choices

- The user's instructions for the current task determine the modeling approach and training objective. These project instructions do not prescribe a particular model family or objective.
- Negative sampling is not fixed by this document. Inspect the current method and the available data, and recommend alternatives with clear trade-offs when appropriate; do not silently change a user-selected approach.
- Confirm whether Qdrant is required for online vector retrieval or whether a local prototype is sufficient for the current stage.
- Confirm which interaction/item fields the API is allowed to store and return.
- The raw MovieLens 100K dataset is already located in `ml-100k/`; keep it unchanged. Confirm the split strategy, artifact format/path, and commands that have actually been verified to work.
