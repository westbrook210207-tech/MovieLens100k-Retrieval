"""Shared settings for training and the API, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Field `pg_host` is read from env var `PG_HOST` (matching is case-insensitive).
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Paths (project-relative defaults)
    raw_data_dir: Path = PROJECT_ROOT / "ml-100k"  # original MovieLens 100K, read-only
    prepared_data_dir: Path = PROJECT_ROOT / "training" / "data"
    artifact_dir: Path = PROJECT_ROOT / "training" / "artifacts" / "two_tower"

    # Postgres
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "recsys"
    pg_user: str = "recsys"
    # SecretStr keeps the value out of repr()/logs. No real default: set PG_PASSWORD in .env.
    pg_password: SecretStr = SecretStr("")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "items"       # item vectors
    qdrant_user_collection: str = "users"  # precomputed user vectors

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    @property
    def pg_conninfo(self) -> str:
        """libpq connection string. Contains the password: never log it."""
        password = self.pg_password.get_secret_value()
        if not password:
            raise RuntimeError("PG_PASSWORD is not set. Copy .env.example to .env and set it.")
        # make_conninfo quotes and escapes each value, so spaces, quotes and backslashes are safe.
        return make_conninfo(host=self.pg_host, port=self.pg_port, dbname=self.pg_db,
                             user=self.pg_user, password=password)


@lru_cache
def get_settings() -> Settings:
    """Shared Settings instance. lru_cache on a no-argument function runs the body once
    and returns the same object afterwards, so .env is read and validated a single time.
    In tests, call get_settings.cache_clear() after changing environment variables."""
    return Settings()
