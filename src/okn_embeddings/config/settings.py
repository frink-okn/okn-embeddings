from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is src/okn_embeddings/config/settings.py, so the repo root
# (where .env lives) is four levels up. That only resolves for a source
# checkout; an installed copy has no repo root and falls back to the
# defaults below.
PROJECT_ROOT = Path(__file__).parents[3]
LOCAL_ENV = PROJECT_ROOT / ".env"


class AppSettings(BaseSettings):
    """Runtime configuration, overridable by environment or .env file.

    Defaults live here rather than in a checked-in env file so that an
    installed copy of the package works without one. Precedence is
    environment variables, then .env, then these.
    """

    model_config = SettingsConfigDict(env_file=LOCAL_ENV)

    qdrant_location: str = "http://127.0.0.1:6663"
    qdrant_hnsw_ef: int = 500
    qdrant_collection: str = "OKN-Graph"
    qdrant_timeout: int = 30
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Embedding backend knobs. `embed_device` picks the torch device:
    # `auto` -> CUDA if present, else MPS on Apple Silicon, else CPU; the
    # concrete values `cpu`, `cuda`, `mps` pin. `embed_batch_size` is the
    # encode-time batch handed to torch; larger batches saturate a GPU but
    # trade latency on the query path. `embed_threads` only applies on CPU
    # (caps torch's intra-op thread pool); GPU devices ignore it.
    embed_device: str = "auto"
    embed_batch_size: int = 256
    embed_threads: int | None = None


def load_settings():
    return AppSettings()
