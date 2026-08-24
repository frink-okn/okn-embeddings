from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from ..core.backend import QdrantBackend, SearchBackend, SidecarBackend
from ..core.embedding import Embedder, make_embedder
from ..core.graphs import get_graphs
from .settings import AppSettings, load_settings


@dataclass
class AppContext:
    client: QdrantClient
    embedder: Embedder
    settings: AppSettings
    backend: SearchBackend

    @staticmethod
    def from_env() -> "AppContext":
        return AppContext.from_settings(load_settings())

    @staticmethod
    def from_settings(settings: AppSettings) -> "AppContext":
        # The client is constructed either way (it connects lazily); the
        # backend decides where similarity queries actually run.
        client = QdrantClient(
            location=settings.qdrant_location,
            timeout=settings.qdrant_timeout,
        )
        embedder = make_embedder(settings)

        backend: SearchBackend
        if settings.search_backend == "sidecar":
            paths = sorted(Path().glob(settings.sidecar_paths))
            if not paths:
                raise ValueError(
                    "SEARCH_BACKEND=sidecar but SIDECAR_PATHS matched no "
                    f"files: {settings.sidecar_paths!r}"
                )
            backend = SidecarBackend.from_paths(paths)
        elif settings.search_backend == "qdrant":
            backend = QdrantBackend(client, settings)
        else:
            raise ValueError(
                f"unknown SEARCH_BACKEND {settings.search_backend!r}; "
                "expected 'qdrant' or 'sidecar'"
            )

        return AppContext(
            client=client,
            embedder=embedder,
            settings=settings,
            backend=backend,
        )

    @property
    def graphs(self) -> list[str]:
        return get_graphs(self)
