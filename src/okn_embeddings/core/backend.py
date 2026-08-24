"""Search backends: where similarity queries actually run.

`run_similarity_search` (core/query.py) is the single entry point for all
three interfaces; this module is the seam underneath it. A backend answers
vector searches over one collection of graphs -- either a Qdrant server or
a set of local sidecar artifacts -- and returns interface-agnostic
`ResultRow`s, so nothing above this layer knows which one is in use.

The Qdrant path preserves the previous behavior exactly (same filters,
same search params, same batched survey). The sidecar path serves the
same queries from embed Parquet files + usearch indexes on local disk,
with no server involved: graph include/exclude reduces to choosing which
stores to query, since each file is one graph.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    QuantizationSearchParams,
    QueryRequest,
    SearchParams,
)

from ..ann.sidecar import SidecarStore
from .graphs import GraphFacet
from .results import ResultRow, summarize_point

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class SearchBackend(Protocol):
    """Vector search over a collection of graphs."""

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        offset: int = 0,
        include_graphs: list[str] | None = None,
        exclude_graphs: list[str] | None = None,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[ResultRow]: ...

    def survey(
        self,
        vector: np.ndarray,
        graphs: list[str],
        *,
        limit: int,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[tuple[str, list[ResultRow]]]:
        """Top-`limit` hits per graph, in the given graph order."""
        ...

    def vector_for_iri(self, iri: str) -> np.ndarray | None: ...

    def graph_facets(self) -> list[GraphFacet]: ...


# --- Qdrant ---------------------------------------------------------------


def build_graph_filter(
    include_graphs: list[str] | None,
    exclude_graphs: list[str] | None,
) -> Filter | None:
    """Build a Qdrant filter on the `graph` payload field.

    Include and exclude are mutually exclusive (enforced upstream by
    `Query`).
    """
    if include_graphs:
        return Filter(
            must=[
                FieldCondition(key="graph", match=MatchAny(any=include_graphs))
            ]
        )

    if exclude_graphs:
        return Filter(
            must_not=[
                FieldCondition(key="graph", match=MatchAny(any=exclude_graphs))
            ]
        )

    return None


class QdrantBackend:
    """Search served by a Qdrant collection."""

    def __init__(self, client: QdrantClient, settings: "AppSettings"):
        self.client = client
        self.settings = settings

    def _search_params(self, hnsw_ef: int | None, exact: bool) -> SearchParams:
        return SearchParams(
            hnsw_ef=(
                self.settings.qdrant_hnsw_ef if hnsw_ef is None else hnsw_ef
            ),
            exact=exact,
            quantization=QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=3.0,
            ),
        )

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        offset: int = 0,
        include_graphs: list[str] | None = None,
        exclude_graphs: list[str] | None = None,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[ResultRow]:
        resp = self.client.query_points(
            query=vector.tolist(),
            collection_name=self.settings.qdrant_collection,
            query_filter=build_graph_filter(include_graphs, exclude_graphs),
            with_payload=True,
            limit=limit,
            offset=offset,
            search_params=self._search_params(hnsw_ef, exact),
            timeout=self.settings.qdrant_timeout,
        )
        return [summarize_point(p) for p in resp.points]

    def survey(
        self,
        vector: np.ndarray,
        graphs: list[str],
        *,
        limit: int,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[tuple[str, list[ResultRow]]]:
        # One filtered query per graph, batched so Qdrant parallelizes
        # them server-side.
        params = self._search_params(hnsw_ef, exact)
        requests = [
            QueryRequest(
                query=vector.tolist(),
                filter=build_graph_filter([graph], None),
                limit=limit,
                with_payload=True,
                params=params,
            )
            for graph in graphs
        ]
        responses = self.client.query_batch_points(
            collection_name=self.settings.qdrant_collection,
            requests=requests,
            timeout=self.settings.qdrant_timeout,
        )
        return [
            (graph, [summarize_point(p) for p in resp.points])
            for graph, resp in zip(graphs, responses, strict=True)
        ]

    def vector_for_iri(self, iri: str) -> np.ndarray | None:
        points, _ = self.client.scroll(
            collection_name=self.settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="iri", match=MatchValue(value=iri))]
            ),
            limit=1,
            with_vectors=True,
        )
        if not points:
            return None
        return np.array(points[0].vector, dtype=np.float32)

    def graph_facets(self) -> list[GraphFacet]:
        res = self.client.facet(
            collection_name=self.settings.qdrant_collection,
            key="graph",
            limit=100,
        )
        return [
            GraphFacet(str(hit.value).strip(), hit.count) for hit in res.hits
        ]


# --- Sidecar --------------------------------------------------------------


class SidecarBackend:
    """Search served by local embed Parquet + usearch sidecar files.

    Each store is one graph, so graph include/exclude is store selection
    rather than a filter. Results from multiple stores merge by score,
    which is meaningful because opening validates that every store shares
    one embedding convention.
    """

    def __init__(self, stores: list[SidecarStore]):
        if not stores:
            raise ValueError("sidecar backend needs at least one store")
        conventions = {s.metadata.get("convention_id") for s in stores}
        if len(conventions) > 1:
            listed = sorted(map(str, conventions))
            raise ValueError(
                "sidecar stores span multiple embedding conventions; "
                f"their scores are not comparable: {listed}"
            )
        self.stores = {store.graph: store for store in stores}

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "SidecarBackend":
        return cls([SidecarStore.open(path) for path in paths])

    def _selected(
        self,
        include_graphs: list[str] | None,
        exclude_graphs: list[str] | None,
    ) -> list[SidecarStore]:
        if include_graphs:
            return [self.stores[g] for g in include_graphs if g in self.stores]
        excluded = set(exclude_graphs or [])
        return [s for g, s in self.stores.items() if g not in excluded]

    def _store_search(
        self,
        store: SidecarStore,
        vector: np.ndarray,
        k: int,
        exact: bool,
        hnsw_ef: int | None,
    ) -> list[ResultRow]:
        if hnsw_ef is not None and store.index is not None:
            store.index.expansion_search = hnsw_ef
        return store.search(vector, k, exact=exact)

    def search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        offset: int = 0,
        include_graphs: list[str] | None = None,
        exclude_graphs: list[str] | None = None,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[ResultRow]:
        fetch = limit + offset
        rows = [
            row
            for store in self._selected(include_graphs, exclude_graphs)
            for row in self._store_search(store, vector, fetch, exact, hnsw_ef)
        ]
        rows.sort(key=lambda r: (-(r.score or 0.0), r.id))
        return rows[offset : offset + limit]

    def survey(
        self,
        vector: np.ndarray,
        graphs: list[str],
        *,
        limit: int,
        exact: bool = False,
        hnsw_ef: int | None = None,
    ) -> list[tuple[str, list[ResultRow]]]:
        return [
            (
                graph,
                self._store_search(
                    self.stores[graph], vector, limit, exact, hnsw_ef
                )
                if graph in self.stores
                else [],
            )
            for graph in graphs
        ]

    def vector_for_iri(self, iri: str) -> np.ndarray | None:
        for store in self.stores.values():
            vector = store.vector_for_iri(iri)
            if vector is not None:
                return vector
        return None

    def graph_facets(self) -> list[GraphFacet]:
        return [
            GraphFacet(graph, store.count)
            for graph, store in sorted(self.stores.items())
        ]
