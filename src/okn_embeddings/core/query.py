import time

import numpy as np
from loguru import logger

from ..config import AppContext
from .errors import URINotFoundError
from .models import (
    Feature,
    NodeFeature,
    Query,
    TextFeature,
)
from .results import TimedSearchResult


def get_embedding(
    ctx: AppContext,
    feature: Feature,
) -> np.ndarray:
    """The query vector for a feature: embed text, or look up a node's.

    Node lookup goes through the backend so both interfaces resolve IRIs
    against whatever actually stores the vectors.
    """
    match feature:
        case TextFeature(type="text"):
            return ctx.embedder.embed(feature.value)
        case NodeFeature(type="node"):
            vector = ctx.backend.vector_for_iri(feature.value)
            if vector is None:
                raise URINotFoundError(f"URI not found: {feature.value}")
            return vector
        case _:
            raise ValueError("Unsupported feature type")


def run_similarity_search(
    ctx: AppContext,
    query_obj: Query,
    hnsw_ef: int | None = None,
    exact: bool = False,
) -> TimedSearchResult:
    """The single entry point for similarity search, on any backend."""
    vector = get_embedding(ctx, query_obj.feature)

    start_time = time.perf_counter()
    rows = ctx.backend.search(
        vector,
        limit=query_obj.limit,
        offset=query_obj.offset,
        include_graphs=query_obj.include_graphs,
        exclude_graphs=query_obj.exclude_graphs,
        exact=exact,
        hnsw_ef=hnsw_ef,
    )
    query_time = time.perf_counter() - start_time

    logger.debug(f"{query_time:.3f}s for query: {query_obj}")

    return TimedSearchResult(rows=rows, time=query_time)
