import time
from dataclasses import dataclass

from loguru import logger

from ..config import AppContext
from .graphs import get_graphs
from .models import Feature
from .query import get_embedding
from .results import ResultRow


@dataclass
class GraphSurveyResult:
    graph: str
    rows: list[ResultRow]


def resolve_target_graphs(
    all_graphs: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """Resolve which graphs a survey should cover.

    `include` wins if given; otherwise all graphs minus `exclude`; otherwise
    every graph.
    """
    if include:
        return list(include)

    if exclude:
        excluded = set(exclude)
        return [g for g in all_graphs if g not in excluded]

    return list(all_graphs)


def _best_score(rows: list[ResultRow]) -> float:
    # Backends return each graph's rows best-first.
    if rows and rows[0].score is not None:
        return rows[0].score
    return float("-inf")


def run_survey(
    ctx: AppContext,
    feature: Feature,
    *,
    include_graphs: list[str] | None = None,
    exclude_graphs: list[str] | None = None,
    limit: int = 5,
    exact: bool = False,
    hnsw_ef: int | None = None,
) -> list[GraphSurveyResult]:
    """Search each target graph independently for `feature`.

    Embeds the query once and asks the backend for per-graph top hits.
    Returns one `GraphSurveyResult` per graph, sorted by best hit score
    descending.
    """
    all_graphs = get_graphs(ctx) if not include_graphs else []
    graphs = resolve_target_graphs(all_graphs, include_graphs, exclude_graphs)

    if not graphs:
        return []

    query_vector = get_embedding(ctx, feature)

    start_time = time.perf_counter()
    per_graph = ctx.backend.survey(
        query_vector,
        graphs,
        limit=limit,
        exact=exact,
        hnsw_ef=hnsw_ef,
    )
    elapsed = time.perf_counter() - start_time

    logger.debug(f"{elapsed:.3f}s for survey over {len(graphs)} graphs")

    results = [
        GraphSurveyResult(graph=graph, rows=rows) for graph, rows in per_graph
    ]
    results.sort(key=lambda r: _best_score(r.rows), reverse=True)

    return results
