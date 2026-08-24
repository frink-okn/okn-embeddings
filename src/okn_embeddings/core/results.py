from dataclasses import dataclass

from qdrant_client.models import ScoredPoint


@dataclass
class ResultRow:
    """Interface-agnostic view of a single search hit."""

    id: str
    score: float | None
    iris: list[str]
    primary_uri: str
    iri_count: int
    label: str
    graph: str
    repr: str


@dataclass
class TimedSearchResult:
    """What `run_similarity_search` returns: hits plus wall time."""

    rows: list[ResultRow]
    time: float


def summarize_point(p: ScoredPoint) -> ResultRow:
    """Extract the fields every interface needs from a scored point.

    `iri` is a list of IRIs that share this embedding. It may be truncated
    relative to `iri_count` (the total before the storage cap). The first IRI
    is the representative used for "find similar" and external links.
    """
    payload = p.payload or {}

    iris = payload.get("iri") or []
    if isinstance(iris, str):
        iris = [iris]
    primary = iris[0] if iris else ""

    return ResultRow(
        id=str(p.id),
        score=float(p.score) if p.score is not None else None,
        iris=iris,
        primary_uri=primary,
        iri_count=payload.get("iri_count", len(iris)),
        label=payload.get("label") or "",
        graph=payload.get("graph") or "",
        repr=payload.get("repr") or "",
    )
