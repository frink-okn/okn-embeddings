from dataclasses import dataclass
from typing import TYPE_CHECKING

from cachetools import TTLCache, cached

if TYPE_CHECKING:
    from ..config import AppContext


@dataclass(frozen=True)
class GraphFacet:
    graph: str
    count: int


@cached(
    cache=TTLCache(maxsize=4, ttl=60 * 10),
    key=lambda ctx: id(ctx.backend),
)
def get_graph_facets(ctx: "AppContext") -> list[GraphFacet]:
    return ctx.backend.graph_facets()


def get_graphs(ctx: "AppContext") -> list[str]:
    return [facet.graph for facet in get_graph_facets(ctx)]
