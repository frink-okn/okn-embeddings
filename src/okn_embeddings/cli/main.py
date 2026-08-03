import json
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from ..ann.sidecar import SidecarStore
from ..config import AppContext
from ..config.settings import load_settings
from ..core.embedding import make_embedder
from ..core.errors import friendly_error
from ..core.explore import GraphSurveyResult, run_survey
from ..core.graphs import get_graph_facets
from ..core.models import build_feature, build_query
from ..core.query import run_similarity_search
from ..core.results import ResultRow, summarize_point

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


class FeatureType(str, Enum):
    text = "text"
    node = "node"


class GraphSort(str, Enum):
    NAME = "name"
    COUNT = "count"


def _fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a nonzero status."""
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _print_results_table(rows: list[ResultRow], show_repr: bool) -> None:
    if not rows:
        typer.echo("No results.")
        return

    table = Table()
    table.add_column("Score", justify="right")
    table.add_column("Label")
    table.add_column("Graph")
    table.add_column("IRI", overflow="fold")
    if show_repr:
        table.add_column("Embedding text", overflow="fold")

    for row in rows:
        score = f"{row.score:.4f}" if row.score is not None else ""
        iri = row.primary_uri
        if row.iri_count > 1:
            iri += f"  (+{row.iri_count - 1} more)"
        cells = [score, row.label, row.graph, iri]
        if show_repr:
            cells.append(row.repr)
        table.add_row(*cells)

    Console().print(table)


def _row_to_dict(row: ResultRow, show_repr: bool) -> dict:
    item = {
        "score": row.score,
        "label": row.label,
        "graph": row.graph,
        "primary_uri": row.primary_uri,
        "iris": row.iris,
        "iri_count": row.iri_count,
    }
    if show_repr:
        item["repr"] = row.repr
    return item


def _print_results_json(rows: list[ResultRow], show_repr: bool) -> None:
    typer.echo(json.dumps([_row_to_dict(row, show_repr) for row in rows]))


@app.command()
def search(
    term: Annotated[
        str,
        typer.Argument(help="Query text, or a node IRI when --type node."),
    ],
    feature_type: Annotated[
        FeatureType,
        typer.Option(
            "--type",
            "-t",
            help="Search by free text or by an existing node's IRI.",
        ),
    ] = FeatureType.text,
    graph: Annotated[
        list[str] | None,
        typer.Option(
            "--graph",
            "-g",
            help="Restrict search to these graphs (repeatable).",
        ),
    ] = None,
    exclude_graph: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-graph",
            "-x",
            help="Exclude these graphs (repeatable).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", min=1, help="Maximum results."),
    ] = 10,
    offset: Annotated[
        int,
        typer.Option("--offset", min=0, help="Results offset (pagination)."),
    ] = 0,
    exact: Annotated[
        bool,
        typer.Option("--exact", help="Exact (kNN) search instead of ANN."),
    ] = False,
    show_repr: Annotated[
        bool,
        typer.Option(
            "--show-repr",
            help="Include each result's embedding text.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output JSON instead of a table."),
    ] = False,
):
    """Search the embeddings by text or by an existing node's IRI."""
    try:
        query = build_query(
            feature_type=feature_type.value,
            value=term,
            include_graphs=graph,
            exclude_graphs=exclude_graph,
            limit=limit,
            offset=offset,
        )
    except ValidationError as e:
        msg = "; ".join(err.get("msg", "") for err in e.errors())
        raise typer.BadParameter(msg) from e

    ctx = AppContext.from_env()

    try:
        response = run_similarity_search(ctx, query_obj=query, exact=exact)
    except Exception as e:
        _fail(friendly_error(e))

    rows = [summarize_point(p) for p in response.points]

    if as_json:
        _print_results_json(rows, show_repr)
    else:
        _print_results_table(rows, show_repr)


@app.command()
def local(
    term: Annotated[
        str,
        typer.Argument(help="Query text, or a node IRI when --type node."),
    ],
    paths: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "Embed Parquet files to search. A usearch index beside a "
                "file (<stem>.usearch) is used when present; otherwise "
                "search is an exact scan."
            ),
        ),
    ],
    feature_type: Annotated[
        FeatureType,
        typer.Option(
            "--type",
            "-t",
            help="Search by free text or by an existing node's IRI.",
        ),
    ] = FeatureType.text,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", min=1, help="Maximum results."),
    ] = 10,
    offset: Annotated[
        int,
        typer.Option("--offset", min=0, help="Results offset (pagination)."),
    ] = 0,
    exact: Annotated[
        bool,
        typer.Option("--exact", help="Exact (kNN) search instead of ANN."),
    ] = False,
    show_repr: Annotated[
        bool,
        typer.Option(
            "--show-repr",
            help="Include each result's embedding text.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output JSON instead of a table."),
    ] = False,
):
    """Search local embed Parquet artifacts instead of Qdrant.

    Results from multiple files are merged by score (comparable when the
    files share one embedding model) with each row's graph taken from the
    file's metadata. No server is contacted.
    """
    stores: list[SidecarStore] = []
    for path in paths:
        if not path.exists():
            _fail(f"Input file not found: {path}")
        try:
            stores.append(SidecarStore.open(path))
        except ValueError as e:
            _fail(str(e))

    if feature_type == FeatureType.node:
        vector = None
        for store in stores:
            vector = store.vector_for_iri(term)
            if vector is not None:
                break
        if vector is None:
            _fail(f"IRI not found in any given file: {term}")
    else:
        # The only step that needs the model; node search is model-free.
        vector = make_embedder(load_settings()).embed(term)

    fetch = limit + offset
    rows = [
        row
        for store in stores
        for row in store.search(vector, fetch, exact=exact)
    ]
    rows.sort(key=lambda r: (-(r.score or 0.0), r.id))
    rows = rows[offset : offset + limit]

    if as_json:
        _print_results_json(rows, show_repr)
    else:
        _print_results_table(rows, show_repr)


def _print_survey_table(
    results: list[GraphSurveyResult], show_repr: bool
) -> None:
    if not results:
        typer.echo("No graphs to search.")
        return

    console = Console()
    for r in results:
        rows = [summarize_point(p) for p in r.points]
        best = (
            f"{rows[0].score:.4f}"
            if rows and rows[0].score is not None
            else "—"
        )
        console.print(
            f"[bold]{r.graph}[/bold]  ({len(rows)} hits, best {best})"
        )

        if not rows:
            console.print("  no results\n")
            continue

        table = Table()
        table.add_column("Score", justify="right")
        table.add_column("Label")
        table.add_column("IRI", overflow="fold")
        if show_repr:
            table.add_column("Embedding text", overflow="fold")

        for row in rows:
            score = f"{row.score:.4f}" if row.score is not None else ""
            iri = row.primary_uri
            if row.iri_count > 1:
                iri += f"  (+{row.iri_count - 1} more)"
            cells = [score, row.label, iri]
            if show_repr:
                cells.append(row.repr)
            table.add_row(*cells)

        console.print(table)
        console.print()


def _print_survey_json(
    results: list[GraphSurveyResult], show_repr: bool
) -> None:
    out = [
        {
            "graph": r.graph,
            "results": [
                _row_to_dict(summarize_point(p), show_repr) for p in r.points
            ],
        }
        for r in results
    ]
    typer.echo(json.dumps(out))


@app.command()
def survey(
    term: Annotated[
        str,
        typer.Argument(help="Query text, or a node IRI when --type node."),
    ],
    feature_type: Annotated[
        FeatureType,
        typer.Option(
            "--type",
            "-t",
            help="Search by free text or by an existing node's IRI.",
        ),
    ] = FeatureType.text,
    graph: Annotated[
        list[str] | None,
        typer.Option(
            "--graph",
            "-g",
            help="Limit the survey to these graphs (repeatable).",
        ),
    ] = None,
    exclude_graph: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-graph",
            "-x",
            help="Survey all graphs except these (repeatable).",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", min=1, help="Top results per graph."),
    ] = 5,
    exact: Annotated[
        bool,
        typer.Option("--exact", help="Exact (kNN) search instead of ANN."),
    ] = False,
    show_repr: Annotated[
        bool,
        typer.Option(
            "--show-repr",
            help="Include each result's embedding text.",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output JSON instead of a table."),
    ] = False,
):
    """Search every graph (or a -g/-x subset) for the top matches per graph."""
    if graph and exclude_graph:
        raise typer.BadParameter("Use only one of --graph / --exclude-graph.")

    try:
        feature = build_feature(feature_type.value, term)
    except ValidationError as e:
        msg = "; ".join(err.get("msg", "") for err in e.errors())
        raise typer.BadParameter(msg) from e

    ctx = AppContext.from_env()

    try:
        results = run_survey(
            ctx,
            feature,
            include_graphs=graph,
            exclude_graphs=exclude_graph,
            limit=limit,
            exact=exact,
        )
    except Exception as e:
        _fail(friendly_error(e))

    if as_json:
        _print_survey_json(results, show_repr)
    else:
        _print_survey_table(results, show_repr)


@app.command("list-graphs")
def list_graphs(
    sort: Annotated[
        GraphSort,
        typer.Option(
            "--sort",
            help="Sort by graph name (asc) or point count (desc).",
        ),
    ] = GraphSort.NAME,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output JSON instead of a table."),
    ] = False,
):
    """List the graphs in the collection and their point counts."""
    ctx = AppContext.from_env()

    try:
        facets = get_graph_facets(ctx)
    except Exception as e:
        _fail(friendly_error(e))

    if sort == GraphSort.COUNT:
        facets = sorted(facets, key=lambda f: f.count, reverse=True)
    else:
        facets = sorted(facets, key=lambda f: f.graph)

    if as_json:
        out = [{"graph": f.graph, "count": f.count} for f in facets]
        typer.echo(json.dumps(out))
        return

    if not facets:
        typer.echo("No graphs.")
        return

    table = Table()
    table.add_column("Graph")
    table.add_column("Count", justify="right")
    for f in facets:
        table.add_row(f.graph, f"{f.count:,}")

    Console().print(table)


if __name__ == "__main__":
    app()
