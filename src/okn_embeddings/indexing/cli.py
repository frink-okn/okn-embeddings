import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from loguru import logger

from ..ann.build import VECTOR_DTYPES, build_index
from ..ann.eval import (
    DEFAULT_EFS,
    DEFAULT_KS,
    DEFAULT_QUERIES,
    DEFAULT_SEED,
    evaluate_and_record,
)
from ..ann.sketch import MAX_K, build_sketch
from ..config import AppContext
from ..config.settings import load_settings
from ..core.embedding import make_embedder
from ..core.errors import friendly_error
from .embed import embed_file
from .manifest import build_manifest, manifest_path, write_manifest
from .models import MaterializationConfiguration
from .output import write_json, write_jsonl, write_text
from .parallel import materialize_records_to_path
from .reader import load_graph
from .sample import (
    sample_targets,
    sample_types,
    write_sample_targets_json,
    write_sample_targets_text,
    write_sample_types_json,
    write_sample_types_text,
)
from .textify import materialize_records
from .upload import upload_files

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.callback()
def main():
    pass


def _fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a nonzero status."""
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


@app.command()
def textify(
    hdt_file: Annotated[
        Path,
        typer.Argument(help="Input HDT graph file."),
    ],
    config_toml: Annotated[
        Path,
        typer.Argument(help="Indexing TOML config."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path."),
    ],
    text: Annotated[
        bool,
        typer.Option(
            "--text",
            help="Write debug text output instead of JSON.",
        ),
    ] = False,
    jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl",
            help="Write JSON Lines output, one grouped record per line.",
        ),
    ] = False,
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help="Name of one target from the config to materialize.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of root nodes to process per target.",
        ),
    ] = None,
    max_iris_per_record: Annotated[
        int,
        typer.Option(
            "--max-iris-per-record",
            min=1,
            help="Maximum source IRIs to keep for each grouped output record.",
        ),
    ] = 10,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            min=0,
            help=(
                "Worker processes for materialization. 1 = single process; "
                "0 = one per CPU. Parallel runs stream output via temp shards, "
                "bounding memory; their record order differs from --jobs 1."
            ),
        ),
    ] = 1,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            min=1,
            help="Root IRIs handed to each worker task (with --jobs != 1).",
        ),
    ] = 1000,
    progress: Annotated[
        bool,
        typer.Option("--progress", help="Show a progress bar."),
    ] = False,
    manifest: Annotated[
        bool,
        typer.Option(
            "--manifest/--no-manifest",
            help=(
                "Also write a provenance manifest (<output stem>.meta.json) "
                "beside the output, recording the graph, config, and run "
                "bounds that produced it."
            ),
        ),
    ] = False,
):
    if text and jsonl:
        raise typer.BadParameter("--text and --jsonl cannot be used together")

    config = MaterializationConfiguration.from_toml(config_toml)

    if jobs == 1:
        graph = load_graph(hdt_file)
        records = materialize_records(
            graph,
            config,
            target=target,
            limit=limit,
            max_iris_per_record=max_iris_per_record,
            progress=progress,
        )
        count = len(records)
        if text:
            write_text(records, output)
        elif jsonl:
            write_jsonl(records, output)
        else:
            write_json(records, output)
    else:
        count = materialize_records_to_path(
            hdt_file,
            config_toml,
            config,
            output,
            text=text,
            jsonl=jsonl,
            target=target,
            limit=limit,
            jobs=jobs,
            chunk_size=chunk_size,
            progress=progress,
            max_iris_per_record=max_iris_per_record,
        )

    typer.echo(f"Wrote {count} records to {output}")

    if manifest:
        data = build_manifest(
            hdt_file,
            config_toml,
            config,
            target=target,
            limit=limit,
            max_iris_per_record=max_iris_per_record,
            record_count=count,
        )
        write_manifest(manifest_path(output), data)
        typer.echo(f"Wrote manifest to {manifest_path(output)}")


@app.command("sample-types")
def sample_types_cmd(
    hdt_file: Annotated[
        Path,
        typer.Argument(help="Input HDT graph file."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path."),
    ],
    text: Annotated[
        bool,
        typer.Option(
            "--text",
            help="Write debug text output instead of JSON.",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of subject IRIs to sample per type.",
        ),
    ] = 5,
    values_limit: Annotated[
        int,
        typer.Option(
            "--values-limit",
            min=1,
            help="Maximum literal examples to include per predicate.",
        ),
    ] = 3,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Sampling seed. Defaults to one derived from the graph "
                "itself, so repeated runs report the same subjects; pass a "
                "different value to draw a different sample."
            ),
        ),
    ] = None,
):
    graph = load_graph(hdt_file)
    records = sample_types(
        graph,
        limit=limit,
        values_limit=values_limit,
        seed=seed,
    )

    if text:
        write_sample_types_text(records, output)
    else:
        write_sample_types_json(records, output)

    typer.echo(f"Wrote {len(records)} type samples to {output}")


@app.command("sample-targets")
def sample_targets_cmd(
    hdt_file: Annotated[
        Path,
        typer.Argument(help="Input HDT graph file."),
    ],
    config_toml: Annotated[
        Path,
        typer.Argument(help="Indexing TOML config."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path."),
    ],
    text: Annotated[
        bool,
        typer.Option(
            "--text",
            help="Write debug text output instead of JSON.",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of root nodes to sample per target.",
        ),
    ] = 5,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Sampling seed. Defaults to one derived from the graph "
                "itself, so repeated runs sample the same roots; pass a "
                "different value to draw a different sample."
            ),
        ),
    ] = None,
):
    graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)
    records = sample_targets(graph, config, limit=limit, seed=seed)

    if text:
        write_sample_targets_text(records, output)
    else:
        write_sample_targets_json(records, output)

    typer.echo(f"Wrote {len(records)} target samples to {output}")


@app.command()
def embed(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "JSONL embedding-record files to embed. Each file is one "
                "graph, named by its stem (e.g. my-graph.jsonl -> my-graph)."
            ),
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Directory for the output Parquet files (default: next to "
                "each input)."
            ),
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of records to embed at once.",
        ),
    ] = 256,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum records to embed per input file.",
        ),
    ] = None,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show a per-graph record progress bar.",
        ),
    ] = True,
):
    """Embed materialized JSONL records into self-described Parquet files.

    Writes one `<graph>.parquet` per input: the record fields plus a `vector`
    column, with the model identity in the file metadata. Downstream steps
    (Qdrant upload, ANN index builds) consume this artifact instead of
    re-embedding.
    """
    settings = load_settings()
    # Deliberately not AppContext.from_env(): embedding needs no Qdrant
    # connection, only the model.
    embedder = make_embedder(settings)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        if not path.exists():
            _fail(f"Input file not found: {path}")
        output = (output_dir or path.parent) / f"{path.stem}.parquet"
        try:
            count = embed_file(
                embedder,
                path,
                output,
                model_name=settings.model_name,
                batch_size=batch_size,
                limit=limit,
                progress_enabled=progress,
            )
        except ValueError as e:
            _fail(str(e))
        typer.echo(f"Wrote {count} records to {output}")


@app.command("build-index")
def build_index_cmd(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "Embed Parquet files to index. Each gets a usearch index "
                "(<stem>.usearch) and a manifest (<stem>.usearch.meta.json) "
                "written beside it."
            ),
        ),
    ],
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            help=(
                "usearch storage type: f32 keeps full precision; f16 and i8 "
                "quantize on ingest for a smaller index."
            ),
        ),
    ] = "f32",
    connectivity: Annotated[
        int | None,
        typer.Option(
            "--connectivity",
            min=2,
            help="HNSW connectivity (m). usearch's default if unset.",
        ),
    ] = None,
    expansion_add: Annotated[
        int | None,
        typer.Option(
            "--expansion-add",
            min=1,
            help=(
                "HNSW build-time beam width (ef_construction). usearch's "
                "default if unset."
            ),
        ),
    ] = None,
    expansion_search: Annotated[
        int | None,
        typer.Option(
            "--expansion-search",
            min=1,
            help=(
                "Default query-time beam width (ef_search) stored in the "
                "index; also overridable at query time. usearch's default "
                "if unset."
            ),
        ),
    ] = None,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show a per-file vector progress bar.",
        ),
    ] = True,
):
    """Build a usearch ANN index beside each embed Parquet artifact.

    The index keys are Parquet row ordinals, so a search hit resolves to its
    record by reading that row of the Parquet file. The manifest records the
    build parameters and the parent file's sha256; the index is rebuildable
    from the Parquet at any time, with different parameters if needed.
    """
    if dtype not in VECTOR_DTYPES:
        raise typer.BadParameter(
            f"dtype must be one of {', '.join(VECTOR_DTYPES)}"
        )

    for path in inputs:
        if not path.exists():
            _fail(f"Input file not found: {path}")
        try:
            index_file, manifest_file, count = build_index(
                path,
                dtype=dtype,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
                progress_enabled=progress,
            )
        except ValueError as e:
            _fail(f"{path}: {e}")
        typer.echo(f"Indexed {count} vectors into {index_file}")
        typer.echo(f"Wrote manifest to {manifest_file}")


@app.command("eval-index")
def eval_index_cmd(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help="Embed Parquet files whose usearch indexes to evaluate.",
        ),
    ],
    queries: Annotated[
        int,
        typer.Option(
            "--queries",
            min=1,
            help="Stored vectors to sample as queries.",
        ),
    ] = DEFAULT_QUERIES,
    k: Annotated[
        list[int] | None,
        typer.Option(
            "--k",
            min=1,
            help="Recall@k values to measure (repeatable).",
        ),
    ] = None,
    ef: Annotated[
        list[int] | None,
        typer.Option(
            "--ef",
            min=1,
            help="expansion_search values to sweep (repeatable).",
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Query sampling seed."),
    ] = DEFAULT_SEED,
    write: Annotated[
        bool,
        typer.Option(
            "--write/--no-write",
            help="Record the evaluation in the index manifest.",
        ),
    ] = True,
):
    """Measure index recall against exact search, and record it.

    Samples stored vectors as queries, computes exact ground truth by brute
    force over the Parquet vectors, and reports recall@k across an
    expansion_search sweep, plus per-query timings for both paths. The
    result is written into the index manifest's `evaluation` block, so the
    artifact itself says what "approximate" means for it.
    """
    ks = k or list(DEFAULT_KS)
    efs = ef or list(DEFAULT_EFS)

    for path in inputs:
        if not path.exists():
            _fail(f"Input file not found: {path}")
        try:
            block = evaluate_and_record(
                path,
                queries=queries,
                ks=ks,
                efs=efs,
                seed=seed,
                write=write,
            )
        except ValueError as e:
            _fail(f"{path}: {e}")

        flat_ms = block["flat"]["mean_query_ms"]
        typer.echo(
            f"{path}: {block['queries']} queries, flat scan {flat_ms} ms/query"
        )
        for step in block["sweep"]:
            recalls = "  ".join(
                f"recall@{k_}={value:.4f}"
                for k_, value in step["recall"].items()
            )
            typer.echo(
                f"  ef={step['expansion_search']:<5} {recalls}  "
                f"{step['mean_query_ms']} ms/query"
            )
        if write:
            typer.echo("Recorded evaluation in the index manifest")


@app.command("build-sketch")
def build_sketch_cmd(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "Embed Parquet files to sketch. Each gets a routing sketch "
                "(<stem>.sketch.parquet) written beside it."
            ),
        ),
    ],
    k: Annotated[
        int | None,
        typer.Option(
            "--k",
            min=1,
            max=MAX_K,
            help=(
                "Number of centroids. Unset picks K automatically from "
                "the radius-vs-K curve."
            ),
        ),
    ] = None,
    seed: Annotated[
        int,
        typer.Option("--seed", help="k-means sampling seed."),
    ] = 42,
    radius_quantile: Annotated[
        float,
        typer.Option(
            "--radius-quantile",
            min=0.0,
            max=1.0,
            help=(
                "Quantile radius recorded per cluster alongside the max, "
                "used for yield ranking."
            ),
        ),
    ] = 0.9,
):
    """Build a routing sketch beside each embed Parquet artifact.

    The sketch answers "is this graph worth an ANN probe for this query"
    from a few hundred KB: a certified no when the query provably clears
    every cluster, and a yield ranking otherwise. Only comparable across
    graphs sharing the parent's convention_id.
    """
    for path in inputs:
        if not path.exists():
            _fail(f"Input file not found: {path}")
        try:
            sketch_file, clusters, unsaturated = build_sketch(
                path,
                k=k,
                seed=seed,
                radius_quantile=radius_quantile,
            )
        except ValueError as e:
            _fail(f"{path}: {e}")
        suffix = " (unsaturated: every vector is its own centroid)"
        typer.echo(
            f"Sketched {clusters} centroids into {sketch_file}"
            + (suffix if unsaturated else "")
        )


@app.command()
def upload(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help=(
                "JSONL embedding-record files to upload. Each file is one "
                "graph, named by its stem (e.g. my-graph.jsonl -> my-graph)."
            ),
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of records to embed at once.",
        ),
    ] = 256,
    upload_batch_size: Annotated[
        int,
        typer.Option(
            "--upload-batch-size",
            min=1,
            help="Number of embedded points to upsert at once.",
        ),
    ] = 256,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum records to upload per input file.",
        ),
    ] = None,
    create_collection: Annotated[
        bool,
        typer.Option(
            "--create-collection/--no-create-collection",
            help="Create the configured collection if it does not exist.",
        ),
    ] = False,
    payload_indexes: Annotated[
        bool,
        typer.Option(
            "--payload-indexes/--no-payload-indexes",
            help="Create keyword payload indexes for graph and iri.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Read and embed records without writing to Qdrant.",
        ),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show a per-graph record progress bar.",
        ),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level."),
    ] = "INFO",
    log_every: Annotated[
        int,
        typer.Option(
            "--log-every",
            min=1,
            help="Log an upload progress line after this many records.",
        ),
    ] = 10_000,
):
    """Embed materialized JSONL records and upsert them into Qdrant."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    ctx = AppContext.from_env()
    try:
        total = upload_files(
            ctx,
            [path.resolve() for path in inputs],
            batch_size=batch_size,
            upload_batch_size=upload_batch_size,
            limit=limit,
            create_collection=create_collection,
            payload_indexes=payload_indexes,
            dry_run=dry_run,
            progress_enabled=progress,
            log_every=log_every,
        )
    except FileNotFoundError as e:
        _fail(f"Input file not found: {e}")
    except Exception as e:
        _fail(friendly_error(e))

    action = "Prepared" if dry_run else "Uploaded"
    typer.echo(
        f"{action} {total} points into {ctx.settings.qdrant_collection} "
        f"at {ctx.settings.qdrant_location}"
    )


@app.command("download-model")
def download_model():
    """Ensure the configured embedding model is in the local cache.

    Building the embedder fetches the model when it is missing, so this is a
    no-op once the cache is warm. It exists so an image build can prime the
    cache in its own layer, instead of every container downloading the model
    the first time it embeds anything. Deliberately avoids AppContext, which
    would open a Qdrant connection this has no use for.
    """
    settings = load_settings()

    try:
        make_embedder(settings)
    except Exception as e:
        _fail(f"Could not download model {settings.model_name}: {e}")

    typer.echo(f"Model {settings.model_name} is available")


if __name__ == "__main__":
    app()
