import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from loguru import logger

from ..config import AppContext
from ..config.settings import load_settings
from ..core.embedding import make_embedder
from ..core.errors import friendly_error
from .embed import embed_file
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
