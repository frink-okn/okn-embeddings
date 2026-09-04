"""Embed materialized JSONL records and upsert them into Qdrant.

This is the producer's second half: `textify`/`materialize` emit text-only
embedding records, and this module embeds each record's `embedding_text`
through the shared `core.embedding` seam (so stored vectors match the query
path) and upserts the points.

Each input file is one graph (`graph = path.stem`). Point IDs are derived
deterministically from `(graph, iri, embedding_text)`, so re-running upserts in
place rather than duplicating.
"""

import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from ..config import AppContext
from .records import chunks, count_jsonl, iter_jsonl

__all__ = [
    "chunks",
    "count_jsonl",
    "iter_jsonl",
    "point_id",
    "payload_for_record",
    "upload_file",
    "upload_files",
]

POINT_ID_NAMESPACE = uuid.UUID("223675f1-af1e-4ce9-b7b6-849178e8c69e")


def point_id(graph: str, record: dict[str, Any]) -> str:
    """A stable UUID5 for a record, so re-uploads upsert in place."""
    iris = record.get("iris")
    iri_key = ""
    if isinstance(iris, list) and iris:
        iri_key = str(iris[0])
    elif isinstance(record.get("iri"), str):
        iri_key = record["iri"]

    embedding_text = str(record.get("embedding_text", ""))
    point_key = f"{graph}\0{iri_key}\0{embedding_text}"
    return str(uuid.uuid5(POINT_ID_NAMESPACE, point_key))


def payload_for_record(graph: str, record: dict[str, Any]) -> dict[str, Any]:
    """Build the Qdrant payload, matching what `summarize_point` reads.

    The query side reads `iri` (the IRI list), `repr`, `label`, and `graph`.
    `embedding_text`/`iris` are renamed to `repr`/`iri`; everything else on the
    record is carried through.
    """
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"embedding_text", "iris"}
    }
    payload["graph"] = graph

    if "iri" not in payload and "iris" in record:
        payload["iri"] = record["iris"]
    if "repr" not in payload and "embedding_text" in record:
        payload["repr"] = record["embedding_text"]

    return payload


def points_for_batch(
    ctx: AppContext,
    graph: str,
    records: list[dict[str, Any]],
) -> list[PointStruct]:
    """Embed a batch of records and turn them into Qdrant points."""
    texts: list[str] = []
    for record in records:
        text = record.get("embedding_text")
        if not isinstance(text, str) or not text:
            raise ValueError(
                f"{graph}: record is missing non-empty embedding_text"
            )
        texts.append(text)

    vectors = ctx.embedder.embed_many(texts)

    return [
        PointStruct(
            id=point_id(graph, record),
            vector=vector.tolist(),
            payload=payload_for_record(graph, record),
        )
        for record, vector in zip(records, vectors, strict=True)
    ]


def ensure_collection(ctx: AppContext, create_if_missing: bool) -> None:
    """Verify (or create) the configured collection before uploading."""
    collection = ctx.settings.qdrant_collection
    if ctx.client.collection_exists(collection):
        logger.info("using existing Qdrant collection {}", collection)
        return

    if not create_if_missing:
        raise RuntimeError(f"Qdrant collection does not exist: {collection}")

    vector_size = int(ctx.embedder.embed("dimension probe").shape[0])
    ctx.client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    logger.info(
        "created Qdrant collection {} with vector size {}",
        collection,
        vector_size,
    )


def ensure_payload_indexes(ctx: AppContext) -> None:
    """Create the keyword payload indexes the graph filter relies on."""
    collection = ctx.settings.qdrant_collection
    for field_name in ("graph", "iri"):
        logger.info(
            "creating payload index {} on collection {}",
            field_name,
            collection,
        )
        ctx.client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def upload_file(
    ctx: AppContext,
    path: Path,
    *,
    batch_size: int,
    upload_batch_size: int,
    limit: int | None,
    dry_run: bool,
    progress_enabled: bool,
    log_every: int,
) -> int:
    """Embed and upsert one graph file. Returns the number of points handled."""
    graph = path.stem
    uploaded = 0
    total_records = count_jsonl(path, limit=limit)
    logger.info(
        "starting graph={} file={} records={} batch_size={} "
        "upload_batch_size={} dry_run={}",
        graph,
        path,
        total_records,
        batch_size,
        upload_batch_size,
        dry_run,
    )

    progress = tqdm(
        total=total_records,
        desc=graph,
        unit="record",
        disable=not progress_enabled,
        dynamic_ncols=True,
    )
    pending_points: list[PointStruct] = []
    last_logged = 0
    for record_batch in chunks(iter_jsonl(path, limit=limit), batch_size):
        pending_points.extend(points_for_batch(ctx, graph, record_batch))

        while len(pending_points) >= upload_batch_size:
            upload_batch = pending_points[:upload_batch_size]
            pending_points = pending_points[upload_batch_size:]
            if not dry_run:
                ctx.client.upsert(
                    collection_name=ctx.settings.qdrant_collection,
                    points=upload_batch,
                    wait=True,
                )
            uploaded += len(upload_batch)
            progress.update(len(upload_batch))
            if uploaded - last_logged >= log_every:
                logger.info("uploaded graph={} records={}", graph, uploaded)
                last_logged = uploaded

    if pending_points:
        if not dry_run:
            ctx.client.upsert(
                collection_name=ctx.settings.qdrant_collection,
                points=pending_points,
                wait=True,
            )
        uploaded += len(pending_points)
        progress.update(len(pending_points))

    progress.close()
    logger.info("finished graph={} records={}", graph, uploaded)

    return uploaded


def upload_files(
    ctx: AppContext,
    paths: list[Path],
    *,
    batch_size: int,
    upload_batch_size: int,
    limit: int | None,
    create_collection: bool,
    payload_indexes: bool,
    dry_run: bool,
    progress_enabled: bool,
    log_every: int,
) -> int:
    """Prepare the collection and upload every input file. Returns the total."""
    if not dry_run:
        ensure_collection(ctx, create_if_missing=create_collection)
        if payload_indexes:
            ensure_payload_indexes(ctx)

    total = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        total += upload_file(
            ctx,
            path,
            batch_size=batch_size,
            upload_batch_size=upload_batch_size,
            limit=limit,
            dry_run=dry_run,
            progress_enabled=progress_enabled,
            log_every=log_every,
        )

    return total
