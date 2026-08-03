"""Embed materialized records into self-described Parquet artifacts.

`textify` emits text-only grouped records; this step runs each record's
`embedding_text` through the shared `core.embedding` seam (so stored vectors
match the query path) and writes one Parquet file per graph: the record
fields plus a fixed-width `vector` column, with the model identity and vector
properties recorded as file-level metadata. The file is the canonical
artifact between materialization and any vector consumer -- Qdrant upload and
ANN index builds both read from it, so vectors are computed once.

Row order is input order, which `textify` already makes deterministic, so a
row's ordinal is a stable key for index artifacts built over this file.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ..core.embedding import Embedder
from .manifest import read_manifest
from .records import chunks, count_jsonl, iter_jsonl

METADATA_PREFIX = "okn-embeddings."

# The vector column is meaningless without knowing how it was produced, so
# `embed_file` stamps these keys (prefixed) into the Parquet footer metadata.
METADATA_KEYS = (
    "format",  # version of this artifact layout
    "graph",  # graph name, from the input file stem
    "model",  # embedding model identity
    "dim",  # vector width
    "metric",  # distance the model is trained for; the query side uses this
    "normalized",  # "true" when every stored vector is unit-length
    "record_count",
)

# When a textify provenance manifest sits beside the input records, these are
# also stamped (prefixed): the full manifest JSON plus the two hashes pulled
# out for cheap comparison across artifacts.
MANIFEST_METADATA_KEYS = (
    "textify_manifest",
    "config_sha256",
    "graph_sha256",
)

# Vectors whose L2 norms all sit within this tolerance of 1.0 are recorded as
# normalized. Measured over every vector rather than asserted from the model
# name, so the metadata stays honest if the configured model changes.
NORMALIZED_TOLERANCE = 1e-3

# Rows are buffered and written in groups of this many, bounding memory while
# keeping row groups large enough for efficient scans.
FLUSH_ROWS = 4096

# The vector column is written uncompressed and without dictionary encoding
# so a reader can memory-map the file and view the values zero-copy;
# float32 embeddings are near-incompressible, so nothing is given up. The
# text columns keep compression.
PARQUET_COMPRESSION = {
    "iris.list.element": "SNAPPY",
    "iri_count": "SNAPPY",
    "label": "SNAPPY",
    "embedding_text": "SNAPPY",
    "vector.list.element": "NONE",
}
PARQUET_DICTIONARY_COLUMNS = ["iris.list.element", "label", "embedding_text"]


def vector_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("iris", pa.list_(pa.string())),
            pa.field("iri_count", pa.int32()),
            pa.field("label", pa.string()),
            pa.field("embedding_text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def record_fields(
    graph: str, record: dict[str, Any]
) -> tuple[list[str], int, str, str]:
    """Validate one record and pull the fields the artifact keeps.

    Accepts both the grouped `iris` list schema and the older singular `iri`.
    A missing `iri_count` defaults to the number of listed IRIs.
    """
    text = record.get("embedding_text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{graph}: record is missing non-empty embedding_text")

    iris = record.get("iris")
    if not isinstance(iris, list) or not iris:
        iri = record.get("iri")
        iris = [iri] if isinstance(iri, str) and iri else []
    if not iris:
        raise ValueError(f"{graph}: record is missing iris")

    iri_count = int(record.get("iri_count", len(iris)))
    label = str(record.get("label", ""))
    return [str(iri) for iri in iris], iri_count, label, text


def rows_to_table(
    schema: pa.Schema,
    rows: list[tuple[list[str], int, str, str]],
    vectors: list[np.ndarray],
) -> pa.Table:
    iris, iri_counts, labels, texts = zip(*rows, strict=True)
    dim = schema.field("vector").type.list_size

    flat = np.concatenate(
        [np.asarray(v, dtype=np.float32) for v in vectors]
    )
    vector_array = pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)

    return pa.Table.from_arrays(
        [
            pa.array(list(iris), type=pa.list_(pa.string())),
            pa.array(list(iri_counts), type=pa.int32()),
            pa.array(list(labels), type=pa.string()),
            pa.array(list(texts), type=pa.string()),
            vector_array,
        ],
        schema=schema,
    )


def manifest_metadata(records_path: Path) -> dict[str, str]:
    """Metadata entries from the manifest beside the records file, if any."""
    manifest = read_manifest(records_path)
    if manifest is None:
        return {}

    entries = {
        METADATA_PREFIX
        + "textify_manifest": json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")
        )
    }
    config_sha = (manifest.get("config") or {}).get("sha256")
    if config_sha:
        entries[METADATA_PREFIX + "config_sha256"] = config_sha
    graph_sha = (manifest.get("graph") or {}).get("sha256")
    if graph_sha:
        entries[METADATA_PREFIX + "graph_sha256"] = graph_sha
    return entries


def embed_file(
    embedder: Embedder,
    path: Path,
    output: Path,
    *,
    model_name: str,
    batch_size: int,
    limit: int | None = None,
    flush_rows: int = FLUSH_ROWS,
    progress_enabled: bool = False,
) -> int:
    """Embed one graph's records into a Parquet file. Returns the row count."""
    graph = path.stem
    dim = int(embedder.embed("dimension probe").shape[0])
    schema = vector_schema(dim)

    total_records = count_jsonl(path, limit=limit)
    progress = tqdm(
        total=total_records,
        desc=graph,
        unit="record",
        disable=not progress_enabled,
        dynamic_ncols=True,
    )

    rows: list[tuple[list[str], int, str, str]] = []
    vectors: list[np.ndarray] = []
    count = 0
    max_norm_deviation = 0.0

    with pq.ParquetWriter(
        output,
        schema,
        compression=PARQUET_COMPRESSION,
        use_dictionary=PARQUET_DICTIONARY_COLUMNS,
    ) as writer:

        def flush() -> None:
            nonlocal rows, vectors
            if rows:
                writer.write_table(rows_to_table(schema, rows, vectors))
                rows = []
                vectors = []

        for record_batch in chunks(iter_jsonl(path, limit=limit), batch_size):
            fields = [record_fields(graph, r) for r in record_batch]
            batch_vectors = embedder.embed_many([f[3] for f in fields])

            norms = np.linalg.norm(np.stack(batch_vectors), axis=1)
            deviation = float(np.abs(norms - 1.0).max())
            max_norm_deviation = max(max_norm_deviation, deviation)

            rows.extend(fields)
            vectors.extend(batch_vectors)
            count += len(fields)
            progress.update(len(fields))

            if len(rows) >= flush_rows:
                flush()

        flush()

        normalized = max_norm_deviation <= NORMALIZED_TOLERANCE
        metadata = {
            METADATA_PREFIX + "format": "1",
            METADATA_PREFIX + "graph": graph,
            METADATA_PREFIX + "model": model_name,
            METADATA_PREFIX + "dim": str(dim),
            METADATA_PREFIX + "metric": "cosine",
            METADATA_PREFIX + "normalized": str(normalized).lower(),
            METADATA_PREFIX + "record_count": str(count),
        }
        metadata.update(manifest_metadata(path))
        writer.add_key_value_metadata(metadata)

    progress.close()
    return count
