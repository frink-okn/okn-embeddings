"""Build a usearch ANN index beside an embed Parquet artifact.

The index is a derived, rebuildable artifact: it adds no information to the
Parquet file it is built from, so it can be dropped and rebuilt with
different parameters without touching the vectors. HNSW construction is
multithreaded and therefore not byte-reproducible; the index's identity claim
is its manifest, which records the build parameters and the sha256 of the
parent Parquet file rather than a hash of the index file itself.

Keys are Parquet row ordinals: `textify` output order is deterministic and
`embed` preserves it, so key k is row k, and a search hit resolves to its
record by reading that row -- no separate id-mapping table.
"""

from pathlib import Path
from typing import Any, Generator

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from usearch.index import Index

from .embed import METADATA_PREFIX
from .manifest import file_sha256, package_version, write_manifest

INDEX_SUFFIX = ".usearch"
INDEX_MANIFEST_FORMAT = 1

# usearch storage types the CLI accepts. The Parquet vectors stay float32;
# `i8`/`f16` quantize on ingest, trading recall (measurable with eval-index)
# for a smaller index file.
VECTOR_DTYPES = ("f32", "f16", "i8")

# Parquet metadata `metric` values mapped to usearch metric names.
METRICS = {"cosine": "cos"}


def index_path_for(parquet_path: Path) -> Path:
    """`graph.parquet` -> `graph.usearch`, beside the Parquet file."""
    return parquet_path.with_suffix(INDEX_SUFFIX)


def index_manifest_path(index_path: Path) -> Path:
    """`graph.usearch` -> `graph.usearch.meta.json`.

    Appended rather than substituted so it cannot collide with the textify
    manifest (`graph.meta.json`) when everything shares a stem.
    """
    return index_path.with_name(index_path.name + ".meta.json")


def parquet_okn_metadata(parquet_file: pq.ParquetFile) -> dict[str, str]:
    """The `okn-embeddings.*` footer entries, with the prefix stripped."""
    raw = parquet_file.metadata.metadata or {}
    entries = {}
    for key, value in raw.items():
        decoded = key.decode()
        if decoded.startswith(METADATA_PREFIX):
            entries[decoded.removeprefix(METADATA_PREFIX)] = value.decode()
    return entries


def iter_vector_batches(
    parquet_file: pq.ParquetFile,
    dim: int,
) -> Generator[np.ndarray, None, None]:
    """Yield each row group's vectors as a (rows, dim) float32 matrix."""
    for group in range(parquet_file.metadata.num_row_groups):
        column = parquet_file.read_row_group(group, columns=["vector"])
        for chunk in column.column("vector").chunks:
            flat = chunk.flatten().to_numpy(zero_copy_only=False)
            yield flat.reshape(-1, dim)


def resolve_index_config(
    parquet_file: pq.ParquetFile, dtype: str
) -> tuple[int, str]:
    """Validate the artifact against what the index build assumes.

    Returns (dim, usearch metric). The vector column's width is the truth;
    the footer metadata, when present, must agree with it.
    """
    if dtype not in VECTOR_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {VECTOR_DTYPES}"
        )

    field = parquet_file.schema_arrow.field("vector")
    dim = field.type.list_size

    metadata = parquet_okn_metadata(parquet_file)
    declared_dim = metadata.get("dim")
    if declared_dim is not None and int(declared_dim) != dim:
        raise ValueError(
            f"metadata dim {declared_dim} does not match "
            f"vector column width {dim}"
        )

    declared_metric = metadata.get("metric", "cosine")
    metric = METRICS.get(declared_metric)
    if metric is None:
        raise ValueError(
            f"unsupported metric {declared_metric!r}; "
            f"expected one of {sorted(METRICS)}"
        )

    return dim, metric


def build_index(
    parquet_path: Path,
    output: Path | None = None,
    *,
    dtype: str = "f32",
    connectivity: int | None = None,
    expansion_add: int | None = None,
    expansion_search: int | None = None,
    progress_enabled: bool = False,
) -> tuple[Path, Path, int]:
    """Build a usearch index over one embed Parquet file.

    Returns (index path, manifest path, indexed row count). Parameters left
    None take usearch's defaults; the manifest records the effective values
    either way.
    """
    if output is None:
        output = index_path_for(parquet_path)

    parquet_file = pq.ParquetFile(parquet_path)
    dim, metric = resolve_index_config(parquet_file, dtype)

    options: dict[str, Any] = {}
    if connectivity is not None:
        options["connectivity"] = connectivity
    if expansion_add is not None:
        options["expansion_add"] = expansion_add
    if expansion_search is not None:
        options["expansion_search"] = expansion_search
    index = Index(ndim=dim, metric=metric, dtype=dtype, **options)

    progress = tqdm(
        total=parquet_file.metadata.num_rows,
        desc=output.stem,
        unit="vector",
        disable=not progress_enabled,
        dynamic_ncols=True,
    )

    count = 0
    for vectors in iter_vector_batches(parquet_file, dim):
        keys = np.arange(count, count + len(vectors), dtype=np.int64)
        index.add(keys, vectors)
        count += len(vectors)
        progress.update(len(vectors))
    progress.close()

    index.save(str(output))

    manifest = build_index_manifest(
        parquet_path,
        parquet_file,
        output,
        index,
        dtype=dtype,
        dim=dim,
        metric=metric,
        count=count,
    )
    manifest_file = index_manifest_path(output)
    write_manifest(manifest_file, manifest)

    return output, manifest_file, count


def build_index_manifest(
    parquet_path: Path,
    parquet_file: pq.ParquetFile,
    index_path: Path,
    index: Index,
    *,
    dtype: str,
    dim: int,
    metric: str,
    count: int,
) -> dict[str, Any]:
    """Assemble the sidecar manifest for one built index.

    The parent block is the artifact's real identity: the index file is not
    byte-reproducible, so consumers compare `parent.sha256` (plus the build
    parameters) to decide whether an index still matches its vectors.
    """
    return {
        "format": INDEX_MANIFEST_FORMAT,
        "tool": {
            "name": "okn-embeddings",
            "version": package_version("okn-embeddings"),
        },
        "index": {
            "library": "usearch",
            "library_version": package_version("usearch"),
            "file": index_path.name,
            "metric": metric,
            "dtype": dtype,
            "ndim": dim,
            "connectivity": int(index.connectivity),
            "expansion_add": int(index.expansion_add),
            "expansion_search": int(index.expansion_search),
            "count": count,
            "keys": "parquet-row-ordinal",
        },
        "parent": {
            "file": parquet_path.name,
            "bytes": parquet_path.stat().st_size,
            "sha256": file_sha256(parquet_path),
            "metadata": parquet_okn_metadata(parquet_file),
        },
    }
