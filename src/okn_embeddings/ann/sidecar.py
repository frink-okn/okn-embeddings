"""Read-side access to embed Parquet artifacts and their ANN sidecars.

`SidecarStore` opens one graph's Parquet file and serves cosine top-k over
it: through the usearch index when one sits beside the file, and by exact
flat scan when none does (or when exact search is requested). Small graphs
therefore need no index at all -- the Parquet alone is a complete, exact
"index" -- and deleting a .usearch file degrades service to exact search
rather than breaking it.

Everything is lazy and memory-map friendly. Opening a store reads only the
Parquet footer and (when present) the usearch index, which is mapped, not
loaded. Vectors materialize on first flat scan as zero-copy views over the
mapped file, one per row group -- the vector column is written uncompressed
for exactly this reason -- and search hits resolve to records by decoding
only the row group they land in. An ANN search on a warm store touches no
record data beyond its k hits.

Index staleness: hits resolve to records by row ordinal, so serving with an
index built from different vectors would return wrong records with high
confidence. Cheap structural checks (parent file size, row count, dimension)
always run; the full parent-sha256 comparison hashes the whole Parquet file,
so it runs only when `verify` is requested.
"""

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from usearch.index import Index

from ..core.results import ResultRow
from ..indexing.manifest import file_sha256
from .build import index_manifest_path, index_path_for, parquet_okn_metadata

_RECORD_COLUMNS = ("iris", "iri_count", "label", "embedding_text")


class SidecarStore:
    """One graph's vectors and records, searchable."""

    def __init__(
        self,
        path: Path,
        parquet_file: pq.ParquetFile,
        *,
        graph: str,
        metadata: dict[str, str],
        dim: int,
        index: Index | None,
        index_manifest: dict | None,
    ):
        self.path = path
        self.graph = graph
        self.metadata = metadata
        self.dim = dim
        self.index = index
        self.index_manifest = index_manifest

        self._parquet = parquet_file
        # Row-group start ordinals, for ordinal -> row group resolution.
        starts = []
        total = 0
        for group in range(parquet_file.metadata.num_row_groups):
            starts.append(total)
            total += parquet_file.metadata.row_group(group).num_rows
        self._group_starts = starts
        self._group_cache: dict[int, dict[str, list]] = {}
        self._vector_parts: list[np.ndarray] | None = None
        self._part_starts: list[int] = []
        self._iri_to_row: dict[str, int] | None = None

    @classmethod
    def open(
        cls,
        parquet_path: Path,
        *,
        use_index: bool = True,
        verify: bool = False,
    ):
        """Open a store over one embed Parquet file.

        Reads only the footer; record data and vectors load lazily. A
        usearch index beside the file is memory-mapped when `use_index` is
        set. `verify` additionally hashes the Parquet file and compares it
        to the index manifest's parent sha256.
        """
        parquet_file = pq.ParquetFile(parquet_path, memory_map=True)
        metadata = parquet_okn_metadata(parquet_file)
        dim = parquet_file.schema_arrow.field("vector").type.list_size

        index = None
        index_manifest = None
        index_file = index_path_for(parquet_path)
        if use_index and index_file.exists():
            index, index_manifest = cls._open_index(
                index_file,
                parquet_path,
                dim=dim,
                rows=parquet_file.metadata.num_rows,
                verify=verify,
            )

        return cls(
            parquet_path,
            parquet_file,
            graph=metadata.get("graph", parquet_path.stem),
            metadata=metadata,
            dim=dim,
            index=index,
            index_manifest=index_manifest,
        )

    @staticmethod
    def _open_index(
        index_file: Path,
        parquet_path: Path,
        *,
        dim: int,
        rows: int,
        verify: bool,
    ) -> tuple[Index, dict]:
        manifest_file = index_manifest_path(index_file)
        if not manifest_file.exists():
            raise ValueError(
                f"{index_file} has no manifest ({manifest_file.name}); "
                "rebuild it with `okn-indexing build-index`"
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        parent = manifest.get("parent") or {}

        stale = (
            f"{index_file} does not match {parquet_path.name}; rebuild it "
            "with `okn-indexing build-index` or delete it to fall back to "
            "exact search"
        )
        expected_bytes = parent.get("bytes")
        if (
            expected_bytes is not None
            and expected_bytes != parquet_path.stat().st_size
        ):
            raise ValueError(f"{stale} (parent size mismatch)")
        if verify and parent.get("sha256") != file_sha256(parquet_path):
            raise ValueError(f"{stale} (parent sha256 mismatch)")

        index = Index.restore(str(index_file), view=True)
        if len(index) != rows:
            raise ValueError(
                f"{stale} (index has {len(index)} keys, Parquet has "
                f"{rows} rows)"
            )
        if index.ndim != dim:
            raise ValueError(
                f"{index_file} has ndim {index.ndim}, expected {dim}"
            )
        return index, manifest

    @property
    def count(self) -> int:
        return self._parquet.metadata.num_rows

    @property
    def has_index(self) -> bool:
        return self.index is not None

    @property
    def vector_parts(self) -> list[np.ndarray]:
        """The stored vectors, one (rows, dim) matrix per row group.

        Zero-copy views over the mapped file where the format allows
        (uncompressed vector column, unit-normalized vectors); parts are
        never concatenated, which would force a heap copy.
        """
        if self._vector_parts is None:
            parts = []
            for group in range(self._parquet.metadata.num_row_groups):
                column = self._parquet.read_row_group(
                    group, columns=["vector"]
                ).column("vector")
                for chunk in column.chunks:
                    parts.append(
                        chunk.flatten()
                        .to_numpy(zero_copy_only=False)
                        .reshape(-1, self.dim)
                    )
            if self.metadata.get("normalized") != "true":
                normalized = []
                for part in parts:
                    norms = np.linalg.norm(part, axis=1, keepdims=True)
                    norms[norms == 0.0] = 1.0
                    normalized.append(part / norms)
                parts = normalized

            starts = []
            total = 0
            for part in parts:
                starts.append(total)
                total += len(part)
            self._vector_parts = parts
            self._part_starts = starts
        return self._vector_parts

    def vector_matrix(self) -> np.ndarray:
        """All vectors as one matrix. May copy when there are several row
        groups; prefer `vector_parts` for scanning."""
        parts = self.vector_parts
        if not parts:
            return np.empty((0, self.dim), dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        return np.vstack(parts)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        *,
        exact: bool = False,
    ) -> list[ResultRow]:
        """Top-k nearest records by cosine similarity, best first.

        Uses the ANN index when present unless `exact` forces the flat
        scan. Scores are cosine similarities either way.
        """
        if k <= 0 or self.count == 0:
            return []

        query = np.asarray(vector, dtype=np.float32).reshape(-1)

        if self.index is None or exact:
            norm = float(np.linalg.norm(query))
            if norm:
                query = query / norm
            similarities = np.empty(self.count, dtype=np.float32)
            position = 0
            for part in self.vector_parts:
                similarities[position : position + len(part)] = part @ query
                position += len(part)
            k = min(k, len(similarities))
            top = np.argpartition(-similarities, k - 1)[:k]
            top = top[np.argsort(-similarities[top])]
            return [
                self._row(int(i), float(similarities[i])) for i in top
            ]

        matches = self.index.search(query, k)
        return [
            self._row(int(key), 1.0 - float(distance))
            for key, distance in zip(
                matches.keys, matches.distances, strict=True
            )
        ]

    def vector_for_iri(self, iri: str) -> np.ndarray | None:
        """The stored vector of the record containing `iri`, if any."""
        if self._iri_to_row is None:
            mapping: dict[str, int] = {}
            iris_column = self._parquet.read(columns=["iris"]).column("iris")
            for row, row_iris in enumerate(iris_column.to_pylist()):
                for candidate in row_iris:
                    mapping.setdefault(candidate, row)
            self._iri_to_row = mapping

        row = self._iri_to_row.get(iri)
        if row is None:
            return None
        parts = self.vector_parts
        part = bisect_right(self._part_starts, row) - 1
        return parts[part][row - self._part_starts[part]]

    def _record_fields(
        self, ordinal: int
    ) -> tuple[list[str], int, str, str]:
        """Decode one row's record columns, one row group at a time."""
        group = bisect_right(self._group_starts, ordinal) - 1
        cached = self._group_cache.get(group)
        if cached is None:
            table = self._parquet.read_row_group(
                group, columns=list(_RECORD_COLUMNS)
            )
            cached = {
                name: table.column(name).to_pylist()
                for name in _RECORD_COLUMNS
            }
            self._group_cache[group] = cached
        offset = ordinal - self._group_starts[group]
        return (
            cached["iris"][offset],
            cached["iri_count"][offset],
            cached["label"][offset],
            cached["embedding_text"][offset],
        )

    def _row(self, ordinal: int, score: float) -> ResultRow:
        iris, iri_count, label, text = self._record_fields(ordinal)
        return ResultRow(
            id=f"{self.graph}:{ordinal}",
            score=score,
            iris=iris,
            primary_uri=iris[0] if iris else "",
            iri_count=iri_count,
            label=label,
            graph=self.graph,
            repr=text,
        )
