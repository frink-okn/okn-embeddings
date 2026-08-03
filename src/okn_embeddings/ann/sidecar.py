"""Read-side access to embed Parquet artifacts and their ANN sidecars.

`SidecarStore` opens one graph's Parquet file and serves similarity search
over it: through the usearch index when one sits beside the file, and by
exact flat scan when none does (or when exact search is requested). Small
graphs therefore need no index at all -- the Parquet alone is a complete,
exact "index" -- and deleting a .usearch file degrades service to exact
search rather than breaking it.

An index is only trusted when its manifest's parent sha256 matches the
Parquet beside it. Hits resolve to records by row ordinal, so serving with a
mismatched index would return wrong records with high confidence; a mismatch
is an error, not a warning.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from usearch.index import Index

from ..core.results import ResultRow
from ..indexing.manifest import file_sha256
from .build import index_manifest_path, index_path_for, parquet_okn_metadata


class SidecarStore:
    """One graph's vectors and records, searchable."""

    def __init__(
        self,
        path: Path,
        *,
        graph: str,
        metadata: dict[str, str],
        labels: list[str],
        iris: list[list[str]],
        iri_counts: list[int],
        texts: list[str],
        vectors: np.ndarray,
        index: Index | None,
        index_manifest: dict | None,
    ):
        self.path = path
        self.graph = graph
        self.metadata = metadata
        self.labels = labels
        self.iris = iris
        self.iri_counts = iri_counts
        self.texts = texts
        self.vectors = vectors
        self.index = index
        self.index_manifest = index_manifest
        self._iri_to_row: dict[str, int] | None = None

    @classmethod
    def open(cls, parquet_path: Path, *, use_index: bool = True):
        """Open a store over one embed Parquet file.

        Loads the usearch index beside the file (memory-mapped) when
        `use_index` is set and one exists; its manifest must name this
        Parquet's sha256 as its parent.
        """
        parquet_file = pq.ParquetFile(parquet_path, memory_map=True)
        metadata = parquet_okn_metadata(parquet_file)
        table = parquet_file.read()

        dim = table.schema.field("vector").type.list_size
        if table.num_rows:
            vectors = np.vstack(
                [
                    chunk.flatten()
                    .to_numpy(zero_copy_only=False)
                    .reshape(-1, dim)
                    for chunk in table.column("vector").chunks
                ]
            )
        else:
            vectors = np.empty((0, dim), dtype=np.float32)

        # Flat-scan scoring assumes unit rows; normalize if the artifact
        # does not promise them.
        if metadata.get("normalized") != "true" and len(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms

        index = None
        index_manifest = None
        index_file = index_path_for(parquet_path)
        if use_index and index_file.exists():
            index, index_manifest = cls._open_index(
                index_file, parquet_path, dim
            )

        return cls(
            parquet_path,
            graph=metadata.get("graph", parquet_path.stem),
            metadata=metadata,
            labels=table.column("label").to_pylist(),
            iris=table.column("iris").to_pylist(),
            iri_counts=table.column("iri_count").to_pylist(),
            texts=table.column("embedding_text").to_pylist(),
            vectors=vectors,
            index=index,
            index_manifest=index_manifest,
        )

    @staticmethod
    def _open_index(
        index_file: Path, parquet_path: Path, dim: int
    ) -> tuple[Index, dict]:
        manifest_file = index_manifest_path(index_file)
        if not manifest_file.exists():
            raise ValueError(
                f"{index_file} has no manifest ({manifest_file.name}); "
                "rebuild it with `okn-indexing build-index`"
            )
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        expected = (manifest.get("parent") or {}).get("sha256")
        if expected != file_sha256(parquet_path):
            raise ValueError(
                f"{index_file} was built from a different Parquet file "
                "(parent sha256 mismatch); rebuild it with "
                "`okn-indexing build-index` or delete it to fall back to "
                "exact search"
            )

        index = Index.restore(str(index_file), view=True)
        if index.ndim != dim:
            raise ValueError(
                f"{index_file} has ndim {index.ndim}, expected {dim}"
            )
        return index, manifest

    @property
    def count(self) -> int:
        return len(self.labels)

    @property
    def has_index(self) -> bool:
        return self.index is not None

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
            similarities = self.vectors @ query
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
            for row, row_iris in enumerate(self.iris):
                for candidate in row_iris:
                    mapping.setdefault(candidate, row)
            self._iri_to_row = mapping

        row = self._iri_to_row.get(iri)
        if row is None:
            return None
        return self.vectors[row]

    def _row(self, ordinal: int, score: float) -> ResultRow:
        iris = self.iris[ordinal]
        return ResultRow(
            id=f"{self.graph}:{ordinal}",
            score=score,
            iris=iris,
            primary_uri=iris[0] if iris else "",
            iri_count=self.iri_counts[ordinal],
            label=self.labels[ordinal],
            graph=self.graph,
            repr=self.texts[ordinal],
        )
