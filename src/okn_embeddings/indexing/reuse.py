"""Vectors from an earlier embed-texts file, looked up by text digest.

Embedding is the slow stage of a KGF build, and the verbalizer addresses
each text by its sha256, so a text unchanged since an earlier build needs
no model run: its vector is in that build's file under the same digest.

An earlier file is a source only if it is the same vector space, and the
name of the model is not evidence of that (see `probes`). It must be this
layout, name the same model at the same width, and carry probes that the
current embedder reproduces within `PROBE_TOLERANCE`. Anything else is
refused with the reason -- the caller embeds everything instead, which is
slower and never wrong.

Lookups hold the earlier file's digests sorted in memory (32 bytes and a
row number per text) and its vectors as zero-copy views of the mapped file,
one per row group; a vector is copied only when it is reused.
"""

from bisect import bisect_right
from pathlib import Path
from typing import NoReturn

import numpy as np
import pyarrow.parquet as pq

from ..core.embedding import Embedder
from .embed import METADATA_PREFIX
from .embed_texts import DIGEST_BYTES, TEXTS_FORMAT
from .probes import PROBE_TOLERANCE, probe_deviation, probe_vectors


class ReuseRefused(Exception):
    """An earlier file that is readable but not this vector space."""


class ReuseSource:
    def __init__(
        self,
        path: Path,
        sorted_digests: np.ndarray,
        rows: np.ndarray,
        vector_groups: list[np.ndarray],
        group_starts: list[int],
    ):
        self.path = path
        self._sorted = sorted_digests
        self._rows = rows
        self._groups = vector_groups
        self._starts = group_starts

    @classmethod
    def open(
        cls,
        path: Path,
        embedder: Embedder,
        *,
        model_name: str,
        dim: int,
    ) -> "ReuseSource":
        """Open `path` as a source for `embedder`'s space.

        Raises ReuseRefused when the file is another space or layout;
        OSError and pyarrow's errors pass through, since a path the
        operator named that cannot be read is a mistake to report, not a
        reason to quietly embed everything.
        """
        parquet = pq.ParquetFile(path, memory_map=True)
        raw = parquet.metadata.metadata or {}
        meta = {
            key.decode().removeprefix(METADATA_PREFIX): value.decode()
            for key, value in raw.items()
            if key.decode().startswith(METADATA_PREFIX)
        }

        def refuse(reason: str) -> NoReturn:
            raise ReuseRefused(f"{path}: {reason}")

        if meta.get("format") != str(TEXTS_FORMAT):
            refuse(
                f"layout format {meta.get('format')!r}, not the "
                f"embed-texts layout {TEXTS_FORMAT}"
            )
        if meta.get("model") != model_name:
            refuse(f"embedded with {meta.get('model')!r}, not {model_name!r}")
        if meta.get("dim") != str(dim):
            refuse(f"{meta.get('dim')}-dimensional, not {dim}")
        probes = meta.get("probes")
        if probes is None:
            refuse("carries no probes to check the vector space by")
        deviation = probe_deviation(probes, probe_vectors(embedder))
        if deviation is None:
            refuse("its probes are not the current probe set")
        if deviation > PROBE_TOLERANCE:
            refuse(
                f"the current embedder moves its probes by {deviation:.2e}, "
                f"over the {PROBE_TOLERANCE:.0e} tolerance: the same model "
                "name, but not the same vectors"
            )

        # Fixed-width and non-null, so the column's data buffer is the
        # digests back to back; numpy's S32 compares them bytewise.
        digest_array = (
            parquet.read(columns=["text_digest"])
            .column("text_digest")
            .combine_chunks()
        )
        digests = np.frombuffer(
            digest_array.buffers()[1],
            dtype=f"S{DIGEST_BYTES}",
            count=len(digest_array),
            offset=digest_array.offset * DIGEST_BYTES,
        )
        order = np.argsort(digests, kind="stable")

        groups: list[np.ndarray] = []
        starts: list[int] = []
        total = 0
        for group in range(parquet.metadata.num_row_groups):
            column = parquet.read_row_group(group, columns=["vector"])
            for chunk in column.column("vector").chunks:
                flat = chunk.flatten().to_numpy(zero_copy_only=True)
                starts.append(total)
                groups.append(flat.reshape(-1, dim))
                total += len(chunk)

        return cls(path, digests[order], order, groups, starts)

    def __len__(self) -> int:
        return len(self._sorted)

    def lookup(self, digests: list[bytes]) -> list[np.ndarray | None]:
        """Each digest's earlier vector (a copy), or None if it has none."""
        if not digests or not len(self._sorted):
            return [None] * len(digests)
        wanted = np.array(digests, dtype=f"S{DIGEST_BYTES}")
        at = np.searchsorted(self._sorted, wanted)
        found: list[np.ndarray | None] = []
        for digest, i in zip(wanted, at, strict=True):
            if i < len(self._sorted) and self._sorted[i] == digest:
                found.append(self._vector(int(self._rows[i])))
            else:
                found.append(None)
        return found

    def _vector(self, row: int) -> np.ndarray:
        group = bisect_right(self._starts, row) - 1
        return np.array(self._groups[group][row - self._starts[group]])
