"""Embed a kgf verbalizer texts stream into a digest-keyed Parquet file.

`kgf verbalize --texts` writes one JSON line per distinct text, `{"digest":
<hex sha256 of text>, "text": ...}`. This step embeds each through the shared
`core.embedding` seam -- the same code the query path runs -- and writes one
row per text: its 32-byte `text_digest` and its `vector`, in input order.

This is the vector store a KGF bundle publishes, and the file `kgf build`
builds its usearch index and id map from. It deliberately carries no IRIs,
labels, or text: which entity has which text is the graph's business (kgf
publishes that separately), and the text is a function of the bundle and its
config. What the file holds is exactly the model's output, keyed by the one
thing that decides a vector -- the text's digest -- so a later build can reuse
any vector whose text is unchanged.

Row order is input order, so kgf can check row i against the i-th text it
wrote and use i as the index key. Digest uniqueness is not checked here (that
would hold every digest in memory); kgf, which wrote the stream, checks it.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ..core.embedding import Embedder
from .embed import (
    FLUSH_ROWS,
    METADATA_PREFIX,
    NORMALIZED_TOLERANCE,
    convention_id,
)
from .probes import encode_probes, probe_vectors
from .records import chunks, count_jsonl, iter_jsonl

if TYPE_CHECKING:
    # reuse imports this module's layout constants.
    from .reuse import ReuseSource

# The layout `embed` writes is format 1: one row per grouped record, with its
# IRIs, label, and text. This is a different layout, not a revision of that
# one, but readers dispatch on the same footer key.
TEXTS_FORMAT = 2

DIGEST_BYTES = 32

# As in `embed`: the vector column stays uncompressed and plain so a reader
# can map it zero-copy. Digests are random bytes, so compressing them gains
# nothing either.
PARQUET_COMPRESSION = "NONE"


def texts_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("text_digest", pa.binary(DIGEST_BYTES), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=False),
        ]
    )


def text_fields(
    path: Path, ordinal: int, record: dict[str, Any]
) -> tuple[bytes, str]:
    """Validate one texts-stream line; return (digest bytes, text).

    The digest is recomputed rather than trusted: it is the key a vector is
    stored and later reused under, so a line whose digest does not match its
    text would file a vector under another text's name.
    """
    where = f"{path}: text {ordinal}"
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{where}: line is missing non-empty text")
    digest_hex = record.get("digest")
    if not isinstance(digest_hex, str):
        raise ValueError(f"{where}: line is missing its digest")
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as e:
        raise ValueError(f"{where}: digest is not hex: {e}") from e
    if len(digest) != DIGEST_BYTES:
        raise ValueError(
            f"{where}: digest is {len(digest)} bytes, not {DIGEST_BYTES}"
        )
    if hashlib.sha256(text.encode()).digest() != digest:
        raise ValueError(f"{where}: digest is not the sha256 of the text")
    return digest, text


def rows_to_table(
    schema: pa.Schema, digests: list[bytes], vectors: list[np.ndarray]
) -> pa.Table:
    dim = schema.field("vector").type.list_size
    flat = np.concatenate([np.asarray(v, dtype=np.float32) for v in vectors])
    return pa.Table.from_arrays(
        [
            pa.array(digests, type=pa.binary(DIGEST_BYTES)),
            pa.FixedSizeListArray.from_arrays(pa.array(flat), dim),
        ],
        schema=schema,
    )


@dataclass(frozen=True)
class EmbedTextsResult:
    count: int
    """Rows written: one per text."""
    truncated: int
    """Texts longer than the model reads, embedded from their first
    `max_seq_length` tokens. Nonzero says the verbalizer config is too
    generous for this model -- a `predicate_limit` or a template to cut."""
    reused: int = 0
    """Rows whose vector came from the reuse source, not the model."""


def embedding_dim(embedder: Embedder) -> int:
    return int(embedder.embed("dimension probe").shape[0])


def embed_texts_file(
    embedder: Embedder,
    path: Path,
    output: Path,
    *,
    model_name: str,
    batch_size: int,
    graph: str,
    reuse: "ReuseSource | None" = None,
    flush_rows: int = FLUSH_ROWS,
    progress_enabled: bool = False,
) -> EmbedTextsResult:
    """Embed one texts stream into a Parquet file.

    With `reuse`, a text whose digest the source holds takes its vector
    from there instead of the model. The output does not otherwise depend
    on it: the source has passed the probe check, so a reused vector is the
    one the model would produce, to within re-batching noise.
    """
    dim = embedding_dim(embedder)
    schema = texts_schema(dim)

    progress = tqdm(
        total=count_jsonl(path),
        desc=graph,
        unit="text",
        disable=not progress_enabled,
        dynamic_ncols=True,
    )

    digests: list[bytes] = []
    vectors: list[np.ndarray] = []
    count = 0
    truncated = 0
    reused = 0
    max_seq_length = embedder.max_seq_length
    max_norm_deviation = 0.0

    # Texts read but not yet written, in input order, each with its reused
    # vector or None. Texts needing the model accumulate until there is a
    # full batch of them, so heavy reuse does not shrink every model call to
    # the few misses in one read.
    pending: list[tuple[bytes, str, np.ndarray | None]] = []
    misses = 0

    with pq.ParquetWriter(
        output,
        schema,
        compression=PARQUET_COMPRESSION,
        use_dictionary=False,
    ) as writer:

        def flush() -> None:
            nonlocal digests, vectors
            if digests:
                writer.write_table(rows_to_table(schema, digests, vectors))
                digests = []
                vectors = []

        def drain() -> None:
            nonlocal misses, max_norm_deviation
            if not pending:
                return
            to_embed = [text for _, text, vector in pending if vector is None]
            fresh = iter(embedder.embed_many(to_embed) if to_embed else [])
            filled = [
                next(fresh) if vector is None else vector
                for _, _, vector in pending
            ]

            norms = np.linalg.norm(np.stack(filled), axis=1)
            deviation = float(np.abs(norms - 1.0).max())
            max_norm_deviation = max(max_norm_deviation, deviation)

            digests.extend(digest for digest, _, _ in pending)
            vectors.extend(filled)
            progress.update(len(pending))
            pending.clear()
            misses = 0
            if len(digests) >= flush_rows:
                flush()

        # Numbered by position in the stream, which is the row the text will
        # occupy (plus one); iter_jsonl's own errors carry line numbers.
        numbered = enumerate(iter_jsonl(path), start=1)
        for batch in chunks(numbered, batch_size):
            fields = [text_fields(path, n, record) for n, record in batch]
            texts = [text for _, text in fields]
            truncated += sum(
                n > max_seq_length for n in embedder.token_counts(texts)
            )
            found = (
                reuse.lookup([digest for digest, _ in fields])
                if reuse is not None
                else [None] * len(fields)
            )
            for (digest, text), vector in zip(fields, found, strict=True):
                pending.append((digest, text, vector))
            hits = sum(vector is not None for vector in found)
            reused += hits
            misses += len(fields) - hits
            count += len(fields)

            if misses >= batch_size or len(pending) >= flush_rows:
                drain()

        drain()
        flush()

        normalized = max_norm_deviation <= NORMALIZED_TOLERANCE
        writer.add_key_value_metadata(
            {
                METADATA_PREFIX + "format": str(TEXTS_FORMAT),
                METADATA_PREFIX + "graph": graph,
                METADATA_PREFIX + "model": model_name,
                METADATA_PREFIX + "dim": str(dim),
                METADATA_PREFIX + "metric": "cosine",
                METADATA_PREFIX + "normalized": str(normalized).lower(),
                METADATA_PREFIX + "record_count": str(count),
                METADATA_PREFIX + "convention_id": convention_id(
                    model_name, dim, "cosine", normalized
                ),
                METADATA_PREFIX + "max_seq_length": str(max_seq_length),
                METADATA_PREFIX + "probes": encode_probes(
                    probe_vectors(embedder)
                ),
                METADATA_PREFIX + "truncated_records": str(truncated),
                METADATA_PREFIX + "reused_records": str(reused),
            }
        )

    progress.close()
    return EmbedTextsResult(count=count, truncated=truncated, reused=reused)
