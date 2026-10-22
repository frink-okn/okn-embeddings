"""Known-answer probes: a vector space's claim made checkable.

`convention_id` hashes the model's *name* with the vector shape. That is what
two parties say they share; it is not evidence they compute the same thing.
A torch or sentence-transformers upgrade, a different device, or a different
build of nominally the same model can shift every vector without changing
the name, and nothing fails -- vectors from the two sides just compare a
little wrong.

So an artifact also carries a fixed set of texts and the vectors this
embedder produced for them. Anyone who believes they share the space -- a
later build reusing these vectors, a query-side client -- embeds the same
texts and compares, in milliseconds.

The texts are embedded as one batch in this order every time: the same text
embedded in a different batch differs in the last float bits (padding
changes the arithmetic), and fixing the batch keeps that out of the check.
"""

import json
from typing import Any

import numpy as np

from ..core.embedding import Embedder

# Short and long, prose and the verbalizer's line shape, identifiers,
# numbers, non-ASCII. Changing this list changes every artifact's probes;
# do it only with PROBES_FORMAT.
PROBE_TEXTS = (
    "diabetes",
    "insulin resistance in skeletal muscle",
    "label: Advanced Center for Psychotherapy Inc\ntype: Treatment Provider",
    "label: Cover crops\ntype: Treatment\nhas location: GRACEnet (ALAUEVS)",
    "label: soil organic carbon\nunit: g/kg\nvalue: 12.5",
    "CHEBI:38769",
    "http://purl.obolibrary.org/obo/MONDO_0005148",
    "The quick brown fox jumps over the lazy dog.",
    "Zürich, São Paulo, Kraków, 東京",
    "2026-10-22",
    "a",
    "protein kinase activity regulates cell cycle progression in yeast",
)

PROBES_FORMAT = 1

# Largest per-component difference two embedders may show on the probes and
# still be one vector space. Re-batching the same model on the same device
# moves components by ~1e-7; a different model moves them by ~1e-1.
PROBE_TOLERANCE = 1e-5


def probe_vectors(embedder: Embedder) -> np.ndarray:
    """The probes under `embedder`, one row per text, as one batch."""
    return np.stack(embedder.embed_many(list(PROBE_TEXTS)))


def encode_probes(vectors: np.ndarray) -> str:
    """The probes as the JSON an artifact's metadata carries.

    Each float32 is written as the shortest decimal that reads back to the
    same float32's float64 widening, so the round trip is exact.
    """
    return json.dumps(
        {
            "format": PROBES_FORMAT,
            "texts": list(PROBE_TEXTS),
            "vectors": [[float(x) for x in row] for row in vectors],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_probes(encoded: str) -> tuple[list[str], np.ndarray]:
    """(texts, vectors) from `encode_probes` output."""
    data: dict[str, Any] = json.loads(encoded)
    if data.get("format") != PROBES_FORMAT:
        raise ValueError(
            f"probes format {data.get('format')!r}; expected {PROBES_FORMAT}"
        )
    return list(data["texts"]), np.asarray(data["vectors"], dtype=np.float32)


def probe_deviation(encoded: str, embedder: Embedder) -> float | None:
    """How far `embedder` lands from the recorded probes.

    The largest per-component difference, or None when the two cannot be
    compared at all (other texts, other width) -- which is a mismatch, not
    a small deviation.
    """
    texts, recorded = decode_probes(encoded)
    if tuple(texts) != PROBE_TEXTS:
        return None
    current = probe_vectors(embedder)
    if current.shape != recorded.shape:
        return None
    return float(np.abs(current - recorded).max())
