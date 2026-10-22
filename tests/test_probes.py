import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from okn_embeddings.core.embedding import SentenceTransformerEmbedder
from okn_embeddings.indexing.embed import METADATA_PREFIX
from okn_embeddings.indexing.embed_texts import embed_texts_file
from okn_embeddings.indexing.probes import (
    PROBE_TEXTS,
    PROBE_TOLERANCE,
    decode_probes,
    encode_probes,
    probe_deviation,
    probe_vectors,
)


class _Shifted:
    """The real embedder, nudged: a stand-in for a library upgrade that
    moves vectors without renaming the model."""

    def __init__(self, inner: SentenceTransformerEmbedder, by: float):
        self.inner = inner
        self.by = by

    def embed_many(self, texts):
        return [v + np.float32(self.by) for v in self.inner.embed_many(texts)]


def test_encoding_round_trips_exactly(embedder: SentenceTransformerEmbedder):
    vectors = probe_vectors(embedder)

    texts, decoded = decode_probes(encode_probes(vectors))

    assert tuple(texts) == PROBE_TEXTS
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, vectors)


def test_the_same_embedder_reproduces_its_probes(
    embedder: SentenceTransformerEmbedder,
):
    encoded = encode_probes(probe_vectors(embedder))

    assert probe_deviation(encoded, embedder) == 0.0


def test_a_shifted_embedder_is_caught(embedder: SentenceTransformerEmbedder):
    encoded = encode_probes(probe_vectors(embedder))

    deviation = probe_deviation(encoded, _Shifted(embedder, 1e-3))

    assert deviation is not None
    assert deviation > PROBE_TOLERANCE


def test_other_probe_texts_cannot_be_compared(
    embedder: SentenceTransformerEmbedder,
):
    data = json.loads(encode_probes(probe_vectors(embedder)))
    data["texts"][0] = "something else"

    assert probe_deviation(json.dumps(data), embedder) is None


def test_embed_texts_stamps_its_probes(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    text = "label: L"
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.write_text(json.dumps({"digest": digest, "text": text}) + "\n")
    output = tmp_path / "embeddings.parquet"

    embed_texts_file(
        embedder, path, output, model_name="m", batch_size=2, graph="g"
    )
    raw = pq.read_metadata(output).metadata
    encoded = raw[(METADATA_PREFIX + "probes").encode()].decode()

    assert probe_deviation(encoded, embedder) == 0.0
