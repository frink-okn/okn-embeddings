import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.core.embedding import FastEmbedEmbedder
from okn_embeddings.indexing.embed import METADATA_PREFIX, embed_file

_MODEL = "test-model"


def _write_records(path, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "iris": [f"urn:{i}", f"urn:{i}-alias"],
                        "iri_count": 3,
                        "label": f"L{i}",
                        "embedding_text": f"label: L{i}\ntype: Thing",
                    }
                )
                + "\n"
            )


def _embed(embedder, path, output, **kwargs) -> int:
    kwargs.setdefault("model_name", _MODEL)
    kwargs.setdefault("batch_size", 2)
    return embed_file(embedder, path, output, **kwargs)


def _metadata(path) -> dict[str, str]:
    raw = pq.read_metadata(path).metadata or {}
    return {
        key.decode(): value.decode()
        for key, value in raw.items()
        if key.decode().startswith(METADATA_PREFIX)
    }


def test_embed_file_preserves_records_in_input_order(
    embedder: FastEmbedEmbedder, tmp_path
):
    path = tmp_path / "my-graph.jsonl"
    _write_records(path, 5)
    output = tmp_path / "my-graph.parquet"

    count = _embed(embedder, path, output)
    table = pq.read_table(output)

    assert count == 5
    assert table.num_rows == 5
    assert table.column_names == [
        "iris",
        "iri_count",
        "label",
        "embedding_text",
        "vector",
    ]
    assert table.column("embedding_text").to_pylist() == [
        f"label: L{i}\ntype: Thing" for i in range(5)
    ]
    assert table.column("iris").to_pylist()[0] == ["urn:0", "urn:0-alias"]
    assert table.column("iri_count").to_pylist() == [3] * 5
    assert table.column("label").to_pylist() == [f"L{i}" for i in range(5)]


def test_stored_vectors_match_embedder(embedder: FastEmbedEmbedder, tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 3)
    output = tmp_path / "g.parquet"

    _embed(embedder, path, output)
    table = pq.read_table(output)

    texts = table.column("embedding_text").to_pylist()
    expected = embedder.embed_many(texts)
    stored = table.column("vector").to_pylist()

    assert len(stored) == 3
    for row, want in zip(stored, expected, strict=True):
        assert np.array_equal(np.asarray(row, dtype=np.float32), want)


def test_file_metadata_describes_the_vectors(
    embedder: FastEmbedEmbedder, tmp_path
):
    path = tmp_path / "my-graph.jsonl"
    _write_records(path, 4)
    output = tmp_path / "my-graph.parquet"

    _embed(embedder, path, output)
    meta = _metadata(output)

    dim = int(embedder.embed("dimension probe").shape[0])
    assert meta[METADATA_PREFIX + "format"] == "1"
    assert meta[METADATA_PREFIX + "graph"] == "my-graph"
    assert meta[METADATA_PREFIX + "model"] == _MODEL
    assert meta[METADATA_PREFIX + "dim"] == str(dim)
    assert meta[METADATA_PREFIX + "metric"] == "cosine"
    # all-MiniLM-L6-v2 emits unit vectors; the flag is measured, not assumed.
    assert meta[METADATA_PREFIX + "normalized"] == "true"
    assert meta[METADATA_PREFIX + "record_count"] == "4"

    vector_type = pq.read_schema(output).field("vector").type
    assert vector_type.list_size == dim


def test_embed_file_honors_limit(embedder: FastEmbedEmbedder, tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 10)
    output = tmp_path / "g.parquet"

    count = _embed(embedder, path, output, limit=3)

    assert count == 3
    assert pq.read_table(output).num_rows == 3
    assert _metadata(output)[METADATA_PREFIX + "record_count"] == "3"


def test_small_flush_rows_still_writes_every_row(
    embedder: FastEmbedEmbedder, tmp_path
):
    path = tmp_path / "g.jsonl"
    _write_records(path, 5)
    output = tmp_path / "g.parquet"

    _embed(embedder, path, output, flush_rows=2)
    parquet = pq.ParquetFile(output)

    assert parquet.metadata.num_row_groups > 1
    table = parquet.read()
    assert table.num_rows == 5
    assert table.column("embedding_text").to_pylist() == [
        f"label: L{i}\ntype: Thing" for i in range(5)
    ]


def test_vector_column_is_mmappable(embedder: FastEmbedEmbedder, tmp_path):
    # The vector column must stay uncompressed and plain-encoded so readers
    # can memory-map the file and view the values zero-copy.
    path = tmp_path / "g.jsonl"
    _write_records(path, 3)
    output = tmp_path / "g.parquet"

    _embed(embedder, path, output)

    parquet = pq.ParquetFile(output, memory_map=True)
    codecs = {}
    for i in range(parquet.metadata.row_group(0).num_columns):
        column = parquet.metadata.row_group(0).column(i)
        codecs[column.path_in_schema] = column.compression
    assert codecs["vector.list.element"] == "UNCOMPRESSED"
    assert codecs["embedding_text"] == "SNAPPY"

    chunk = parquet.read(columns=["vector"]).column("vector").chunks[0]
    view = chunk.flatten().to_numpy(zero_copy_only=True)  # raises if copied
    assert view.dtype == np.float32


def test_accepts_singular_iri_schema(embedder: FastEmbedEmbedder, tmp_path):
    # materialize.py emits a singular `iri`; index.py emits an `iris` list.
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"iri": "urn:a", "embedding_text": "hi"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "g.parquet"

    _embed(embedder, path, output)
    table = pq.read_table(output)

    assert table.column("iris").to_pylist() == [["urn:a"]]
    assert table.column("iri_count").to_pylist() == [1]
    assert table.column("label").to_pylist() == [""]


def test_missing_embedding_text_is_an_error(
    embedder: FastEmbedEmbedder, tmp_path
):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"iris": ["urn:a"], "label": "L"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="embedding_text"):
        _embed(embedder, path, tmp_path / "g.parquet")
