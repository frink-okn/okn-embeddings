import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.core.embedding import SentenceTransformerEmbedder
from okn_embeddings.indexing.embed import METADATA_PREFIX, convention_id
from okn_embeddings.indexing.embed_texts import (
    TEXTS_FORMAT,
    EmbedTextsResult,
    embed_texts_file,
)

_MODEL = "test-model"


def _text(i: int) -> str:
    return f"label: L{i}\ntype: Thing"


def _line(text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return json.dumps({"digest": digest, "text": text}) + "\n"


def _write_texts(path, n: int) -> list[str]:
    texts = [_text(i) for i in range(n)]
    path.write_text("".join(_line(t) for t in texts), encoding="utf-8")
    return texts


def _embed(embedder, path, output, **kwargs) -> EmbedTextsResult:
    kwargs.setdefault("model_name", _MODEL)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("graph", "g")
    return embed_texts_file(embedder, path, output, **kwargs)


def _metadata(path) -> dict[str, str]:
    raw = pq.read_metadata(path).metadata or {}
    return {
        key.decode().removeprefix(METADATA_PREFIX): value.decode()
        for key, value in raw.items()
        if key.decode().startswith(METADATA_PREFIX)
    }


def test_rows_are_digest_and_vector_in_input_order(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 5)
    output = tmp_path / "embeddings.parquet"

    result = _embed(embedder, path, output)
    table = pq.read_table(output)

    assert result.count == 5
    assert result.truncated == 0
    assert table.column_names == ["text_digest", "vector"]
    assert table.column("text_digest").to_pylist() == [
        hashlib.sha256(t.encode()).digest() for t in texts
    ]


def test_stored_vectors_match_embedder(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 3)
    output = tmp_path / "embeddings.parquet"

    # One batch, so the comparison is bit-exact: the same text embedded in
    # a different batch differs in the last float bits.
    _embed(embedder, path, output, batch_size=3)
    stored = pq.read_table(output).column("vector").to_pylist()

    for row, want in zip(stored, embedder.embed_many(texts), strict=True):
        assert np.array_equal(np.asarray(row, dtype=np.float32), want)


def test_file_metadata_describes_the_vectors(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    _write_texts(path, 4)
    output = tmp_path / "embeddings.parquet"

    _embed(embedder, path, output, graph="sockg")
    meta = _metadata(output)

    dim = int(embedder.embed("dimension probe").shape[0])
    assert meta["format"] == str(TEXTS_FORMAT)
    assert meta["graph"] == "sockg"
    assert meta["model"] == _MODEL
    assert meta["dim"] == str(dim)
    assert meta["normalized"] == "true"
    assert meta["record_count"] == "4"
    assert meta["convention_id"] == convention_id(_MODEL, dim, "cosine", True)
    assert meta["max_seq_length"] == str(embedder.max_seq_length)
    assert meta["truncated_records"] == "0"


def test_columns_are_uncompressed_and_vectors_map_zero_copy(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    _write_texts(path, 3)
    output = tmp_path / "embeddings.parquet"

    _embed(embedder, path, output)

    parquet = pq.ParquetFile(output, memory_map=True)
    group = parquet.metadata.row_group(0)
    for i in range(group.num_columns):
        assert group.column(i).compression == "UNCOMPRESSED"
    chunk = parquet.read(columns=["vector"]).column("vector").chunks[0]
    view = chunk.flatten().to_numpy(zero_copy_only=True)  # raises if copied
    assert view.dtype == np.float32


def test_small_flush_rows_still_writes_every_row(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 5)
    output = tmp_path / "embeddings.parquet"

    _embed(embedder, path, output, flush_rows=2)
    parquet = pq.ParquetFile(output)

    assert parquet.metadata.num_row_groups > 1
    assert parquet.read().column("text_digest").to_pylist() == [
        hashlib.sha256(t.encode()).digest() for t in texts
    ]


def test_texts_past_the_model_window_are_counted(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    # The model reads its first max_seq_length tokens and drops the rest
    # without a word; the count is how a too-generous config shows itself.
    long = "label: L\n" + "has value: something\n" * embedder.max_seq_length
    path = tmp_path / "texts.jsonl"
    path.write_text(_line("label: short") + _line(long), encoding="utf-8")
    output = tmp_path / "embeddings.parquet"

    result = _embed(embedder, path, output)

    assert result.truncated == 1
    assert _metadata(output)["truncated_records"] == "1"


def test_a_digest_that_is_not_the_texts_is_refused(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    wrong = hashlib.sha256(b"something else").hexdigest()
    path.write_text(
        _line("first") + json.dumps({"digest": wrong, "text": "second"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="text 2: digest is not the sha256"):
        _embed(embedder, path, tmp_path / "embeddings.parquet")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"digest": "00" * 32}, "missing non-empty text"),
        ({"digest": "00" * 32, "text": ""}, "missing non-empty text"),
        ({"text": "t"}, "missing its digest"),
        ({"digest": "zz", "text": "t"}, "not hex"),
        ({"digest": "00" * 31, "text": "t"}, "31 bytes, not 32"),
    ],
)
def test_malformed_lines_are_refused(
    embedder: SentenceTransformerEmbedder, tmp_path, line, message
):
    path = tmp_path / "texts.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _embed(embedder, path, tmp_path / "embeddings.parquet")


def test_a_run_that_fails_leaves_the_last_output_whole(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    output = tmp_path / "embeddings.parquet"
    good = tmp_path / "good.jsonl"
    _write_texts(good, 3)
    _embed(embedder, good, output)
    before = output.read_bytes()

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "".join(_line(_text(i)) for i in range(4)) + '{"text": "t"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing its digest"):
        _embed(embedder, bad, output, flush_rows=2)

    assert output.read_bytes() == before
