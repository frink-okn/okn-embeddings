import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.core.embedding import SentenceTransformerEmbedder
from okn_embeddings.indexing.embed import embed_file
from okn_embeddings.indexing.embed_texts import embed_texts_file
from okn_embeddings.indexing.probes import PROBE_TEXTS, probe_vectors
from okn_embeddings.indexing.reuse import (
    ReuseFile,
    ReuseRefused,
    ReuseSource,
)

_MODEL = "test-model"


class _Counting:
    """The real embedder, recording every text that reaches the model
    other than the dimension probe and the known-answer probes."""

    def __init__(self, inner: SentenceTransformerEmbedder, shift: float = 0):
        self.inner = inner
        self.shift = np.float32(shift)
        self.calls: list[list[str]] = []

    @property
    def max_seq_length(self) -> int:
        return self.inner.max_seq_length

    def token_counts(self, texts):
        return self.inner.token_counts(texts)

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        if texts != ["dimension probe"] and tuple(texts) != PROBE_TEXTS:
            self.calls.append(list(texts))
        return [v + self.shift for v in self.inner.embed_many(texts)]

    @property
    def embedded(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _text(i: int) -> str:
    return f"label: L{i}\ntype: Thing"


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def _write_texts(path, indices) -> list[str]:
    texts = [_text(i) for i in indices]
    path.write_text(
        "".join(
            json.dumps({"digest": _digest(t).hex(), "text": t}) + "\n"
            for t in texts
        ),
        encoding="utf-8",
    )
    return texts


def _embed(embedder, path, output, **kwargs):
    kwargs.setdefault("model_name", _MODEL)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("graph", "g")
    return embed_texts_file(embedder, path, output, **kwargs)


def _open_file(path, embedder, model_name=_MODEL, dim=None) -> ReuseFile:
    if dim is None:
        dim = len(embedder.embed("dimension probe"))
    return ReuseFile.open(
        path, probe_vectors(embedder), model_name=model_name, dim=dim
    )


def _open(path, embedder, **kwargs) -> ReuseSource:
    return ReuseSource([_open_file(path, embedder, **kwargs)])


def _earlier(embedder, tmp_path, indices):
    path = tmp_path / "earlier.jsonl"
    _write_texts(path, indices)
    output = tmp_path / "earlier.parquet"
    _embed(embedder, path, output)
    return output


def _rows(path) -> dict[bytes, np.ndarray]:
    table = pq.read_table(path)
    return {
        digest: np.asarray(vector, dtype=np.float32)
        for digest, vector in zip(
            table.column("text_digest").to_pylist(),
            table.column("vector").to_pylist(),
            strict=True,
        )
    }


def test_unchanged_texts_never_reach_the_model(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(5))
    path = tmp_path / "texts.jsonl"
    _write_texts(path, range(5))
    counting = _Counting(embedder)

    result = _embed(
        counting,
        path,
        tmp_path / "out.parquet",
        reuse=_open(earlier, counting),
    )

    assert counting.embedded == []
    assert result.reused == 5
    out, before = _rows(tmp_path / "out.parquet"), _rows(earlier)
    assert out.keys() == before.keys()
    for digest, vector in out.items():
        assert np.array_equal(vector, before[digest])


def test_only_changed_texts_are_embedded_and_order_is_kept(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(5))
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, range(2, 8))
    counting = _Counting(embedder)
    output = tmp_path / "out.parquet"

    result = _embed(counting, path, output, reuse=_open(earlier, counting))

    assert sorted(counting.embedded) == [_text(i) for i in range(5, 8)]
    assert (result.count, result.reused) == (6, 3)
    digests = pq.read_table(output).column("text_digest").to_pylist()
    assert digests == [_digest(t) for t in texts]
    meta = pq.read_metadata(output).metadata
    assert meta[b"okn-embeddings.reused_records"] == b"3"


def test_misses_are_gathered_into_full_batches(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    # Earlier holds the even texts; the input alternates hit, miss. Batched
    # per read, every model call would carry one text.
    earlier = _earlier(embedder, tmp_path, range(0, 12, 2))
    path = tmp_path / "texts.jsonl"
    _write_texts(path, range(12))
    counting = _Counting(embedder)

    _embed(
        counting,
        path,
        tmp_path / "out.parquet",
        batch_size=4,
        reuse=_open(earlier, counting),
    )

    assert [len(call) for call in counting.calls] == [4, 2]


def test_a_reused_run_matches_a_fresh_one(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(4))
    path = tmp_path / "texts.jsonl"
    _write_texts(path, range(8))

    _embed(embedder, path, tmp_path / "fresh.parquet")
    _embed(
        embedder,
        path,
        tmp_path / "reused.parquet",
        reuse=_open(earlier, embedder),
    )

    fresh = _rows(tmp_path / "fresh.parquet")
    reused = _rows(tmp_path / "reused.parquet")
    assert fresh.keys() == reused.keys()
    for digest in fresh:
        np.testing.assert_allclose(reused[digest], fresh[digest], atol=1e-6)


def test_another_model_name_is_refused(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(2))

    with pytest.raises(ReuseRefused, match="embedded with 'test-model'"):
        _open(earlier, embedder, model_name="other-model")


def test_another_width_is_refused(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(2))

    with pytest.raises(ReuseRefused, match="-dimensional, not 7"):
        _open(earlier, embedder, dim=7)


def test_the_same_name_with_moved_vectors_is_refused(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier = _earlier(embedder, tmp_path, range(2))

    with pytest.raises(ReuseRefused, match="moves its probes"):
        _open(earlier, _Counting(embedder, shift=1e-3))


def test_the_record_layout_is_refused(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({"iris": ["urn:a"], "embedding_text": "hi"}) + "\n"
    )
    output = tmp_path / "records.parquet"
    embed_file(embedder, records, output, model_name=_MODEL, batch_size=2)

    with pytest.raises(ReuseRefused, match="layout format '1'"):
        _open(output, embedder)


def test_digests_differing_only_in_trailing_zero_bytes_are_distinct(
    tmp_path,
):
    # numpy's S dtype strips trailing NULs from scalars; with every digest
    # exactly 32 bytes that must still be a bytewise comparison.
    a = b"a" + b"\0" * 31
    b = b"a" + b"\0" * 30 + b"\1"
    keys = np.array([a, b], dtype="S32")
    vectors = np.arange(4, dtype=np.float32).reshape(2, 2)
    source = ReuseSource([ReuseFile(tmp_path, keys, [vectors], [0])])

    found = source.lookup([b, a, b"a" + b"\0" * 30 + b"\2"])

    assert [None if v is None else v.tolist() for v in found] == [
        [2.0, 3.0],
        [0.0, 1.0],
        None,
    ]


def test_several_files_merge_into_one_lookup(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a = _earlier(embedder, first, range(0, 3))
    b = _earlier(embedder, second, range(3, 6))
    path = tmp_path / "texts.jsonl"
    _write_texts(path, range(7))
    counting = _Counting(embedder)

    result = _embed(
        counting,
        path,
        tmp_path / "out.parquet",
        reuse=ReuseSource([_open_file(a, counting), _open_file(b, counting)]),
    )

    assert counting.embedded == [_text(6)]
    assert result.reused == 6


def test_where_files_share_a_digest_the_first_wins(tmp_path):
    digest = b"d" * 32
    one = ReuseFile(
        tmp_path, np.array([digest], dtype="S32"), [np.ones((1, 2))], [0]
    )
    two = ReuseFile(
        tmp_path, np.array([digest], dtype="S32"), [np.zeros((1, 2))], [0]
    )

    (found,) = ReuseSource([one, two]).lookup([digest])
    (found_reversed,) = ReuseSource([two, one]).lookup([digest])

    assert found is not None and found.tolist() == [1.0, 1.0]
    assert found_reversed is not None and found_reversed.tolist() == [0, 0]


def test_no_files_finds_nothing():
    assert ReuseSource([]).lookup([b"d" * 32]) == [None]
