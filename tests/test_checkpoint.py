import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from okn_embeddings.core.embedding import SentenceTransformerEmbedder
from okn_embeddings.indexing.checkpoint import Checkpoint
from okn_embeddings.indexing.cli import app
from okn_embeddings.indexing.embed_texts import embed_texts_file
from okn_embeddings.indexing.probes import probe_vectors
from okn_embeddings.indexing.reuse import ReuseFile, ReuseSource

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

runner = CliRunner()


class _Crash(Exception):
    pass


class _DiesAfter:
    """The real embedder, until it has embedded `limit` input texts."""

    def __init__(self, inner: SentenceTransformerEmbedder, limit: int):
        self.inner = inner
        self.limit = limit
        self.embedded: list[str] = []

    @property
    def max_seq_length(self) -> int:
        return self.inner.max_seq_length

    def token_counts(self, texts):
        return self.inner.token_counts(texts)

    def embed(self, text):
        return self.inner.embed(text)

    def embed_many(self, texts):
        if any(t.startswith("label: L") for t in texts):
            if len(self.embedded) + len(texts) > self.limit:
                raise _Crash
            self.embedded.extend(texts)
        return self.inner.embed_many(texts)


def _text(i: int) -> str:
    return f"label: L{i}\ntype: Thing"


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def _write_texts(path, n: int) -> list[str]:
    texts = [_text(i) for i in range(n)]
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


def _open_parts(checkpoint, embedder) -> list[ReuseFile]:
    probes = probe_vectors(embedder)
    dim = len(embedder.embed("dimension probe"))
    return [
        ReuseFile.open(p, probes, model_name=_MODEL, dim=dim)
        for p in checkpoint.existing
    ]


def test_open_creates_the_directory_and_ignores_other_files(tmp_path):
    directory = tmp_path / "ckpt"
    checkpoint = Checkpoint.open(directory)
    assert directory.is_dir()
    assert checkpoint.existing == []

    (directory / "part-000003.parquet").write_bytes(b"")
    (directory / "part-000001.parquet").write_bytes(b"")
    (directory / "notes.txt").write_text("keep")
    (directory / "part-000004.parquet.partial").write_bytes(b"")

    reopened = Checkpoint.open(directory)

    assert [p.name for p in reopened.existing] == [
        "part-000001.parquet",
        "part-000003.parquet",
    ]
    # A part the crash caught mid-write never reached its name.
    assert not (directory / "part-000004.parquet.partial").exists()
    assert (directory / "notes.txt").exists()


def test_a_crashed_run_resumes_without_re_embedding_its_parts(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 10)
    output = tmp_path / "out.parquet"
    directory = tmp_path / "ckpt"

    dying = _DiesAfter(embedder, limit=6)
    with pytest.raises(_Crash):
        _embed(
            dying,
            path,
            output,
            checkpoint=Checkpoint.open(directory),
            checkpoint_rows=4,
        )
    assert not output.exists()

    # Six embedded, but only whole parts survive: one of four.
    checkpoint = Checkpoint.open(directory)
    assert len(checkpoint.existing) == 1
    parts = _open_parts(checkpoint, embedder)
    assert parts[0].digests.tolist() == [_digest(t) for t in texts[:4]]

    resumed = _DiesAfter(embedder, limit=100)
    result = _embed(
        resumed,
        path,
        output,
        reuse=ReuseSource(parts),
        checkpoint=checkpoint,
        checkpoint_rows=4,
    )

    assert resumed.embedded == texts[4:]
    assert (result.count, result.reused) == (10, 4)
    table = pq.read_table(output)
    assert table.column("text_digest").to_pylist() == [
        _digest(t) for t in texts
    ]


def test_parts_hold_only_freshly_embedded_vectors(
    embedder: SentenceTransformerEmbedder, tmp_path
):
    earlier_texts = tmp_path / "earlier.jsonl"
    _write_texts(earlier_texts, 4)
    earlier = tmp_path / "earlier.parquet"
    _embed(embedder, earlier_texts, earlier)

    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 8)
    checkpoint = Checkpoint.open(tmp_path / "ckpt")
    probes = probe_vectors(embedder)
    dim = len(embedder.embed("dimension probe"))

    _embed(
        embedder,
        path,
        tmp_path / "out.parquet",
        reuse=ReuseSource(
            [ReuseFile.open(earlier, probes, model_name=_MODEL, dim=dim)]
        ),
        checkpoint=checkpoint,
        checkpoint_rows=2,
    )

    written = [
        ReuseFile.open(p, probes, model_name=_MODEL, dim=dim)
        for p in checkpoint.written
    ]
    assert [d for f in written for d in f.digests.tolist()] == [
        _digest(t) for t in texts[4:]
    ]


def test_remove_leaves_parts_it_did_not_use(tmp_path):
    for n in (1, 2):
        (tmp_path / f"part-00000{n}.parquet").write_bytes(b"")
    checkpoint = Checkpoint.open(tmp_path)
    used, refused = checkpoint.existing

    checkpoint.remove([used])

    assert not used.exists()
    assert refused.exists()


def test_the_cli_resumes_and_clears_its_checkpoint(
    embedder: SentenceTransformerEmbedder, tmp_path, monkeypatch
):
    monkeypatch.setenv("MODEL_NAME", _MODEL)
    monkeypatch.setenv("EMBED_DEVICE", "cpu")
    path = tmp_path / "texts.jsonl"
    texts = _write_texts(path, 6)
    output = tmp_path / "out.parquet"
    directory = tmp_path / "ckpt"

    with pytest.raises(_Crash):
        _embed(
            _DiesAfter(embedder, limit=3),
            path,
            output,
            checkpoint=Checkpoint.open(directory),
            checkpoint_rows=2,
        )
    assert len(list(directory.iterdir())) == 1

    result = runner.invoke(
        app,
        [
            "embed-texts",
            str(path),
            "-o",
            str(output),
            "--graph",
            "g",
            "--no-progress",
            "--checkpoint",
            str(directory),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Resuming: 1 of 1 checkpoint parts" in result.output
    assert "Reused 2 vectors; embedded 4" in result.output
    assert list(directory.iterdir()) == []
    table = pq.read_table(output)
    assert table.column("text_digest").to_pylist() == [
        _digest(t) for t in texts
    ]
    vectors = np.asarray(table.column("vector").to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(
        vectors, np.stack(embedder.embed_many(texts)), atol=1e-6
    )
