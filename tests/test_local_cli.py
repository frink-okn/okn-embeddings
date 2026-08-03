import json

import numpy as np
import pyarrow.parquet as pq
from typer.testing import CliRunner

from okn_embeddings.cli.main import app
from okn_embeddings.core.embedding import FastEmbedEmbedder
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    rows_to_table,
    vector_schema,
)

runner = CliRunner()

_DIM = 4


def _unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def _write_parquet(path, graph, labeled_vectors, dim=_DIM):
    """labeled_vectors: list of (label, vector); IRIs derive from labels."""
    schema = vector_schema(dim)
    rows = [
        ([f"urn:{graph}:{label}"], 1, label, f"label: {label}")
        for label, _ in labeled_vectors
    ]
    vectors = [np.asarray(v, dtype=np.float32) for _, v in labeled_vectors]
    metadata = {
        METADATA_PREFIX + "format": "1",
        METADATA_PREFIX + "graph": graph,
        METADATA_PREFIX + "model": "test-model",
        METADATA_PREFIX + "dim": str(dim),
        METADATA_PREFIX + "metric": "cosine",
        METADATA_PREFIX + "normalized": "true",
        METADATA_PREFIX + "record_count": str(len(rows)),
    }
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(rows_to_table(schema, rows, vectors))
        writer.add_key_value_metadata(metadata)


def _two_graphs(tmp_path):
    a = tmp_path / "graph-a.parquet"
    b = tmp_path / "graph-b.parquet"
    _write_parquet(
        a,
        "graph-a",
        [("A0", _unit([1, 0, 0, 0])), ("A1", _unit([0, 1, 0, 0]))],
    )
    _write_parquet(
        b,
        "graph-b",
        [("B0", _unit([1, 0.1, 0, 0])), ("B1", _unit([0, 0, 1, 0]))],
    )
    return a, b


def test_local_node_search_merges_graphs_by_score(tmp_path):
    a, b = _two_graphs(tmp_path)

    result = runner.invoke(
        app,
        ["local", "urn:graph-a:A0", str(a), str(b), "-t", "node", "--json"],
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    # Self hit first, then graph-b's nearby vector, across file boundaries.
    assert [r["label"] for r in rows[:2]] == ["A0", "B0"]
    assert [r["graph"] for r in rows[:2]] == ["graph-a", "graph-b"]
    assert rows[0]["primary_uri"] == "urn:graph-a:A0"
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_local_limit_and_offset_paginate_merged_results(tmp_path):
    a, b = _two_graphs(tmp_path)
    args = ["local", "urn:graph-a:A0", str(a), str(b), "-t", "node", "--json"]

    full = json.loads(runner.invoke(app, args).stdout)
    page = json.loads(
        runner.invoke(app, args + ["--limit", "2", "--offset", "1"]).stdout
    )

    assert [r["label"] for r in page] == [r["label"] for r in full[1:3]]


def test_local_show_repr_includes_embedding_text(tmp_path):
    a, b = _two_graphs(tmp_path)

    result = runner.invoke(
        app,
        [
            "local",
            "urn:graph-b:B1",
            str(a),
            str(b),
            "-t",
            "node",
            "--json",
            "--show-repr",
        ],
    )

    rows = json.loads(result.stdout)
    assert rows[0]["repr"] == "label: B1"


def test_local_unknown_iri_fails(tmp_path):
    a, b = _two_graphs(tmp_path)

    result = runner.invoke(
        app, ["local", "urn:nope", str(a), str(b), "-t", "node"]
    )

    assert result.exit_code == 1
    assert "IRI not found" in result.output


def test_local_missing_file_fails(tmp_path):
    result = runner.invoke(
        app, ["local", "x", str(tmp_path / "absent.parquet")]
    )

    assert result.exit_code == 1
    assert "not found" in result.output


def test_local_text_search_uses_the_model(
    embedder: FastEmbedEmbedder, tmp_path
):
    # Vectors produced by the same model the CLI will load, so a text query
    # must rank its own document first.
    texts = {"Aspirin": "label: Aspirin", "Volcano": "label: Volcano"}
    vectors = dict(
        zip(texts, embedder.embed_many(list(texts.values())), strict=True)
    )
    path = tmp_path / "g.parquet"
    _write_parquet(
        path,
        "g",
        [(label, vectors[label]) for label in texts],
        dim=len(next(iter(vectors.values()))),
    )

    result = runner.invoke(app, ["local", "aspirin", str(path), "--json"])

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert rows[0]["label"] == "Aspirin"
