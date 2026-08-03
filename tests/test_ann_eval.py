import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.ann.build import (
    build_index,
    index_manifest_path,
    index_path_for,
)
from okn_embeddings.ann.eval import evaluate, evaluate_and_record
from okn_embeddings.ann.sidecar import SidecarStore
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    PARQUET_COMPRESSION,
    PARQUET_DICTIONARY_COLUMNS,
    rows_to_table,
    vector_schema,
)

_DIM = 16
_COUNT = 60


def _random_unit_vectors(n, dim, seed=7):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _write_parquet(path, vectors):
    schema = vector_schema(_DIM)
    rows = [
        ([f"urn:{i}"], 1, f"L{i}", f"text {i}") for i in range(len(vectors))
    ]
    metadata = {
        METADATA_PREFIX + "format": "1",
        METADATA_PREFIX + "graph": "g",
        METADATA_PREFIX + "model": "test-model",
        METADATA_PREFIX + "dim": str(_DIM),
        METADATA_PREFIX + "metric": "cosine",
        METADATA_PREFIX + "normalized": "true",
        METADATA_PREFIX + "record_count": str(len(vectors)),
    }
    with pq.ParquetWriter(
        path,
        schema,
        compression=PARQUET_COMPRESSION,
        use_dictionary=PARQUET_DICTIONARY_COLUMNS,
    ) as writer:
        writer.write_table(rows_to_table(schema, rows, list(vectors)))
        writer.add_key_value_metadata(metadata)


@pytest.fixture
def indexed_parquet(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _random_unit_vectors(_COUNT, _DIM))
    build_index(parquet)
    return parquet


def test_evaluate_reports_recall_and_timings(indexed_parquet):
    store = SidecarStore.open(indexed_parquet)

    block = evaluate(
        store, queries=20, ks=(5, 10), efs=(8, 64), seed=1
    )

    assert block["queries"] == 20
    assert block["seed"] == 1
    assert block["ks"] == [5, 10]
    assert block["flat"]["mean_query_ms"] >= 0
    assert [s["expansion_search"] for s in block["sweep"]] == [8, 64]
    for step in block["sweep"]:
        assert set(step["recall"]) == {"5", "10"}
        for value in step["recall"].values():
            assert 0.0 <= value <= 1.0
        assert step["mean_query_ms"] >= 0
    # Not exactly 1.0 even with ef > corpus size: HNSW makes no
    # exhaustiveness guarantee, and float32 boundary ties order differently
    # in usearch than in the numpy ground truth. High recall is all that a
    # nondeterministically-built index can promise.
    assert block["sweep"][1]["recall"]["10"] >= 0.8


def test_evaluate_is_deterministic_apart_from_timings(indexed_parquet):
    store = SidecarStore.open(indexed_parquet)

    a = evaluate(store, queries=15, ks=(10,), efs=(64,), seed=3)
    b = evaluate(store, queries=15, ks=(10,), efs=(64,), seed=3)

    assert a["sweep"][0]["recall"] == b["sweep"][0]["recall"]


def test_evaluate_restores_expansion_search(indexed_parquet):
    store = SidecarStore.open(indexed_parquet)
    original = store.index.expansion_search

    evaluate(store, queries=5, ks=(5,), efs=(7,), seed=1)

    assert store.index.expansion_search == original


def test_ks_are_capped_at_corpus_size(indexed_parquet):
    store = SidecarStore.open(indexed_parquet)

    block = evaluate(store, queries=5, ks=(10, 10_000), efs=(64,), seed=1)

    assert block["ks"] == [10, _COUNT]


def test_evaluate_and_record_updates_the_manifest(indexed_parquet):
    block = evaluate_and_record(
        indexed_parquet, queries=10, ks=(5,), efs=(32,), seed=2
    )

    manifest_file = index_manifest_path(index_path_for(indexed_parquet))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    assert manifest["evaluation"] == block
    # The rest of the manifest survives the rewrite.
    assert manifest["index"]["keys"] == "parquet-row-ordinal"
    assert manifest["parent"]["file"] == "g.parquet"


def test_no_write_leaves_manifest_untouched(indexed_parquet):
    manifest_file = index_manifest_path(index_path_for(indexed_parquet))
    before = manifest_file.read_text(encoding="utf-8")

    evaluate_and_record(
        indexed_parquet, queries=5, ks=(5,), efs=(16,), seed=1, write=False
    )

    assert manifest_file.read_text(encoding="utf-8") == before


def test_missing_index_is_an_error(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _random_unit_vectors(10, _DIM))

    store = SidecarStore.open(parquet)
    with pytest.raises(ValueError, match="no index"):
        evaluate(store)
