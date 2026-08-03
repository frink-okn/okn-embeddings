import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.ann.build import build_index, index_manifest_path
from okn_embeddings.ann.sidecar import SidecarStore
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    rows_to_table,
    vector_schema,
)

_DIM = 4


def _unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


_VECTORS = [
    _unit([1, 0, 0, 0]),
    _unit([0, 1, 0, 0]),
    _unit([0, 0, 1, 0]),
    _unit([0, 0, 0, 1]),
    _unit([1, 1, 0, 0]),
]


def _write_parquet(path, vectors, metadata_overrides=None):
    schema = vector_schema(_DIM)
    rows = [
        ([f"urn:{i}", f"urn:{i}-alias"], 2, f"L{i}", f"text {i}")
        for i in range(len(vectors))
    ]
    metadata = {
        METADATA_PREFIX + "format": "1",
        METADATA_PREFIX + "graph": "test-graph",
        METADATA_PREFIX + "model": "test-model",
        METADATA_PREFIX + "dim": str(_DIM),
        METADATA_PREFIX + "metric": "cosine",
        METADATA_PREFIX + "normalized": "true",
        METADATA_PREFIX + "record_count": str(len(vectors)),
    }
    metadata.update(metadata_overrides or {})
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(rows_to_table(schema, rows, list(vectors)))
        writer.add_key_value_metadata(metadata)


def test_flat_search_without_index(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    store = SidecarStore.open(parquet)

    assert not store.has_index
    assert store.count == len(_VECTORS)
    assert store.graph == "test-graph"

    rows = store.search(_VECTORS[2], 3)
    assert len(rows) == 3
    assert rows[0].id == "test-graph:2"
    assert rows[0].score == pytest.approx(1.0, abs=1e-5)
    assert rows[0].label == "L2"
    assert rows[0].iris == ["urn:2", "urn:2-alias"]
    assert rows[0].primary_uri == "urn:2"
    assert rows[0].iri_count == 2
    assert rows[0].repr == "text 2"
    assert rows[0].graph == "test-graph"
    # Best-first ordering.
    scores = [r.score for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_ann_search_with_index(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)

    store = SidecarStore.open(parquet)

    assert store.has_index
    rows = store.search(_VECTORS[1], 2)
    assert rows[0].id == "test-graph:1"
    assert rows[0].score == pytest.approx(1.0, abs=1e-3)


def test_exact_flag_forces_flat_scan(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)

    store = SidecarStore.open(parquet)
    ann = store.search(_VECTORS[0], len(_VECTORS))
    exact = store.search(_VECTORS[0], len(_VECTORS), exact=True)

    # The top two are unambiguous ([1,0,0,0] then [1,1,0,0]); the rest tie
    # at score 0, where ordering is backend-dependent.
    assert [r.id for r in ann[:2]] == [r.id for r in exact[:2]]
    assert {r.id for r in ann} == {r.id for r in exact}


def test_use_index_false_skips_the_index(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)

    store = SidecarStore.open(parquet, use_index=False)
    assert not store.has_index
    assert store.search(_VECTORS[3], 1)[0].id == "test-graph:3"


def test_stale_index_is_rejected(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)
    # Regenerate the parquet with different vectors; the index is now stale.
    _write_parquet(parquet, list(reversed(_VECTORS)))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        SidecarStore.open(parquet)


def test_index_without_manifest_is_rejected(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    index_file, _, _ = build_index(parquet)
    index_manifest_path(index_file).unlink()

    with pytest.raises(ValueError, match="manifest"):
        SidecarStore.open(parquet)


def test_unnormalized_vectors_are_normalized_at_open(tmp_path):
    parquet = tmp_path / "g.parquet"
    scaled = [v * 7.5 for v in _VECTORS]
    _write_parquet(
        parquet,
        scaled,
        metadata_overrides={METADATA_PREFIX + "normalized": "false"},
    )

    store = SidecarStore.open(parquet)
    rows = store.search(np.asarray([0, 0, 1, 0], dtype=np.float32), 1)

    assert rows[0].id == "test-graph:2"
    assert rows[0].score == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_count_returns_all_rows(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    store = SidecarStore.open(parquet)
    assert len(store.search(_VECTORS[0], 100)) == len(_VECTORS)


def test_vector_for_iri(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    store = SidecarStore.open(parquet)

    vector = store.vector_for_iri("urn:3-alias")
    assert vector is not None
    assert np.allclose(vector, _VECTORS[3])
    assert store.vector_for_iri("urn:nope") is None
