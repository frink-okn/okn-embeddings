import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.ann.build import build_index, index_manifest_path
from okn_embeddings.ann.sidecar import SidecarStore
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    PARQUET_COMPRESSION,
    PARQUET_DICTIONARY_COLUMNS,
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


def _write_parquet(path, vectors, row_groups=1, metadata_overrides=None):
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
    per_group = -(-len(vectors) // row_groups)
    with pq.ParquetWriter(
        path,
        schema,
        compression=PARQUET_COMPRESSION,
        use_dictionary=PARQUET_DICTIONARY_COLUMNS,
    ) as writer:
        for start in range(0, len(vectors), per_group):
            writer.write_table(
                rows_to_table(
                    schema,
                    rows[start : start + per_group],
                    list(vectors[start : start + per_group]),
                )
            )
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
    scores = [r.score for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_open_reads_nothing_and_ann_search_touches_no_vectors(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)

    store = SidecarStore.open(parquet)
    assert store._vector_parts is None
    assert store._group_cache == {}

    rows = store.search(_VECTORS[1], 2)
    assert rows[0].id == "test-graph:1"
    assert rows[0].score == pytest.approx(1.0, abs=1e-3)
    # The ANN path resolved its hits without materializing vectors.
    assert store._vector_parts is None
    assert store._group_cache


def test_flat_scan_spans_row_groups(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS, row_groups=3)
    assert pq.ParquetFile(parquet).metadata.num_row_groups > 1

    store = SidecarStore.open(parquet)
    rows = store.search(_VECTORS[4], 1)

    # A hit in the last row group resolves to the right global record.
    assert rows[0].id == "test-graph:4"
    assert rows[0].label == "L4"
    assert len(store.vector_parts) > 1


def test_vector_parts_are_zero_copy_views(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS, row_groups=2)

    store = SidecarStore.open(parquet)

    for part in store.vector_parts:
        # A view over the mapped file owns no data.
        assert part.base is not None


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


def test_row_count_mismatch_is_always_rejected(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)
    _write_parquet(parquet, _VECTORS[:3])

    with pytest.raises(ValueError, match="rebuild"):
        SidecarStore.open(parquet)


def test_content_swap_needs_verify_to_be_caught(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)
    build_index(parquet)
    # Same schema, same row count, same size -- only the content differs.
    _write_parquet(parquet, list(reversed(_VECTORS)))
    if parquet.stat().st_size != _load_parent_bytes(parquet):
        pytest.skip("rewrite changed the file size; covered by size check")

    SidecarStore.open(parquet)  # cheap checks cannot see it

    with pytest.raises(ValueError, match="sha256 mismatch"):
        SidecarStore.open(parquet, verify=True)


def _load_parent_bytes(parquet):
    import json

    from okn_embeddings.ann.build import index_path_for

    manifest = index_manifest_path(index_path_for(parquet))
    return json.loads(manifest.read_text(encoding="utf-8"))["parent"]["bytes"]


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
    _write_parquet(parquet, _VECTORS, row_groups=2)

    store = SidecarStore.open(parquet)

    vector = store.vector_for_iri("urn:3-alias")
    assert vector is not None
    assert np.allclose(vector, _VECTORS[3])
    assert store.vector_for_iri("urn:nope") is None


def test_vector_matrix_collects_all_rows(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS, row_groups=2)

    store = SidecarStore.open(parquet)
    matrix = store.vector_matrix()

    assert matrix.shape == (len(_VECTORS), _DIM)
    assert np.allclose(matrix, np.stack(_VECTORS))
