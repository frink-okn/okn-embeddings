import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from usearch.index import Index

from okn_embeddings.ann.build import (
    build_index,
    index_manifest_path,
    index_path_for,
)
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    rows_to_table,
    vector_schema,
)
from okn_embeddings.indexing.manifest import file_sha256


def _load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))

_DIM = 4


def _unit(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# Distinct directions, so exact nearest neighbors are unambiguous.
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
        ([f"urn:{i}"], 1, f"L{i}", f"text {i}")
        for i in range(len(vectors))
    ]
    metadata = {
        METADATA_PREFIX + "format": "1",
        METADATA_PREFIX + "graph": path.stem,
        METADATA_PREFIX + "model": "test-model",
        METADATA_PREFIX + "dim": str(_DIM),
        METADATA_PREFIX + "metric": "cosine",
        METADATA_PREFIX + "normalized": "true",
        METADATA_PREFIX + "record_count": str(len(vectors)),
    }
    metadata.update(metadata_overrides or {})

    per_group = -(-len(vectors) // row_groups)
    with pq.ParquetWriter(path, schema) as writer:
        for start in range(0, len(vectors), per_group):
            group_rows = rows[start : start + per_group]
            group_vectors = list(vectors[start : start + per_group])
            writer.write_table(rows_to_table(schema, group_rows, group_vectors))
        writer.add_key_value_metadata(metadata)


def test_default_paths_sit_beside_the_parquet(tmp_path):
    parquet = tmp_path / "g.parquet"
    assert index_path_for(parquet) == tmp_path / "g.usearch"
    # Appended, not substituted: must not collide with the textify manifest
    # g.meta.json.
    assert index_manifest_path(tmp_path / "g.usearch") == (
        tmp_path / "g.usearch.meta.json"
    )


def test_build_index_keys_are_row_ordinals(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    index_file, manifest_file, count = build_index(parquet)

    assert count == len(_VECTORS)
    assert index_file == tmp_path / "g.usearch"
    assert manifest_file == tmp_path / "g.usearch.meta.json"

    restored = Index.restore(str(index_file), view=True)
    assert len(restored) == len(_VECTORS)
    for ordinal, vector in enumerate(_VECTORS):
        matches = restored.search(vector, 1)
        assert matches.keys[0] == ordinal


def test_multiple_row_groups_keep_keys_continuous(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS, row_groups=3)
    assert pq.ParquetFile(parquet).metadata.num_row_groups > 1

    index_file, _, count = build_index(parquet)
    restored = Index.restore(str(index_file), view=True)

    assert count == len(_VECTORS)
    # A vector from the last row group still maps to its global ordinal.
    matches = restored.search(_VECTORS[4], 1)
    assert matches.keys[0] == 4


def test_manifest_identifies_parent_and_parameters(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    index_file, _, _ = build_index(
        parquet, connectivity=8, expansion_add=64, expansion_search=32
    )
    manifest = _load_manifest(index_manifest_path(index_file))

    assert manifest["format"] == 1
    assert manifest["index"]["library"] == "usearch"
    assert manifest["index"]["file"] == "g.usearch"
    assert manifest["index"]["metric"] == "cos"
    assert manifest["index"]["dtype"] == "f32"
    assert manifest["index"]["ndim"] == _DIM
    assert manifest["index"]["connectivity"] == 8
    assert manifest["index"]["expansion_add"] == 64
    assert manifest["index"]["expansion_search"] == 32
    assert manifest["index"]["count"] == len(_VECTORS)
    assert manifest["index"]["keys"] == "parquet-row-ordinal"

    assert manifest["parent"]["file"] == "g.parquet"
    assert manifest["parent"]["sha256"] == file_sha256(parquet)
    assert manifest["parent"]["metadata"]["model"] == "test-model"
    assert manifest["parent"]["metadata"]["record_count"] == "5"


def test_default_parameters_are_recorded(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    _, manifest_file, _ = build_index(parquet)
    manifest = _load_manifest(manifest_file)

    # Whatever usearch chose, the manifest states it explicitly.
    assert manifest["index"]["connectivity"] > 0
    assert manifest["index"]["expansion_add"] > 0
    assert manifest["index"]["expansion_search"] > 0


def test_i8_dtype_builds_and_searches(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    index_file, manifest_file, _ = build_index(parquet, dtype="i8")
    restored = Index.restore(str(index_file), view=True)

    matches = restored.search(_VECTORS[2], 1)
    assert matches.keys[0] == 2
    assert _load_manifest(manifest_file)["index"]["dtype"] == "i8"


def test_unsupported_dtype_is_an_error(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, _VECTORS)

    with pytest.raises(ValueError, match="dtype"):
        build_index(parquet, dtype="f64")


def test_unsupported_metric_is_an_error(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(
        parquet,
        _VECTORS,
        metadata_overrides={METADATA_PREFIX + "metric": "euclidean"},
    )

    with pytest.raises(ValueError, match="metric"):
        build_index(parquet)


def test_dim_mismatch_is_an_error(tmp_path):
    parquet = tmp_path / "g.parquet"
    _write_parquet(
        parquet,
        _VECTORS,
        metadata_overrides={METADATA_PREFIX + "dim": "999"},
    )

    with pytest.raises(ValueError, match="dim"):
        build_index(parquet)
