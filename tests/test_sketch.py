import numpy as np
import pyarrow.parquet as pq
import pytest

from okn_embeddings.ann.sketch import (
    SketchStore,
    build_sketch,
    choose_k,
    cosine_to_distance,
    minibatch_kmeans,
    sketch_path_for,
)
from okn_embeddings.indexing.embed import (
    METADATA_PREFIX,
    rows_to_table,
    vector_schema,
)

_DIM = 8


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _clustered_vectors(
    n_clusters: int, per_cluster: int, spread: float, seed: int = 7
) -> np.ndarray:
    """Unit vectors in tight bundles around random directions."""
    rng = np.random.default_rng(seed)
    anchors = _unit_rows(rng.normal(size=(n_clusters, _DIM)))
    points = np.repeat(anchors, per_cluster, axis=0)
    points = points + rng.normal(scale=spread, size=points.shape)
    return _unit_rows(points).astype(np.float32)


def _write_parquet(path, vectors, metadata_overrides=None):
    schema = vector_schema(_DIM)
    rows = [
        ([f"urn:{i}"], 1, f"L{i}", f"text {i}") for i in range(len(vectors))
    ]
    metadata = {
        METADATA_PREFIX + "format": "1",
        METADATA_PREFIX + "graph": path.stem,
        METADATA_PREFIX + "model": "test-model",
        METADATA_PREFIX + "dim": str(_DIM),
        METADATA_PREFIX + "metric": "cosine",
        METADATA_PREFIX + "normalized": "true",
        METADATA_PREFIX + "record_count": str(len(vectors)),
        METADATA_PREFIX + "convention_id": "abc123",
    }
    metadata.update(metadata_overrides or {})
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(rows_to_table(schema, rows, list(vectors)))
        writer.add_key_value_metadata(metadata)


def test_sketch_path_sits_beside_the_parquet(tmp_path):
    assert sketch_path_for(tmp_path / "g.parquet") == (
        tmp_path / "g.sketch.parquet"
    )


def test_unsaturated_sketch_is_the_vector_set(tmp_path):
    vectors = _clustered_vectors(4, 2, spread=0.05)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, clusters, unsaturated = build_sketch(parquet, k=100)

    assert unsaturated
    assert clusters == len(vectors)
    store = SketchStore.open(sketch_file)
    assert store.unsaturated
    assert np.allclose(
        np.sort(store.centroids, axis=0), np.sort(vectors, axis=0), atol=1e-6
    )
    assert (store.max_radii == 0).all()
    assert (store.counts == 1).all()


def test_radius_soundness(tmp_path):
    # The one conformance obligation: every vector must lie within its
    # cluster's max_radius of some centroid.
    vectors = _clustered_vectors(6, 50, spread=0.1)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, unsaturated = build_sketch(parquet, k=6)
    assert not unsaturated
    store = SketchStore.open(sketch_file)

    for x in vectors:
        distances = np.linalg.norm(store.centroids - x, axis=1)
        assert (distances <= store.max_radii + 1e-6).any()


def test_counts_cover_every_vector(tmp_path):
    vectors = _clustered_vectors(5, 40, spread=0.08)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, _ = build_sketch(parquet, k=5)
    store = SketchStore.open(sketch_file)

    assert store.counts.sum() == len(vectors)


def test_certified_no_is_never_wrong(tmp_path):
    # A certified no must agree with the flat scan: whenever the sketch says
    # "no match above the threshold", brute force must concur.
    vectors = _clustered_vectors(5, 40, spread=0.05, seed=3)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, _ = build_sketch(parquet, k=5)
    store = SketchStore.open(sketch_file)

    rng = np.random.default_rng(11)
    queries = _unit_rows(rng.normal(size=(200, _DIM))).astype(np.float32)
    threshold = 0.8

    for q in queries:
        if store.certified_no(q, threshold):
            best = float((vectors @ q).max())
            assert best < threshold


def test_best_distance_bound_is_a_true_lower_bound(tmp_path):
    vectors = _clustered_vectors(4, 30, spread=0.1, seed=5)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, _ = build_sketch(parquet, k=4)
    store = SketchStore.open(sketch_file)

    rng = np.random.default_rng(13)
    queries = _unit_rows(rng.normal(size=(50, _DIM))).astype(np.float32)
    for q in queries:
        true_best = float(np.linalg.norm(vectors - q, axis=1).min())
        assert store.best_distance_bound(q) <= true_best + 1e-6


def test_metadata_carries_identity_and_provenance(tmp_path):
    vectors = _clustered_vectors(3, 20, spread=0.05)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, _ = build_sketch(parquet, k=3, radius_quantile=0.75)
    store = SketchStore.open(sketch_file)

    assert store.convention_id == "abc123"
    assert store.metadata["kind"] == "routing-sketch"
    assert store.metadata["graph"] == "g"
    assert store.metadata["radius_quantile"] == "0.75"
    assert store.metadata["parent_file"] == "g.parquet"
    assert len(store.metadata["parent_sha256"]) == 64


def test_yield_estimate_prefers_matching_clusters(tmp_path):
    vectors = _clustered_vectors(2, 50, spread=0.02, seed=9)
    parquet = tmp_path / "g.parquet"
    _write_parquet(parquet, vectors)

    sketch_file, _, _ = build_sketch(parquet, k=2)
    store = SketchStore.open(sketch_file)

    # A query sitting on a cluster anchor should see yield from that
    # cluster; an orthogonal-ish query far from both should see none.
    on_cluster = vectors[0]
    assert store.yield_estimate(on_cluster, 0.9) >= 1

    rng = np.random.default_rng(21)
    for _ in range(50):
        q = _unit_rows(rng.normal(size=(1, _DIM)))[0].astype(np.float32)
        if float((vectors @ q).max()) < 0.5:
            assert store.yield_estimate(q, 0.99) == 0
            break
    else:
        pytest.skip("no sufficiently distant query found")


def test_choose_k_stops_when_radii_collapse():
    # 8 distinct points, each repeated 100 times: at the first sweep step
    # every cluster's radius is exactly zero, so doubling K cannot improve
    # anything and the sweep must stop at the starting K instead of paying
    # for centroids that cannot prune better.
    rng_data = np.random.default_rng(17)
    anchors = _unit_rows(rng_data.normal(size=(8, _DIM))).astype(np.float32)
    vectors = np.repeat(anchors, 100, axis=0)
    rng = np.random.default_rng(0)

    k = choose_k(vectors, rng=rng, radius_quantile=0.9, max_k=256)

    assert k == 64


def test_cosine_to_distance_endpoints():
    assert cosine_to_distance(1.0) == pytest.approx(0.0)
    assert cosine_to_distance(0.0) == pytest.approx(np.sqrt(2.0))
    assert cosine_to_distance(-1.0) == pytest.approx(2.0)


def test_minibatch_kmeans_shape_and_dtype():
    vectors = _clustered_vectors(4, 25, spread=0.1)
    rng = np.random.default_rng(1)
    centroids = minibatch_kmeans(vectors, 4, rng=rng, iterations=20)
    assert centroids.shape == (4, _DIM)
    assert centroids.dtype == np.float32
