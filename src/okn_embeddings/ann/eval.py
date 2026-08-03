"""Measure an ANN index's fidelity against exact search.

The index is approximate; this module says by how much, so the number can
live in the artifact instead of in folklore. Sampled stored vectors are the
queries (index fidelity is a vector-space property -- no model, no curated
query set), exact brute force over the Parquet vectors is the ground truth,
and recall@k is measured across a sweep of `expansion_search` values. Flat
scan time is measured alongside, so each evaluation also answers whether the
index earns its keep over exact search at this corpus size.

Results are written into the index manifest under `"evaluation"`, making the
sidecar self-describing about what "approximate" means: params, measured
recall, and timings, next to the build parameters they refer to.
"""

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..indexing.manifest import write_manifest
from .build import index_manifest_path, index_path_for
from .sidecar import SidecarStore

DEFAULT_QUERIES = 200
DEFAULT_KS = (10, 100)
DEFAULT_EFS = (16, 32, 64, 128, 256)
DEFAULT_SEED = 42

# Ground-truth queries are scored in batches of this many, bounding the
# (rows x batch) similarity matrix.
_QUERY_BATCH = 64


def evaluate(
    store: SidecarStore,
    *,
    queries: int = DEFAULT_QUERIES,
    ks: Sequence[int] = DEFAULT_KS,
    efs: Sequence[int] = DEFAULT_EFS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate a store's index against exact search over its own vectors.

    Returns the evaluation block. Deterministic for a given (store, seed)
    apart from the timing fields.
    """
    if not store.has_index:
        raise ValueError(
            f"{store.path} has no index to evaluate; build one with "
            "`okn-indexing build-index`"
        )
    if store.count == 0:
        raise ValueError(f"{store.path} has no records")

    ks = sorted({min(k, store.count) for k in ks})
    k_max = max(ks)
    matrix = store.vector_matrix()

    rng = np.random.default_rng(seed)
    query_ids = np.sort(
        rng.choice(store.count, size=min(queries, store.count), replace=False)
    )
    query_vectors = matrix[query_ids]

    exact_ranked, flat_ms = _exact_ground_truth(matrix, query_vectors, k_max)

    index = store.index
    assert index is not None  # has_index checked above
    original_ef = index.expansion_search
    sweep = []
    try:
        for ef in efs:
            index.expansion_search = ef
            ann_ranked = []
            started = time.perf_counter()
            for vector in query_vectors:
                ann_ranked.append(np.asarray(index.search(vector, k_max).keys))
            ann_ms = (
                (time.perf_counter() - started) * 1000 / len(query_vectors)
            )

            recall = {
                str(k): float(
                    np.mean(
                        [
                            len(
                                set(ann[:k].tolist())
                                & set(exact[:k].tolist())
                            )
                            / k
                            for ann, exact in zip(
                                ann_ranked, exact_ranked, strict=True
                            )
                        ]
                    )
                )
                for k in ks
            }
            sweep.append(
                {
                    "expansion_search": ef,
                    "recall": recall,
                    "mean_query_ms": round(ann_ms, 4),
                }
            )
    finally:
        index.expansion_search = original_ef

    return {
        "queries": len(query_vectors),
        "seed": seed,
        "ks": list(ks),
        "flat": {"mean_query_ms": round(flat_ms, 4)},
        "sweep": sweep,
    }


def _exact_ground_truth(
    matrix: np.ndarray,
    query_vectors: np.ndarray,
    k_max: int,
) -> tuple[list[np.ndarray], float]:
    """Exact top-k_max ids per query (by cosine, best first) and the mean
    per-query flat-scan time in milliseconds."""
    ranked: list[np.ndarray] = []
    total_seconds = 0.0

    for start in range(0, len(query_vectors), _QUERY_BATCH):
        batch = query_vectors[start : start + _QUERY_BATCH]
        began = time.perf_counter()
        similarities = matrix @ batch.T
        if k_max < len(matrix):
            top = np.argpartition(-similarities, k_max - 1, axis=0)[:k_max]
        else:
            top = np.argsort(-similarities, axis=0)
        total_seconds += time.perf_counter() - began

        for column in range(batch.shape[0]):
            ids = top[:, column]
            order = np.argsort(-similarities[ids, column])
            ranked.append(ids[order])

    return ranked, total_seconds * 1000 / len(query_vectors)


def evaluate_and_record(
    parquet_path: Path,
    *,
    queries: int = DEFAULT_QUERIES,
    ks: Sequence[int] = DEFAULT_KS,
    efs: Sequence[int] = DEFAULT_EFS,
    seed: int = DEFAULT_SEED,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate one artifact's index and record the result in its manifest."""
    store = SidecarStore.open(parquet_path)
    block = evaluate(store, queries=queries, ks=ks, efs=efs, seed=seed)

    if write:
        manifest = dict(store.index_manifest or {})
        manifest["evaluation"] = block
        manifest_file = index_manifest_path(index_path_for(parquet_path))
        write_manifest(manifest_file, manifest)

    return block
