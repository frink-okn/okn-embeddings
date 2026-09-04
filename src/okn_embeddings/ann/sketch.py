"""Routing sketches: which graphs are worth an ANN probe.

A sketch is K centroids from k-means over one graph's vectors, each with a
member count, a max radius, and a quantile radius, written as a small
Parquet file beside the embed artifact. Its one guarantee comes from the
triangle inequality: every member of a cluster lies within `max_radius` of
its centroid, so a query farther than `max_radius + threshold` from the
centroid provably has no match inside -- a *certified no*. The minimum of
`dist(q, centroid) - max_radius` over all clusters lower-bounds the best
match the whole graph could hold. "Yes" answers are only a ranking (by
expected yield from counts and quantile radii), never a guarantee.

Construction is deliberately non-normative: conformance is the radius
soundness obligation (`max_radius` truly bounds every member), not
byte-identical k-means output. A graph with at most K vectors stores each
vector as its own centroid (count 1, radius 0) and is marked unsaturated;
every bound it produces is exact.

Sketches are only comparable within one embedding convention: the parent's
`convention_id` is copied into the sketch footer, and readers must not
score a query against a sketch built under a different id.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..indexing.embed import METADATA_PREFIX
from ..indexing.manifest import file_sha256, package_version
from .build import iter_vector_batches, parquet_okn_metadata

SKETCH_SUFFIX = ".sketch.parquet"
SKETCH_FORMAT = 1

# Profile cap from the design doc: sketches stay small and flat in dataset
# size, so K never exceeds this regardless of what the caller asks for.
MAX_K = 4096

# --k auto sweeps doubling K values and stops when the quantile radius
# improves by less than this fraction per doubling -- the elbow of the
# radius curve, which is where extra centroids stop buying pruning power.
AUTO_K_MIN_IMPROVEMENT = 0.05
AUTO_K_START = 64


def sketch_path_for(parquet_path: Path) -> Path:
    """`graph.parquet` -> `graph.sketch.parquet`, beside the Parquet file."""
    return parquet_path.with_suffix(SKETCH_SUFFIX)


def load_vectors(parquet_path: Path) -> np.ndarray:
    """All vectors from an embed Parquet as one (n, dim) float32 matrix."""
    parquet_file = pq.ParquetFile(parquet_path)
    dim = parquet_file.schema_arrow.field("vector").type.list_size
    parts = list(iter_vector_batches(parquet_file, dim))
    return np.vstack(parts) if parts else np.empty((0, dim), np.float32)


def _pairwise_sq_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared L2 distances between rows of `a` and rows of `b`."""
    sq = (
        (a * a).sum(axis=1)[:, None]
        - 2.0 * (a @ b.T)
        + (b * b).sum(axis=1)[None, :]
    )
    return np.maximum(sq, 0.0)


def _kmeans_pp_init(
    vectors: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding over a subsample, for stable starts."""
    sample_size = min(len(vectors), max(10 * k, 1024))
    sample = vectors[rng.choice(len(vectors), sample_size, replace=False)]

    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    centroids[0] = sample[rng.integers(len(sample))]
    closest = _pairwise_sq_distances(sample, centroids[:1]).ravel()
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            centroids[i:] = sample[rng.integers(len(sample), size=k - i)]
            break
        probabilities = closest / total
        choice = rng.choice(len(sample), p=probabilities)
        centroids[i] = sample[choice]
        distance = _pairwise_sq_distances(sample, centroids[i : i + 1]).ravel()
        np.minimum(closest, distance, out=closest)
    return centroids


def minibatch_kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    batch_size: int = 4096,
    iterations: int = 100,
) -> np.ndarray:
    """Mini-batch k-means centroids. Plain numpy, no sklearn.

    Construction is non-normative (the artifact's guarantee comes from the
    radii measured afterwards, not from clustering quality), so this favors
    simplicity and portability over squeezing out the last percent of
    quantization error.
    """
    centroids = _kmeans_pp_init(vectors, k, rng)
    counts = np.zeros(k, dtype=np.int64)

    for _ in range(iterations):
        batch = vectors[rng.integers(len(vectors), size=batch_size)]
        labels = _pairwise_sq_distances(batch, centroids).argmin(axis=1)

        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, batch)
        batch_counts = np.bincount(labels, minlength=k)

        seen = batch_counts > 0
        counts[seen] += batch_counts[seen]
        rates = (batch_counts[seen] / counts[seen])[:, None]
        means = sums[seen] / batch_counts[seen][:, None]
        centroids[seen] += rates * (means - centroids[seen])

    return centroids


def assign_and_measure(
    vectors: np.ndarray,
    centroids: np.ndarray,
    *,
    radius_quantile: float,
    chunk_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One pass over all vectors: per-cluster count, max and quantile radius.

    Returns (kept_centroids, counts, max_radii, quantile_radii) with empty
    clusters dropped. `max_radii` is the soundness-critical output: it must
    upper-bound every member's distance, which holds by construction because
    the radii come from the same assignment this function performs.
    """
    n = len(vectors)
    labels = np.empty(n, dtype=np.int64)
    distances = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        chunk = vectors[start : start + chunk_size]
        sq = _pairwise_sq_distances(chunk, centroids)
        chunk_labels = sq.argmin(axis=1)
        labels[start : start + chunk_size] = chunk_labels
        distances[start : start + chunk_size] = np.sqrt(
            sq[np.arange(len(chunk)), chunk_labels]
        )

    kept, counts, max_radii, quantile_radii = [], [], [], []
    for cluster in range(len(centroids)):
        member_distances = distances[labels == cluster]
        if len(member_distances) == 0:
            continue
        kept.append(centroids[cluster])
        counts.append(len(member_distances))
        max_radii.append(float(member_distances.max()))
        quantile_radii.append(
            float(np.quantile(member_distances, radius_quantile))
        )

    return (
        np.asarray(kept, dtype=np.float32),
        np.asarray(counts, dtype=np.int64),
        np.asarray(max_radii, dtype=np.float64),
        np.asarray(quantile_radii, dtype=np.float64),
    )


def choose_k(
    vectors: np.ndarray,
    *,
    rng: np.random.Generator,
    radius_quantile: float,
    max_k: int = MAX_K,
) -> int:
    """Pick K at the elbow of the radius-vs-K curve.

    Doubles K from `AUTO_K_START` and stops when the median cluster's
    quantile radius improves by less than `AUTO_K_MIN_IMPROVEMENT` per
    doubling. On clustered data radii collapse early and K stays small; on
    diffuse data radii barely move and the sweep stops immediately rather
    than paying for centroids that cannot prune.
    """
    ceiling = min(max_k, len(vectors))
    k = min(AUTO_K_START, ceiling)
    _, _, _, quantile_radii = assign_and_measure(
        vectors,
        minibatch_kmeans(vectors, k, rng=rng),
        radius_quantile=radius_quantile,
    )
    best_radius = float(np.median(quantile_radii))

    while k * 2 <= ceiling:
        candidate = k * 2
        _, _, _, quantile_radii = assign_and_measure(
            vectors,
            minibatch_kmeans(vectors, candidate, rng=rng),
            radius_quantile=radius_quantile,
        )
        radius = float(np.median(quantile_radii))
        if best_radius <= 0:
            break
        if (best_radius - radius) / best_radius < AUTO_K_MIN_IMPROVEMENT:
            break
        k, best_radius = candidate, radius

    return k


def build_sketch(
    parquet_path: Path,
    output: Path | None = None,
    *,
    k: int | None = None,
    seed: int = 42,
    radius_quantile: float = 0.9,
) -> tuple[Path, int, bool]:
    """Build a routing sketch beside one embed Parquet file.

    `k=None` selects K automatically from the radius curve. Returns
    (sketch path, cluster count, unsaturated flag).
    """
    if output is None:
        output = sketch_path_for(parquet_path)
    if k is not None and not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")

    parquet_file = pq.ParquetFile(parquet_path)
    parent_metadata = parquet_okn_metadata(parquet_file)
    vectors = load_vectors(parquet_path)
    rng = np.random.default_rng(seed)

    if k is None and len(vectors) > AUTO_K_START:
        k = choose_k(vectors, rng=rng, radius_quantile=radius_quantile)
    elif k is None:
        k = max(len(vectors), 1)

    unsaturated = len(vectors) <= k
    if unsaturated:
        centroids = vectors
        counts = np.ones(len(vectors), dtype=np.int64)
        max_radii = np.zeros(len(vectors), dtype=np.float64)
        quantile_radii = np.zeros(len(vectors), dtype=np.float64)
    else:
        centroids, counts, max_radii, quantile_radii = assign_and_measure(
            vectors,
            minibatch_kmeans(vectors, k, rng=rng),
            radius_quantile=radius_quantile,
        )

    dim = vectors.shape[1] if len(vectors) else 0
    flat = centroids.ravel() if len(centroids) else np.empty(0, np.float32)
    table = pa.Table.from_arrays(
        [
            pa.FixedSizeListArray.from_arrays(pa.array(flat), dim),
            pa.array(counts, type=pa.int64()),
            pa.array(max_radii, type=pa.float64()),
            pa.array(quantile_radii, type=pa.float64()),
        ],
        names=["centroid", "count", "max_radius", "quantile_radius"],
    )

    metadata = {
        METADATA_PREFIX + "format": str(SKETCH_FORMAT),
        METADATA_PREFIX + "kind": "routing-sketch",
        METADATA_PREFIX + "k": str(k),
        METADATA_PREFIX + "clusters": str(len(centroids)),
        METADATA_PREFIX + "count": str(len(vectors)),
        METADATA_PREFIX + "dim": str(dim),
        METADATA_PREFIX + "unsaturated": str(unsaturated).lower(),
        METADATA_PREFIX + "radius_quantile": str(radius_quantile),
        METADATA_PREFIX + "seed": str(seed),
        METADATA_PREFIX + "tool_version": package_version("okn-embeddings")
        or "",
        METADATA_PREFIX + "parent_file": parquet_path.name,
        METADATA_PREFIX + "parent_sha256": file_sha256(parquet_path),
    }
    for key in ("graph", "convention_id", "model"):
        value = parent_metadata.get(key)
        if value is not None:
            metadata[METADATA_PREFIX + key] = value

    pq.write_table(
        table.replace_schema_metadata(metadata),
        output,
        compression="NONE",
    )
    return output, len(centroids), unsaturated


def cosine_to_distance(min_cosine: float) -> float:
    """The L2 distance equivalent of a cosine threshold on unit vectors.

    d^2 = 2 - 2*cos, so any vector at least `min_cosine` similar to the
    query lies within this distance of it.
    """
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * min_cosine)))


class SketchStore:
    """One graph's routing sketch, loaded for query-side bounds."""

    def __init__(
        self,
        centroids: np.ndarray,
        counts: np.ndarray,
        max_radii: np.ndarray,
        quantile_radii: np.ndarray,
        metadata: dict[str, str],
    ):
        self.centroids = centroids
        self.counts = counts
        self.max_radii = max_radii
        self.quantile_radii = quantile_radii
        self.metadata = metadata

    @classmethod
    def open(cls, path: Path) -> "SketchStore":
        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read()
        metadata = parquet_okn_metadata(parquet_file)

        dim = table.schema.field("centroid").type.list_size
        flat = table.column("centroid").combine_chunks().flatten()
        centroids = flat.to_numpy(zero_copy_only=False).reshape(-1, dim)
        return cls(
            centroids.astype(np.float32, copy=False),
            table.column("count").to_numpy(),
            table.column("max_radius").to_numpy(),
            table.column("quantile_radius").to_numpy(),
            metadata,
        )

    @property
    def convention_id(self) -> str | None:
        return self.metadata.get("convention_id")

    @property
    def unsaturated(self) -> bool:
        return self.metadata.get("unsaturated") == "true"

    def _centroid_distances(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        return np.sqrt(_pairwise_sq_distances(q, self.centroids).ravel())

    def best_distance_bound(self, query: np.ndarray) -> float:
        """Certified lower bound on the graph's best match distance.

        No vector in the graph can be closer to the query than this: for a
        member x of the cluster around centroid mu with radius r, the
        triangle inequality gives dist(q, x) >= dist(q, mu) - r.
        """
        if len(self.centroids) == 0:
            return np.inf
        bounds = self._centroid_distances(query) - self.max_radii
        return float(np.maximum(bounds, 0.0).min())

    def certified_no(self, query: np.ndarray, min_cosine: float) -> bool:
        """True when the graph provably holds no match above `min_cosine`."""
        return self.best_distance_bound(query) > cosine_to_distance(min_cosine)

    def yield_estimate(self, query: np.ndarray, min_cosine: float) -> int:
        """Expected-yield score for ranking graphs, higher probes first.

        Sums the member counts of clusters whose typical member (within the
        quantile radius) could still beat the threshold. A heuristic for
        ordering, not a guarantee -- the confirm rung is the actual ANN
        probe.
        """
        if len(self.centroids) == 0:
            return 0
        threshold = cosine_to_distance(min_cosine)
        optimistic = self._centroid_distances(query) - self.quantile_radii
        return int(self.counts[optimistic <= threshold].sum())
