"""Interactive UMAP map of one or more embed artifacts.

`build_map` samples vectors from each graph's embed Parquet, projects them
(plus each graph's sketch centroids, when a sketch sits beside the Parquet)
to 2-D with UMAP, and writes a self-contained interactive HTML page: a
canvas scatter with pan/zoom, hover tooltips, per-graph legend with
isolate, label search, and a centroid overlay toggle.

UMAP is a dev-group dependency (`evaluation`), imported lazily so the
installed package never needs it.
"""

import json
import string
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .build import parquet_okn_metadata
from .sketch import SketchStore, load_vectors, sketch_path_for


def sample_graph(
    parquet_path: Path,
    sample: int,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray, list[dict], int, str | None]:
    """(graph name, sampled vectors, row dicts, total, convention_id).

    Each row dict carries the display fields a clicked point shows:
    label, embedding text (the verbalization), and the primary IRI.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_okn_metadata(parquet_file)
    graph = metadata.get("graph", parquet_path.stem)

    vectors = load_vectors(parquet_path)
    table = parquet_file.read(columns=["label", "embedding_text", "iris"])
    labels = table.column("label").to_pylist()
    texts = table.column("embedding_text").to_pylist()
    iris = table.column("iris").to_pylist()

    total = len(vectors)
    indexes = range(total)
    if total > sample:
        picks = np.sort(rng.choice(total, size=sample, replace=False))
        vectors = vectors[picks]
        indexes = [int(i) for i in picks]

    rows = [
        {
            "label": labels[i],
            "text": texts[i],
            "iri": iris[i][0] if iris[i] else "",
        }
        for i in indexes
    ]
    return graph, vectors, rows, total, metadata.get("convention_id")


def disambiguate(names: list[str], paths: list[Path]) -> list[str]:
    """Make duplicate graph names unique by prefixing a directory name.

    Graph metadata is stamped from the input file stem, so artifacts kept
    in per-graph directories (rural-kg/graph-embed/records.parquet,
    soc-kg/graph-embed/records.parquet) all share one name. For each
    group of duplicates, walk up the directory chain to the first level
    where the ancestor names diverge and prefix that name.
    """
    result = list(names)
    for name in set(names):
        indexes = [i for i, n in enumerate(names) if n == name]
        if len(indexes) < 2:
            continue
        chains = {
            i: [p.name for p in paths[i].resolve().parents if p.name]
            for i in indexes
        }
        depth = 0
        max_depth = max(len(c) for c in chains.values())
        while depth < max_depth:
            level = [
                chains[i][depth] if depth < len(chains[i]) else ""
                for i in indexes
            ]
            if len(set(level)) > 1:
                for i, ancestor in zip(indexes, level, strict=True):
                    if ancestor:
                        result[i] = f"{ancestor}/{name}"
                break
            depth += 1
    return result


def build_map(
    parquet_paths: list[Path],
    output: Path,
    *,
    sample: int = 2000,
    seed: int = 42,
    neighbors: int = 15,
    min_dist: float = 0.1,
) -> Path:
    """Project sampled vectors + sketch centroids to 2-D and write HTML."""
    from umap import UMAP

    rng = np.random.default_rng(seed)

    graphs: list[dict] = []
    all_vectors: list[np.ndarray] = []
    conventions: set[str | None] = set()
    for path in parquet_paths:
        graph, vectors, rows, total, convention = sample_graph(
            path, sample, rng
        )
        conventions.add(convention)

        centroids = np.empty((0, vectors.shape[1]), dtype=np.float32)
        sketch_file = sketch_path_for(path)
        if sketch_file.exists():
            centroids = SketchStore.open(sketch_file).centroids

        graphs.append(
            {
                "name": graph,
                "rows": rows,
                "total": total,
                "sampled": len(vectors),
                "centroid_count": len(centroids),
            }
        )
        all_vectors.append(vectors)
        all_vectors.append(centroids)

    names = disambiguate([g["name"] for g in graphs], parquet_paths)
    for graph, name in zip(graphs, names, strict=True):
        graph["name"] = name

    if len(conventions) > 1:
        raise ValueError(
            "artifacts span multiple embedding conventions; their vectors "
            f"are not comparable in one map: {sorted(map(str, conventions))}"
        )

    stacked = np.vstack(all_vectors)
    reducer = UMAP(
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    projected = reducer.fit_transform(stacked).astype(np.float32)

    points = []
    centroid_points = []
    offset = 0
    for index, graph in enumerate(graphs):
        n, c = graph["sampled"], graph["centroid_count"]
        for position, row in zip(
            projected[offset : offset + n], graph["rows"], strict=True
        ):
            points.append(
                [
                    index,
                    round(float(position[0]), 3),
                    round(float(position[1]), 3),
                    row["label"],
                    row["text"],
                    row["iri"],
                ]
            )
        for position in projected[offset + n : offset + n + c]:
            centroid_points.append(
                [
                    index,
                    round(float(position[0]), 3),
                    round(float(position[1]), 3),
                ]
            )
        offset += n + c

    payload = {
        "graphs": [
            {
                "name": g["name"],
                "sampled": g["sampled"],
                "total": g["total"],
                "centroids": g["centroid_count"],
            }
            for g in graphs
        ],
        "points": points,
        "centroids": centroid_points,
        "umap": {
            "neighbors": neighbors,
            "min_dist": min_dist,
            "seed": seed,
            "metric": "cosine",
        },
    }

    template = Path(__file__).with_name("plot_template.html").read_text("utf-8")
    html = string.Template(template).substitute(
        payload=json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    output.write_text(html, encoding="utf-8")
    return output
