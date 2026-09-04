import json

import pytest
from qdrant_client import QdrantClient

from okn_embeddings.config.context import AppContext
from okn_embeddings.config.settings import AppSettings
from okn_embeddings.core.backend import SidecarBackend
from okn_embeddings.core.errors import URINotFoundError
from okn_embeddings.core.explore import run_survey
from okn_embeddings.core.models import build_feature, build_query
from okn_embeddings.core.query import run_similarity_search
from okn_embeddings.indexing.embed import embed_file

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write_graph(embedder, tmp_path, graph: str, texts: list[str]):
    jsonl = tmp_path / f"{graph}.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(
                json.dumps(
                    {
                        "iris": [f"urn:{graph}:{i}"],
                        "label": text,
                        "embedding_text": text,
                    }
                )
                + "\n"
            )
    parquet = tmp_path / f"{graph}.parquet"
    embed_file(
        embedder,
        jsonl,
        parquet,
        model_name=_MODEL,
        batch_size=8,
        graph=graph,
    )
    return parquet


@pytest.fixture
def backend(embedder, tmp_path) -> SidecarBackend:
    animals = _write_graph(
        embedder, tmp_path, "animals", ["cat", "dog", "horse"]
    )
    foods = _write_graph(
        embedder, tmp_path, "foods", ["pizza", "salad", "cake", "soup"]
    )
    return SidecarBackend.from_paths([animals, foods])


@pytest.fixture
def sidecar_ctx(embedder, backend) -> AppContext:
    settings = AppSettings(model_name=_MODEL, search_backend="sidecar")
    return AppContext(
        client=QdrantClient(":memory:"),
        embedder=embedder,
        settings=settings,
        backend=backend,
    )


def test_search_merges_across_graphs(embedder, backend):
    rows = backend.search(embedder.embed("puppy"), limit=3)

    assert rows[0].label == "dog"
    assert {r.graph for r in rows} <= {"animals", "foods"}


def test_include_graphs_selects_stores(embedder, backend):
    rows = backend.search(
        embedder.embed("puppy"), limit=3, include_graphs=["foods"]
    )
    assert all(r.graph == "foods" for r in rows)


def test_exclude_graphs_selects_stores(embedder, backend):
    rows = backend.search(
        embedder.embed("puppy"), limit=10, exclude_graphs=["animals"]
    )
    assert all(r.graph == "foods" for r in rows)


def test_offset_skips_from_the_merged_ranking(embedder, backend):
    vector = embedder.embed("puppy")
    all_rows = backend.search(vector, limit=4)
    shifted = backend.search(vector, limit=3, offset=1)
    assert [r.id for r in shifted] == [r.id for r in all_rows[1:4]]


def test_vector_for_iri_searches_every_store(backend):
    assert backend.vector_for_iri("urn:foods:0") is not None
    assert backend.vector_for_iri("urn:nowhere:9") is None


def test_graph_facets_report_counts(backend):
    facets = {f.graph: f.count for f in backend.graph_facets()}
    assert facets == {"animals": 3, "foods": 4}


def test_mixed_conventions_are_refused(embedder, tmp_path, monkeypatch):
    a = _write_graph(embedder, tmp_path, "a", ["x"])
    b = _write_graph(embedder, tmp_path, "b", ["y"])

    import pyarrow.parquet as pq

    table = pq.read_table(b)
    meta = dict(table.schema.metadata or {})
    meta[b"okn-embeddings.convention_id"] = b"different"
    pq.write_table(table.replace_schema_metadata(meta), b)

    with pytest.raises(ValueError, match="conventions"):
        SidecarBackend.from_paths([a, b])


def test_run_similarity_search_over_sidecars(sidecar_ctx):
    result = run_similarity_search(
        sidecar_ctx, build_query("text", "kitten", limit=2)
    )

    assert result.rows[0].label == "cat"
    assert result.time > 0


def test_node_feature_resolves_via_backend(sidecar_ctx):
    result = run_similarity_search(
        sidecar_ctx, build_query("node", "urn:animals:0", limit=2)
    )
    # The node's own vector is its best match.
    assert result.rows[0].label == "cat"


def test_unknown_node_raises(sidecar_ctx):
    with pytest.raises(URINotFoundError):
        run_similarity_search(sidecar_ctx, build_query("node", "urn:missing:0"))


def test_survey_returns_per_graph_hits(sidecar_ctx):
    results = run_survey(sidecar_ctx, build_feature("text", "puppy"), limit=2)

    assert [r.graph for r in results] == ["animals", "foods"]
    assert results[0].rows[0].label == "dog"
    assert all(len(r.rows) <= 2 for r in results)
