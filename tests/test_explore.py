import math

import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client.models import Filter, PointStruct

from okn_embeddings.config.context import AppContext
from okn_embeddings.core.explore import (
    resolve_target_graphs,
    run_survey,
)
from okn_embeddings.core.models import (
    NodeFeature,
    TextFeature,
    build_feature,
)
from okn_embeddings.core.backend import build_graph_filter


def test_resolve_target_graphs_include_wins():
    assert resolve_target_graphs(["a", "b", "c"], ["b", "c"], None) == [
        "b",
        "c",
    ]


def test_resolve_target_graphs_exclude():
    assert resolve_target_graphs(["a", "b", "c"], None, ["b"]) == ["a", "c"]


def test_resolve_target_graphs_all():
    assert resolve_target_graphs(["a", "b"], None, None) == ["a", "b"]


def test_build_graph_filter_include():
    f = build_graph_filter(["a", "b"], None)
    assert isinstance(f, Filter)
    assert f.must is not None
    assert f.must_not is None


def test_build_graph_filter_exclude():
    f = build_graph_filter(None, ["a"])
    assert isinstance(f, Filter)
    assert f.must_not is not None
    assert f.must is None


def test_build_graph_filter_none():
    assert build_graph_filter(None, None) is None


def test_build_feature_text():
    feat = build_feature("text", "hello")
    assert isinstance(feat, TextFeature)
    assert feat.value == "hello"


def test_build_feature_node():
    assert isinstance(build_feature("node", "urn:x"), NodeFeature)


def test_build_feature_bad_type():
    with pytest.raises(ValidationError):
        build_feature("bogus", "x")


# --- run_survey against in-memory Qdrant ---


def _unit(dim: int, cosine: float) -> list[float]:
    # A unit vector whose cosine similarity to [1, 0, 0, ...] is `cosine`.
    v = [0.0] * dim
    v[0] = cosine
    v[1] = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return v


def _seed(ctx: AppContext, rows: list[tuple[str, float]]) -> int:
    # Upsert one point per (graph, cosine-to-the-query) row. Returns the dim.
    dim = len(ctx.embedder.embed("probe"))
    ctx.client.upsert(
        ctx.settings.qdrant_collection,
        points=[
            PointStruct(
                id=i, vector=_unit(dim, cosine), payload={"graph": graph}
            )
            for i, (graph, cosine) in enumerate(rows)
        ],
    )
    return dim


def test_run_survey_orders_graphs_by_best_score(ctx: AppContext, monkeypatch):
    dim = _seed(
        ctx,
        [
            ("g_high", 1.0),
            ("g_high", 0.9),
            ("g_mid", 0.5),
            ("g_low", 0.2),
            ("g_low", 0.1),
        ],
    )

    calls = {"n": 0}

    def embed(_ctx, _feature):
        calls["n"] += 1
        return np.array(_unit(dim, 1.0), dtype=np.float32)

    monkeypatch.setattr("okn_embeddings.core.explore.get_embedding", embed)

    results = run_survey(
        ctx,
        build_feature("text", "q"),
        include_graphs=["g_low", "g_high", "g_mid"],
        limit=2,
    )

    assert calls["n"] == 1  # embedded once, reused for every graph
    assert [r.graph for r in results] == ["g_high", "g_mid", "g_low"]
    assert len(results[0].rows) == 2  # per-graph limit honored
    assert results[0].rows[0].score == pytest.approx(1.0, abs=1e-3)


def test_run_survey_with_no_filter_surveys_every_graph(
    ctx: AppContext, monkeypatch
):
    dim = _seed(ctx, [("a", 0.3), ("b", 0.7)])
    monkeypatch.setattr(
        "okn_embeddings.core.explore.get_embedding",
        lambda _ctx, _feature: np.array(_unit(dim, 1.0), dtype=np.float32),
    )

    # No include/exclude: the graph list is discovered via get_graphs (facet).
    results = run_survey(ctx, build_feature("text", "q"))

    assert [r.graph for r in results] == ["b", "a"]
