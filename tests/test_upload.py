import json

import numpy as np
import pytest
from qdrant_client.models import ScoredPoint

from okn_embeddings.config.context import AppContext
from okn_embeddings.config.settings import AppSettings
from okn_embeddings.core.embedding import (
    SentenceTransformerEmbedder,
    make_embedder,
)
from okn_embeddings.core.results import summarize_point
from okn_embeddings.indexing.upload import (
    chunks,
    iter_jsonl,
    payload_for_record,
    point_id,
    upload_file,
)

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write_records(path, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "iris": [f"urn:{i}"],
                        "label": f"L{i}",
                        "embedding_text": f"text {i}",
                    }
                )
                + "\n"
            )


# --- point_id: deterministic and idempotent ---


def test_point_id_is_deterministic():
    rec = {"iris": ["urn:a"], "embedding_text": "hello"}
    assert point_id("g", rec) == point_id("g", rec)


def test_point_id_varies_by_graph_iri_and_text():
    base = {"iris": ["urn:a"], "embedding_text": "hello"}
    pid = point_id("g", base)
    assert pid != point_id("other", base)
    assert pid != point_id("g", {"iris": ["urn:b"], "embedding_text": "hello"})
    assert pid != point_id("g", {"iris": ["urn:a"], "embedding_text": "bye"})


def test_point_id_accepts_singular_iri_schema():
    # materialize.py emits a singular `iri`; index.py emits an `iris` list.
    singular = point_id("g", {"iri": "urn:a", "embedding_text": "hi"})
    plural = point_id("g", {"iris": ["urn:a"], "embedding_text": "hi"})
    assert singular == plural


# --- payload_for_record: matches what summarize_point reads ---


def test_payload_maps_to_query_side_fields():
    record = {
        "iris": ["urn:a", "urn:b"],
        "label": "A Thing",
        "embedding_text": "label: A Thing\ntype: Foo",
    }
    payload = payload_for_record("my-graph", record)

    assert payload["graph"] == "my-graph"
    assert payload["iri"] == ["urn:a", "urn:b"]
    assert payload["repr"] == "label: A Thing\ntype: Foo"
    assert payload["label"] == "A Thing"
    assert "embedding_text" not in payload
    assert "iris" not in payload


def test_payload_round_trips_through_summarize_point():
    record = {
        "iris": ["urn:a", "urn:b"],
        "label": "A Thing",
        "embedding_text": "some text",
    }
    point = ScoredPoint(
        id="pid",
        version=0,
        score=0.5,
        payload=payload_for_record("my-graph", record),
    )

    row = summarize_point(point)
    assert row.graph == "my-graph"
    assert row.iris == ["urn:a", "urn:b"]
    assert row.primary_uri == "urn:a"
    assert row.label == "A Thing"
    assert row.repr == "some text"


# --- iter_jsonl / chunks ---


def test_iter_jsonl_reports_line_numbers(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"g\.jsonl:2: invalid JSON"):
        list(iter_jsonl(path))


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in iter_jsonl(path)] == [1, 2]


def test_chunks_batches_with_remainder():
    batches = list(chunks(iter([{"i": i} for i in range(5)]), 2))
    assert [len(b) for b in batches] == [2, 2, 1]


# --- upload_file against in-memory Qdrant (the shared `ctx` fixture) ---


def test_upload_file_upserts_all_points(ctx: AppContext, tmp_path):
    collection = ctx.settings.qdrant_collection
    path = tmp_path / "my-graph.jsonl"
    _write_records(path, 5)

    uploaded = upload_file(
        ctx,
        path,
        batch_size=2,
        upload_batch_size=2,
        limit=None,
        dry_run=False,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 5
    assert ctx.client.count(collection).count == 5
    # graph comes from the file stem; payload mapping lands what the query side
    # reads.
    points, _ = ctx.client.scroll(collection, limit=10, with_payload=True)
    payloads = [p.payload or {} for p in points]
    assert all(p["graph"] == "my-graph" for p in payloads)
    assert {p["repr"] for p in payloads} == {f"text {i}" for i in range(5)}


def test_upload_file_dry_run_skips_upsert(ctx: AppContext, tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 3)

    uploaded = upload_file(
        ctx,
        path,
        batch_size=2,
        upload_batch_size=2,
        limit=None,
        dry_run=True,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 3
    assert ctx.client.count(ctx.settings.qdrant_collection).count == 0


def test_upload_file_honors_limit(ctx: AppContext, tmp_path):
    path = tmp_path / "g.jsonl"
    _write_records(path, 10)

    uploaded = upload_file(
        ctx,
        path,
        batch_size=4,
        upload_batch_size=4,
        limit=3,
        dry_run=False,
        progress_enabled=False,
        log_every=10_000,
    )

    assert uploaded == 3
    assert ctx.client.count(ctx.settings.qdrant_collection).count == 3


def test_reupload_is_idempotent(ctx: AppContext, tmp_path):
    # Stable UUID5 point IDs mean re-running upserts in place, not duplicating.
    path = tmp_path / "my-graph.jsonl"
    _write_records(path, 5)

    for _ in range(2):
        upload_file(
            ctx,
            path,
            batch_size=4,
            upload_batch_size=4,
            limit=None,
            dry_run=False,
            progress_enabled=False,
            log_every=10_000,
        )

    assert ctx.client.count(ctx.settings.qdrant_collection).count == 5


# --- seam: single embed agrees with batch embed_many ---


def test_embed_matches_embed_many(embedder: SentenceTransformerEmbedder):
    one = embedder.embed("diabetes")
    many = embedder.embed_many(["diabetes", "insulin"])

    assert len(many) == 2
    assert np.array_equal(one, many[0])


# --- embedding backend knobs ---


def test_make_embedder_passes_backend_settings():
    embedder = make_embedder(
        AppSettings(
            model_name=_MODEL,
            embed_device="cpu",
            embed_batch_size=32,
            embed_threads=2,
        )
    )

    assert embedder.device == "cpu"
    assert embedder.batch_size == 32
    assert embedder.threads == 2


def test_backend_settings_default_to_auto_device():
    embedder = make_embedder(
        AppSettings(model_name=_MODEL, embed_device="cpu")
    )

    assert embedder.batch_size == 256
    assert embedder.threads is None
