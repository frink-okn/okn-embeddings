import json

import pyarrow.parquet as pq

from okn_embeddings.core.embedding import FastEmbedEmbedder
from okn_embeddings.indexing.embed import METADATA_PREFIX, embed_file
from okn_embeddings.indexing.manifest import (
    build_manifest,
    manifest_path,
    read_manifest,
    write_manifest,
)
from okn_embeddings.indexing.models import MaterializationConfiguration

_CONFIG_TOML = """\
[defaults]
predicate_limit = 3

[targets.thing]
type = "https://example.org/Thing"
"""


def _fixture_files(tmp_path):
    hdt = tmp_path / "my-graph.hdt"
    hdt.write_bytes(b"not really an hdt file")
    config_file = tmp_path / "config.toml"
    config_file.write_text(_CONFIG_TOML, encoding="utf-8")
    config = MaterializationConfiguration.from_toml(config_file)
    return hdt, config_file, config


def _build(tmp_path, **overrides):
    hdt, config_file, config = _fixture_files(tmp_path)
    kwargs = {
        "target": None,
        "limit": 500,
        "max_iris_per_record": 10,
        "record_count": 946,
    }
    kwargs.update(overrides)
    return build_manifest(hdt, config_file, config, **kwargs)


def test_manifest_path_sits_beside_records_file(tmp_path):
    assert manifest_path(tmp_path / "g.jsonl") == tmp_path / "g.meta.json"
    assert manifest_path(tmp_path / "out" / "g.json") == (
        tmp_path / "out" / "g.meta.json"
    )


def test_build_manifest_records_provenance(tmp_path):
    manifest = _build(tmp_path)

    assert manifest["format"] == 1
    assert manifest["graph"]["file"] == "my-graph.hdt"
    assert manifest["graph"]["bytes"] == len(b"not really an hdt file")
    assert len(manifest["graph"]["sha256"]) == 64
    assert manifest["config"]["file"] == "config.toml"
    assert len(manifest["config"]["sha256"]) == 64
    assert manifest["config"]["parsed"]["targets"]["thing"]["type"] == (
        "https://example.org/Thing"
    )
    assert manifest["run"] == {
        "target": None,
        "limit": 500,
        "max_iris_per_record": 10,
    }
    assert manifest["record_count"] == 946


def test_build_manifest_is_deterministic(tmp_path):
    assert _build(tmp_path) == _build(tmp_path)


def test_manifest_round_trips(tmp_path):
    manifest = _build(tmp_path)
    records_path = tmp_path / "g.jsonl"

    write_manifest(manifest_path(records_path), manifest)

    assert read_manifest(records_path) == manifest


def test_read_manifest_returns_none_when_absent(tmp_path):
    assert read_manifest(tmp_path / "g.jsonl") is None


def test_embed_folds_manifest_into_parquet_metadata(
    embedder: FastEmbedEmbedder, tmp_path
):
    records_path = tmp_path / "g.jsonl"
    records_path.write_text(
        json.dumps({"iris": ["urn:a"], "embedding_text": "hi"}) + "\n",
        encoding="utf-8",
    )
    manifest = _build(tmp_path, record_count=1)
    write_manifest(manifest_path(records_path), manifest)
    output = tmp_path / "g.parquet"

    embed_file(
        embedder, records_path, output, model_name="m", batch_size=2
    )
    raw = pq.read_metadata(output).metadata
    meta = {k.decode(): v.decode() for k, v in raw.items()}

    assert meta[METADATA_PREFIX + "config_sha256"] == (
        manifest["config"]["sha256"]
    )
    assert meta[METADATA_PREFIX + "graph_sha256"] == (
        manifest["graph"]["sha256"]
    )
    assert json.loads(meta[METADATA_PREFIX + "textify_manifest"]) == manifest


def test_embed_without_manifest_omits_manifest_keys(
    embedder: FastEmbedEmbedder, tmp_path
):
    records_path = tmp_path / "g.jsonl"
    records_path.write_text(
        json.dumps({"iris": ["urn:a"], "embedding_text": "hi"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "g.parquet"

    embed_file(
        embedder, records_path, output, model_name="m", batch_size=2
    )
    raw = pq.read_metadata(output).metadata
    keys = {k.decode() for k in raw}

    assert METADATA_PREFIX + "record_count" in keys
    assert METADATA_PREFIX + "textify_manifest" not in keys
    assert METADATA_PREFIX + "config_sha256" not in keys
