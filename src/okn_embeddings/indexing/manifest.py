"""Optional provenance manifest for textify outputs.

`textify --manifest` writes a small JSON file beside the records file
(`graph.jsonl` -> `graph.meta.json`) recording what produced it: the source
graph's identity, the indexing config (file hash plus parsed form), the
invocation bounds, and the record count. The records file itself stays a pure
record stream.

`embed` folds the manifest into the Parquet footer metadata when one is
present, so the recipe -> records -> vectors provenance chain survives into
the vector artifact. No manifest, no error: it is optional end to end.
"""

import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .models import MaterializationConfiguration

MANIFEST_FORMAT = 1
MANIFEST_SUFFIX = ".meta.json"


def manifest_path(records_path: Path) -> Path:
    """`graph.jsonl` -> `graph.meta.json`, beside the records file."""
    return records_path.with_suffix(MANIFEST_SUFFIX)


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_manifest(
    hdt_file: Path,
    config_file: Path,
    config: MaterializationConfiguration,
    *,
    target: str | None,
    limit: int | None,
    max_iris_per_record: int,
    record_count: int,
) -> dict[str, Any]:
    """Assemble the manifest for one textify run.

    Deliberately carries no timestamp, so identical inputs produce an
    identical manifest.
    """
    try:
        version = importlib_metadata.version("okn-embeddings")
    except importlib_metadata.PackageNotFoundError:
        version = None

    return {
        "format": MANIFEST_FORMAT,
        "tool": {"name": "okn-embeddings", "version": version},
        "graph": {
            "file": hdt_file.name,
            "bytes": hdt_file.stat().st_size,
            "sha256": file_sha256(hdt_file),
        },
        "config": {
            "file": config_file.name,
            "sha256": file_sha256(config_file),
            "parsed": config.model_dump(mode="json"),
        },
        "run": {
            "target": target,
            "limit": limit,
            "max_iris_per_record": max_iris_per_record,
        },
        "record_count": record_count,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_manifest(records_path: Path) -> dict[str, Any] | None:
    """Load the manifest beside a records file, or None if there is none."""
    path = manifest_path(records_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
