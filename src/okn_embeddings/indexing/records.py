"""Reading materialized record files.

`textify` writes one JSONL file of grouped records per graph. These helpers
are shared by every consumer of those files (`embed`, `upload`), so consumers
that never touch Qdrant do not have to import the upload module.
"""

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(
    path: Path,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield each JSON object from a JSONL file, with line-numbered errors."""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if limit is not None and idx > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{idx}: invalid JSON: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{idx}: expected a JSON object")
            yield record


def count_jsonl(path: Path, limit: int | None = None) -> int:
    """Count non-empty lines, so the progress bar knows its total up front."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def chunks(records: Iterable[dict[str, Any]], size: int):
    """Group an iterable of records into lists of at most `size`."""
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
