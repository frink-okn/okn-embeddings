import json
import multiprocessing
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from tqdm import tqdm

from .models import GraphConfiguration, MaterializationConfiguration
from .output import (
    OutputRecord,
    WorkerRecord,
    write_json_record,
    write_jsonl_record,
    write_text_record,
)
from .reader import graph_reader, load_graph
from .textify import (
    Textifier,
    finish_records,
    merge_worker_record,
    target_config_items,
    warn_truncated_records,
)

# Each worker opens the graph once and materializes chunks of root IRIs,
# sharding its records to temp files by text-digest so identical text lands in
# one shard. The parent then merges one shard at a time and streams the output,
# keeping its memory bounded to a single shard rather than the whole
# (potentially multi-GB) record set.

_WORKER_TEXTIFIER: Textifier | None = None
_WORKER_TARGET_CONFIGS: dict[str, GraphConfiguration] = {}


@contextmanager
def suppress_native_output():
    """Silence the HDT C library's stdout/stderr (e.g. on graph load)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def chunked(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def shard_for_digest(digest: str, shard_count: int) -> int:
    return int(digest[:8], 16) % shard_count


def _init_worker(hdt_file: Path, config_toml: Path) -> None:
    global _WORKER_TEXTIFIER, _WORKER_TARGET_CONFIGS
    with suppress_native_output():
        graph = load_graph(hdt_file)
    config = MaterializationConfiguration.from_toml(config_toml)
    _WORKER_TEXTIFIER = Textifier(graph_reader(graph), config)
    # Resolve each target's config once per worker so the per-record cache keys
    # (which use `id(config)`) stay stable across chunks.
    _WORKER_TARGET_CONFIGS = {
        name: config.for_target(name) for name in config.targets
    }


def _materialize_chunk_to_shards(
    target_name: str,
    temp_dir: str,
    shard_count: int,
    iris: list[str],
) -> int:
    textifier = _WORKER_TEXTIFIER
    if textifier is None:
        raise RuntimeError("materialization worker was not initialized")

    target_config = _WORKER_TARGET_CONFIGS[target_name]
    rows_by_shard: defaultdict[int, list[str]] = defaultdict(list)
    for iri in iris:
        record = textifier.materialize_one(iri, target_config)
        shard = shard_for_digest(record.digest, shard_count)
        rows_by_shard[shard].append(
            json.dumps(asdict(record), ensure_ascii=False)
        )

    pid = os.getpid()
    for shard, rows in rows_by_shard.items():
        path = Path(temp_dir) / f"shard-{shard:04d}-{pid}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(row)
                f.write("\n")

    return len(iris)


def _grouped_records_from_shard(
    temp_dir: Path,
    shard: int,
    max_iris_per_record: int,
) -> list[OutputRecord]:
    by_digest: dict[str, OutputRecord] = {}
    for path in sorted(temp_dir.glob(f"shard-{shard:04d}-*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                merge_worker_record(
                    by_digest,
                    WorkerRecord(**json.loads(line)),
                    max_iris_per_record=max_iris_per_record,
                )
    records = finish_records(by_digest)
    warn_truncated_records(records, max_iris_per_record=max_iris_per_record)
    return records


def materialize_records_to_path(
    hdt_file: Path,
    config_toml: Path,
    config: MaterializationConfiguration,
    output_path: Path,
    *,
    text: bool = False,
    jsonl: bool = False,
    target: str | None = None,
    limit: int | None = None,
    jobs: int = 0,
    chunk_size: int = 1000,
    progress: bool = False,
    max_iris_per_record: int = 10,
    shard_count: int = 256,
) -> int:
    reader = graph_reader(load_graph(hdt_file))
    worker_count = multiprocessing.cpu_count() if jobs == 0 else jobs

    output_count = 0
    with TemporaryDirectory(prefix="okn-textify-") as temp_name:
        temp_dir = Path(temp_name)
        with multiprocessing.Pool(
            processes=worker_count,
            initializer=_init_worker,
            initargs=(hdt_file, config_toml),
        ) as pool:
            for target_name, target_config in target_config_items(
                config, target
            ):
                root_iter = reader.root_iris(target_config.type)
                total = reader.root_count(target_config.type)
                if limit is not None:
                    root_iter = islice(root_iter, limit)
                    total = limit if total is None else min(total, limit)
                worker = partial(
                    _materialize_chunk_to_shards,
                    target_name,
                    str(temp_dir),
                    shard_count,
                )
                bar = (
                    tqdm(desc=target_name, unit=" roots", total=total)
                    if progress
                    else None
                )
                try:
                    for done in pool.imap_unordered(
                        worker, chunked(root_iter, chunk_size)
                    ):
                        if bar is not None:
                            bar.update(done)
                finally:
                    if bar is not None:
                        bar.close()

        shard_iter: Iterable[int] = range(shard_count)
        if progress:
            shard_iter = tqdm(shard_iter, desc="merge", unit=" shards")

        with output_path.open("w", encoding="utf-8") as f:
            first = True
            if not text and not jsonl:
                f.write("[")
            for shard in shard_iter:
                for record in _grouped_records_from_shard(
                    temp_dir, shard, max_iris_per_record
                ):
                    if text:
                        write_text_record(record, f)
                    elif jsonl:
                        write_jsonl_record(record, f)
                    else:
                        first = write_json_record(record, f, first)
                    output_count += 1
            if not text and not jsonl:
                if not first:
                    f.write("\n")
                f.write("]")

    return output_count
