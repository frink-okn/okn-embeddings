"""Checkpoint parts: what a crashed embed-texts run keeps.

A Parquet file is unreadable until its footer is written, so a run cannot
resume from its own half-written output. Instead, every `CHECKPOINT_ROWS`
freshly embedded texts are also written to a small file of their own -- a
complete embed-texts file, probes and all, written beside its name and
renamed into place -- in a directory the operator names. A rerun opens the
parts it finds there as reuse sources, so the texts they hold skip the
model; resuming is reuse. A crash loses at most the texts embedded since
the last part.

Only freshly embedded vectors are written: a reused one is still in the
file it came from. Once a run succeeds, its output holds every vector the
parts do, and the parts it wrote or reused are removed.
"""

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PART_PATTERN = re.compile(r"part-(\d{6})\.parquet")

# ~100 MB of 384-dimensional float32 vectors per part: a crash costs at most
# this many texts of model time, and a run of 10^8 texts writes ~1,500 parts.
CHECKPOINT_ROWS = 65_536


class Checkpoint:
    def __init__(self, directory: Path, existing: list[Path]):
        self.directory = directory
        self.existing = existing
        """Parts found when the directory was opened, in the order written."""
        self.written: list[Path] = []
        self._next = 1 + max(
            (_number(p) for p in existing),
            default=-1,
        )

    @classmethod
    def open(cls, directory: Path) -> "Checkpoint":
        """Open `directory`, creating it if missing.

        A part caught mid-write by the crash never reached its name; its
        `.partial` is removed here.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("part-*.parquet.partial"):
            if PART_PATTERN.fullmatch(stale.name.removesuffix(".partial")):
                stale.unlink()
        existing = sorted(
            (p for p in directory.iterdir() if PART_PATTERN.fullmatch(p.name)),
            key=_number,
        )
        return cls(directory, existing)

    def write(self, table: pa.Table, metadata: dict[str, str]) -> Path:
        """Write one part, whole or not at all."""
        path = self.directory / f"part-{self._next:06d}.parquet"
        partial = path.with_name(path.name + ".partial")
        pq.write_table(
            table.replace_schema_metadata(metadata),
            partial,
            compression="NONE",
            use_dictionary=False,
        )
        partial.replace(path)
        self._next += 1
        self.written.append(path)
        return path

    def remove(self, used: list[Path]) -> None:
        """Remove the parts this run wrote, and of those it found, `used` --
        the ones it reused. A part refused as another vector space is left:
        it is that space's work, not this run's."""
        for path in [*used, *self.written]:
            path.unlink(missing_ok=True)
        self.written = []


def _number(path: Path) -> int:
    match = PART_PATTERN.fullmatch(path.name)
    assert match is not None
    return int(match.group(1))
