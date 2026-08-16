"""Reader for the open Voxel51 mirror of Egocentric-10K.

The main repo is gated. This mirror is not, and it carries factory 051 in
full: 8 workers x 52 clips, 37 GB of flat .mp4 files under data/.

Exposes the same shape as shards.py -- a list of units, and a per-unit
sample stream -- so the pipeline is indifferent to which source it reads.
Grouping is per worker, which is the unit redundancy is measured within.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from huggingface_hub import HfApi, HfFileSystem

from shards import Sample

SUBSET_REPO = "Voxel51/Egocentric_10K_subset"
NAME_PATTERN = re.compile(r"factory(?P<factory>\d+)_worker(?P<worker>\d+)_(?P<idx>\d+)\.mp4$")


@dataclass
class WorkerUnit:
    """All clips belonging to one worker."""

    factory: str
    worker: str
    paths: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"factory{self.factory}_worker{self.worker}"


def list_units(
    max_workers: int | None = None,
    clips_per_worker: int | None = None,
    worker_offset: int = 0,
) -> list[WorkerUnit]:
    """Group the flat file listing into per-worker units.

    `worker_offset` skips the first N workers, so an interrupted run can be
    resumed from where it died instead of re-streaming 37 GB.
    """
    api = HfApi()
    info = api.dataset_info(SUBSET_REPO)

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sib in info.siblings or []:
        match = NAME_PATTERN.search(sib.rfilename)
        if match is None:
            continue
        key = (match.group("factory"), match.group("worker"))
        grouped[key].append(sib.rfilename)

    units = [
        WorkerUnit(
            factory=factory,
            worker=worker,
            paths=sorted(paths)[:clips_per_worker],
        )
        for (factory, worker), paths in sorted(grouped.items())
    ]
    units = units[worker_offset:]
    return units[:max_workers] if max_workers else units


def stream_unit(unit: WorkerUnit) -> Iterator[Sample]:
    """Yield each of the worker's clips as in-memory bytes.

    Clips average 90 MB, so one at a time stays well inside RAM and nothing
    is written to disk.
    """
    fs = HfFileSystem()
    for path in unit.paths:
        full = f"datasets/{SUBSET_REPO}/{path}"
        try:
            with fs.open(full, "rb") as handle:
                blob = handle.read()
        except Exception:  # noqa: BLE001 - a single unreadable clip is not fatal
            continue

        key = path.rsplit("/", 1)[-1].removesuffix(".mp4")
        yield Sample(
            key=key,
            video=blob,
            meta={
                "factory_id": unit.factory,
                "worker_id": unit.worker,
                "source": SUBSET_REPO,
            },
        )
