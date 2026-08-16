"""Enumerate and stream Egocentric-10K WebDataset shards from Hugging Face.

The dataset is ~18 TB. Nothing is persisted to disk: shards are read as a
byte stream, samples are yielded one at a time, and the buffer is dropped
as soon as a sample has been embedded.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass
from typing import Iterator

from huggingface_hub import HfFileSystem

from config import REPO_ID, SHARD_SUFFIX

# factory_012/workers/worker_003/factory012_worker003_part00.tar
SHARD_PATTERN = re.compile(
    r"factory_(?P<factory>\d+)/workers/worker_(?P<worker>\d+)/[^/]+\.tar$"
)


@dataclass(frozen=True)
class ShardRef:
    """A single tar shard, tagged with the factory and worker it came from."""

    path: str
    factory: str
    worker: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass
class Sample:
    """One clip: H.265 video bytes plus its sidecar metadata."""

    key: str
    video: bytes
    meta: dict


def _fs() -> HfFileSystem:
    return HfFileSystem()


def list_shards(
    max_factories: int | None = None,
    shards_per_worker: int = 1,
    workers_per_factory: int = 1,
) -> list[ShardRef]:
    """List shards, spread across factories rather than concentrated in one.

    Redundancy is measured *within* a worker, *across* workers in a factory,
    and *across* factories. Sampling breadth-first keeps all three levels
    measurable instead of over-representing whichever factory sorts first.
    """
    fs = _fs()
    root = f"datasets/{REPO_ID}"

    factory_dirs = sorted(
        p for p in fs.ls(root, detail=False) if "/factory_" in p
    )
    if max_factories is not None:
        factory_dirs = factory_dirs[:max_factories]

    shards: list[ShardRef] = []
    for factory_dir in factory_dirs:
        try:
            worker_dirs = sorted(
                fs.ls(f"{factory_dir}/workers", detail=False)
            )[:workers_per_factory]
        except FileNotFoundError:
            continue

        for worker_dir in worker_dirs:
            tars = sorted(
                p
                for p in fs.ls(worker_dir, detail=False)
                if p.endswith(SHARD_SUFFIX)
            )[:shards_per_worker]

            for tar_path in tars:
                rel = tar_path.split(f"{REPO_ID}/", 1)[-1]
                match = SHARD_PATTERN.search(rel)
                if match is None:
                    continue
                shards.append(
                    ShardRef(
                        path=tar_path,
                        factory=match.group("factory"),
                        worker=match.group("worker"),
                    )
                )
    return shards


def stream_shard(shard: ShardRef) -> Iterator[Sample]:
    """Yield paired .mp4/.json samples from one shard.

    Tar members arrive in stream order, so pairs are buffered by key until
    both halves are present. WebDataset writes each pair contiguously, so
    the buffer holds at most a couple of entries in practice.
    """
    pending: dict[str, dict] = {}
    fs = _fs()

    with fs.open(shard.path, "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                name = member.name
                key, _, ext = name.rpartition(".")
                if ext not in ("mp4", "json"):
                    continue

                payload = tar.extractfile(member)
                if payload is None:
                    continue
                blob = payload.read()

                slot = pending.setdefault(key, {})
                slot[ext] = blob

                if "mp4" in slot and "json" in slot:
                    try:
                        meta = json.loads(slot["json"].decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        meta = {}
                    yield Sample(key=key, video=slot["mp4"], meta=meta)
                    del pending[key]


def open_video(sample: Sample) -> io.BytesIO:
    """Wrap the in-memory video bytes for a decoder that wants a file object."""
    return io.BytesIO(sample.video)
