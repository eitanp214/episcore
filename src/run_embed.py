"""Stream shards, sample keyframes, embed them, persist vectors + provenance.

Usage:
    python src/run_embed.py --factories 20 --workers 2 --shards 1

Nothing but embeddings and metadata touches the disk. Raw video is held in
memory only for as long as it takes to decode, which is what makes an 18 TB
corpus workable on a laptop.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, OUT_DIR  # noqa: E402
from embed import Embedder  # noqa: E402
from frames import sample_frames  # noqa: E402
from shards import list_shards, stream_shard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed Egocentric-10K keyframes.")
    parser.add_argument(
        "--source",
        choices=("subset", "gated"),
        default="subset",
        help="'subset' = open Voxel51 mirror; 'gated' = full repo (needs HF token)",
    )
    parser.add_argument("--factories", type=int, default=10, help="gated source only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shards", type=int, default=1, help="gated source only")
    parser.add_argument(
        "--factory-offset", type=int, default=0,
        help="skip the first N factories (gated source) - for resuming",
    )
    parser.add_argument("--max-clips", type=int, default=None, help="cap per unit")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap keyframes decoded per clip; decoding is the bottleneck on "
             "long clips, so this trades sample depth for wall-clock directly",
    )
    parser.add_argument(
        "--worker-offset",
        type=int,
        default=0,
        help="skip the first N workers (subset source) — for resuming a dead run",
    )
    parser.add_argument("--tag", type=str, default="run1")
    return parser.parse_args()


def _persist(vectors: list[np.ndarray], rows: list[dict], tag: str):
    """Write embeddings + manifest atomically, returning both and their paths.

    Both files are staged under temporary names and only swapped into place
    once both have been written. Vectors without their manifest have no
    provenance and are unusable, so a checkpoint interrupted between the two
    writes must leave the previous good pair intact rather than a half-updated
    pair -- which is exactly what an earlier non-atomic version produced.
    """
    matrix = np.concatenate(vectors, axis=0)
    manifest = pd.DataFrame(rows)
    emb_path = EMB_DIR / f"{tag}.npy"
    man_path = EMB_DIR / f"{tag}.parquet"

    tmp_emb = emb_path.with_suffix(".npy.tmp")
    tmp_man = man_path.with_suffix(".parquet.tmp")
    # np.save appends '.npy' unless the path already ends in it, which would
    # write to <tag>.npy.tmp.npy and leave the rename below with no source.
    # Handing it an open file object suppresses that.
    with open(tmp_emb, "wb") as handle:
        np.save(handle, matrix)
    manifest.to_parquet(tmp_man, index=False)

    tmp_emb.replace(emb_path)
    tmp_man.replace(man_path)
    return matrix, manifest, emb_path, man_path


def resolve_source(args: argparse.Namespace):
    """Return (units, stream_fn) for the chosen source.

    Both sources expose units carrying .factory / .worker / .name, so the
    embedding loop below does not care which one it is reading.
    """
    if args.source == "subset":
        from subset_source import list_units, stream_unit

        units = list_units(
            max_workers=args.workers,
            clips_per_worker=args.max_clips,
            worker_offset=args.worker_offset,
        )
        return units, stream_unit

    units = list_shards(
        max_factories=args.factories,
        shards_per_worker=args.shards,
        workers_per_factory=args.workers,
        factory_offset=args.factory_offset,
    )
    return units, stream_shard


def main() -> int:
    args = parse_args()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"listing units from '{args.source}' source...", flush=True)
    shards, stream_fn = resolve_source(args)
    print(f"  {len(shards)} units across "
          f"{len({s.factory for s in shards})} factories, "
          f"{len({(s.factory, s.worker) for s in shards})} workers", flush=True)
    if not shards:
        print("no units found — check repo layout / access", file=sys.stderr)
        return 1

    embedder = Embedder()
    print(f"embedder on {embedder.device}", flush=True)

    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    started = time.time()

    for n, shard in enumerate(shards, start=1):
        clip_count = 0
        frame_count = 0
        shard_start = time.time()

        # islice, not a break inside the loop: the generator downloads the
        # next clip before control returns, so checking the cap after the
        # yield pulls one extra 200 MB clip per unit and throws it away.
        source = stream_fn(shard)
        if args.max_clips:
            source = islice(source, args.max_clips)

        try:
            for sample in source:
                frames = (
                    sample_frames(sample.video, max_frames=args.max_frames)
                    if args.max_frames
                    else sample_frames(sample.video)
                )
                clip_count += 1
                if len(frames) == 0:
                    continue

                vecs = embedder.embed(frames.images)
                vectors.append(vecs)
                for pos, (ts, sharp) in enumerate(
                    zip(frames.timestamps, frames.sharpness)
                ):
                    rows.append(
                        {
                            "factory": shard.factory,
                            "worker": shard.worker,
                            "shard": shard.name,
                            "clip": sample.key,
                            "frame_pos": pos,
                            "timestamp_sec": ts,
                            "sharpness": sharp,
                            "duration_sec": sample.meta.get("duration_sec"),
                        }
                    )
                frame_count += len(frames)
        except Exception as exc:  # noqa: BLE001 - one bad shard must not kill a long run
            print(f"  [{n}/{len(shards)}] {shard.name} FAILED: {exc}", flush=True)
            continue

        elapsed = time.time() - shard_start
        print(
            f"  [{n}/{len(shards)}] {shard.name} "
            f"f{shard.factory}/w{shard.worker}: "
            f"{clip_count} clips, {frame_count} frames, {elapsed:.0f}s",
            flush=True,
        )

        # Checkpoint after every unit. A long run is network-bound and can
        # die on any single clip; losing an hour of embedding to that would
        # be avoidable waste. Rewriting ~10 MB each pass costs nothing.
        if vectors:
            _persist(vectors, rows, args.tag)

    if not vectors:
        print("no frames embedded", file=sys.stderr)
        return 1

    matrix, manifest, emb_path, man_path = _persist(vectors, rows, args.tag)

    print(
        f"\n{matrix.shape[0]} frames from {manifest['clip'].nunique()} clips, "
        f"{manifest['worker'].nunique()} workers, "
        f"{manifest['factory'].nunique()} factories "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    print(f"  {emb_path}\n  {man_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
