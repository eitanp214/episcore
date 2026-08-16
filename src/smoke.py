"""Plumbing check: listing -> streaming -> decode -> embed, on one shard.

Verifies the repo layout assumptions before committing to a long run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import Embedder  # noqa: E402
from frames import sample_frames  # noqa: E402
from shards import list_shards, stream_shard  # noqa: E402


def main() -> int:
    print("1. listing shards ...", flush=True)
    t0 = time.time()
    shards = list_shards(max_factories=3, shards_per_worker=1, workers_per_factory=1)
    print(f"   {len(shards)} shards in {time.time() - t0:.1f}s")
    for s in shards[:5]:
        print(f"   f{s.factory}/w{s.worker}  {s.name}")
    if not shards:
        print("   FAIL: no shards — repo layout assumption is wrong", file=sys.stderr)
        return 1

    print("\n2. streaming first shard ...", flush=True)
    t0 = time.time()
    samples = []
    for sample in stream_shard(shards[0]):
        samples.append(sample)
        if len(samples) >= 2:
            break
    print(f"   {len(samples)} samples in {time.time() - t0:.1f}s")
    if not samples:
        print("   FAIL: shard yielded nothing", file=sys.stderr)
        return 1
    first = samples[0]
    print(f"   key={first.key}")
    print(f"   video={len(first.video) / 1e6:.1f} MB")
    print(f"   meta={first.meta}")

    print("\n3. decoding keyframes ...", flush=True)
    t0 = time.time()
    frames = sample_frames(first.video)
    print(f"   {len(frames)} frames in {time.time() - t0:.1f}s")
    if len(frames) == 0:
        print("   FAIL: decoder produced nothing", file=sys.stderr)
        return 1
    print(f"   size={frames.images[0].size}")
    print(f"   timestamps={[round(t, 1) for t in frames.timestamps[:8]]}")

    print("\n4. embedding ...", flush=True)
    t0 = time.time()
    embedder = Embedder()
    print(f"   device={embedder.device}")
    vecs = embedder.embed(frames.images)
    dt = time.time() - t0
    print(f"   {vecs.shape} in {dt:.1f}s")

    norms = (vecs.astype("float32") ** 2).sum(axis=1) ** 0.5
    print(f"   norms min={norms.min():.4f} max={norms.max():.4f} (expect ~1.0)")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
