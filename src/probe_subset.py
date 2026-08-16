"""Inspect the open Voxel51 mirrors: size, coverage, and whether the
evaluation split spans multiple factories (needed as a control group).
"""

import re
from collections import Counter

from huggingface_hub import HfApi

api = HfApi()


def summarise(repo: str) -> None:
    print(f"\n=== {repo}")
    info = api.dataset_info(repo, files_metadata=True)
    sizes = {}
    for sib in info.siblings or []:
        size = sib.size or (sib.lfs.size if sib.lfs else None) or 0
        sizes[sib.rfilename] = size

    total = sum(sizes.values())
    print(f"  files={len(sizes)}  total={total / 1e9:.2f} GB")

    media = {k: v for k, v in sizes.items() if k.endswith((".mp4", ".jpg", ".png"))}
    if media:
        vals = sorted(media.values())
        print(f"  media files={len(media)}  median={vals[len(vals) // 2] / 1e6:.1f} MB")

    # factoryNNN_workerNNN pattern anywhere in the path
    tags = Counter()
    for name in sizes:
        m = re.search(r"factory(\d+)_worker(\d+)", name)
        if m:
            tags[(m.group(1), m.group(2))] += 1
    if tags:
        print(f"  factory/worker combos={len(tags)}")
        for (f, w), n in list(tags.most_common(8)):
            print(f"    factory{f}/worker{w}: {n} files")
    else:
        print("  no factory/worker tags in filenames")

    non_media = [k for k in sizes if not k.endswith((".mp4", ".jpg", ".png"))]
    print(f"  non-media: {non_media[:12]}")


for repo in [
    "Voxel51/Egocentric_10K_subset",
    "builddotai/Egocentric-10K-Evaluation",
]:
    try:
        summarise(repo)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:200]}")
