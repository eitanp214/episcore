"""List file sizes in the open mirrors WITHOUT downloading anything.

A blind hf_hub_download on this repo pulled a 19 GB blob and nearly filled
the system drive. Check sizes first, then fetch only what is small enough
to be worth fetching.
"""

from __future__ import annotations

from huggingface_hub import HfApi

api = HfApi()

REPOS = [
    "builddotai/Egocentric-10K-Evaluation",
    "Voxel51/Egocentric_10K_Evaluation",
    "Voxel51/Egocentric_10K_subset",
]

for repo in REPOS:
    print(f"\n=== {repo}")
    try:
        info = api.dataset_info(repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:160]}")
        continue

    entries = []
    for sib in info.siblings or []:
        size = sib.size or (sib.lfs.size if sib.lfs else None) or 0
        if not sib.rfilename.endswith((".jpg", ".png", ".mp4")):
            entries.append((size, sib.rfilename))

    for size, name in sorted(entries, reverse=True)[:14]:
        print(f"  {size / 1e6:10.1f} MB  {name}")
