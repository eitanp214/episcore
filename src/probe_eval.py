"""Look for an OPEN path to a multi-factory sample.

The cross-factory control is what blocks publication, and the main repo is
gated. Both evaluation mirrors are ungated -- if either spans more than one
factory, the control can run without any token at all.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()


def inspect_parquet(repo: str, filename: str) -> None:
    print(f"\n--- {repo}/{filename}")
    try:
        path = hf_hub_download(repo, filename, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed: {type(exc).__name__}: {str(exc)[:160]}")
        return

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  parse failed: {type(exc).__name__}: {str(exc)[:160]}")
        return

    print(f"  rows={len(df)}  columns={list(df.columns)}")
    for col in df.columns:
        lowered = col.lower()
        if any(k in lowered for k in ("factory", "worker", "video", "clip", "path", "id")):
            vals = df[col].astype(str)
            print(f"  {col}: {vals.nunique()} unique, e.g. {vals.head(3).tolist()}")


def inspect_json(repo: str, filename: str) -> None:
    print(f"\n--- {repo}/{filename}")
    try:
        path = hf_hub_download(repo, filename, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed: {type(exc).__name__}: {str(exc)[:160]}")
        return

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        head = handle.read(400_000)

    import re

    tags = Counter(re.findall(r"factory[_]?(\d{2,3})", head))
    print(f"  read {len(head)} chars; factory ids seen: {dict(tags.most_common(12))}")
    try:
        blob = json.loads(head)
        if isinstance(blob, dict):
            print(f"  top-level keys: {list(blob.keys())[:15]}")
    except json.JSONDecodeError:
        print("  (truncated read, not parsed as JSON)")


inspect_parquet("builddotai/Egocentric-10K-Evaluation", "egocentric_10k.parquet")

for name in ("samples.json", "metadata.json", "fiftyone.yml"):
    inspect_json("Voxel51/Egocentric_10K_Evaluation", name)
