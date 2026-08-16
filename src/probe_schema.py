"""Read the evaluation parquet's schema and first rows over HTTP range reads.

Only the footer and one row group are fetched, so this costs a few MB
instead of the full 1.8 GB. The question being answered: do the open
evaluation mirrors carry a factory label, and do they span more than one
factory? If yes, the cross-factory control runs without a token.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

fs = HfFileSystem()
TARGET = "datasets/builddotai/Egocentric-10K-Evaluation/egocentric_10k.parquet"

print(f"=== {TARGET}")
try:
    with fs.open(TARGET, "rb") as handle:
        pf = pq.ParquetFile(handle)
        print(f"  rows={pf.metadata.num_rows}  row_groups={pf.metadata.num_row_groups}")
        print("  schema:")
        for field in pf.schema_arrow:
            print(f"    {field.name}: {field.type}")

        light = [
            f.name
            for f in pf.schema_arrow
            if not str(f.type).startswith("binary") and "image" not in f.name.lower()
        ]
        print(f"\n  reading row group 0, columns={light}")
        table = pf.read_row_group(0, columns=light)
        df = table.to_pandas()

    print(f"  got {len(df)} rows")
    for col in df.columns:
        vals = df[col].astype(str)
        print(f"    {col}: {vals.nunique()} unique | e.g. {vals.head(2).tolist()}")

    joined = " ".join(df[c].astype(str).head(3000).tolist() for c in df.columns)
    factories = Counter(re.findall(r"factory[_]?(\d{2,3})", joined))
    print(f"\n  FACTORY IDS FOUND: {dict(factories.most_common(20))}")
except Exception as exc:  # noqa: BLE001
    print(f"  ERROR {type(exc).__name__}: {str(exc)[:300]}")

# Small sidecar files from the Voxel51 mirror.
for name in ("metadata.json", "fiftyone.yml"):
    print(f"\n=== Voxel51/Egocentric_10K_Evaluation/{name}")
    try:
        path = hf_hub_download(
            "Voxel51/Egocentric_10K_Evaluation", name, repo_type="dataset"
        )
        text = open(path, "r", encoding="utf-8", errors="replace").read()[:3000]
        print(text[:1500])
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {str(exc)[:200]}")
