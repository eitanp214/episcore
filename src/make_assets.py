"""Downscale the calibration pairs and emit them as base64 for the page.

The published page has to be self-contained -- no external image hosts -- so
the two pairs that carry the blur argument are embedded directly.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR  # noqa: E402

PAIRS_DIR = OUT_DIR / "pairs"
TARGET_WIDTH = 900
QUALITY = 74

WANTED = {
    "genuine_high": "sim0.956_3014_3013.jpg",
    "blur_false": "sim0.906_446_192.jpg",
    "genuine_low": "sim0.872_7584_7558.jpg",
}


def encode(path: Path) -> tuple[str, int]:
    img = Image.open(path).convert("RGB")
    ratio = TARGET_WIDTH / img.width
    img = img.resize((TARGET_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY, optimize=True)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


out = {}
total = 0
for key, name in WANTED.items():
    path = PAIRS_DIR / name
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        continue
    b64, size = encode(path)
    out[key] = b64
    total += size
    print(f"{key:14s} {name:28s} {size / 1024:6.1f} KB")

dest = OUT_DIR / "page_assets.json"
dest.write_text(json.dumps(out), encoding="utf-8")
print(f"\ntotal {total / 1024:.1f} KB raw -> {dest}")
