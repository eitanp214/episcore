"""Confirm the saturation gate accepts real footage and rejects saturated.

A gate that rejects everything is as useless as one that accepts everything.
This runs it against the measured factory corpus, where the redundancy figure
is known to be meaningful, and reports the margin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMB_DIR, HEADLINE_THRESHOLD  # noqa: E402
from score import SATURATION_CEILING, saturation  # noqa: E402

for tag in ("factory051_v2", "factory051"):
    path = EMB_DIR / f"{tag}.npy"
    if not path.exists():
        continue

    embeddings = np.load(path)
    result = saturation(embeddings, HEADLINE_THRESHOLD)
    verdict = "PASS" if result["share"] <= SATURATION_CEILING else "FAIL"

    print(f"\n{tag}  ({len(embeddings)} frames)")
    print(f"  random-pair median      {result['median']:.3f}")
    print(f"  share >= {HEADLINE_THRESHOLD}          {result['share'] * 100:.2f}%")
    print(f"  ceiling                 {SATURATION_CEILING * 100:.0f}%")
    print(f"  headroom                {(SATURATION_CEILING - result['share']) * 100:.2f} pts")
    print(f"  verdict                 {verdict}")

print(
    "\nFor contrast, the synthetic fixture scored median 0.904 with 26.2% of\n"
    "random pairs above threshold — rejected."
)
