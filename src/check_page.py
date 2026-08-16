"""Sanity-check the generated page: valid JSON-LD, no theme-only colours."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR  # noqa: E402

html = (OUT_DIR / "findings.html").read_text(encoding="utf-8")
print(f"page size: {len(html) / 1024:.1f} KB")

# --- structured data -----------------------------------------------------
match = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
if not match:
    print("FAIL: no JSON-LD block")
    raise SystemExit(1)

data = json.loads(match.group(1))
types = [node["@type"] for node in data["@graph"]]
faq = next(n for n in data["@graph"] if n["@type"] == "FAQPage")
print(f"JSON-LD valid: {types}")
print(f"FAQ entries: {len(faq['mainEntity'])}")
for entry in faq["mainEntity"]:
    print(f"  - {entry['name']}")

# --- theme safety --------------------------------------------------------
# A colour whose ONLY definition sits inside a media/[data-theme] block never
# applies in the un-stamped default state, which is the classic unreadable
# artifact bug. Tokens are declared on bare :root, so check that holds.
root_block = re.search(r":root \{(.*?)\}", html, re.S)
tokens_bare = set(re.findall(r"--([\w-]+):", root_block.group(1))) if root_block else set()

dark_media = re.search(
    r"@media \(prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme=\"light\"\]\) \{(.*?)\}",
    html, re.S)
tokens_dark = set(re.findall(r"--([\w-]+):", dark_media.group(1))) if dark_media else set()

stamped = re.search(r':root\[data-theme="dark"\] \{(.*?)\}', html, re.S)
tokens_stamped = set(re.findall(r"--([\w-]+):", stamped.group(1))) if stamped else set()

print(f"\ntokens on bare :root      {len(tokens_bare)}")
print(f"tokens in dark media      {len(tokens_dark)}")
print(f"tokens in [data-theme]    {len(tokens_stamped)}")

orphans = (tokens_dark | tokens_stamped) - tokens_bare
if orphans:
    print(f"FAIL: defined only in a theme block: {sorted(orphans)}")
    raise SystemExit(1)
if tokens_dark != tokens_stamped:
    print(f"WARN: media and [data-theme] blocks disagree: "
          f"{sorted(tokens_dark ^ tokens_stamped)}")

body_bg = re.search(r"body \{[^}]*background: var\(--ground\)", html, re.S)
print(f"body paints explicit background: {bool(body_bg)}")
if not body_bg:
    raise SystemExit(1)

print("\nPAGE OK")
