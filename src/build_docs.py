"""Wrap the findings page as a standalone document for GitHub Pages.

build_page.py emits a fragment, because the Artifact host supplies the
<html>/<head>/<body> shell at publish time. Served directly from Pages there
is no host, so the shell has to be real: doctype, charset, viewport, and the
social-preview tags that decide what a shared link looks like.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR, PROJECT_ROOT  # noqa: E402

REPO = "https://github.com/eitanp214/episcore"
PAGE = "https://eitanp214.github.io/episcore/"

TITLE = "What fraction of a video corpus is content it already contains?"
DESC = (
    "Measured corpus redundancy on 10,000 hours of open egocentric factory "
    "video: 27.2% of sampled frames duplicate earlier content, with a 6.2x "
    "spread between individual operators."
)


def main() -> int:
    fragment = (OUT_DIR / "findings.html").read_text(encoding="utf-8")

    # The fragment leads with its own <title>; the shell owns that now.
    fragment = re.sub(r"^\s*<title>.*?</title>\s*", "", fragment, count=1, flags=re.S)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<link rel="canonical" href="{PAGE}">

<meta property="og:type" content="article">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{PAGE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESC}">

<link rel="icon" href="data:image/svg+xml,\
%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E\
%3Ctext y='.9em' font-size='90'%3E%F0%9F%8E%9E%EF%B8%8F%3C/text%3E%3C/svg%3E">
</head>
<body>
{fragment}
</body>
</html>
"""

    dest = PROJECT_ROOT / "docs" / "index.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(doc, encoding="utf-8")

    # Pages runs Jekyll by default, which skips files and dirs starting with
    # an underscore and can rewrite content. This corpus has neither, but the
    # marker costs nothing and removes a whole class of surprise.
    (dest.parent / ".nojekyll").write_text("", encoding="utf-8")

    print(f"wrote {dest}  ({len(doc) / 1024:.1f} KB)")
    print(f"      {dest.parent / '.nojekyll'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
