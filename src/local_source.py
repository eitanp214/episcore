"""Read a local directory of video files as a scoreable corpus.

The research pipeline reads one dataset's Hugging Face layout. A customer
has a folder. This adapts a folder into the same Sample stream, so the
measurement code is unchanged.

Operator grouping is the important part. The headline finding is that
redundancy varies ~6x between operators, so a report that cannot tell them
apart is not worth producing. Grouping is inferred from directory structure
by default and can be overridden with a filename pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from shards import Sample

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass
class LocalUnit:
    """One operator's clips."""

    factory: str
    worker: str
    paths: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.factory}/{self.worker}" if self.factory != "-" else self.worker


def _iter_videos(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            yield path


def discover(
    root: Path,
    pattern: str | None = None,
    clips_per_operator: int | None = None,
) -> list[LocalUnit]:
    """Group a directory tree into operator units.

    Three strategies, in order of preference:

    1. `pattern` — a regex with a named group `operator` matched against the
       filename. Use when operator identity lives in the name.
    2. Immediate subdirectories of `root` that contain video — one operator
       per directory. This is the common layout.
    3. Everything in one bucket, when neither applies. The report then says
       so rather than inventing groups, because a single-bucket score hides
       exactly the variation that matters.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    groups: dict[str, list[Path]] = {}

    if pattern:
        rx = re.compile(pattern)
        for path in _iter_videos(root):
            match = rx.search(path.name)
            key = match.group("operator") if match and "operator" in (match.groupdict() or {}) else "unmatched"
            groups.setdefault(key, []).append(path)
    else:
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        nested = {d.name: list(_iter_videos(d)) for d in subdirs}
        nested = {k: v for k, v in nested.items() if v}

        if len(nested) >= 2:
            groups = nested
        else:
            found = list(_iter_videos(root))
            if found:
                groups = {"all": found}

    units = [
        LocalUnit(factory="-", worker=name, paths=sorted(paths)[:clips_per_operator])
        for name, paths in sorted(groups.items())
        if paths
    ]
    return units


def stream_unit(unit: LocalUnit) -> Iterator[Sample]:
    """Yield each clip's bytes. Unreadable files are skipped, not fatal."""
    for path in unit.paths:
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        yield Sample(
            key=path.stem,
            video=blob,
            meta={"path": str(path), "operator": unit.worker,
                  "size_bytes": path.stat().st_size},
        )


def describe(units: list[LocalUnit]) -> str:
    """One-line summary used in the report header."""
    clips = sum(len(u.paths) for u in units)
    if len(units) == 1 and units[0].worker == "all":
        return (f"{clips} clips, operator grouping NOT detected — "
                f"per-operator breakdown unavailable")
    return f"{clips} clips across {len(units)} operators"
